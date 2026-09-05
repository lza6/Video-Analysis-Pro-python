"""监控视频画面变化检测引擎 (M1)

针对监控场景的核心优化：长时间无变化的空走廊/空房间不该整段送 AI（NVIDIA
Nemotron Omni 每 2min 分片是昂贵调用）。本模块用本地算法（1fps 抽帧 +
scenedetect 场景切分 + 帧差分 + 昼夜自适应阈值）先找出"画面有变化"的时段，
只对这些时段加 ±N 秒上下文后切片送 AI，跳过空段，省 90%+ AI 调用。

算法链路
  1. _sample_frames      : ffmpeg 1fps 抽帧到临时目录（监控 H.264 流 PPS 头
                           问题让 cv2.VideoCapture 偶发解析不到帧数，ffmpeg
                           fast seek 更鲁棒，复用 surveillance_agent 的调用形状）
  2. _detect_day_night   : 抽样帧灰度均值判昼/夜（亮度>50=昼），返回每帧 day/night
  3. _frame_diff_score   : 相邻帧 cv2.absdiff 灰度均值 → 变化分
  4. 阈值过滤            : 夜间用 night_threshold（更敏感，低噪点放大），昼间用 day_threshold
  5. scenedetect         : AdaptiveDetector 在 1fps 帧上找场景边界（额外补强，
                           帧差分可能漏掉缓变场景）
  6. _merge_to_segments  : 合并相邻变化点为时段，加 context_padding 上下文
  7. _clamp_segment      : 段长不超 NVIDIA 2min 上限，超了再切

与 batch_runner 的接缝
  - BatchRunner._segment_video 不再固定 120s 切，改为调 MotionDetector.detect
    得 MotionSegment 列表，只对这些时段切 mp4 分片送 AI。
  - 无变化时段的视频 detect 返回 []，batch_runner 直接标 done（0 片段，跳过）。

不依赖 torch / sentence_transformers / NVIDIA API，纯本地 cv2+scenedetect+ffmpeg。
scenedetect 缺失时降级为纯帧差分（仍可工作，只是场景切分粗一些）。
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("VideoAnalyzerCore")

# NVIDIA Nemotron Omni 硬上限：mp4≤2min 720p 2fps/256 帧
NVIDIA_MAX_SEGMENT_SEC = 120

# 亮度阈值：灰度均值 > 此值判为昼，否则夜（经验值，监控室内外通用）
DAY_NIGHT_BRIGHTNESS_THRESHOLD = 50

# v5.9：帧文件名时间戳解析（CrowdedSceneDetector 按帧文件找 ts）
import re as _re
_TS_RE = _re.compile(r"f\d{6}_(\d+(?:\.\d+)?)\.jpg$", _re.IGNORECASE)


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class MotionConfig:
    """变化检测配置。

    Attributes:
        sample_fps:         抽帧频率，1.0=1fps（监控场景足够，省算力）
        min_scene_len:      scenedetect AdaptiveDetector 的 min_scene_len（帧数，
                            1fps 下 15 帧=15 秒，过滤秒级抖动）
        day_threshold:      昼间帧差分阈值（变化分>此值判为变化帧）
        night_threshold:    夜间帧差分阈值（更敏感，低光照噪点放大用更敏感阈值）
        context_padding:    变化时段前后加的上下文秒数（让 AI 看到变化前后）
        max_segment_sec:    单段最大秒数，超了再切（对齐 NVIDIA 2min 上限）
        max_frames:         抽帧上限（防爆内存，4 小时视频 1fps=14400 帧）
        brightness_threshold: 灰度均值>此值判昼，否则夜
    """
    sample_fps: float = 1.0
    min_scene_len: int = 15
    day_threshold: float = 15.0   # 昼间：调高防噪点（原 8.0 太敏感，实测把光线微变误判为变化）
    night_threshold: float = 6.0  # 夜间：调高防低光噪点放大（原 3.0）
    context_padding: float = 10.0
    max_segment_sec: int = NVIDIA_MAX_SEGMENT_SEC
    max_frames: int = 14400
    brightness_threshold: int = DAY_NIGHT_BRIGHTNESS_THRESHOLD
    # v5.7：帧持久化目录。非空时 _sample_frames 直接落盘到该目录（不落临时目录），
    # detect() 的 finally 不删该目录，调用方拿到按时间序排列的帧文件用于拼长图证据。
    # 空时（旧调用/单测）：行为不变，临时目录用完即删（零回归）。
    frame_out_dir: Optional[str] = None
    # v5.9 I5.9-skills-1：crowded-scene 密度阈值。变化点密度（变化帧数/总帧数）
    # > 此值时启用 YOLO 物体类别聚类去重（过滤"人来回走动"的重复变化）。
    # 默认 0.6 = 60% 帧有变化才判密集场景。CrowdedSceneDetector 用。
    crowded_density_threshold: float = 0.6


@dataclass
class MotionSegment:
    """一个画面变化时段。

    Attributes:
        start_sec:    时段起始秒（含 padding，已 clamp 到 [0, duration]）
        end_sec:      时段结束秒（含 padding，已 clamp）
        duration:     end_sec - start_sec
        brightness:   'day' / 'night' / 'mixed'（时段内帧亮度分布）
        diff_score:   时段内最大帧差分值（变化强度，越大越显著）
        scene_count:  时段内场景切分数（scenedetect 给的边界数）
        change_points: 触发该时段的变化帧时间戳列表（诊断用）
    """
    start_sec: float
    end_sec: float
    duration: float
    brightness: str
    diff_score: float
    scene_count: int
    change_points: List[float] = field(default_factory=list)


# ----------------------------------------------------------------------
# 检测器
# ----------------------------------------------------------------------
class MotionDetector:
    """监控视频画面变化检测器。

    用法：
        detector = MotionDetector(MotionConfig())
        segments = detector.detect(Path("监控.mp4"))
        # segments 为空 → 整段无变化，跳过 AI
        # segments 非空 → 只对这些时段切片送 AI
    """

    def __init__(self, config: Optional[MotionConfig] = None,
                 ffmpeg_exe: Optional[str] = None) -> None:
        self.config = config or MotionConfig()
        self._ffmpeg = ffmpeg_exe or self._find_ffmpeg()
        if not self._ffmpeg:
            logger.warning("[motion] 未找到 ffmpeg，抽帧将失败")
        # v5.7：帧持久化目录。非空时 _sample_frames 直接落盘到此处，
        # detect() 的 finally 不删它（供 batch_runner 拼长图证据）。空时走旧路径。
        self._frame_out_dir: Optional[Path] = (
            Path(self.config.frame_out_dir) if self.config.frame_out_dir else None
        )

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def detect(self, video_path: Path) -> List[MotionSegment]:
        """检测视频中的画面变化时段。

        Returns:
            MotionSegment 列表（按 start_sec 升序）。无变化时返回 []。
        """
        if not self._ffmpeg:
            logger.error("[motion] 无 ffmpeg，无法抽帧")
            return []
        if not Path(video_path).exists():
            logger.error(f"[motion] 视频不存在: {video_path}")
            return []

        duration = self._probe_duration(video_path)
        if duration <= 0:
            logger.warning(f"[motion] 读不到时长: {video_path.name}")
            return []
        if self._cancel_quick_check(video_path, duration):
            logger.info(f"[motion] {video_path.name} 太短，整段送 AI 不切")
            return [MotionSegment(
                start_sec=0.0, end_sec=duration, duration=duration,
                brightness="day", diff_score=0.0, scene_count=0,
                change_points=[],
            )]

        # 1fps 抽帧——frame_out_dir 非空时落到持久目录（v5.7 长图证据），
        # 否则落临时目录（旧路径，detect 返回前 rmtree 删）。
        tmp_dir: Optional[Path] = None
        if self._frame_out_dir is not None:
            self._frame_out_dir.mkdir(parents=True, exist_ok=True)
            frame_dir = self._frame_out_dir
        else:
            tmp_dir = Path(tempfile.mkdtemp(prefix="motion_"))
            frame_dir = tmp_dir
        try:
            frames = self._sample_frames(video_path, frame_dir, duration)
            if len(frames) < 2:
                logger.info(f"[motion] {video_path.name} 抽帧不足 2 张，整段送 AI")
                return [MotionSegment(
                    start_sec=0.0, end_sec=duration, duration=duration,
                    brightness="day", diff_score=0.0, scene_count=0,
                    change_points=[],
                )]

            # 昼夜判断（每帧）
            day_night_labels = self._detect_day_night(frames)

            # 帧差分序列
            diff_scores, ts_list = self._compute_diff_series(frames)

            # 场景切分（scenedetect 在 1fps 帧上跑 AdaptiveDetector，补强缓变）
            scene_boundaries = self._detect_scenes(video_path, frames)

            # 合并变化点为时段
            segments = self._merge_to_segments(
                diff_scores=diff_scores,
                timestamps=ts_list,
                day_night=day_night_labels,
                scene_boundaries=scene_boundaries,
                duration=duration,
            )
            logger.info(
                f"[motion] {video_path.name} 检测到 {len(segments)} 个变化时段"
                f"（抽帧 {len(frames)} 帧，时长 {duration:.0f}s）")
            return segments
        finally:
            # 只删临时目录；frame_out_dir（持久帧）留给调用方拼长图，不删。
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _cancel_quick_check(self, video_path: Path, duration: float) -> bool:
        """视频太短直接整段送 AI 不切（变化检测对 <30s 视频无意义）。"""
        return duration < self.config.min_scene_len * 2

    # ------------------------------------------------------------------
    # 抽帧
    # ------------------------------------------------------------------
    def _sample_frames(self, video: Path, out_dir: Path,
                      duration: float) -> List[Tuple[float, str]]:
        """ffmpeg 1fps 抽帧到 out_dir，返回 [(timestamp, path)] 升序。

        复用 surveillance_agent._extract_frames 的 ffmpeg fast seek 调用形状。
        每帧 -ss 单独 seek（监控流 seek 到任意点比 select 滤镜更鲁棒）。

        v5.7：out_dir 若为持久 frame_out_dir（非临时），已存在的帧直接复用
        （跳过重复抽帧，支持长图重建/断点续跑不重抽）。
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        fps = self.config.sample_fps
        if fps <= 0:
            fps = 1.0
        interval = 1.0 / fps
        n_targets = min(self.config.max_frames,
                        int(duration / interval) + 1)
        frames: List[Tuple[float, str]] = []
        for i in range(n_targets):
            ts = i * interval
            if ts >= duration:
                break
            fp = out_dir / f"f{i:06d}_{ts:.1f}.jpg"
            # 持久目录下已存在的帧复用（断点续跑/重建长图不重抽，省 IO）
            if fp.exists():
                frames.append((ts, str(fp)))
                continue
            cmd = [self._ffmpeg, "-y", "-ss", f"{ts:.3f}",
                   "-i", str(video), "-frames:v", "1",
                   "-q:v", "3", str(fp)]
            try:
                subprocess.run(cmd, capture_output=True, timeout=15)
                if fp.exists():
                    frames.append((ts, str(fp)))
            except subprocess.TimeoutExpired:
                logger.debug(f"[motion] 抽帧 {ts}s 超时")
        return frames

    # ------------------------------------------------------------------
    # 昼夜判断
    # ------------------------------------------------------------------
    def _detect_day_night(self, frames: List[Tuple[float, str]]) -> List[str]:
        """按帧亮度均值判昼/夜。

        灰度均值 > brightness_threshold → 'day'，否则 'night'。
        监控场景夜间光照低（红外补光画面偏灰偏暗），阈值 50 经验值区分室内外。
        """
        import cv2
        out: List[str] = []
        for _ts, fp in frames:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                out.append("day")  # 读不到按昼处理（保守，不误判夜间为高噪点）
                continue
            mean_val = float(img.mean())
            out.append("day" if mean_val > self.config.brightness_threshold else "night")
        return out

    # ------------------------------------------------------------------
    # 帧差分
    # ------------------------------------------------------------------
    @staticmethod
    def _frame_diff_score(f1_path: str, f2_path: str) -> float:
        """相邻帧 cv2.absdiff 灰度均值 → 变化分。

        返回 0-255 范围的均值差（越小越相似）。监控静态画面 <2，有人经过 >10。
        """
        import cv2
        a = cv2.imread(f1_path, cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(f2_path, cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            return 0.0
        if a.shape != b.shape:
            # 尺寸不一致（极少见，但安全兜底）：resize 对齐
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        diff = cv2.absdiff(a, b)
        return float(diff.mean())

    def _compute_diff_series(
        self, frames: List[Tuple[float, str]]
    ) -> Tuple[List[float], List[float]]:
        """计算相邻帧差分序列。

        Returns:
            diff_scores: 长度 len(frames)-1，第 i 项是 frames[i] 与 frames[i+1] 的差分
            timestamps:  对应的帧时间戳（取前帧 ts，与 diff_scores 对齐）
        """
        diff_scores: List[float] = []
        ts_list: List[float] = []
        for i in range(len(frames) - 1):
            score = self._frame_diff_score(frames[i][1], frames[i + 1][1])
            diff_scores.append(score)
            ts_list.append(frames[i][0])
        return diff_scores, ts_list

    # ------------------------------------------------------------------
    # scenedetect 场景切分（补强）
    # ------------------------------------------------------------------
    def _detect_scenes(self, video: Path,
                       frames: List[Tuple[float, str]]) -> List[float]:
        """scenedetect AdaptiveDetector 在原视频上找场景边界，返回时间戳列表。

        帧差分对"渐变"（光线缓慢变化、缓慢移动）不敏感，scenedetect 的
        AdaptiveDetector 带自适应阈值能补。scenedetect 缺失时返回 []（降级为
        纯帧差分，仍可工作）。
        """
        try:
            from scenedetect import detect, AdaptiveDetector
        except ImportError:
            logger.debug("[motion] scenedetect 不可用，跳过场景切分")
            return []
        try:
            scene_list = detect(
                str(video),
                AdaptiveDetector(min_scene_len=self.config.min_scene_len),
                show_progress=False,
            )
        except Exception as e:
            logger.debug(f"[motion] scenedetect 失败，降级帧差分: {e}")
            return []
        # scene_list 是 [(start, end), ...]，FrameTimecode 对象
        boundaries: List[float] = []
        for start, _end in scene_list:
            try:
                ts = float(start.seconds if hasattr(start, "seconds") else start.get_seconds())
            except Exception:
                continue
            # 过滤 t≈0 的边界：scenedetect 总会把"视频开始"作为第一个场景
            # 边界返回，但视频开始不是"变化点"，混入会让整段空视频被误
            # 判为有变化。只保留 t > epsilon 的真实场景切分点。
            if ts > 0.5:
                boundaries.append(ts)
        return boundaries

    # ------------------------------------------------------------------
    # 合并变化点为时段
    # ------------------------------------------------------------------
    def _merge_to_segments(
        self,
        diff_scores: List[float],
        timestamps: List[float],
        day_night: List[str],
        scene_boundaries: List[float],
        duration: float,
    ) -> List[MotionSegment]:
        """合并相邻变化点为时段，加 padding 上下文，clamp 到视频时长。

        变化点来源：
          - 帧差分 > 该帧昼夜对应阈值
          - scenedetect 场景边界（额外补强）
        """
        cfg = self.config
        # 收集变化帧时间戳（去重 + 排序）
        change_ts: List[float] = []
        for i, score in enumerate(diff_scores):
            if i >= len(day_night):
                break
            label = day_night[i]
            threshold = cfg.day_threshold if label == "day" else cfg.night_threshold
            if score > threshold:
                change_ts.append(timestamps[i])
        # 场景边界也作为变化点（补强缓变）
        change_ts.extend(scene_boundaries)
        if not change_ts:
            return []
        change_ts = sorted(set(change_ts))

        # 合并：相邻变化点间隔 < min_scene_len 视为同一时段
        # （避免一秒内的多帧变化切成多段）
        groups: List[List[float]] = []
        current: List[float] = []
        gap = float(cfg.min_scene_len)
        for ts in change_ts:
            if not current:
                current.append(ts)
            elif ts - current[-1] <= gap:
                current.append(ts)
            else:
                groups.append(current)
                current = [ts]
        if current:
            groups.append(current)

        # 每个 group → MotionSegment（加 padding + clamp）
        segments: List[MotionSegment] = []
        for g in groups:
            start = max(0.0, g[0] - cfg.context_padding)
            end = min(duration, g[-1] + cfg.context_padding)
            if end <= start:
                continue
            # 时段内昼/夜分布
            label = self._group_brightness(g[0], g[-1], day_night, timestamps)
            # 时段内最大差分值（变化强度）
            max_score = self._max_score_in_range(
                g[0], g[-1], diff_scores, timestamps)
            # 时段内场景切分数（诊断）
            scene_count = sum(1 for b in scene_boundaries
                              if g[0] <= b <= g[-1])
            seg = MotionSegment(
                start_sec=start,
                end_sec=end,
                duration=end - start,
                brightness=label,
                diff_score=round(max_score, 2),
                scene_count=scene_count,
                change_points=g,
            )
            segments.extend(self._clamp_segment(seg))
        return segments

    @staticmethod
    def _group_brightness(start_ts: float, end_ts: float,
                          day_night: List[str],
                          timestamps: List[float]) -> str:
        """时段内帧昼/夜分布 → 'day'/'night'/'mixed'。"""
        has_day = False
        has_night = False
        for i, ts in enumerate(timestamps):
            if ts < start_ts or ts > end_ts:
                continue
            if i < len(day_night):
                if day_night[i] == "day":
                    has_day = True
                else:
                    has_night = True
        if has_day and has_night:
            return "mixed"
        return "day" if has_day else "night"

    @staticmethod
    def _max_score_in_range(start_ts: float, end_ts: float,
                            diff_scores: List[float],
                            timestamps: List[float]) -> float:
        """时段内最大帧差分值。"""
        out = 0.0
        for i, ts in enumerate(timestamps):
            if ts < start_ts or ts > end_ts:
                continue
            if i < len(diff_scores) and diff_scores[i] > out:
                out = diff_scores[i]
        return out

    def _clamp_segment(self, seg: MotionSegment) -> List[MotionSegment]:
        """段长不超 max_segment_sec，超了按等长切多段。

        保留首段含 change_points（诊断），后续段 change_points 为空。
        """
        max_sec = self.config.max_segment_sec
        if seg.duration <= max_sec:
            return [seg]
        out: List[MotionSegment] = []
        n = max(1, int(seg.duration // max_sec) +
                (1 if seg.duration % max_sec else 0))
        chunk = seg.duration / n
        for i in range(n):
            s = seg.start_sec + i * chunk
            e = seg.start_sec + (i + 1) * chunk
            out.append(MotionSegment(
                start_sec=round(s, 2),
                end_sec=round(e, 2),
                duration=round(e - s, 2),
                brightness=seg.brightness,
                diff_score=seg.diff_score if i == 0 else 0.0,
                scene_count=seg.scene_count if i == 0 else 0,
                change_points=seg.change_points if i == 0 else [],
            ))
        return out

    # ------------------------------------------------------------------
    # 工具（v6.0.1 修正：从 CrowdedSceneDetector 上移到 MotionDetector 父类，
    # 父类 __init__ 调 self._find_ffmpeg()、detect() 调 self._probe_duration()，
    # 之前这俩方法在子类导致父类实例 AttributeError。）
    # ------------------------------------------------------------------
    @staticmethod
    def _find_ffmpeg() -> Optional[str]:
        """定位 ffmpeg（优先 PATH，其次 imageio-ffmpeg 自带）。"""
        found = shutil.which("ffmpeg")
        if found:
            return found
        env_exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
        if env_exe and Path(env_exe).exists():
            return env_exe
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    def _probe_duration(self, video: Path) -> float:
        """读视频时长。优先 cv2，回退 ffprobe，最后 ffmpeg -i stderr 解析。

        与 batch_runner._probe_duration 同款三级回退（监控 H.264 流 PPS 头
        常让 cv2 读不到 frame count）。
        """
        try:
            import cv2
            cap = cv2.VideoCapture(str(video))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
                if total and fps and total > 0 and fps > 0:
                    return total / fps
        except Exception as e:
            logger.debug(f"[motion] cv2 读时长失败 {video.name}: {e}")
        ffprobe = shutil.which("ffprobe")
        if not ffprobe and self._ffmpeg:
            cand = Path(self._ffmpeg).with_name("ffprobe.exe")
            if cand.exists():
                ffprobe = str(cand)
        if ffprobe:
            try:
                cmd = [ffprobe, "-v", "error", "-show_entries",
                       "format=duration", "-of",
                       "default=noprint_wrappers=1:nokey=1", str(video)]
                out = subprocess.run(cmd, capture_output=True, timeout=15)
                txt = out.stdout.decode("utf-8", "ignore").strip()
                if txt and float(txt) > 0:
                    return float(txt)
            except Exception:
                pass
        if self._ffmpeg:
            try:
                cmd = [self._ffmpeg, "-hide_banner", "-i", str(video)]
                out = subprocess.run(cmd, capture_output=True, timeout=15)
                err = out.stderr.decode("utf-8", "ignore")
                m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
                if m:
                    h, mi, s = m.group(1), m.group(2), m.group(3)
                    return int(h) * 3600 + int(mi) * 60 + float(s)
            except Exception:
                pass
        return 0.0

    # ------------------------------------------------------------------
    # v5.9 I5.9-skills-1：crowded-scene YOLO 去重
    # ------------------------------------------------------------------
    def _detect_objects_yolo(self, frame_paths: List[str]) -> Optional[List[List[str]]]:
        """用 YOLO 检测每帧物体类别列表。

        返回每帧的物体类别集合（如 [['person'], ['person','backpack'], ...]）。
        无 ultralytics / 推理失败返回 None（调用方据此降级纯帧差分）。
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.debug("[motion] ultralytics 不可用，crowded 去重降级纯帧差分")
            return None
        try:
            model = YOLO("yolov8n.pt")  # 最小模型，CPU 也能跑
            results = model(frame_paths, verbose=False)
            out: List[List[str]] = []
            for r in results:
                # r.boxes.data: tensor[N,6] (x1,y1,x2,y2,conf,cls)
                classes = []
                if hasattr(r, "boxes") and r.boxes is not None:
                    try:
                        cls_ids = r.boxes.cls.tolist()
                        names = r.names  # {0:'person',1:'bicycle',...}
                        for cid in cls_ids:
                            classes.append(names.get(int(cid), str(cid)))
                    except Exception:
                        pass
                out.append(sorted(set(classes)))
            return out
        except Exception as e:
            logger.debug(f"[motion] YOLO 推理失败，降级纯帧差分: {e}")
            return None

    @staticmethod
    def _object_set_unchanged(prev_set: List[str],
                               curr_set: List[str]) -> bool:
        """两帧物体类别集合是否相同（去重判定：物体集合不变 = 重复变化）。"""
        return set(prev_set) == set(curr_set)


class CrowdedSceneDetector(MotionDetector):
    """人多密集场景监控检测器（v5.9 I5.9-skills-1）。

    场景：商场/路口/车站，画面持续有人，纯帧差分会把"人来回走动"误判为无数
    变化点，送 AI 的分片爆炸。解法：变化点密度（变化帧数/总帧数）超过
    crowded_density_threshold（默认 0.6）时，用 YOLO 按物体类别聚类去重——
    只保留"新物体类别出现"的变化点，过滤"同类物体来回走动"的重复变化。

    无 ultralytics 时降级为父类纯帧差分（不崩，CI 标准子集无 ultralytics）。
    """

    def _merge_to_segments(
        self,
        diff_scores: List[float],
        timestamps: List[float],
        day_night: List[str],
        scene_boundaries: List[float],
        duration: float,
    ) -> List[MotionSegment]:
        """密集场景：密度高时 YOLO 物体去重，否则走父类逻辑。"""
        cfg = self.config
        # 收集变化帧时间戳（与父类同款逻辑）
        change_ts: List[float] = []
        for i, score in enumerate(diff_scores):
            if i >= len(day_night):
                break
            label = day_night[i]
            threshold = cfg.day_threshold if label == "day" else cfg.night_threshold
            if score > threshold:
                change_ts.append(timestamps[i])
        change_ts.extend(scene_boundaries)
        change_ts = sorted(set(change_ts))

        # 密度判定：变化帧数 / 总帧数
        density = (len(change_ts) / len(timestamps)) if timestamps else 0.0
        if density <= cfg.crowded_density_threshold:
            # 密度低：走父类纯帧差分合并（与 sparse-corridor 同款）
            return super()._merge_to_segments(
                diff_scores, timestamps, day_night,
                scene_boundaries, duration)

        # 密度高：YOLO 物体类别去重
        # 复用已落盘的帧（frame_out_dir）做 YOLO 推理
        frame_paths: List[str] = []
        if self._frame_out_dir is not None:
            for ts in change_ts:
                # 帧文件名 f{i:06d}_{ts:.1f}.jpg，按 ts 找
                for fp in self._frame_out_dir.glob("f*.jpg"):
                    m = _TS_RE.match(fp.name)
                    if m and abs(float(m.group(1)) - ts) < 0.5:
                        frame_paths.append(str(fp))
                        break

        obj_sets = self._detect_objects_yolo(frame_paths) if frame_paths else None
        if obj_sets is None:
            # YOLO 不可用 → 降级父类纯帧差分
            logger.info("[motion] crowded 场景但 YOLO 不可用，降级纯帧差分")
            return super()._merge_to_segments(
                diff_scores, timestamps, day_night,
                scene_boundaries, duration)

        # 物体去重：只在物体类别集合变化时保留变化点
        deduped_ts: List[float] = []
        prev_set: List[str] = []
        for i, ts in enumerate(change_ts):
            idx = min(i, len(obj_sets) - 1)
            curr_set = obj_sets[idx] if idx < len(obj_sets) else []
            if i == 0 or not self._object_set_unchanged(prev_set, curr_set):
                deduped_ts.append(ts)
            prev_set = curr_set
        logger.info(
            f"[motion] crowded 去重: {len(change_ts)} 变化点 → "
            f"{len(deduped_ts)}（密度 {density:.2f}，YOLO 去重生效）")

        # 用去重后的 change_ts 走父类合并（但父类会重新算 change_ts，
        # 这里手动构造 segments 避免重复）
        if not deduped_ts:
            return []
        # 复用父类 _merge_to_segments 的 padding/clamp 逻辑：构造一个
        # 只含 deduped_ts 变化点的伪 diff_scores（超阈值的才保留）
        deduped_set = set(deduped_ts)
        filtered_diff = []
        filtered_ts = []
        for i, ts in enumerate(timestamps):
            if ts in deduped_set and i < len(diff_scores):
                filtered_diff.append(diff_scores[i])
                filtered_ts.append(ts)
        return super()._merge_to_segments(
            filtered_diff, filtered_ts, day_night,
            scene_boundaries, duration)
