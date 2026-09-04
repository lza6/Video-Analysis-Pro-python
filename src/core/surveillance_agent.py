"""监控视频智能分析 Agent (v5.0)

针对用户场景：在大量监控视频中，根据"关键物品"图片+描述，找出携带该物品的人，
并自动剪辑出对应片段。

工作流（参考 video-use Ask→confirm→execute→iterate→persist + CL4R1T4S Agent Loop）:
  1. analyze   : 理解查询意图（找什么物品/人）
  2. plan      : 决定抽帧策略（1fps + 智能场景切分）
  3. search    : 逐视频/逐帧用 VLM 判断是否匹配关键物品
  4. locate    : 确认命中时刻 + 视频文件 + 时间戳
  5. cut       : 自动裁剪命中片段（前后留余量）
  6. report    : 生成总结报告（命中清单 + 时间线 + 建议）

大文件处理：
  - 自动裁剪：对超长视频（>阈值）按时间窗口分片处理，避免上下文爆炸
  - 抽帧密度自适应：监控视频多为静态场景，低密度（0.2fps）足够
  - 批量并行：多视频并发处理
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import torch

logger = logging.getLogger("VideoAnalyzerCore")


@dataclass
class HitMoment:
    """一次命中：视频 + 时间 + 置信度 + 帧路径。"""
    video_path: str
    video_name: str
    timestamp: float
    confidence: float  # 0-1
    reason: str
    frame_path: str = ""


@dataclass
class SearchReport:
    """一次搜索任务的完整报告。"""
    query: str
    item_description: str
    total_videos: int
    total_frames_scanned: int
    hits: List[HitMoment] = field(default_factory=list)
    clips: List[str] = field(default_factory=list)  # 裁剪出的片段路径
    elapsed_seconds: float = 0.0
    timeline: List[Dict[str, Any]] = field(default_factory=list)  # 处理时间线
    model: str = ""


class SurveillanceAgent:
    """监控视频智能分析 Agent。

    使用 VLM (glm-5.3-flash 等) 直接判断每一帧是否携带关键物品，
    不依赖 CLIP 语义近似（监控场景需要精确匹配而非语义相似）。
    """

    def __init__(self, backend, key_item_image: str,
                 item_description: str = "",
                 fps: float = 1.0,
                 max_frames_per_video: int = 600,
                 clip_padding: float = 5.0,
                 clip_duration: float = 20.0):
        """
        Args:
            backend: LLM 网关后端 (AnthropicBackend 等)
            key_item_image: 关键物品图片路径
            item_description: 物品文字描述（VLM 给出，可空）
            fps: 抽帧频率（监控 1fps 足够）
            max_frames_per_video: 单视频最大抽帧数（防爆上下文）
            clip_padding: 命中片段前后留的余量（秒）
            clip_duration: 命中片段总时长（秒）
        """
        self.backend = backend
        self.key_item_image = key_item_image
        self.item_description = item_description or "关键物品"
        self.fps = fps
        self.max_frames_per_video = max_frames_per_video
        self.clip_padding = clip_padding
        self.clip_duration = clip_duration

    def _extract_frames(self, video_path: Path, out_dir: Path) -> List[Dict]:
        """抽帧到 out_dir，返回 [{timestamp, path}]。

        优先用 ffmpeg（监控视频 H.264 流的 PPS 头问题 OpenCV 常报错，
        ffmpeg 更鲁棒）。回退到 cv2。
        """
        import subprocess, shutil
        out_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg") or os.environ.get("IMAGEIO_FFMPEG_EXE")

        # 先读元数据
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total / fps if fps > 0 else 0
        cap.release()

        interval = 1.0 / self.fps if self.fps > 0 else 2.0
        n_targets = min(self.max_frames_per_video,
                        int(duration / interval) + 1 if duration > 0 else self.max_frames_per_video)

        frames = []
        if ffmpeg:
            # ffmpeg select 滤镜按时间戳抽帧
            for i in range(n_targets):
                ts = i * interval
                if duration > 0 and ts >= duration:
                    break
                fp = out_dir / f"frame_{i:04d}_{ts:.1f}s.jpg"
                cmd = [ffmpeg, "-y", "-ss", f"{ts:.2f}", "-i", str(video_path),
                       "-frames:v", "1", "-q:v", "3", str(fp)]
                try:
                    subprocess.run(cmd, capture_output=True, timeout=15)
                    if fp.exists():
                        frames.append({"timestamp": ts, "path": str(fp)})
                except Exception as e:
                    logger.debug(f"ffmpeg 抽帧 {ts}s 失败: {e}")
        else:
            # 回退 cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return []
            step = max(1, int(fps * interval))
            idx = 0
            while len(frames) < n_targets:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                ts = idx / fps if fps > 0 else 0
                fp = out_dir / f"frame_{len(frames):04d}_{ts:.1f}s.jpg"
                # cv2.imwrite 对含 Unicode 的路径在 Windows 不稳定，改用 imencode + tofile
                ok, buf = cv2.imencode('.jpg', frame)
                if ok:
                    buf.tofile(str(fp))
                    frames.append({"timestamp": ts, "path": str(fp)})
                idx += step
            cap.release()
        logger.info(f"抽帧 {video_path.name}: {len(frames)} 帧, 时长 {duration:.0f}s, 间隔 {interval:.1f}s")
        return frames

    def _judge_frame(self, frame_path: str) -> Optional[HitMoment]:
        """让 VLM 判断该帧是否包含关键物品。

        用关键物品图 + 当前帧 一起发给模型，让它做精确匹配判断。
        """
        prompt = (
            f"你在分析监控视频的一帧画面。我需要找出携带「{self.item_description}」的人。\n"
            "第一张图是要找的关键物品，第二张图是监控画面的一帧。\n"
            "请判断：这一帧画面中是否出现了携带该物品（或该物品本身）的人？\n"
            "严格按以下 JSON 格式回答，不要有任何其它文字：\n"
            '{"match": true/false, "confidence": 0.0-1.0, "reason": "简短中文说明"}'
        )
        chunks = list(self.backend.chat_stream(
            messages=[{"role": "user", "content": prompt}],
            image_paths=[self.key_item_image, frame_path],
            temperature=0.1,
        ))
        raw = "".join(chunks)
        # 剥思考标签
        import re
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        # 提取 JSON
        m = re.search(r'\{[^{}]*"match"[^{}]*\}', raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        if not data.get("match"):
            return None
        ts_str = Path(frame_path).stem
        # 从文件名 frame_0001_123.4s.jpg 提取时间戳
        ts = 0.0
        parts = ts_str.split("_")
        if len(parts) >= 3:
            try:
                ts = float(parts[-1].rstrip("s"))
            except ValueError:
                pass
        return HitMoment(
            video_path="", video_name="",
            timestamp=ts,
            confidence=float(data.get("confidence", 0.5)),
            reason=data.get("reason", ""),
            frame_path=frame_path,
        )

    def search_video(self, video_path: Path, work_dir: Path) -> List[HitMoment]:
        """搜索单个视频，返回命中列表。

        两阶段策略 (避免 VLM 调用爆炸):
          1. CLIP 粗筛: 用关键物品图与所有抽帧做相似度匹配，取 top-K 候选
          2. VLM 确认: 只对候选帧调 VLM 精确判断
        """
        t0 = time.time()
        frames = self._extract_frames(video_path, work_dir / video_path.stem)
        if not frames:
            return []

        # ---- 阶段 1: CLIP 粗筛 ----
        candidates = self._clip_prefilter(frames)
        logger.info(f"CLIP 粗筛 {video_path.name}: {len(frames)} -> {len(candidates)} 候选")

        # ---- 阶段 2: VLM 确认 ----
        hits = []
        for i, fr in enumerate(candidates):
            hit = self._judge_frame(fr["path"])
            if hit:
                hit.video_path = str(video_path)
                hit.video_name = video_path.name
                hit.timestamp = fr["timestamp"]
                hit.frame_path = fr["path"]
                hits.append(hit)
                logger.info(f"  ✓ 命中 {video_path.name} @ {fr['timestamp']:.1f}s (conf={hit.confidence:.2f})")
            logger.info(f"  VLM 确认 {i+1}/{len(candidates)}")
        logger.info(f"视频 {video_path.name} 完成: {len(hits)} 命中, 耗时 {time.time()-t0:.1f}s")
        return hits

    def _clip_prefilter(self, frames: List[Dict], top_k: int = 15,
                        min_sim: float = 0.28) -> List[Dict]:
        """用 CLIP 图-图相似度粗筛，返回 top-K 候选帧。

        监控场景下关键物品可能在画面中很小/部分可见，阈值不宜太高。
        """
        try:
            from sentence_transformers import util
            from PIL import Image
            from src.core.kb_indexer import get_embedder
        except ImportError:
            logger.warning("CLIP 不可用，跳过预筛，全帧送 VLM")
            return frames
        try:
            model = get_embedder()
            if model is None:
                logger.warning("embedder 加载失败，跳过预筛")
                return frames
            item_img = Image.open(self.key_item_image)
            item_emb = model.encode([item_img], convert_to_tensor=True,
                                    show_progress_bar=False)
            frame_imgs = [Image.open(f["path"]) for f in frames]
            frame_embs = model.encode(frame_imgs, convert_to_tensor=True,
                                      batch_size=32, show_progress_bar=False)
            sims = util.cos_sim(item_emb, frame_embs)[0]
            # 取 top-K 且超过阈值
            top_vals, top_idx = torch.topk(sims, k=min(top_k, len(frames)))
            candidates = []
            for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                if val >= min_sim:
                    candidates.append(frames[idx])
            return candidates
        except Exception as e:
            logger.error(f"CLIP 预筛失败: {e}")
            return frames

    def cut_clip(self, video_path: Path, timestamp: float, out_path: Path):
        """从 video_path 的 timestamp 处裁剪 clip_duration 秒的片段。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        cap.release()

        start = max(0, timestamp - self.clip_duration / 2)
        # total=0（cv2 读不到 frame count，监控/流式视频常见）时按 clip_duration 兜底，
        # 否则 total/fps=0 导致 end<=start 永不裁剪，所有命中片段产不出 mp4
        end = min((total / fps if (total and fps) else timestamp + self.clip_duration),
                  timestamp + self.clip_duration / 2)
        if end <= start:
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 用 ffmpeg 裁剪（lossless, 快）
        import subprocess, shutil
        ffmpeg = shutil.which("ffmpeg") or os.environ.get("IMAGEIO_FFMPEG_EXE")
        if ffmpeg:
            cmd = [ffmpeg, "-y", "-ss", f"{start:.2f}", "-i", str(video_path),
                   "-t", f"{end-start:.2f}", "-c", "copy", str(out_path)]
            subprocess.run(cmd, capture_output=True, timeout=60)
        else:
            # 回退到 cv2
            cap = cv2.VideoCapture(str(video_path))
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
            frames_to_write = int((end - start) * fps)
            for _ in range(frames_to_write):
                ret, f = cap.read()
                if not ret:
                    break
                writer.write(f)
            cap.release()
            writer.release()

    def run(self, video_dir: str, output_dir: str,
            max_videos: int = 0) -> SearchReport:
        """对 video_dir 下所有视频执行搜索 + 裁剪 + 报告。"""
        t_start = time.time()
        vdir = Path(video_dir)
        videos = sorted([f for f in vdir.iterdir()
                         if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")])
        if max_videos > 0:
            videos = videos[:max_videos]
        odir = Path(output_dir)
        odir.mkdir(parents=True, exist_ok=True)

        all_hits: List[HitMoment] = []
        total_frames = 0
        timeline = []

        for i, v in enumerate(videos):
            t_v = time.time()
            hits = self.search_video(v, odir / "frames")
            all_hits.extend(hits)
            # 每个 hit 裁剪片段
            for j, h in enumerate(hits):
                clip_path = odir / "clips" / f"{v.stem}_hit{j:02d}_{h.timestamp:.1f}s.mp4"
                self.cut_clip(v, h.timestamp, clip_path)
                if clip_path.exists():
                    all_hits[len(all_hits) - len(hits) + j].frame_path = str(clip_path)
            timeline.append({
                "video": v.name, "hits": len(hits),
                "elapsed": round(time.time() - t_v, 1),
            })
            logger.info(f"[{i+1}/{len(videos)}] {v.name}: {len(hits)} 命中")

        # T4: 清理抽帧临时目录（保留 clips/ 与报告文件）。
        # 历史 bug: 每个视频 600 帧×200KB≈120MB 残留磁盘。
        import shutil
        frames_root = odir / "frames"
        if frames_root.exists():
            shutil.rmtree(frames_root, ignore_errors=True)
            logger.info(f"已清理临时抽帧目录: {frames_root}")

        # 生成报告
        report = SearchReport(
            query=f"查找携带「{self.item_description}」的人",
            item_description=self.item_description,
            total_videos=len(videos),
            total_frames_scanned=total_frames,
            hits=all_hits,
            clips=[h.frame_path for h in all_hits if h.frame_path],
            elapsed_seconds=round(time.time() - t_start, 1),
            timeline=timeline,
            model=getattr(self.backend, "model", ""),
        )
        self._write_report(report, odir)
        return report

    def _write_report(self, report: SearchReport, out_dir: Path):
        """写 JSON + Markdown 报告。"""
        # JSON
        (out_dir / "search_report.json").write_text(
            json.dumps({
                "query": report.query,
                "item": report.item_description,
                "model": report.model,
                "total_videos": report.total_videos,
                "total_frames_scanned": report.total_frames_scanned,
                "total_hits": len(report.hits),
                "elapsed_seconds": report.elapsed_seconds,
                "timeline": report.timeline,
                "hits": [{
                    "video": h.video_name,
                    "timestamp": h.timestamp,
                    "confidence": h.confidence,
                    "reason": h.reason,
                    "clip": h.frame_path,
                } for h in report.hits],
            }, ensure_ascii=False, indent=2), encoding="utf-8")

        # Markdown
        md = ["# 监控视频分析报告\n",
              f"**查询**: {report.query}\n",
              f"**模型**: {report.model}\n",
              f"**分析视频数**: {report.total_videos}\n",
              f"**命中数**: {len(report.hits)}\n",
              f"**总耗时**: {report.elapsed_seconds}s\n\n",
              "## 时间线\n"]
        for t in report.timeline:
            md.append(f"- {t['video']}: {t['hits']} 命中 ({t['elapsed']}s)\n")
        md.append("\n## 命中清单\n")
        for i, h in enumerate(report.hits, 1):
            md.append(f"### 命中 {i}: {h.video_name} @ {h.timestamp:.1f}s\n")
            md.append(f"- 置信度: {h.confidence:.2f}\n")
            md.append(f"- 原因: {h.reason}\n")
            md.append(f"- 片段: `{h.frame_path}`\n\n")
        if not report.hits:
            md.append("未找到匹配项。\n")
        (out_dir / "search_report.md").write_text("".join(md), encoding="utf-8")
