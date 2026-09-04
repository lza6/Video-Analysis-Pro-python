"""批量视频分析任务引擎 (T2)

在 `surveillance_agent.SurveillanceAgent`（单视频搜索+裁剪）之上做批量编排，
接 `provider_router.ProviderRouter`（多 key 轮换 + 40/min 限速 + 无限重试）和
`run_store.RunStore`（runs/segments/clips 三表 + WAL）。

设计要点（用户硬性要求）
  - 拉满每分钟 40 请求：视频串行（避免显存/磁盘 IO 争抢），分片判断由 router
    内部多 key 并发处理（router 已自带 ThreadPool 风格的多 key 选择）。
  - 内存回收防泄漏（万级视频场景）：每视频跑完 `gc.collect()` + 清该视频的
    分片临时目录（clean_segments=True 时），命中 clip 单独保留。
  - 断点续跑：run_store 里 status=started/running 的 run，查 segments 已完成的
    seg_idx 跳过，未完成的继续。
  - UI 能查实时进度：QObject 发信号 run_started/video_started/segment_done/
    video_done/batch_progress/batch_finished/error；UI 直接读 run_store.get_progress。

接缝说明（与 surveillance_agent 的关系）
  - surveillance_agent 仍是「单视频 CLIP 粗筛 + VLM 逐帧确认 + 裁剪」的孤立
    后端，**不改其核心**。
  - T2 复用 surveillance_agent 的 ffmpeg 抽帧模式（_extract_frames 的 ffmpeg
    调用形状）和 cut_clip 的裁剪逻辑形状，但走的是 NVIDIA 视频模型直评整段
    分片（mp4≤2min, 720p, 2fps/256帧），**不再逐帧抽帧**——这是与
    surveillance_agent 的关键差异：surveillance_agent 是「抽帧→逐帧 VLM」，
    batch_runner 是「切 2min 分片→整段送 NVIDIA 视频模型」。
  - 因此 batch_runner 不依赖 surveillance_agent 实例，只在裁剪时复用其
    `cut_clip` 的 ffmpeg 命令形状（-ss/-t/-c copy）。

本文件只新增，不改 provider_router / run_store / nvidia_models / surveillance_agent。
"""
from __future__ import annotations

import base64
import gc
import json
import logging
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

# 不在模块顶层 import torch / cv2：headless/无 GPU 环境也要能 import 本模块
# （与 surveillance_agent.py 顶部 import torch 不同——batch_runner 只在切分片
# 时按需 import cv2 读元数据，不强制全局 torch）。
from src.core.motion_detector import MotionConfig, MotionDetector

logger = logging.getLogger("VideoAnalyzerCore")

# ---- 常量 ----
SUPPORTED_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
# NVIDIA Nemotron Omni 限制：mp4≤2min，720p 2fps/256 帧
NVIDIA_MAX_SEGMENT_SEC = 120
NVIDIA_TARGET_HEIGHT = 720
NVIDIA_TARGET_FPS = 2
NVIDIA_MAX_FRAMES = 256
# ffmpeg 路径：优先 PATH 上的 ffmpeg，其次 imageio-ffmpeg 自带
DEFAULT_FFMPEG_ENV = "IMAGEIO_FFMPEG_EXE"


def _find_ffmpeg() -> Optional[str]:
    """定位 ffmpeg 可执行文件。surveillance_agent._extract_frames 同款逻辑。"""
    found = shutil.which("ffmpeg")
    if found:
        return found
    env_exe = os.environ.get(DEFAULT_FFMPEG_ENV)
    if env_exe and Path(env_exe).exists():
        return env_exe
    # imageio-ffmpeg 自带的 ffmpeg.exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
@dataclass
class BatchConfig:
    """批量视频分析配置。

    segment_sec=120 对齐 NVIDIA Nemotron Omni 的 2min 上限；fps_sample=1.0
    是抽帧诊断密度（不送模型，只用于本地预览/调试）；clip_padding=10 是命中
    裁剪时前后留的余量（秒）。
    """
    video_dir: str
    key_item_image: str
    item_description: str = ""
    segment_sec: int = NVIDIA_MAX_SEGMENT_SEC  # 120s，NVIDIA 上限
    fps_sample: float = 1.0  # 抽帧诊断密度（不送模型）
    clip_padding: float = 10.0  # 命中裁剪余量（秒）
    concurrency_per_key: int = 1  # 每 key 并发：router 已限 40/min，这里控制单视频内分片并发
    max_tokens: int = 65536
    reasoning_budget: int = 8192
    clean_segments: bool = True  # 跑完删该视频分片临时目录（留 clip）
    resume: bool = True  # 断点续跑
    model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    enable_thinking: bool = True
    temperature: float = 0.2
    request_timeout: int = 120  # router.post_nvidia 的 timeout
    out_dir: str = "out/batch"  # 产物根目录（clips/<run_id>/、segments/<run_id>/）
    # 视频级并发：多个视频同时跑 motion_detector（CPU）+ 分片判断（AI）。
    # 默认 4：单机多核并行，motion_detector 是 CPU 密集（每视频 ~75s），
    # 串行跑 63 视频要 60min+，并发 4 缩到 ~15min。AI 调用由 router 多 key
    # 限速兜底（11 key × 2 并发 = 22），不会雪崩。
    video_concurrency: int = 4
    # 二次验证：首次判断 match=true 且 confidence≥此阈值时，自动二次送 AI 确认。
    # 二次 prompt 更严格（has_person/has_target_item/description），只两次都 true
    # 才最终算命中。防止首次判断宽松导致的误判（实测 _375/_388 曾误判）。
    confidence_threshold: float = 0.7
    enable_verification: bool = True  # 开关：可关闭二次验证（默认开）


# ----------------------------------------------------------------------
# 批量引擎
# ----------------------------------------------------------------------
class BatchRunner(QObject):
    """批量视频分析引擎。

    线程模型：
      - run_batch 在 QThread.run() 里调用（由调用方包 QThread，或直接在 worker
        线程调用）。本类本身不继承 QThread，避免多重继承陷阱；调用方自行
        `QThread(runner)` 或 `runner.moveToThread(thread)`。
      - 信号通过 pyqtSignal 跨线程回主线程（Qt 自动 QueuedConnection）。
      - cancel() 设标志位，当前视频跑完即停（优雅停止，不强杀正在飞的 HTTP）。

    并发模型：
      - 视频级并发：video_concurrency 个视频同时跑（默认 4），motion_detector
        是 CPU 密集，多核并行；AI 调用由 router 多 key 限速兜底。
      - 单视频内的分片判断：用 ThreadPoolExecutor(concurrency_per_key) 并发
        调 router.post_nvidia，router 内部多 key 轮换 + 40/min 限速。
    """

    # ---- 信号 ----
    run_started = pyqtSignal(str, str)  # run_id, video_name
    video_started = pyqtSignal(str, str)  # run_id, video_name
    segment_done = pyqtSignal(str, int, bool, float)  # run_id, seg_idx, match, conf
    video_done = pyqtSignal(str, str, int)  # run_id, video_name, hits
    batch_progress = pyqtSignal(int, int)  # done, total
    batch_finished = pyqtSignal(int, int)  # total_runs, total_hits
    error = pyqtSignal(str)

    def __init__(self, config: BatchConfig, run_store: Any,
                 router: Any, parent: Optional[QObject] = None):
        """
        Args:
            config: BatchConfig
            run_store: RunStore 实例（写 runs/segments/clips）
            router: ProviderRouter 实例（多 key 轮换 + 限速 + 无限重试）
            parent: QObject 父（可选）
        """
        super().__init__(parent)
        self.config = config
        self.run_store = run_store
        self.router = router
        self._cancel_flag = False
        self._hits_meta: List[Dict[str, Any]] = []  # 命中详情内存（供 MD 报告）
        self._ffmpeg = _find_ffmpeg()
        if not self._ffmpeg:
            logger.warning("[batch] 未找到 ffmpeg，分片切分将失败")
        # 关键物品图编码为 base64 data URL（NVIDIA image_url 接受 data:image/jpeg;base64,...）
        self._key_item_b64: Optional[str] = self._encode_key_item(config.key_item_image)
        self._out_dir = Path(config.out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        # M1：变化检测器配置（昼夜自适应 + context padding）
        # 复用 BatchConfig 的 clip_padding 作为 MotionConfig.context_padding，
        # 保持两个 padding 语义一致（命中裁剪余量 = 变化时段上下文余量）。
        self._motion_config = MotionConfig(
            sample_fps=config.fps_sample,
            context_padding=config.clip_padding,
            max_segment_sec=config.segment_sec,
        )

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run_batch(self, videos: List[Path]) -> Tuple[int, int]:
        """遍历视频，video_concurrency 个并发调 _run_single_video。

        Returns:
            (total_runs, total_hits) 供 UI 总结。
        """
        total = len(videos)
        total_hits = 0
        done = 0
        self._cancel_flag = False
        vc = max(1, self.config.video_concurrency)
        logger.info(f"[batch] 开始批量分析：{total} 个视频，视频并发 {vc}")
        # video_concurrency=1 时在当前线程串行跑（信号 DirectConnection 即时投递，
        # 无需事件循环，测试友好）；>1 时用线程池并发（信号 QueuedConnection，
        # 需调用方跑 Qt 事件循环）
        if vc == 1:
            for i, video in enumerate(videos):
                if self._cancel_flag:
                    logger.info(f"[batch] cancel 已设，跳过剩余 {total - i} 视频")
                    break
                try:
                    hits = self._run_single_video(video)
                    total_hits += hits
                except Exception as e:
                    logger.exception(f"[batch] 视频 {video.name} 异常")
                    self.error.emit(f"视频 {video.name} 异常: {e}")
                done += 1
                self.batch_progress.emit(done, total)
        else:
            with ThreadPoolExecutor(max_workers=vc) as pool:
                futs = {pool.submit(self._run_single_video, v): v for v in videos}
                for fut in as_completed(futs):
                    if self._cancel_flag:
                        for f in futs:
                            f.cancel()
                        break
                    try:
                        hits = fut.result()
                        total_hits += hits
                    except Exception as e:
                        v = futs[fut]
                        logger.exception(f"[batch] 视频 {v.name} 异常")
                        self.error.emit(f"视频 {v.name} 异常: {e}")
                    done += 1
                    self.batch_progress.emit(done, total)
        self.batch_finished.emit(done, total_hits)
        # 批结束写总命中 MD 报告（视频名/起止时分秒/片段路径）
        self._write_hits_report()
        # 批结束再 GC 一次
        self._gc_collect()
        return done, total_hits

    # ------------------------------------------------------------------
    # 单视频
    # ------------------------------------------------------------------
    def _run_single_video(self, video: Path) -> int:
        """跑一个视频：分片→逐片判断→命中裁剪→写 store。

        Returns:
            该视频命中数。
        """
        # 读视频时长
        duration = self._probe_duration(video)
        if duration <= 0:
            self.error.emit(f"无法读取 {video.name} 时长，跳过")
            return 0

        # 断点续跑：若 video_path 已有 run 且 status=started/running，复用 run_id
        run_id = self._find_resumable_run(video)
        if run_id:
            logger.info(f"[batch] 续跑 {video.name} run_id={run_id}")
        else:
            run_id = self.run_store.create_run(
                str(video),
                duration_sec=duration,
                model=self.config.model,
                provider="nvidia",
                mode="batch_surveillance",
                status="running",
            )
        self.run_started.emit(run_id, video.name)
        self.video_started.emit(run_id, video.name)

        # 切分片（M1：基于变化检测，无变化时段返回 0 片段）
        seg_paths, seg_starts, seg_durs = self._segment_video(video, run_id, duration)
        n_segs = len(seg_paths)
        self.run_store.update_run(
            run_id, segments_total=n_segs, status="running")
        logger.info(f"[batch] {video.name} 切成 {n_segs} 片")

        # M1：无变化时段 → 0 片段，直接标 done 跳过 AI（省调用）
        if n_segs == 0:
            finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.run_store.update_run(
                run_id, status="done", finished_at=finished_at,
                segments_ok=0, hits_count=0)
            self.video_done.emit(run_id, video.name, 0)
            self._gc_collect()
            return 0

        # 查已完成的分片（续跑跳过）
        done_idx = self._completed_seg_indices(run_id)
        hits = self._count_existing_hits(run_id)
        seg_ok = len(done_idx & {i for i in range(n_segs)})

        # 并发判断分片
        results: List[Optional[Dict[str, Any]]] = [None] * n_segs
        with ThreadPoolExecutor(max_workers=max(1, self.config.concurrency_per_key)) as pool:
            futures: Dict[Future, int] = {}
            for i, (sp, ss, sd) in enumerate(zip(seg_paths, seg_starts, seg_durs)):
                if i in done_idx:
                    continue  # 续跑跳过
                if self._cancel_flag:
                    break
                fut = pool.submit(self._judge_segment, sp, ss, sd, run_id, i)
                futures[fut] = i
            for fut in futures:
                idx = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    logger.warning(f"[batch] seg {idx} 判断异常: {e}")
                    res = {"match": False, "confidence": 0.0,
                           "reason": f"judge error: {e}", "status": "failed",
                           "error": str(e)}
                results[idx] = res
                # 写 segment
                self._record_segment(run_id, idx, seg_starts[idx],
                                     seg_durs[idx], results[idx] or {})
                if results[idx] and results[idx].get("match"):
                    conf = float(results[idx].get("confidence", 0.5))
                    # 二次验证：confidence 达阈值时再送 AI 确认，防误判
                    if (self.config.enable_verification
                            and conf >= self.config.confidence_threshold):
                        verified = self._verify_segment(
                            seg_paths[idx], results[idx].get("reason", ""))
                        results[idx]["verified"] = verified
                        results[idx]["match"] = bool(verified.get("has_target_item", False))
                        results[idx]["confidence"] = float(verified.get("confidence", conf))
                        results[idx]["reason"] = (
                            f"[二次验证] {verified.get('description', '')}")
                    if results[idx].get("match"):
                        self._cut_and_record_clip(
                            run_id, video, seg_starts[idx], idx,
                            results[idx].get("confidence", 0.5),
                            results[idx].get("reason", ""))
                        hits += 1
                match_flag = bool(results[idx] and results[idx].get("match"))
                conf = float(results[idx].get("confidence", 0.0)) if results[idx] else 0.0
                self.segment_done.emit(run_id, idx, match_flag, conf)
                seg_ok += 1
                self.run_store.update_run(
                    run_id, segments_ok=seg_ok,
                    hits_count=self._count_existing_hits(run_id))

        # 视频结束
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.run_store.update_run(
            run_id, status="done", finished_at=finished_at,
            segments_ok=seg_ok,
            hits_count=self._count_existing_hits(run_id))
        self.video_done.emit(run_id, video.name, hits)

        # 内存回收：清该视频分片临时目录
        if self.config.clean_segments:
            seg_dir = self._out_dir / "segments" / run_id
            if seg_dir.exists():
                shutil.rmtree(seg_dir, ignore_errors=True)
                logger.info(f"[batch] 清理分片目录 {seg_dir}")
        self._gc_collect()
        return hits

    # ------------------------------------------------------------------
    # 分片切分
    # ------------------------------------------------------------------
    def _segment_video(self, video: Path, run_id: str,
                       duration: float) -> Tuple[List[Path], List[float], List[float]]:
        """用 MotionDetector 找画面变化时段，只对变化时段切片送 AI。

        M1 重构：不再固定 120s 切片。改用 motion_detector 找"有人经过/物品
        出现"的变化时段，只对这些时段加 padding 后切 mp4。长时间无变化的
        空走廊直接跳过（不切、不送 AI），省 90%+ AI 调用。

        Returns:
            (seg_paths, seg_starts, seg_durs)
        """
        seg_dir = self._out_dir / "segments" / run_id
        seg_dir.mkdir(parents=True, exist_ok=True)

        # 找变化时段
        detector = MotionDetector(self._motion_config, ffmpeg_exe=self._ffmpeg)
        segments = detector.detect(video)

        # 无变化时段 → 0 片段（直接标 done，跳过 AI）
        if not segments:
            logger.info(f"[batch] {video.name} 无画面变化时段，跳过 AI（0 片段）")
            return [], [], []

        paths: List[Path] = []
        starts: List[float] = []
        durs: List[float] = []
        for i, seg in enumerate(segments):
            out = seg_dir / f"seg_{i:04d}.mp4"
            if not out.exists() and not self._cancel_flag:
                # ffmpeg：-ss 在 -i 前是 fast seek，-an 弃音频，scale 到 720p，
                # fps=2 对齐 NVIDIA 2fps/256 帧上限，libx264 编码
                # NVIDIA_MAX_FRAMES 上限：2fps × min(seg_dur, 120s)
                max_frames = min(NVIDIA_MAX_FRAMES,
                                  int(seg.duration * NVIDIA_TARGET_FPS))
                cmd = [self._ffmpeg, "-y", "-ss", f"{seg.start_sec:.2f}",
                       "-t", f"{seg.duration:.2f}", "-i", str(video),
                       "-an",
                       "-vf", f"scale=-2:{NVIDIA_TARGET_HEIGHT},fps={NVIDIA_TARGET_FPS}",
                       "-frames:v", str(max_frames),
                       "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                       "-movflags", "+faststart", str(out)]
                try:
                    subprocess.run(cmd, capture_output=True, timeout=120)
                except subprocess.TimeoutExpired:
                    logger.warning(f"[batch] 分片 {i} 切分超时")
            if out.exists():
                paths.append(out)
                starts.append(float(seg.start_sec))
                durs.append(float(seg.duration))
        return paths, starts, durs

    # ------------------------------------------------------------------
    # 分片判断
    # ------------------------------------------------------------------
    def _judge_segment(self, seg_path: Path, seg_start: float,
                      seg_dur: float, run_id: str, seg_idx: int) -> Dict[str, Any]:
        """调 router.post_nvidia 发 video_url+image_url，解析 match/confidence/reason。

        NVIDIA Nemotron Omni 原生接受 video_url，但要求视频可公开访问（或 data
        URL）。本地分片无法走公网 URL，这里用 base64 data URL 传分片（2min 720p
        约 1-3MB，base64 后 1.3-4MB，单请求可接受）。

        Returns:
            {"match": bool, "confidence": float, "reason": str,
             "status": "ok"/"failed", "elapsed_sec": float, "usage_json": str}
        """
        if not self._key_item_b64:
            return {"match": False, "confidence": 0.0,
                    "reason": "key item image 不可读", "status": "failed",
                    "error": "no key item image"}
        t0 = time.time()
        first_token_ms: Optional[int] = None
        # 分片转 data URL
        video_b64 = self._encode_file_b64(seg_path)
        if not video_b64:
            return {"match": False, "confidence": 0.0,
                    "reason": f"分片 {seg_path.name} 读取失败",
                    "status": "failed", "error": "seg read fail"}
        video_url = f"data:video/mp4;base64,{video_b64}"
        item_url = f"data:image/jpeg;base64,{self._key_item_b64}"

        prompt = self._build_judge_prompt(seg_start, seg_dur)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": item_url}},
                {"type": "video_url", "video_url": {"url": video_url}},
            ],
        }]
        # 用 nvidia_models.build_nvidia_payload（思考参数拍平到顶层）
        from src.core.nvidia_models import build_nvidia_payload
        payload = build_nvidia_payload(
            self.config.model, messages,
            enable_thinking=self.config.enable_thinking,
            reasoning_budget=self.config.reasoning_budget,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream=False,
        )
        try:
            resp = self.router.post_nvidia(
                payload, timeout=self.config.request_timeout)
            # M1：记录首字到达时间。post_nvidia 当前非流式（整响应一次返回），
            # first_token_ms = 请求发出到收到响应的全程耗时；后续切流式时此
            # hook 点改为"第一个 chunk 到达时刻"即可（run_store 已加该列）。
            first_token_ms = int((time.time() - t0) * 1000)
        except Exception as e:
            logger.warning(f"[batch] run {run_id} seg {seg_idx} NVIDIA 调用失败: {e}")
            return {"match": False, "confidence": 0.0,
                    "reason": f"router error: {e}", "status": "failed",
                    "error": str(e), "elapsed_sec": round(time.time() - t0, 2),
                    "first_token_ms": first_token_ms}

        elapsed = round(time.time() - t0, 2)
        # 解析 OpenAI 兼容响应
        text, usage = self._extract_text_and_usage(resp)
        match, conf, reason = self._parse_judge(text)
        return {"match": match, "confidence": conf, "reason": reason,
                "status": "ok", "elapsed_sec": elapsed,
                "first_token_ms": first_token_ms,
                "usage_json": json.dumps(usage, ensure_ascii=False) if usage else None}

    def _build_judge_prompt(self, seg_start: float, seg_dur: float) -> str:
        desc = self.config.item_description or "关键物品"
        return (
            f"你在分析一段监控视频分片（起始 {seg_start:.1f}s，时长 {seg_dur:.1f}s）。\n"
            f"第一张图是要找的关键物品：{desc}。\n"
            f"第二段视频是监控画面的一段时间窗口。\n"
            "请判断：这段视频中是否出现了携带该物品（或该物品本身）的人？\n"
            "严格按以下 JSON 格式回答，不要有任何其它文字：\n"
            '{"match": true/false, "confidence": 0.0-1.0, "reason": "简短中文说明"}'
        )

    @staticmethod
    def _extract_text_and_usage(resp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """从 OpenAI 兼容响应里抽 text 与 usage。

        NVIDIA 响应形如 {"choices":[{"message":{"content":"..."}}], "usage":{...}}。
        content 里可能含 <think>...</think> 思考标签，剥掉。
        """
        if not isinstance(resp, dict):
            return "", {}
        choices = resp.get("choices") or []
        text = ""
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content", "") or ""
        # 剥思考标签（与 surveillance_agent._judge_frame 同款正则）
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        usage = resp.get("usage") or {}
        return text, usage

    @staticmethod
    def _parse_judge(text: str) -> Tuple[bool, float, str]:
        """从模型输出抽 match/confidence/reason。

        与 surveillance_agent._judge_frame 同款 JSON 提取逻辑。
        """
        m = re.search(r'\{[^{}]*"match"[^{}]*\}', text, re.DOTALL)
        if not m:
            return False, 0.0, "无法解析模型输出"
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return False, 0.0, "JSON 解析失败"
        return (bool(data.get("match")),
                float(data.get("confidence", 0.5)),
                str(data.get("reason", "")))

    def _verify_segment(self, seg_path: Path, first_reason: str) -> Dict[str, Any]:
        """二次验证：把首次判 match=true 的分片重新送 NVIDIA，用更严格 prompt
        分别确认 has_person / has_target_item，防止首次宽松判断误判。

        prompt 同时送关键物品图 + 视频片段，要求 AI 描述画面并分别判断。
        只有 has_target_item=true 才最终算命中。
        """
        from src.core.nvidia_models import build_nvidia_payload  # 局部 import 避循环
        if not self._key_item_b64 or not seg_path.exists():
            return {"has_target_item": False, "confidence": 0.0,
                    "description": "验证跳过：物品图或分片缺失"}
        vid_b64 = self._encode_file_b64(seg_path)
        if not vid_b64:
            return {"has_target_item": False, "confidence": 0.0,
                    "description": "验证跳过：分片读取失败"}
        desc = self.config.item_description or "关键物品"
        prompt = (
            "这是监控视频的一个片段（二次验证）。第一张图是要找的丢失物品："
            f"{desc}。\n请仔细看整段视频，严格 JSON 回答：\n"
            '{"has_person": bool, "has_target_item": bool, '
            '"confidence": 0-1, "description": "画面内容简述"}'
        )
        payload = build_nvidia_payload(
            self.config.model,
            [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{self._key_item_b64}"}},
                {"type": "video_url",
                 "video_url": {"url": f"data:video/mp4;base64,{vid_b64}"}},
            ]}],
            enable_thinking=True, reasoning_budget=4096,
            max_tokens=2048, temperature=0.2, stream=False)
        try:
            resp = self.router.post_nvidia(payload, timeout=self.config.request_timeout)
        except Exception as e:
            logger.warning(f"[batch] 二次验证 NVIDIA 调用失败: {e}")
            return {"has_target_item": False, "confidence": 0.0,
                    "description": f"验证失败: {e}"}
        text, _ = self._extract_text_and_usage(resp)
        m = re.search(r'\{[^{}]*"has_target_item"[^{}]*\}', text, re.DOTALL)
        if not m:
            return {"has_target_item": False, "confidence": 0.0,
                    "description": "验证输出未匹配JSON"}
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"has_target_item": False, "confidence": 0.0,
                    "description": "验证JSON解析失败"}
        logger.info(f"[batch] 二次验证 has_target_item={d.get('has_target_item')} "
                    f"conf={d.get('confidence')} desc={d.get('description','')[:60]}")
        return d

    # ------------------------------------------------------------------
    # 写 store
    # ------------------------------------------------------------------
    def _record_segment(self, run_id: str, seg_idx: int,
                        start_sec: float, dur_sec: float,
                        result: Dict[str, Any]) -> None:
        """把分片判断结果写 run_store.add_segment。"""
        status = result.get("status", "ok")
        match = 1 if result.get("match") else 0
        self.run_store.add_segment(run_id, {
            "seg_idx": seg_idx,
            "start_sec": start_sec,
            "dur_sec": dur_sec,
            "status": status,
            "match": match,
            "confidence": result.get("confidence"),
            "reason": result.get("reason"),
            "attempts": 1,
            "elapsed_sec": result.get("elapsed_sec"),
            "first_token_ms": result.get("first_token_ms"),
            "usage_json": result.get("usage_json"),
            "error": result.get("error"),
        })

    def _cut_and_record_clip(self, run_id: str, video: Path,
                             seg_start: float, seg_idx: int,
                             conf: float, reason: str) -> None:
        """命中裁剪：从原视频 seg_start 处裁 clip_padding*2 秒片段。

        复用 surveillance_agent.cut_clip 的 ffmpeg -ss/-t/-c copy 形状。
        起止时间 = [seg_start - pad/2, seg_start + pad/2]，写 clips 表 + 供
        _write_hits_report 生成总命中 MD 文档（视频名/起止时分秒/片段路径）。
        """
        clip_dir = self._out_dir / "clips" / run_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        clip_path = clip_dir / f"hit_seg{seg_idx:04d}_{seg_start:.0f}s.mp4"
        pad = self.config.clip_padding
        start = max(0.0, seg_start - pad / 2)
        dur = pad
        end = start + dur
        if self._ffmpeg and not clip_path.exists():
            cmd = [self._ffmpeg, "-y", "-ss", f"{start:.2f}",
                   "-i", str(video), "-t", f"{dur:.2f}",
                   "-c", "copy", str(clip_path)]
            try:
                subprocess.run(cmd, capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                logger.warning(f"[batch] clip 裁剪超时 seg {seg_idx}")
        # 记录 hit + clip（add_hit 递增 hits_count 并写 clips 表）
        # abs_timestamp 用秒数（更精确，避免 gmtime 乱跳）
        self.run_store.add_hit(run_id, {
            "hit_idx": seg_idx,
            "abs_timestamp": f"{seg_start:.1f}",
            "clip_path": str(clip_path) if clip_path.exists() else "",
        })
        # 命中详情写内存（供 batch_finished 后生成 MD 报告）
        self._hits_meta.append({
            "run_id": run_id,
            "video_name": video.name,
            "video_path": str(video),
            "seg_idx": seg_idx,
            "seg_start_sec": seg_start,
            "clip_start_sec": round(start, 1),
            "clip_end_sec": round(end, 1),
            "confidence": conf,
            "reason": reason,
            "clip_path": str(clip_path) if clip_path.exists() else "",
        })

    def _fmt_timecode(sec: float) -> str:
        """秒 → HH:MM:SS（24h，监控时间码）。"""
        s = max(0.0, float(sec))
        h = int(s // 3600); m = int((s % 3600) // 60); ss = int(s % 60)
        return f"{h:02d}:{m:02d}:{ss:02d}"

    def _write_hits_report(self) -> None:
        """批量跑完后写总命中 MD 文档：每个命中的视频名/起止时分秒/片段路径/置信度/原因。

        输出到 out_dir/HITS_REPORT.md，用户可直接查所有可靠命中的总览。
        """
        if not self._hits_meta:
            logger.info("[batch] 无命中，跳过 HITS_REPORT.md")
            return
        md = ["# 监控视频分析 — 总命中报告\n\n",
              f"**模型**: {self.config.model}\n",
              f"**物品**: {self.config.item_description or '关键物品'}\n",
              f"**总命中数**: {len(self._hits_meta)}（均经二次验证 has_target_item=true）\n\n",
              "## 命中清单\n\n",
              "| # | 视频文件 | 命中时间 | 片段起 | 片段止 | 置信度 | 片段路径 | AI 描述 |\n",
              "|---|---------|---------|--------|--------|--------|---------|--------|\n"]
        for i, h in enumerate(self._hits_meta, 1):
            tc = BatchRunner._fmt_timecode(h["seg_start_sec"])
            tc_s = BatchRunner._fmt_timecode(h["clip_start_sec"])
            tc_e = BatchRunner._fmt_timecode(h["clip_end_sec"])
            clip = h["clip_path"].replace("\\", "/") if h["clip_path"] else "(未裁剪)"
            reason = (h["reason"] or "").replace("|", "/").replace("\n", " ")[:80]
            md.append(f"| {i} | {h['video_name']} | {tc} | {tc_s} | {tc_e} | "
                      f"{h['confidence']:.2f} | `{clip}` | {reason} |\n")
        md.append("\n## 说明\n\n")
        md.append("- 每个命中均经过二次 NVIDIA Nemotron 验证（has_target_item=true）。\n")
        md.append("- 片段为命中点 ±clip_padding/2 秒，ffmpeg 无损裁剪（-c copy）。\n")
        md.append("- 时间码为视频内相对时间（HH:MM:SS）。\n")
        report_path = self._out_dir / "HITS_REPORT.md"
        report_path.write_text("".join(md), encoding="utf-8")
        logger.info(f"[batch] 总命中报告: {report_path}")

    # ------------------------------------------------------------------
    # 续跑 / 取消
    # ------------------------------------------------------------------
    def _find_resumable_run(self, video: Path) -> Optional[str]:
        """查 run_store 里 video_path 匹配且 status=started/running 的 run。

        用 status 过滤查询（store 有 idx_runs_status 索引），避免在万级 done
        run 里全表扫。
        """
        if not self.config.resume:
            return None
        vname = str(video)
        try:
            for status in ("running", "started"):
                for r in self.run_store.list_runs(limit=200, status=status):
                    if r.get("video_path") == vname:
                        return r.get("run_id")
        except Exception:
            return None
        return None

    def _completed_seg_indices(self, run_id: str) -> set:
        """查 run_store 里该 run 已完成的 seg_idx 集合（status=ok/failed）。"""
        run = self.run_store.get_run(run_id)
        if not run:
            return set()
        out = set()
        for s in run.get("segments", []):
            if s.get("status") in ("ok", "failed", "skipped"):
                out.add(int(s.get("seg_idx", -1)))
        return out

    def _count_existing_hits(self, run_id: str) -> int:
        """查 run_store 里该 run 已记录的命中数。"""
        run = self.run_store.get_run(run_id)
        if not run:
            return 0
        return int(run.get("hits_count", 0) or 0)

    def resume_batch(self) -> Tuple[int, int]:
        """续跑：查 run_store 里 status=started/running 的 run，继续未完成的。

        对每个未完成 run，重新跑其 video_path（分片会跳过已完成 seg_idx）。
        Returns:
            (total_runs, total_hits)
        """
        try:
            runs = self.run_store.list_runs(limit=500)
        except Exception as e:
            self.error.emit(f"resume 查询失败: {e}")
            return 0, 0
        pending = [r for r in runs if r.get("status") in ("started", "running")]
        videos = []
        for r in pending:
            p = r.get("video_path")
            if p and Path(p).exists():
                videos.append(Path(p))
        if not videos:
            logger.info("[batch] resume 无未完成 run")
            return 0, 0
        return self.run_batch(videos)

    def cancel(self) -> None:
        """优雅停止：设标志位，当前视频跑完停（不强杀 HTTP）。"""
        self._cancel_flag = True
        logger.info("[batch] cancel 已请求，当前视频跑完即停")

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_key_item(path: str) -> Optional[str]:
        """关键物品图 → base64（不带 data: 前缀，调用方拼）。"""
        p = Path(path)
        if not p.exists():
            logger.error(f"[batch] 关键物品图不存在: {path}")
            return None
        try:
            return base64.b64encode(p.read_bytes()).decode("utf-8")
        except Exception as e:
            logger.error(f"[batch] 关键物品图编码失败: {e}")
            return None

    @staticmethod
    def _encode_file_b64(path: Path) -> Optional[str]:
        try:
            return base64.b64encode(path.read_bytes()).decode("utf-8")
        except Exception as e:
            logger.warning(f"[batch] 文件 base64 编码失败 {path}: {e}")
            return None

    def _probe_duration(self, video: Path) -> float:
        """读视频时长。优先 cv2，回退 ffprobe，最后回退 ffmpeg -i stderr 解析。

        监控 H.264 流的 PPS 头常让 cv2/ffprobe 读不到（surveillance_agent 注释
        同款问题："No start code is found" / "decode_slice_header error"）。
        ffmpeg -i stderr 的 "Duration: HH:MM:SS.xx" 行最鲁棒（不解码帧）。
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
            logger.debug(f"[batch] cv2 读时长失败 {video.name}: {e}")
        # 回退 1：ffprobe
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
        # 回退 2：ffmpeg -i stderr 解析 Duration
        if self._ffmpeg:
            try:
                cmd = [self._ffmpeg, "-hide_banner", "-i", str(video)]
                out = subprocess.run(cmd, capture_output=True, timeout=15)
                # ffmpeg 无 -output 时 exit code !=0，但 stderr 含元数据
                err = out.stderr.decode("utf-8", "ignore")
                m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
                if m:
                    h, mi, s = m.group(1), m.group(2), m.group(3)
                    return int(h) * 3600 + int(mi) * 60 + float(s)
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _gc_collect() -> None:
        """显式 GC（万级视频防内存泄漏）。记日志但不强断言。"""
        collected = gc.collect()
        logger.debug(f"[batch] gc.collect 回收 {collected} 对象")
