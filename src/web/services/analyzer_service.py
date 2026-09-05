"""分析服务:把 src/core 的同步三阶段流水线包成后台任务 + SSE 事件发射。

这是 GUI 层 ExtractionWorker/AnalysisWorker 的 Web 等价物。
QThread 的 pyqtSignal → 这里是 JobRecord.queue.put_nowait。
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

import cv2

from src.core.logic import (
    VideoProcessor,
    AudioProcessor,
    VideoAnalyzer,
    OllamaClient,
    PromptLoader,
    Frame,
    videocapture_unicode,
)
from src.utils.config_manager import ConfigurationManager

from ..job_store import JobRecord, JobStatus
from ..schemas import SSEEvent

log = logging.getLogger("web.analyzer")


def _probe_duration(video_path: Path) -> float:
    """用 OpenCV 探测视频时长(秒)。复用 headless.py 的逻辑。"""
    try:
        cap = videocapture_unicode(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return total / fps if fps > 0 else 0.0
    except Exception:
        return 0.0


def _run_in_thread(
    target, args, on_start: callable, on_done: callable
) -> threading.Thread:
    """启动 daemon 线程跑 target,on_start 在线程入口设状态,on_done 收尾。"""
    def wrapper():
        try:
            on_start()
            target(*args)
        except Exception as e:
            log.exception(f"analysis job failed: {e}")
            raise
        finally:
            on_done()
    t = threading.Thread(target=wrapper, daemon=True, name="vap-analyze")
    t.start()
    return t


class AnalyzerService:
    """三阶段分析编排。

    用法:
        svc = AnalyzerService(job_store, config_manager, loop)
        job = svc.create_job(video_path, video_name, config)
        # 前端订阅 GET /api/jobs/{job_id}/stream → svc 的事件队列
    """

    def __init__(
        self,
        job_store,
        config_manager: ConfigurationManager,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._store = job_store
        self._cm = config_manager
        self._loop = loop

    def create_job(
        self,
        video_path: Path,
        video_name: str,
        config: dict,
        workdir: Path | None = None,
    ) -> JobRecord:
        """创建作业记录 + 启动后台线程。返回 record(含 job_id)。

        workdir 由路由层分配(web_jobs_root/<uuid>/),保证帧隔离在
        受控目录内 —— 绝不写回用户视频所在目录(local_path 模式)。
        未传时兜底用视频同级 frames/ 目录。
        """
        if workdir is None:
            workdir = Path(video_path).parent
        frames_dir = workdir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        rec = self._store.create(video_name, workdir, frames_dir)

        # 后台线程入口:在子线程里跑,通过 loop.call_soon_threadsafe 推 SSE 事件
        def push(event_type: str, data: dict):
            """从子线程向主 loop 的 queue 推事件。"""
            try:
                self._loop.call_soon_threadsafe(
                    rec.queue.put_nowait, {"type": event_type, "data": data}
                )
            except Exception as e:
                log.debug(f"push failed (job {rec.job_id}): {e}")

        def on_start():
            rec.status = JobStatus.RUNNING
            rec.started_at = time.time()
            push(SSEEvent.PHASE, {"phase": "extraction"})

        def on_done():
            # close 信号由 target 内部发(SSEEvent.DONE / ERROR)
            # 这里只保证 stream_closed 标记 + 释放
            push(SSEEvent._CLOSE, {})
            rec.stream_closed = True

        def run_analysis():
            self._run_pipeline(rec, video_path, config, push)

        _run_in_thread(run_analysis, (), on_start, on_done)
        return rec

    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        rec: JobRecord,
        video_path: Path,
        config: dict,
        push: callable,
    ) -> None:
        """三阶段流水线。任一阶段异常 → ERROR 事件 + 状态 FAILED。"""
        try:
            # ── Phase 1: 数据提取 ──
            push(SSEEvent.PHASE, {"phase": "extraction"})
            push(SSEEvent.LOG, {"level": "info", "msg": f"Phase 1: 抽帧 {video_path.name}"})

            processor = VideoProcessor(video_path, rec.frames_dir)
            if config.get("smart_extraction"):
                frames = processor.extract_smart_keyframes()
            else:
                frames = processor.extract_keyframes(density=config["density"])

            rec.frame_count = len(frames)
            rec.duration = _probe_duration(video_path)
            push(SSEEvent.PROGRESS, {"label": "抽帧完成", "value": 30})

            # 逐帧推给前端(画廊/时间轴实时填充)
            for f in frames:
                payload = self._frame_payload(rec, f)
                rec.frames.append(payload)
                push(SSEEvent.FRAME, payload)

            # 音频转录
            transcript = ""
            if config.get("enable_audio"):
                push(SSEEvent.PHASE, {"phase": "audio"})
                push(SSEEvent.LOG, {"level": "info", "msg": "Phase 1.5: Whisper 音频转录"})
                audio_proc = AudioProcessor()
                audio_path = audio_proc.extract_audio(video_path, rec.workdir)
                if audio_path:
                    tr = audio_proc.transcribe(audio_path)
                    transcript = tr.text if tr else ""
                rec.transcript = transcript
                push(SSEEvent.TRANSCRIPT, {"text": transcript})

            push(SSEEvent.PROGRESS, {"label": "数据提取完成", "value": 50})

            # ── Phase 2: AI 分析(LLM 流式)──
            push(SSEEvent.PHASE, {"phase": "analysis"})
            report = ""
            try:
                client = OllamaClient()
                analyzer = VideoAnalyzer(
                    client,
                    config["model"],
                    PromptLoader(),
                    use_yolo=config.get("enable_yolo", False),
                    use_ocr=config.get("enable_ocr", False),
                )
                chunks: list[str] = []
                for chunk in analyzer.analyze_video(frames, transcript, config.get("custom_prompt")):
                    if chunk.startswith("__"):
                        continue
                    chunks.append(chunk)
                    # 流式 token 直推前端
                    push(SSEEvent.REPORT_TOKEN, {"token": chunk})
                report = "".join(chunks)
            except Exception as e:
                log.warning(f"LLM 阶段降级: {e}")
                report = f"[LLM 阶段不可用: {e}]"
                push(SSEEvent.LOG, {"level": "warning", "msg": str(e)})

            rec.report = report
            push(SSEEvent.PROGRESS, {"label": "AI 分析完成", "value": 90})

            # ── 完成 ──
            rec.status = JobStatus.DONE
            rec.finished_at = time.time()
            push(SSEEvent.DONE, {
                "duration": rec.duration,
                "frame_count": rec.frame_count,
                "report_preview": report[:500],
            })
        except Exception as e:
            rec.status = JobStatus.FAILED
            rec.error = str(e)
            rec.finished_at = time.time()
            push(SSEEvent.ERROR, {"message": str(e)})
            log.exception(f"pipeline failed (job {rec.job_id})")

    def _frame_payload(self, rec: JobRecord, f: Frame) -> dict:
        """单帧 SSE 载荷。url 用 job_id + 帧文件名(前端拼 /api/frames/...)。"""
        return {
            "timestamp": round(f.timestamp, 2),
            "metrics": {k: round(v, 2) for k, v in f.metrics.items()},
            "url": f"/api/jobs/{rec.job_id}/frames/{f.path.name}",
            "vision_content": f.vision_content,
            "ocr_text": f.ocr_text,
        }
