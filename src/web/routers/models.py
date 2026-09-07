"""模型管理路由。

  GET  /api/models              → 4 张必下模型卡状态 + 本地扫描模型列表
  POST /api/models/{id}/download → 启动下载(SSE 进度) + SHA256 校验
  GET  /api/models/{id}/download/stream → SSE: progress/verify/done/error
  POST /api/models/{id}/verify  → SHA256 校验
  POST /api/models/detect-type  → 探测本地模型类型(VL/Text-only)

复用 src/core/logic.ModelManager(download_model + verify_model_integrity)。
下载在后台线程跑,进度通过 asyncio.Queue → SSE。
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..deps import get_job_store, web_jobs_root
from ..job_store import JobStatus
from ..schemas import SSEEvent
from ..security import require_auth, require_rate_limit

log = logging.getLogger("web.models")

router = APIRouter(prefix="/api/models", tags=["models"])


# 4 张必下模型卡(与 ModelManager.EXPECTED_SHA256 / download urls 对齐)
MODEL_CARDS = [
    {"id": "yolo_v11n", "name": "YOLOv11n", "desc": "物体检测", "size_hint": "~6MB"},
    {"id": "whisper_base", "name": "Whisper Base", "desc": "音频转录", "size_hint": "~140MB"},
    {"id": "st_minilm", "name": "Sentence-Transformer", "desc": "语义嵌入", "size_hint": "~90MB"},
    {"id": "ffmpeg", "name": "FFmpeg", "desc": "视频解码", "size_hint": "~80MB"},
]


def _get_manager():
    """延迟构造 ModelManager(避免启动时即扫描 models 目录)。"""
    from src.core.logic import ModelManager
    return ModelManager()


@router.get("", dependencies=[Depends(require_auth)])
def list_models() -> dict:
    """模型卡状态 + 本地扫描列表。"""
    mgr = _get_manager()
    cards = []
    for c in MODEL_CARDS:
        path = mgr.get_model_path(c["id"])
        exists = path is not None and Path(path).exists()
        cards.append({
            **c,
            "exists": exists,
            "path": str(path) if path else None,
            "size_mb": round(Path(path).stat().st_size / 1024 / 1024, 1) if exists and path else 0,
            "sha256_expected": mgr.EXPECTED_SHA256.get(c["id"]) is not None,
        })
    local_models = [
        {"name": n, "type": mgr.detect_model_type(n)}
        for n in mgr.list_local_models()
    ]
    return {"cards": cards, "local_models": local_models}


@router.post(
    "/{model_id}/download",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_auth), Depends(require_rate_limit)],
)
def start_download(model_id: str, request: Request) -> dict:  # type: ignore[no-untyped-def]
    """启动模型下载。返回 job_id,前端订阅 stream 路由看进度。"""
    if model_id not in {c["id"] for c in MODEL_CARDS}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"unknown model_id: {model_id}"},
        )
    store = get_job_store()

    job_id = f"dl_{model_id}_{int(time.time())}"
    workdir = web_jobs_root() / job_id
    workdir.mkdir(parents=True, exist_ok=True)
    rec = store.create(model_id, workdir, workdir)
    rec.job_id = job_id
    # 用可读 id 覆盖 store 里的随机 id 映射
    store._jobs[job_id] = rec  # type: ignore[attr-defined]
    # 删掉 create 返回的旧随机 id 记录(store.create 用 uuid 生成 job_id,
    # 我们改成了可读 id,旧 key 需要清掉避免两条指向同一 record)
    stale = [k for k in store._jobs if k != job_id and store._jobs[k] is rec]  # type: ignore[attr-defined]
    for k in stale:
        del store._jobs[k]  # type: ignore[attr-defined]

    _launch_download(model_id, rec, _get_app_loop(request))
    return {"job_id": job_id, "model_id": model_id, "status": "running"}


@router.get("/{model_id}/download/stream", dependencies=[Depends(require_auth)])
async def download_stream(model_id: str, request: Request):
    """SSE 下载进度。"""
    from ..deps import get_job_store
    store = get_job_store()
    job_id_prefix = f"dl_{model_id}_"
    rec = None
    for jid, r in store._jobs.items():  # type: ignore[attr-defined]
        if jid.startswith(job_id_prefix):
            rec = r
            break
    if rec is None:
        raise HTTPException(status_code=404, detail={"error": "no active download"})

    from sse_starlette.sse import EventSourceResponse
    from ..sse import stream_job_events

    # v10.2.0:支持断线续连(读 Last-Event-ID)
    last_event_id: int | None = None
    raw = request.headers.get("last-event-id")
    if raw:
        try:
            last_event_id = int(raw)
        except ValueError:
            last_event_id = None

    async def _gen():
        async for evt in stream_job_events(rec, last_event_id=last_event_id):
            if await request.is_disconnected():
                return
            yield evt

    return EventSourceResponse(_gen())


@router.post("/{model_id}/verify", dependencies=[Depends(require_auth)])
def verify_model(model_id: str) -> dict:
    """手动触发 SHA256 校验。"""
    if model_id not in {c["id"] for c in MODEL_CARDS}:
        raise HTTPException(status_code=400, detail={"error": f"unknown model_id: {model_id}"})
    mgr = _get_manager()
    path = mgr.get_model_path(model_id)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail={"error": "model not found"})
    ok = mgr.verify_model_integrity(model_id)
    return {"model_id": model_id, "valid": ok, "path": str(path)}


@router.post("/detect-type", dependencies=[Depends(require_auth)])
def detect_type(payload: dict) -> dict:
    """探测本地模型类型。"""
    filename = payload.get("filename", "")
    if not filename:
        raise HTTPException(status_code=400, detail={"error": "filename required"})
    return {"filename": filename, "type": _get_manager().detect_model_type(filename)}


# ============================ 后台下载 ============================

def _get_app_loop(request):
    return request.app.state.analyzer_service._loop


def _launch_download(model_id: str, rec, loop: asyncio.AbstractEventLoop) -> None:
    """后台线程跑 download_model,进度推 SSE。"""
    def push(event_type: str, data: dict):
        # v10.2.0:走 rec.put_event 统一分配 seq + 维护缓冲,支持断线续连
        try:
            loop.call_soon_threadsafe(rec.put_event, event_type, data)
        except Exception as e:
            log.debug(f"push failed: {e}")

    def run():
        rec.status = JobStatus.RUNNING
        rec.started_at = time.time()
        push(SSEEvent.LOG, {"level": "info", "msg": f"开始下载 {model_id}"})
        try:
            from src.core.logic import ModelManager
            mgr = ModelManager()

            def on_progress(pct: int):
                push("progress", {"value": pct, "label": f"下载中 {pct}%"})

            ok = mgr.download_model(model_id, progress_callback=on_progress)
            if ok:
                push("verify", {"ok": True, "msg": "SHA256 校验通过"})
                push(SSEEvent.DONE, {"model_id": model_id, "ok": True})
            else:
                push(SSEEvent.ERROR, {"message": "下载或校验失败,见日志"})
        except Exception as e:
            log.exception(f"download failed: {e}")
            push(SSEEvent.ERROR, {"message": str(e)})
        finally:
            rec.status = JobStatus.DONE if rec.status == JobStatus.RUNNING else rec.status
            rec.finished_at = time.time()
            push(SSEEvent._CLOSE, {})

    threading.Thread(target=run, daemon=True, name=f"dl-{model_id}").start()
