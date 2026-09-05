"""监控分析路由(RTSP 实时流)。

  POST /api/surveillance/start  → 启动 RtspMonitor(后台线程拉流+运动检测+VLM)
  GET  /api/surveillance/events → 已累积事件列表(时间/类型/详情/置信度)
  GET  /api/surveillance/stream → SSE 实时命中事件流
  POST /api/surveillance/stop   → 停止监控

复用 src/core/rtsp_stream.RtspMonitor + llm_gateway.build_backend。
VLM backend 走 NVIDIA 多 key 路由(.env VAP_NV_API_KEYS)。
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..security import require_auth, require_rate_limit

log = logging.getLogger("web.surveillance")

router = APIRouter(prefix="/api/surveillance", tags=["surveillance"])

# 进程级单例(同时只跑一个监控,单 GPU 安全)
_monitor = None
_monitor_lock = threading.Lock()
# SSE 订阅者队列(命中事件广播)
_event_subscribers: list[asyncio.Queue] = []
_sub_lock = threading.Lock()


class StartReq(BaseModel):
    rtsp_url: str
    key_item_image: str = ""
    item_description: str = "关键物品"
    fps: float = 1.0
    motion_threshold: float = 25.0
    vlm_cooldown: float = 30.0


@router.post(
    "/start",
    status_code=201,
    dependencies=[Depends(require_auth), Depends(require_rate_limit)],
)
def start_surveillance(req: StartReq, request: Request) -> dict:
    """启动监控。若已在跑,先停掉旧的。"""
    global _monitor
    loop = request.app.state.analyzer_service._loop

    with _monitor_lock:
        if _monitor is not None:
            _stop_monitor(_monitor)
        _monitor = _build_and_start(req, loop)
        if _monitor is None:
            raise HTTPException(status_code=500, detail={"error": "启动监控失败,见日志"})
    return {"status": "running", "rtsp_url": _sanitize(req.rtsp_url)}


@router.post("/stop", dependencies=[Depends(require_auth)])
def stop_surveillance() -> dict:
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            return {"status": "idle"}
        _stop_monitor(_monitor)
        _monitor = None
    return {"status": "stopped"}


@router.get("/events", dependencies=[Depends(require_auth)])
def list_events(limit: int = 100) -> dict:
    with _monitor_lock:
        m = _monitor
    if m is None:
        return {"events": [], "running": False}
    return {
        "events": [_event_to_dict(e) for e in list(m.events)[-limit:]],
        "running": True,
        "count": len(m.events),
    }


@router.get("/stream", dependencies=[Depends(require_auth)])
async def event_stream(request: Request):
    """SSE 实时命中事件流。"""
    q: asyncio.Queue = asyncio.Queue()
    with _sub_lock:
        _event_subscribers.append(q)

    async def _gen():
        try:
            # 首推已累积事件
            with _monitor_lock:
                m = _monitor
            if m is not None:
                for e in list(m.events)[-50:]:
                    yield {
                        "event": "hit",
                        "data": __import__("json").dumps(_event_to_dict(e), ensure_ascii=False),
                    }
            while True:
                if await request.is_disconnected():
                    return
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield {"event": "hit", "data": __import__("json").dumps(ev, ensure_ascii=False)}
                except asyncio.TimeoutError:
                    yield {"comment": "keepalive"}
        finally:
            with _sub_lock:
                if q in _event_subscribers:
                    _event_subscribers.remove(q)

    return EventSourceResponse(_gen())


# ============================ helpers ============================

def _build_and_start(req: StartReq, loop: asyncio.AbstractEventLoop):
    """构造 backend + RtspMonitor,启动拉流。"""
    try:
        backend = _build_backend()
        if backend is None:
            log.error("无可用 VLM backend(配 .env VAP_NV_API_KEYS)")
            return None
        from src.core.rtsp_stream import RtspMonitor
        monitor = RtspMonitor(
            rtsp_url=req.rtsp_url,
            backend=backend,
            key_item_image=req.key_item_image,
            item_description=req.item_description,
            motion_threshold=req.motion_threshold,
            vlm_cooldown=req.vlm_cooldown,
        )
        # 轮询线程:把新事件广播给 SSE 订阅者
        last_count = 0
        def poller():
            nonlocal last_count
            while True:
                with _monitor_lock:
                    if _monitor is not monitor:
                        return
                cur = len(monitor.events)
                if cur > last_count:
                    new_events = monitor.events[last_count:]
                    last_count = cur
                    for e in new_events:
                        d = _event_to_dict(e)
                        with _sub_lock:
                            dead = []
                            for q in _event_subscribers:
                                try:
                                    loop.call_soon_threadsafe(q.put_nowait, d)
                                except Exception:
                                    dead.append(q)
                            for q in dead:
                                _event_subscribers.remove(q)
                time.sleep(1.0)

        monitor.start(fps=req.fps)
        threading.Thread(target=poller, daemon=True, name="rtsp-poll").start()
        return monitor
    except Exception as e:
        log.exception(f"start surveillance failed: {e}")
        return None


def _build_backend():
    """构建 VLM backend:优先 NVIDIA 多 key 路由。"""
    try:
        from src.core.provider_router import load_from_env
        from src.core.llm_gateway import build_backend
        nv_keys = [k for k in load_from_env() if k.provider == "nvidia"]
        if nv_keys:
            key = nv_keys[0]
            return build_backend("openai_chat", key.api_key, key.base_url, "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
        return None
    except Exception as e:
        log.warning(f"build_backend failed: {e}")
        return None


def _stop_monitor(monitor) -> None:
    try:
        monitor.stop()
    except Exception as e:
        log.warning(f"stop monitor failed: {e}")


def _event_to_dict(e) -> dict:
    return {
        "timestamp": float(getattr(e, "timestamp", 0) or 0),
        "kind": str(getattr(e, "kind", "") or ""),
        "detail": str(getattr(e, "detail", "") or ""),
        "confidence": float(getattr(e, "confidence", 0) or 0),
        "frame_url": _frame_to_url(getattr(e, "frame_path", "") or ""),
    }


def _frame_to_url(frame_path: str) -> str:
    """把监控帧磁盘路径转成可访问 URL(注册到 web_jobs 内容根)。"""
    if not frame_path:
        return ""
    p = Path(frame_path).resolve()
    from ..security import register_content_root
    try:
        register_content_root("rtsp_frames", p.parent)
    except Exception:
        pass
    return f"/api/surveillance/frame/{p.name}"


@router.get("/frame/{name}", dependencies=[Depends(require_auth)])
def frame_image(name: str):
    """监控帧图片(路径消毒)。"""
    from ..security import resolve_within_root, ALLOWED_IMAGE_EXTS
    # 帧在 cache/rtsp/ 下
    rtsp_root = Path("cache") / "rtsp"
    rtsp_root.mkdir(parents=True, exist_ok=True)
    full = resolve_within_root(rtsp_root.resolve(), name, ALLOWED_IMAGE_EXTS)
    if not full.is_file():
        raise HTTPException(status_code=404, detail={"error": "frame not found"})
    from fastapi.responses import FileResponse
    return FileResponse(full, media_type="image/jpeg")


def _sanitize(url: str) -> str:
    """隐藏 RTSP 凭据(不回传完整 url)。"""
    if "@" in url:
        try:
            scheme, rest = url.split("://", 1)
            host = rest.split("@", 1)[1] if "@" in rest else rest
            return f"{scheme}://***@{host}"
        except Exception:
            return "***"
    return url
