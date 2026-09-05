"""系统日志路由。

  GET  /api/logs/stream  → SSE 实时日志流(转发 src.core logger)
  GET  /api/logs?limit=N → 最近 N 条日志(内存环形缓冲)

后端日志通过内存环形缓冲 + asyncio.Queue 推送,替代 PyQt6 的 log_signal。
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from typing import Deque

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from ..schemas import SSEEvent
from ..security import require_auth

log = logging.getLogger("web.logs")

router = APIRouter(prefix="/api/logs", tags=["logs"])

# 进程级环形缓冲 + 订阅者列表(多客户端可同时订阅)
_BUFFER: Deque[dict] = deque(maxlen=500)
_BUFFER_LOCK = threading.Lock()
_SUBSCRIBERS: list[asyncio.Queue] = []
_SUB_LOCK = threading.Lock()


class _LogBridge(logging.Handler):
    """把 Python logging 记录桥接到 SSE 订阅者 + 环形缓冲。"""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.setFormatter(logging.Formatter("%(name)s [%(levelname)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts": record.created,
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": self.format(record),
        }
        with _BUFFER_LOCK:
            _BUFFER.append(entry)
        with _SUB_LOCK:
            dead: list[asyncio.Queue] = []
            for q in _SUBSCRIBERS:
                try:
                    q.put_nowait(entry)
                except Exception:
                    dead.append(q)
            for q in dead:
                _SUBSCRIBERS.remove(q)


_bridge_installed = False


def install_log_bridge() -> None:
    """启动时安装一次日志桥(挂到 root logger)。"""
    global _bridge_installed
    if _bridge_installed:
        return
    logging.getLogger().addHandler(_LogBridge())
    _bridge_installed = True


@router.get("", dependencies=[Depends(require_auth)])
def list_logs(limit: int = 100) -> dict:
    """最近 N 条日志(环形缓冲)。确保桥已安装。"""
    install_log_bridge()
    with _BUFFER_LOCK:
        items = list(_BUFFER)[-limit:]
    return {"logs": items, "count": len(items)}


@router.get("/stream", dependencies=[Depends(require_auth)])
async def stream_logs(request: Request):
    """SSE 实时日志流。客户端断开时退出。"""
    install_log_bridge()
    q: asyncio.Queue = asyncio.Queue()
    with _SUB_LOCK:
        _SUBSCRIBERS.append(q)

    async def _gen():
        try:
            while True:
                if await request.is_disconnected():
                    return
                try:
                    entry = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield {
                        "event": SSEEvent.LOG,
                        "data": __import__("json").dumps(entry, ensure_ascii=False),
                    }
                except asyncio.TimeoutError:
                    yield {"comment": "keepalive"}
        finally:
            with _SUB_LOCK:
                if q in _SUBSCRIBERS:
                    _SUBSCRIBERS.remove(q)

    return EventSourceResponse(_gen())
