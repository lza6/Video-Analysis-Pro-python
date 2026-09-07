"""系统日志路由。

  GET  /api/logs/stream  → SSE 实时日志流(转发 src.core logger)
  GET  /api/logs?limit=N → 最近 N 条日志(内存环形缓冲)
  POST /api/logs         → 前端上报日志(错误边界/前端异常,落环形缓冲)

后端日志通过内存环形缓冲 + asyncio.Queue 推送,经 SSE 转发给前端。
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from typing import Deque

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..schemas import SSEEvent
from ..security import require_auth, require_rate_limit

log = logging.getLogger("web.logs")

router = APIRouter(prefix="/api/logs", tags=["logs"])

# 进程级环形缓冲 + 订阅者列表(多客户端可同时订阅)
_BUFFER: Deque[dict] = deque(maxlen=500)
_BUFFER_LOCK = threading.Lock()
_SUBSCRIBERS: list[asyncio.Queue] = []
_SUB_LOCK = threading.Lock()


class _LogBridge(logging.Handler):
    """把 Python logging 记录桥接到 SSE 订阅者 + 环形缓冲。

    v10.1.1:透传 trace_id/span_id(从 contextvars 读),让前端 /api/logs
    能按一次 Agent turn 串日志(用户启动日志实证 trace_id 全 null)。
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.setFormatter(logging.Formatter("%(name)s [%(levelname)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        # 延迟 import 避免循环依赖(structured_log 不依赖 web 层)
        try:
            from src.core.runtime.structured_log import TRACE, SPAN
            trace_id = TRACE.get()
            span_id = SPAN.get()
        except Exception:  # noqa: BLE001
            trace_id = None
            span_id = None
        entry = {
            "ts": record.created,
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": self.format(record),
            "trace_id": trace_id,
            "span_id": span_id,
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


class FrontendLogEntry(BaseModel):
    """前端上报的日志条目(错误边界/未捕获异常)。

    v10.1.0:前端 error.tsx 边界 POST 此结构到 /api/logs,
    落入环形缓冲供 /api/logs 查询 + SSE 订阅者收到。
    """

    level: str = Field("error", description="日志级别(error/warn/info)")
    message: str = Field(..., min_length=1, max_length=4000)
    digest: str | None = Field(None, description="Next.js error digest")
    stack: str | None = Field(None, description="stack trace 截断片段")
    source: str = Field("frontend", description="来源标记")


@router.post(
    "",
    dependencies=[Depends(require_auth), Depends(require_rate_limit)],
)
def report_log(entry: FrontendLogEntry) -> dict:
    """前端上报日志(错误边界用)。落环形缓冲 + 推 SSE 订阅者。

    不真实写库(环形缓冲内存级,够前端排查),失败不抛。
    """
    install_log_bridge()
    import time
    level = entry.level.lower()
    # 归一到 logging 合法级别(前端只报 error/warn/info)
    if level not in ("error", "warn", "info", "debug"):
        level = "info"
    frontend_entry = {
        "ts": time.time(),
        "level": level,
        "logger": "frontend",
        "msg": entry.message,
        "digest": entry.digest,
        "source": entry.source,
        "stack": entry.stack,
    }
    with _BUFFER_LOCK:
        _BUFFER.append(frontend_entry)
    with _SUB_LOCK:
        dead: list[asyncio.Queue] = []
        for q in _SUBSCRIBERS:
            try:
                q.put_nowait(frontend_entry)
            except Exception:  # noqa: BLE001
                dead.append(q)
        for q in dead:
            _SUBSCRIBERS.remove(q)
    # 同时落 Python logging(走结构化 JSON formatter)
    getattr(log, level if level != "warn" else "warning",
            log.info)(f"[frontend] {entry.message}")
    return {"ok": True, "level": level}


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
