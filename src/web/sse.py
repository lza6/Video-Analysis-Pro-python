"""SSE(Server-Sent Events)辅助。

把 JobRecord.queue 的事件转成 sse-starlette 的 ServerSentEvent,
并在收到 __close__ 哨兵时结束流。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from sse_starlette.sse import ServerSentEvent

from .job_store import JobRecord
from .schemas import SSEEvent

log = logging.getLogger("web.sse")


async def stream_job_events(
    rec: JobRecord,
    heartbeat_sec: float = 15.0,
) -> AsyncIterator[ServerSentEvent]:
    """从作业队列持续 yield SSE 事件,直到收到 __close__ 或心跳超时。

    - 每个事件带 event=<type>, data=<json>
    - 每 heartbeat_sec 无事件时发一个 comment 保活(防代理断连)
    - 收到 __close__ 哨兵 → 结束
    """
    while True:
        try:
            evt = await asyncio.wait_for(rec.queue.get(), timeout=heartbeat_sec)
        except asyncio.TimeoutError:
            # 保活注释(前端 EventSource 忽略以冒号开头的行)
            yield ServerSentEvent(comment="keepalive")
            continue

        etype = evt.get("type")
        data = evt.get("data", {})

        if etype == SSEEvent._CLOSE:
            return
        if etype is None:
            continue

        yield ServerSentEvent(
            event=etype,
            data=json.dumps(data, ensure_ascii=False),
        )
