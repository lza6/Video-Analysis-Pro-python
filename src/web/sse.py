"""SSE(Server-Sent Events)辅助。

把 JobRecord.queue 的事件转成 sse-starlette 的 ServerSentEvent,
并在收到 __close__ 哨兵时结束流。

v10.2.0:断线续连。每个 yield 的 ServerSentEvent 带 id=<seq>,
客户端断线重连带 Last-Event-ID 时,先从 JobRecord.recent_events
重放 seq > last_event_id 的事件,再继续 live 流。

重放 vs live 一致性:put_event 同时入 recent_events 和 queue。
重连时若只重放 recent_events,live 阶段从 queue 取到旧事件会重复。
解法:live 阶段跟踪 replayed_max_seq,跳过 seq <= 已重放/已 ACK
的事件,保证 Last-Event-ID 单调递增不重复。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

from sse_starlette.sse import ServerSentEvent

from .job_store import JobRecord
from .schemas import SSEEvent

log = logging.getLogger("web.sse")


async def stream_job_events(
    rec: JobRecord,
    heartbeat_sec: float = 15.0,
    last_event_id: Optional[int] = None,
) -> AsyncIterator[ServerSentEvent]:
    """从作业队列持续 yield SSE 事件,直到收到 __close__ 或流已关闭。

    - 每个事件带 event=<type>, data=<json>, id=<seq>
    - 若 last_event_id 给定:先从 rec.recent_events 重放 seq > last_event_id
      的事件(历史快照),再继续 live 流
    - 每 heartbeat_sec 无事件时发一个 comment 保活(防代理断连)
    - 收到 __close__ 哨兵 → 结束
    - 若 rec.stream_closed 且 live queue 已排空(无新事件)→ 结束

    重放阶段不发心跳(重放是同步快照,瞬时完成,无需保活)。
    live 阶段跟踪 replayed_max_seq,跳过 seq <= 已重放/已 ACK 的事件,
    防止 put_event 同时入 queue 和 recent_events 导致重连后重复消费。
    """
    # 已发/已 ACK 的最大 seq。live 阶段跳过 <= 此值的事件。
    replayed_max_seq = last_event_id if last_event_id is not None else 0

    # ── 重放阶段:从环形缓冲补发断线期间漏掉的事件 ──
    if last_event_id is not None:
        missed = rec.replay_since(last_event_id)
        for snap in missed:
            etype = snap.get("type")
            data = snap.get("data", {})
            seq = snap.get("seq")
            if etype is None or etype == SSEEvent._CLOSE or seq is None:
                continue
            yield ServerSentEvent(
                event=etype,
                data=json.dumps(data, ensure_ascii=False),
                id=str(seq),
            )
            if seq > replayed_max_seq:
                replayed_max_seq = seq

    # ── live 阶段:持续 await queue ──
    while True:
        # 流已关闭且 queue 为空 → 重放已覆盖到末尾,直接结束(避免卡心跳)
        if rec.stream_closed and _queue_drained(rec):
            return

        try:
            evt = await asyncio.wait_for(rec.queue.get(), timeout=heartbeat_sec)
        except asyncio.TimeoutError:
            # 保活注释(前端 EventSource 忽略以冒号开头的行)
            yield ServerSentEvent(comment="keepalive")
            continue

        etype = evt.get("type")
        data = evt.get("data", {})
        seq = evt.get("seq")

        if etype == SSEEvent._CLOSE:
            return
        if etype is None:
            continue

        # 跳过已重放或已 ACK 的事件(put_event 同时入 queue 和 recent_events,
        # 重连后 queue 里可能残留旧事件,必须过滤防止重复)
        if seq is not None and seq <= replayed_max_seq:
            continue

        if seq is not None and seq > replayed_max_seq:
            replayed_max_seq = seq

        yield ServerSentEvent(
            event=etype,
            data=json.dumps(data, ensure_ascii=False),
            id=str(seq) if seq is not None else None,
        )


def _queue_drained(rec: JobRecord) -> bool:
    """queue 里是否还有未消费事件。stream_closed 时用于判断是否可收尾。

    asyncio.Queue.empty() 是近似值(非线程安全),但 stream_job_events
    在主 loop 线程跑,与 put_event 的 call_soon_threadsafe 调度同线程,
    判断足够准确。即使误判为非空,也只是多等一轮心跳后取到 None/旧事件,
    不会破坏正确性(旧事件会被 seq 过滤跳过)。
    """
    try:
        return rec.queue.empty()
    except Exception:
        return False
