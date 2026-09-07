"""作业状态仓库(进程内,线程安全)。

阶段 0:内存实现,满足单机单用户的核心闭环。
阶段 3 批量处理时桥接 src/core/run_store.py(SQLite 断点续跑)。

每个 JobRecord 持有一个 asyncio.Queue,SSE 端点 await queue.get(),
后台分析线程通过 loop.call_soon_threadsafe 把事件推入队列。

v10.2.0:SSE 断线续连。每个事件带自增 seq,JobRecord 维护
last_event_seq + recent_events 环形缓冲(最近 100 条带 seq 的快照)。
客户端断线重连时带 Last-Event-ID,服务端从 recent_events 重放
seq > last_event_id 的事件,再继续 live 流。
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger("web.job_store")

# recent_events 环形缓冲容量。最近 N 条带 seq 的事件快照,供断线重连续推。
# 太小:长断线丢事件;太大:内存占用。100 条够覆盖一次心跳周期 + 重连窗口。
RECENT_EVENTS_CAP = 100


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class JobRecord:
    job_id: str
    video_name: str
    workdir: Path
    frames_dir: Path
    video_path: Optional[str] = None  # 原始视频绝对路径(metrics/media 回溯用)
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    # 分析结果摘要(供 GET /api/jobs/{id} 非流式查询)
    duration: float = 0.0
    frame_count: int = 0
    transcript: str = ""
    report: str = ""
    # 已提取帧的载荷列表(供 GET /api/jobs/{id}/frames 非流式查询)
    frames: list[dict] = field(default_factory=list)
    # metrics 产物(get_advanced_video_metrics 的 avg + 图表 png 路径)
    metrics_avg: dict = field(default_factory=dict)
    metrics_chart_path: Optional[str] = None
    # 摘要媒体产物(create_summary_media_artifacts)
    media_clips: list[str] = field(default_factory=list)
    media_summary_video: Optional[str] = None
    media_gif: Optional[str] = None
    # SSE 事件队列:后台线程 push,SSE 端点 pull。无界,防丢事件。
    # 在 start_job 时用所属 loop 构造。
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=0))
    # 标记事件流是否已关闭(收到 __close__)
    stream_closed: bool = False
    # ── SSE 断线续连(v10.2.0)──
    # 最近发出的事件 seq(单调递增,每次 put 自增)。Last-Event-ID 比对基准。
    last_event_seq: int = 0
    # 环形缓冲:最近 RECENT_EVENTS_CAP 条带 seq 的事件快照。
    # 重连时按 seq > last_event_id 过滤后重放,避免丢事件。
    # 元素结构:{"seq": int, "type": str, "data": dict}
    recent_events: list[dict] = field(default_factory=list)
    # 保护 last_event_seq + recent_events 的锁(后台线程 call_soon_threadsafe
    # 与 SSE 端点并发读写,需互斥)。queue 自身线程安全,但 seq + 缓冲需一致快照。
    _seq_lock: threading.Lock = field(default_factory=threading.Lock)

    def put_event(self, event_type: str, data: dict) -> int:
        """线程安全地推事件到 queue + 维护 seq/缓冲。返回该事件 seq。

        由 analyzer_service.push / batch / surveillance 等后台线程调用,
        经 loop.call_soon_threadsafe 调度到主 loop。本方法在主 loop 线程执行,
        但 SSE 端点也可能并发读 recent_events,故仍加锁保一致。

        __close__ 哨兵不分配 seq 也不入缓冲(它是流结束信号,不应重放)。
        """
        with self._seq_lock:
            if event_type == "__close__":
                # 哨兵不入缓冲,直接入队。无 seq。
                self.queue.put_nowait({"type": event_type, "data": data, "seq": None})
                return 0
            self.last_event_seq += 1
            seq = self.last_event_seq
            snapshot = {"seq": seq, "type": event_type, "data": data}
            self.recent_events.append(snapshot)
            # 环形缓冲:超容量丢弃最旧
            if len(self.recent_events) > RECENT_EVENTS_CAP:
                del self.recent_events[: len(self.recent_events) - RECENT_EVENTS_CAP]
        self.queue.put_nowait({"type": event_type, "data": data, "seq": seq})
        return seq

    def replay_since(self, last_event_id: int) -> list[dict]:
        """返回 recent_events 中 seq > last_event_id 的快照(按 seq 升序)。

        客户端断线重连带 Last-Event-ID,服务端先重放这些事件再继续 live 流。
        加锁保证读到一致快照(并发 put 不会撕裂 list)。
        """
        with self._seq_lock:
            return [e for e in self.recent_events if (e.get("seq") or 0) > last_event_id]

    def summary(self) -> dict:
        """非流式摘要(不含 queue)。"""
        return {
            "job_id": self.job_id,
            "video_name": self.video_name,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": round(self.duration, 2),
            "frame_count": self.frame_count,
            "transcript_preview": self.transcript[:500],
            "report_preview": self.report[:500],
            "error": self.error,
        }


class JobStore:
    """线程安全的作业注册表。

    注:asyncio.Queue 在 record 创建时构造,但创建可能发生在非 async 上下文
    (POST handler 是 async,所以 OK)。queue 的真正消费发生在 SSE 端点的 loop。
    """

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()  # 保护 dict 增删

    def create(self, video_name: str, workdir: Path, frames_dir: Path) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        rec = JobRecord(
            job_id=job_id,
            video_name=video_name,
            workdir=workdir,
            frames_dir=frames_dir,
        )
        self._jobs[job_id] = rec
        log.info(f"job created: {job_id} ({video_name})")
        return rec

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def list_all(self, limit: int = 50) -> list[JobRecord]:
        # 按 created_at 倒序
        items = sorted(self._jobs.values(), key=lambda r: r.created_at, reverse=True)
        return items[:limit]

    def delete(self, job_id: str) -> bool:
        rec = self._jobs.pop(job_id, None)
        if rec is None:
            return False
        # 清理工作目录(帧图片)
        import shutil as _sh
        _sh.rmtree(rec.workdir, ignore_errors=True)
        return True

    def cleanup_stale(self, max_age_sec: float = 3600.0) -> int:
        """清理超过 max_age_sec 的已完成/失败作业。"""
        now = time.time()
        stale = [
            jid for jid, rec in self._jobs.items()
            if rec.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED)
            and now - (rec.finished_at or rec.created_at) > max_age_sec
        ]
        for jid in stale:
            self.delete(jid)
        return len(stale)
