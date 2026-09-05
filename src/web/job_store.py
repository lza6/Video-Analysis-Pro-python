"""作业状态仓库(进程内,线程安全)。

阶段 0:内存实现,满足单机单用户的核心闭环。
阶段 3 批量处理时桥接 src/core/run_store.py(SQLite 断点续跑)。

每个 JobRecord 持有一个 asyncio.Queue,SSE 端点 await queue.get(),
后台分析线程通过 loop.call_soon_threadsafe 把事件推入队列。
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger("web.job_store")


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
    # SSE 事件队列:后台线程 push,SSE 端点 pull。无界,防丢事件。
    # 在 start_job 时用所属 loop 构造。
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=0))
    # 标记事件流是否已关闭(收到 __close__)
    stream_closed: bool = False

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
