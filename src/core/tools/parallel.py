"""工具并行执行 + 读写锁。

参考 codex `tools/parallel.rs:27`：工具并发执行，共享资源读锁共享、
独占资源写锁独占。Python asyncio 版用 asyncio.Lock 实现 RWLock 语义。

设计：
  - AsyncRWLock: 多读单写。读锁可多个同时持有；写锁独占，与所有读锁互斥。
  - ToolCallRuntime: 单次工具调用的运行态（持有锁种类、超时）。
  - ParallelExecutor: asyncio.gather 并发跑多个 ToolCall。

实现简洁优先：用一个 asyncio.Lock 保护内部计数器 + 一个 asyncio.Event
做"无写者"信号。读者计数 > 0 时写者等待；写者活跃时读者等待。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from src.core.tools.waterfall import ToolDenied


class AsyncRWLock:
    """asyncio 读写锁：多读单写。

    语义：
      - acquire_read: 共享读，多个可同时持有，与写互斥。
      - acquire_write: 独占写，与所有读和另一个写互斥。

    实现：用一把内部 asyncio.Lock 保护 (readers / writer_active) 计数，
    加一个"可读"信号 event。写者在 writer_active=True 时拉低 event，
    读者在 event 上等；读者计数降到 0 时唤醒写者。
    """

    def __init__(self) -> None:
        # 内部状态锁（保护 readers / writer_active 字段读写）
        self._state_lock = asyncio.Lock()
        self._readers = 0
        self._writer_active = False
        # 写者排队等待：用 Condition 在 state_lock 上 wait
        self._writer_cond = asyncio.Condition(self._state_lock)
        # 读者排队等待：同上
        self._reader_cond = asyncio.Condition(self._state_lock)

    async def acquire_read(self) -> "AsyncRWLock.ReadLock":
        async with self._reader_cond:
            while self._writer_active:
                await self._reader_cond.wait()
            self._readers += 1
        return AsyncRWLock.ReadLock(self)

    async def acquire_write(self) -> "AsyncRWLock.WriteLock":
        async with self._writer_cond:
            while self._writer_active or self._readers > 0:
                await self._writer_cond.wait()
            self._writer_active = True
        return AsyncRWLock.WriteLock(self)

    async def _release_read(self) -> None:
        async with self._reader_cond:
            self._readers -= 1
            if self._readers == 0:
                # 唤醒写者
                self._writer_cond.notify_all()

    async def _release_write(self) -> None:
        async with self._reader_cond:
            self._writer_active = False
            # 优先唤醒所有读者（读者优先语义）
            self._reader_cond.notify_all()
            self._writer_cond.notify_all()

    @property
    def readers(self) -> int:
        return self._readers

    @property
    def writer_active(self) -> bool:
        return self._writer_active

    class ReadLock:
        def __init__(self, owner: "AsyncRWLock") -> None:
            self._owner = owner

        async def __aenter__(self) -> "AsyncRWLock.ReadLock":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            await self._owner._release_read()

    class WriteLock:
        def __init__(self, owner: "AsyncRWLock") -> None:
            self._owner = owner

        async def __aenter__(self) -> "AsyncRWLock.WriteLock":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            await self._owner._release_write()


@dataclass
class ToolCallRuntime:
    """单次工具调用的运行态。

    持有锁种类（read/write/shared exclusive）与超时。lock_resolver 由 registry
    在 execute 前调用，返回 (rw_lock, mode) 元组或 (None, None) 表示无锁。

    Attributes:
        lock: 该工具关联的 AsyncRWLock（同一资源多个工具共享）。
        mode: "read" | "write" | None。
        timeout_sec: 超时秒数，None 表示不超时。
    """

    lock: Optional[AsyncRWLock] = None
    mode: Optional[str] = None
    timeout_sec: Optional[float] = None


# 工具→锁解析器：registry 用它给每个 call 决定锁策略
LockResolver = Callable[[str, dict], Tuple[Optional[AsyncRWLock], Optional[str]]]


@dataclass
class _PendingCall:
    """并发执行的单条工具调用任务。"""

    name: str
    args: dict
    callback: Callable[..., Awaitable[Any]]
    runtime: ToolCallRuntime


class ParallelExecutor:
    """并发跑多个工具调用，按锁策略共享/独占资源。

    用法：executor = ParallelExecutor(); results = await executor.run(calls)
    """

    def __init__(self) -> None:
        self._runtime_locks: dict[str, AsyncRWLock] = {}

    def get_or_create_lock(self, lock_key: str) -> AsyncRWLock:
        if lock_key not in self._runtime_locks:
            self._runtime_locks[lock_key] = AsyncRWLock()
        return self._runtime_locks[lock_key]

    async def run(self, calls: List[_PendingCall]) -> List[Any]:
        """并发跑所有 call，按 lock/mode 串行化冲突项，返回与 calls 同序结果列表。

        异常被捕获转成 ToolDenied(f"tool {name} failed: {e}")——上层可见到。
        """
        coros = [self._run_one(c) for c in calls]
        results = await asyncio.gather(*coros, return_exceptions=True)
        return list(results)

    async def _run_one(self, call: _PendingCall) -> Any:
        try:
            if call.runtime.lock is not None and call.runtime.mode == "read":
                async with await call.runtime.lock.acquire_read():
                    return await self._invoke(call)
            elif call.runtime.lock is not None and call.runtime.mode == "write":
                async with await call.runtime.lock.acquire_write():
                    return await self._invoke(call)
            return await self._invoke(call)
        except ToolDenied:
            raise
        except Exception as e:  # noqa: BLE001
            return ToolDenied(f"tool {call.name} failed: {e}")

    async def _invoke(self, call: _PendingCall) -> Any:
        if call.runtime.timeout_sec is not None:
            return await asyncio.wait_for(
                call.callback(**call.args), timeout=call.runtime.timeout_sec)
        return await call.callback(**call.args)
