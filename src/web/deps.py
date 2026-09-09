"""依赖注入(Depends providers)。

集中提供 Settings / ConfigurationManager / 能力矩阵 / JobStore,
便于路由声明依赖、便于测试替换。
"""
from __future__ import annotations

import asyncio
import functools
import importlib
import itertools
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from src.utils.config_manager import ConfigurationManager
from src.utils.constants import CACHE_DIR
from src.core.logic import (
    CLIP_AVAILABLE,
    NVIDIA_GPU_AVAILABLE,
    ADVANCED_FEATURES_AVAILABLE,
    FFMPEG_AVAILABLE,
)

from .job_store import JobStore

log = logging.getLogger("web.deps")

# 单例 JobStore(进程内,线程安全)
_job_store: JobStore | None = None


def get_job_store() -> JobStore:
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store


@functools.lru_cache(maxsize=1)
def _config_manager() -> ConfigurationManager:
    cm = ConfigurationManager()
    cm.load_main_config()
    return cm


def get_config_manager() -> ConfigurationManager:
    return _config_manager()


def capability_matrix() -> dict:
    """能力矩阵。复用 headless.py 的语义,/api/health 必须永远 <50ms。

    Ollama 探测留给 /api/analyze 真正调用时(/healthz 不探,避免阻塞)。
    """
    return {
        "clip_semantic": CLIP_AVAILABLE,
        "nvidia_gpu": NVIDIA_GPU_AVAILABLE,
        "advanced_media": ADVANCED_FEATURES_AVAILABLE,
        "ffmpeg": FFMPEG_AVAILABLE,
        "ocr": _module_available("paddleocr"),
        "llm_backend": "unknown",
    }


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def web_jobs_root() -> Path:
    """所有 Web 分析作业的工作目录根。注册为 'frames' 内容根。

    放在 CACHE_DIR 下(CACHE_DIR = "cache",已 gitignore),跨重启可清理。
    """
    root = Path(CACHE_DIR) / "web_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def disk_free_gb() -> float:
    try:
        usage = shutil.disk_usage(os.getcwd())
        return round(usage.free / 1024**3, 1)
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# ApprovalBus — 工具审批事件订阅/推送(线程安全 ring buffer + 唯一 pin)
# ---------------------------------------------------------------------------
#
# 供 agent router 用:经 SSE 把审批请求投递前端,前端回调后由主控装配
# approval_fn。线程安全(asyncio 事件循环 / 工具线程池并发调用):
#   - request_approval(ask) -> pin     开一个 pending,等前端确认
#   - wait_decision(pin, timeout)       轮询等待,超时返回 None
#   - decide(pin, allow)                前端回调,解除等待
# ring buffer 存最近 _RING_CAP 条决策记录(审计/历史),decide 即记账。


class ApprovalBus:
    """线程安全工具审批总线(环形缓冲 + 唯一 pin 回调)。

    生命周期:单例(get_approval_bus)。每个审批请求生成唯一 pin,
    前端经 SSE 收到后调 decide(pin, allow),wait_decision 解除阻塞。

    v10.2 (B-FIN-2 MAJOR-2):新增 asyncio 事件通知——
      - request_approval 额外建一个 asyncio.Event(挂在 _events[pin])
      - decide 时 set 该 Event(无论当前在哪个线程,线程安全)
      - wait_decision_async 用 asyncio.wait_for(event.wait(), timeout) 等,
        等待期间不占用事件循环(uvicorn 能继续处理 decide 请求,不再死锁)

    API:
      request_approval(ask: dict) -> str pin
      wait_decision(pin, timeout=None) -> bool | None   # 同步轮询,超时 None
      wait_decision_async(pin, timeout=None) -> bool | None  # 异步等,超时 None
      decide(pin, allow: bool) -> bool                   # 首次决定 True,重复 False
    """

    def __init__(self, capacity: int = 512) -> None:
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._decisions: dict[str, bool] = {}
        self._events: dict[str, "asyncio.Event"] = {}
        # 最近决策环形缓冲(审计用,容量 capacity)
        self._ring = [None] * max(1, capacity)
        self._ring_index = 0
        self._pin_counter = itertools.count(1)

    def _new_pin(self) -> str:
        with self._lock:
            return f"aprv-{next(self._pin_counter)}"

    # ---- 审批发起 ----
    def request_approval(self, ask: dict) -> str:
        """登记一个待审批请求,返回唯一 pin(投递到前端 / SSE)。

        同时创建一个 asyncio.Event 供 wait_decision_async 等待。Event 在
        首次 wait_decision_async 调用时才绑定当前 running loop(若该 loop
        上未存在绑定);request 阶段不绑定 loop,避免跨线程 loop 错配。
        """
        pin = self._new_pin()
        with self._lock:
            self._pending[pin] = {
                "ask": dict(ask),
                "state": "pending",
                "decided": None,
            }
            self._events[pin] = asyncio.Event()
        return pin

    def pending_count(self) -> int:
        """当前未决审批数(SSE 心跳/健康检查用)。"""
        with self._lock:
            return len(self._pending)

    def pending_requests(self) -> list[dict]:
        """所有未决请求快照(SSE 首次连接推送当前待办)。"""
        with self._lock:
            return [
                {"pin": p, **self._pending[p]["ask"]}
                for p in self._pending
            ]

    # ---- 等待 ----
    def wait_decision(self, pin: str, timeout: float | None = None) -> bool | None:
        """阻塞轮询等待 pin 的审批决定。

        线程安全:等待用 time.sleep 轮询(不依赖事件循环,工具线程池 /
        独立线程都能调),决定由线程锁保护。

        Args:
            pin: request_approval 返回的 pin。
            timeout: 秒;None 无限等;0 立即查一次;超时返回 None(未决定)。

        Returns:
            bool(前端 decide 传入) | None(超时/未知 pin 尚未决定)。
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                pending = self._pending.get(pin)
                if pending is None:
                    return None
                if pending["decided"] is not None:
                    # 已决定,记账后移除(防泄漏),返回决定
                    decided = pending["decided"]
                    self._ring[self._ring_index % len(self._ring)] = {
                        "pin": pin,
                        "ask": dict(pending["ask"]),
                        "decided": decided,
                    }
                    self._ring_index += 1
                    del self._pending[pin]
                    self._events.pop(pin, None)
                    return decided
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(0.01)

    async def wait_decision_async(self, pin: str, timeout: float | None = None) -> bool | None:
        """异步等待 pin 的审批决定(不阻塞事件循环)。

        v10.2 (B-FIN-2 MAJOR-2):等待用 asyncio.Event.set 通知 + asyncio.wait_for,
        等待期间事件循环可继续处理 decide 请求。此前同步轮询版在事件循环上
        await 会冻结服务至多 60s,decide 请求进不来形成死锁。

        Args:
            pin: request_approval 返回的 pin。
            timeout: 秒;None 无限等;0 立即查一次;超时返回 None(未决定)。

        Returns:
            bool(前端 decide 传入) | None(超时/未知 pin 尚未决定)。
        """
        # 已决定/未知 pin 快路径
        with self._lock:
            pending = self._pending.get(pin)
            if pending is None:
                return None
            if pending["decided"] is not None:
                decided = pending["decided"]
                self._ring[self._ring_index % len(self._ring)] = {
                    "pin": pin,
                    "ask": dict(pending["ask"]),
                    "decided": decided,
                }
                self._ring_index += 1
                del self._pending[pin]
                self._events.pop(pin, None)
                return decided
            ev = self._events.get(pin)

        # 等 Event(不占用事件循环)
        if ev is not None:
            # 绑定当前 running loop(Python 3.10+ asyncio.Event 首次 wait
            # 自动绑定调用线程的 running loop;若已在别的 loop wait 过则
            # 保持原绑定)。request 阶段不绑定,避免跨线程 loop 错配。
            try:
                if ev._loop is None:
                    ev._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            try:
                if timeout is None:
                    await ev.wait()
                elif timeout > 0:
                    await asyncio.wait_for(ev.wait(), timeout=timeout)
                else:
                    # timeout <= 0:立即查一次(不等待)
                    return self._peek_decision(pin)
            except asyncio.TimeoutError:
                return None
            except RuntimeError:
                # Event 绑定的 loop 与当前 loop 不一致(跨线程),退同步快查
                return self._peek_decision(pin)

        # 事件通知后取决定(决定由 decide 写入,这里只读 + 记账)
        return self._settle(pin)

    def _peek_decision(self, pin: str) -> bool | None:
        """立即查一次 pin 是否已决定(不等待、不记账)。"""
        with self._lock:
            pending = self._pending.get(pin)
            if pending is None or pending["decided"] is None:
                return None
            return pending["decided"]

    def _settle(self, pin: str) -> bool | None:
        """决定已就绪时记账 + 移除,返回决定。"""
        with self._lock:
            pending = self._pending.get(pin)
            if pending is None or pending["decided"] is None:
                return None
            decided = pending["decided"]
            self._ring[self._ring_index % len(self._ring)] = {
                "pin": pin,
                "ask": dict(pending["ask"]),
                "decided": decided,
            }
            self._ring_index += 1
            del self._pending[pin]
            self._events.pop(pin, None)
            return decided

    # ---- 前端回调 ----
    def decide(self, pin: str, allow: bool) -> bool:
        """前端回调:确认 pin 的审批决定。首次决定 True,重复/未知 False。"""
        with self._lock:
            pending = self._pending.get(pin)
            if pending is None or pending["decided"] is not None:
                return False
            pending["decided"] = bool(allow)
            # 通知异步等待者(线程安全,decide 可在任意线程被调)
            ev = self._events.get(pin)
        if ev is not None:
            try:
                ev.set()
            except Exception:  # noqa: BLE001 — 通知失败不影响决定已落库
                pass
        return True

    # ---- 审计 ----
    def recent_decisions(self, limit: int = 20) -> list[dict]:
        """最近 limit 条决策记录(倒序,新在前)。"""
        with self._lock:
            records = [r for r in self._ring if r is not None]
            return list(reversed(records[-limit:]))

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)


# 单例 bus(进程内,线程安全)
_approval_bus: ApprovalBus | None = None


def get_approval_bus() -> ApprovalBus:
    """进程内 ApprovalBus 单例(测试可注入新实例覆盖)。"""
    global _approval_bus
    if _approval_bus is None:
        _approval_bus = ApprovalBus()
    return _approval_bus
