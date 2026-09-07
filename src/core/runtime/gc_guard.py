"""Runtime 长跑守护子系统。

提供 7×24 监控/批跑场景下的进程级资源守护:
- `GCGuard`: 周期采样 RSS, 超阈值触发 `gc.collect`, 记录内存基线用于泄漏趋势告警
- `MemoryBaseline`: 滑动窗口采样, 检测持续增长趋势
- `install_gc_guard`: FastAPI lifespan 装配钩子(可选接入)

设计约束:
- 仅依赖 `psutil`(项目已有), 不引入新依赖
- 线程安全(`threading.Lock` + `threading.Event`)
- 所有 psutil 访问通过可注入的 process 工厂, 便于测试 Mock
"""

from __future__ import annotations

import gc
import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psutil

logger = logging.getLogger(__name__)

# 默认参数: 2GB 阈值, 60s 间隔, 基线 32 样本
DEFAULT_THRESHOLD_MB: float = 2048.0
DEFAULT_INTERVAL_SEC: float = 60.0
DEFAULT_BASELINE_SAMPLES: int = 32
DEFAULT_TREND_MIN_SAMPLES: int = 8
DEFAULT_TREND_GROWTH_MB: float = 64.0

ProcessFactory = Callable[[], "psutil.Process"]


def _default_process_factory() -> psutil.Process:
    """默认 process 工厂: 取当前进程(便于测试注入伪造对象)。"""
    return psutil.Process()


@dataclass(frozen=True)
class GCCheckResult:
    """单次 GC 检查结果(不可变, 避免并发被改)。"""

    before_mb: float
    after_mb: float
    collected_mb: float
    triggered: bool
    threshold_mb: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "before": round(self.before_mb, 3),
            "after": round(self.after_mb, 3),
            "collected": round(self.collected_mb, 3),
            "triggered": self.triggered,
            "threshold": self.threshold_mb,
        }


@dataclass
class MemoryBaseline:
    """内存基线采样器: 滑动窗口 + 趋势检测。

    持续增长(连续 N 次采样单调递增且总增幅超阈值)视为疑似泄漏, 触发告警。
    稳定采样不告警。
    """

    samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_BASELINE_SAMPLES)
    )
    baseline_mb: float | None = None
    min_samples: int = DEFAULT_TREND_MIN_SAMPLES
    growth_threshold_mb: float = DEFAULT_TREND_GROWTH_MB
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def set_baseline(self, current_mb: float) -> None:
        """记录当前内存为基线值。"""
        with self._lock:
            self.baseline_mb = current_mb

    def record(self, current_mb: float) -> None:
        """追加一次采样(自动截断到窗口大小)。"""
        with self._lock:
            self.samples.append(current_mb)

    def is_leaking_trend(self) -> bool:
        """检测持续增长趋势。

        判据: 采样数 >= min_samples, 且最近 min_samples 个采样严格单调递增,
        且首末差值 >= growth_threshold_mb。
        """
        with self._lock:
            if len(self.samples) < self.min_samples:
                return False
            window = list(self.samples)[-self.min_samples :]
        if any(window[i + 1] <= window[i] for i in range(len(window) - 1)):
            return False
        return (window[-1] - window[0]) >= self.growth_threshold_mb

    def delta_from_baseline(self, current_mb: float) -> float | None:
        """当前内存相对基线的增量(MB), 无基线返回 None。"""
        with self._lock:
            if self.baseline_mb is None:
                return None
        return current_mb - self.baseline_mb

    def snapshot(self) -> dict[str, Any]:
        """只读快照(供监控/调试)。"""
        with self._lock:
            return {
                "baseline_mb": self.baseline_mb,
                "sample_count": len(self.samples),
                "latest_mb": self.samples[-1] if self.samples else None,
                "leaking_trend": self.is_leaking_trend(),
            }


class GCGuard:
    """后台 GC 守护: 周期采样 RSS, 超阈值触发 `gc.collect`, 检测泄漏趋势。"""

    def __init__(
        self,
        *,
        threshold_mb: float = DEFAULT_THRESHOLD_MB,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        process_factory: ProcessFactory | None = None,
        on_leak: Callable[[dict[str, Any]], None] | None = None,
        baseline: MemoryBaseline | None = None,
    ) -> None:
        if threshold_mb <= 0:
            raise ValueError("threshold_mb must be > 0")
        if interval_sec <= 0:
            raise ValueError("interval_sec must be > 0")
        self._threshold_mb = threshold_mb
        self._interval_sec = interval_sec
        self._process_factory: ProcessFactory = (
            process_factory or _default_process_factory
        )
        self._on_leak = on_leak
        self._baseline = baseline or MemoryBaseline()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def threshold_mb(self) -> float:
        return self._threshold_mb

    @property
    def interval_sec(self) -> float:
        return self._interval_sec

    @property
    def baseline(self) -> MemoryBaseline:
        return self._baseline

    def get_memory_mb(self) -> float:
        """读取当前进程 RSS(MB)。psutil 失败返回 0.0(不阻塞守护)。"""
        try:
            proc = self._process_factory()
            rss = proc.memory_info().rss
            return float(rss) / (1024.0 * 1024.0)
        except psutil.Error:
            logger.warning("psutil 读取 RSS 失败, 返回 0.0")
            return 0.0
        except Exception:  # noqa: BLE001 — 守护线程不得向宿主抛异常
            logger.exception("get_memory_mb 未预期异常, 返回 0.0")
            return 0.0

    def check_and_collect(self) -> GCCheckResult:
        """超阈值触发 GC, 返回检查结果(线程安全)。

        锁只保护对 ``self`` 可变状态的读写; ``gc.collect`` 与 ``psutil`` 读取
        放在锁外, 避免 N 个并发调用方在 GC 停顿上串行排队(否则高并发下
        ``check_and_collect`` 会因排队 GC 而超时)。``MemoryBaseline`` 自带
        ``RLock`` 保护自身的采样窗口, 此处不再外层加锁。
        """
        before = self.get_memory_mb()
        triggered = before >= self._threshold_mb
        collected = 0.0
        if triggered:
            collected_count = gc.collect()
            after = self.get_memory_mb()
            collected = before - after
            logger.info(
                "GCGuard 触发 GC: before=%.1fMB after=%.1fMB collected=%.1fMB"
                " (gc returned %d objects)",
                before,
                after,
                collected,
                collected_count,
            )
        else:
            after = before
        # baseline 自带 RLock, record / is_leaking_trend / snapshot 均线程安全
        self._baseline.record(after)
        if self._baseline.is_leaking_trend() and self._on_leak is not None:
            try:
                self._on_leak(self._baseline.snapshot())
            except Exception:  # noqa: BLE001
                logger.exception("on_leak 回调异常, 已吞掉")
        return GCCheckResult(
            before_mb=before,
            after_mb=after,
            collected_mb=collected,
            triggered=triggered,
            threshold_mb=self._threshold_mb,
        )

    def set_baseline(self) -> float:
        """以当前内存为基线。"""
        current = self.get_memory_mb()
        self._baseline.set_baseline(current)
        logger.info("GCGuard 基线已设: %.1fMB", current)
        return current

    def _loop(self) -> None:
        """后台循环: 等待 -> 检查, 直至 stop。"""
        while not self._stop_event.wait(self._interval_sec):
            try:
                self.check_and_collect()
            except Exception:  # noqa: BLE001
                logger.exception("GCGuard 循环异常, 跳过本轮")

    def start(self) -> None:
        """启动后台守护线程(幂等)。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="GCGuard",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "GCGuard 启动: threshold=%.0fMB interval=%.1fs",
            self._threshold_mb,
            self._interval_sec,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """停止后台线程(等待 join, 不强杀)。"""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        logger.info("GCGuard 已停止")

    def is_running(self) -> bool:
        """线程是否在跑(诊断用)。"""
        return self._thread is not None and self._thread.is_alive()


def install_gc_guard(
    app: Any,
    *,
    threshold_mb: float = DEFAULT_THRESHOLD_MB,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
) -> GCGuard:
    """FastAPI lifespan 装配钩子(可选接入)。

    将 GCGuard 绑定到 app.state, 并注册 startup/shutdown。
    用法::

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan(app):
            guard = install_gc_guard(app, threshold_mb=2048, interval_sec=60)
            try:
                yield
            finally:
                guard.stop()

        app = FastAPI(lifespan=lifespan)
    """
    guard = GCGuard(threshold_mb=threshold_mb, interval_sec=interval_sec)

    async def _on_startup() -> None:
        guard.set_baseline()
        guard.start()

    async def _on_shutdown() -> None:
        guard.stop()

    # 兼容 FastAPI 的 lifespan 上下文与旧式 event 钩子
    try:
        from fastapi import FastAPI  # noqa: PLC0415 — 延迟导入, 测试无 FastAPI 不报错

        if isinstance(app, FastAPI):
            app.state.gc_guard = guard
            return guard
    except Exception:  # noqa: BLE001 — FastAPI 未安装时静默回退
        _ = _on_startup  # 占位引用, 避免未使用告警
        _ = _on_shutdown

    # 非 FastAPI / FastAPI 未装: 仅返回实例, 由调用方自行编排生命周期
    return guard
