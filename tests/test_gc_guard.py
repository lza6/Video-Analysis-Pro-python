"""GCGuard / MemoryBaseline 单元测试。

策略:
- Mock psutil.Process 与 memory_info(), 不真实跑 2GB 内存
- patch gc.collect 计数真实调用
- 用线程安全的 FakeProcess 控制 RSS 序列
- 趋势检测用显式序列驱动
"""

from __future__ import annotations

import gc
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import psutil
import pytest

from src.core.runtime import gc_guard
from src.core.runtime.gc_guard import (
    DEFAULT_THRESHOLD_MB,
    GCCheckResult,
    GCGuard,
    MemoryBaseline,
    install_gc_guard,
)


class FakeMemoryInfo:
    """伪 memory_info, 返回固定 RSS(bytes)。"""

    def __init__(self, rss_bytes: int) -> None:
        self.rss = rss_bytes


@dataclass
class FakeProcess:
    """伪 psutil.Process, 按预设 RSS 序列返回。"""

    rss_sequence: list[int]
    _idx: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def memory_info(self) -> FakeMemoryInfo:
        with self._lock:
            idx = self._idx
            if idx < len(self.rss_sequence):
                rss = self.rss_sequence[idx]
            else:
                rss = self.rss_sequence[-1]
            self._idx = idx + 1
        return FakeMemoryInfo(rss)


@dataclass
class FakeProcessFactory:
    """可被多次调用返回同一伪 process 的工厂。"""

    process: FakeProcess

    def __call__(self) -> FakeProcess:
        return self.process


def _make_guard(
    rss_sequence: list[int],
    *,
    threshold_mb: float = DEFAULT_THRESHOLD_MB,
    interval_sec: float = 60.0,
    baseline: MemoryBaseline | None = None,
    on_leak: Any = None,
) -> tuple[GCGuard, FakeProcess]:
    """构造一个注入伪 psutil 的 GCGuard, 返回 (guard, fake_process)。"""
    fake = FakeProcess(rss_sequence=list(rss_sequence))
    factory = FakeProcessFactory(fake)
    guard = GCGuard(
        threshold_mb=threshold_mb,
        interval_sec=interval_sec,
        process_factory=factory,
        on_leak=on_leak,
        baseline=baseline,
    )
    return guard, fake


@contextmanager
def _patch_process(factory: FakeProcessFactory) -> Iterator[None]:
    """patch gc_guard._default_process_factory(未注入时回退路径用)。"""
    with patch.object(gc_guard, "_default_process_factory", factory):
        yield


# ---------------------------------------------------------------- GCGuard 生命周期


def test_start_stop_lifecycle() -> None:
    """start 起线程, stop 干净收尾, is_running 反映状态。"""
    guard, _ = _make_guard([100 * 1024 * 1024], interval_sec=0.01)
    assert not guard.is_running()
    guard.start()
    assert guard.is_running()
    guard.stop(timeout=2.0)
    assert not guard.is_running()


def test_start_is_idempotent() -> None:
    """重复 start 不起第二个线程。"""
    guard, _ = _make_guard([100 * 1024 * 1024], interval_sec=0.01)
    guard.start()
    first_thread = guard._thread
    guard.start()
    assert guard._thread is first_thread
    guard.stop()


def test_stop_without_start_is_safe() -> None:
    """未 start 直接 stop 不报错。"""
    guard, _ = _make_guard([100 * 1024 * 1024])
    guard.stop()  # 不应抛


# ---------------------------------------------------------------- check_and_collect


def test_check_and_collect_under_threshold_no_gc() -> None:
    """低于阈值不触发 GC, before == after, collected == 0。"""
    # 100MB RSS, 阈值 2048MB
    rss = 100 * 1024 * 1024
    guard, _ = _make_guard([rss, rss, rss], threshold_mb=2048)
    with patch("src.core.runtime.gc_guard.gc.collect", return_value=0) as mock_gc:
        result = guard.check_and_collect()
    assert isinstance(result, GCCheckResult)
    assert result.triggered is False
    assert result.before_mb == pytest.approx(100.0, abs=0.01)
    assert result.after_mb == pytest.approx(100.0, abs=0.01)
    assert result.collected_mb == 0.0
    mock_gc.assert_not_called()


def test_check_and_collect_over_threshold_triggers_gc() -> None:
    """超阈值触发 gc.collect, after < before。"""
    # RSS 序列: 3000MB(触发) -> 2000MB(回收后)
    before_bytes = 3000 * 1024 * 1024
    after_bytes = 2000 * 1024 * 1024
    guard, _ = _make_guard(
        [before_bytes, after_bytes, after_bytes], threshold_mb=2048
    )
    with patch("src.core.runtime.gc_guard.gc.collect", return_value=42) as mock_gc:
        result = guard.check_and_collect()
    assert result.triggered is True
    assert result.before_mb == pytest.approx(3000.0, abs=0.01)
    assert result.after_mb == pytest.approx(2000.0, abs=0.01)
    assert result.collected_mb == pytest.approx(1000.0, abs=0.01)
    # 触发分支会调两次 gc.collect(显式 + 计数), 验证至少被调
    assert mock_gc.call_count >= 1


def test_check_and_collect_real_gc_runs() -> None:
    """不 mock gc.collect 时, 真实 gc.collect 被调用且不抛。"""
    before_bytes = 3000 * 1024 * 1024
    after_bytes = 2500 * 1024 * 1024
    guard, _ = _make_guard(
        [before_bytes, after_bytes, after_bytes], threshold_mb=2048
    )
    result = guard.check_and_collect()
    assert result.triggered is True
    # collected 可能为 0(无垃圾), 但 before/after 正确
    assert result.before_mb == pytest.approx(3000.0, abs=0.01)


def test_as_dict_shape() -> None:
    """as_dict 包含所有键。"""
    guard, _ = _make_guard([100 * 1024 * 1024], threshold_mb=2048)
    result = guard.check_and_collect()
    d = result.as_dict()
    assert set(d.keys()) == {"before", "after", "collected", "triggered", "threshold"}


# ---------------------------------------------------------------- get_memory_mb


def test_get_memory_mb_reads_psutil() -> None:
    """get_memory_mb 通过注入的 process_factory 读 RSS。"""
    rss = 512 * 1024 * 1024  # 512MB
    guard, fake = _make_guard([rss])
    mb = guard.get_memory_mb()
    assert mb == pytest.approx(512.0, abs=0.01)
    # fake._idx 已自增
    assert fake._idx == 1


def test_get_memory_mb_psutil_error_returns_zero() -> None:
    """psutil.Error 时返回 0.0, 不抛。"""

    class ErrorProcess:
        def memory_info(self) -> FakeMemoryInfo:
            raise psutil.AccessDenied("denied")

    class ErrorFactory:
        def __call__(self) -> ErrorProcess:
            return ErrorProcess()

    guard = GCGuard(process_factory=ErrorFactory())  # type: ignore[arg-type]
    mb = guard.get_memory_mb()
    assert mb == 0.0


def test_get_memory_mb_unexpected_exception_returns_zero() -> None:
    """非 psutil 异常也被吞, 返回 0.0。"""

    class BoomProcess:
        def memory_info(self) -> FakeMemoryInfo:
            raise RuntimeError("boom")

    class BoomFactory:
        def __call__(self) -> BoomProcess:
            return BoomProcess()

    guard = GCGuard(process_factory=BoomFactory())  # type: ignore[arg-type]
    assert guard.get_memory_mb() == 0.0


# ---------------------------------------------------------------- MemoryBaseline


def test_baseline_record_and_leaking_trend_growing() -> None:
    """连续单调增长且增幅超阈值 -> 告警。"""
    from collections import deque

    baseline = MemoryBaseline(
        min_samples=4,
        growth_threshold_mb=10.0,
        samples=deque(maxlen=8),
    )
    for mb in (100.0, 105.0, 110.0, 120.0):
        baseline.record(mb)
    assert baseline.is_leaking_trend() is True


def test_baseline_stable_no_leak() -> None:
    """稳定序列不告警。"""
    from collections import deque

    baseline = MemoryBaseline(
        min_samples=4, growth_threshold_mb=10.0, samples=deque(maxlen=8)
    )
    for mb in (100.0, 101.0, 100.5, 100.8, 101.2, 100.9):
        baseline.record(mb)
    assert baseline.is_leaking_trend() is False


def test_baseline_too_few_samples_no_trend() -> None:
    """采样不足 min_samples 不判定泄漏。"""
    baseline = MemoryBaseline(min_samples=8, growth_threshold_mb=1.0)
    baseline.record(100.0)
    baseline.record(200.0)
    assert baseline.is_leaking_trend() is False


def test_baseline_decreasing_no_leak() -> None:
    """单调递减不算泄漏。"""
    from collections import deque

    baseline = MemoryBaseline(
        min_samples=4, growth_threshold_mb=1.0, samples=deque(maxlen=8)
    )
    for mb in (200.0, 180.0, 150.0, 100.0):
        baseline.record(mb)
    assert baseline.is_leaking_trend() is False


def test_baseline_set_baseline_delta() -> None:
    """set_baseline 后 delta_from_baseline 正确计算。"""
    baseline = MemoryBaseline()
    baseline.set_baseline(100.0)
    assert baseline.delta_from_baseline(150.0) == 50.0
    assert baseline.delta_from_baseline(80.0) == -20.0
    baseline2 = MemoryBaseline()
    assert baseline2.delta_from_baseline(100.0) is None


def test_baseline_snapshot_shape() -> None:
    """snapshot 返回只读字段。"""
    from collections import deque

    baseline = MemoryBaseline(
        min_samples=2, growth_threshold_mb=1.0, samples=deque(maxlen=8)
    )
    baseline.set_baseline(100.0)
    baseline.record(100.0)
    baseline.record(110.0)
    snap = baseline.snapshot()
    assert snap["baseline_mb"] == 100.0
    assert snap["sample_count"] == 2
    assert snap["latest_mb"] == 110.0
    assert isinstance(snap["leaking_trend"], bool)


def test_baseline_maxlen_truncates() -> None:
    """deque maxlen 截断旧样本。"""
    from collections import deque

    baseline = MemoryBaseline(samples=deque(maxlen=3))
    for mb in (10.0, 20.0, 30.0, 40.0, 50.0):
        baseline.record(mb)
    assert len(baseline.samples) == 3
    assert list(baseline.samples) == [30.0, 40.0, 50.0]


# ---------------------------------------------------------------- 泄漏回调


def test_on_leak_callback_invoked() -> None:
    """持续增长趋势触发 on_leak 回调, 传入 snapshot。"""
    from collections import deque

    received: list[dict[str, Any]] = []
    baseline = MemoryBaseline(
        min_samples=3, growth_threshold_mb=1.0, samples=deque(maxlen=8)
    )
    # fake 序列: 第二个值 250MB 延续预灌入的增长趋势(>200)
    guard, _ = _make_guard(
        [100 * 1024 * 1024, 250 * 1024 * 1024, 300 * 1024 * 1024],
        threshold_mb=2048,
        baseline=baseline,
        on_leak=lambda snap: received.append(snap),
    )
    # 第一次 check: fake 返回 100MB, 不超阈值, record(100) -> samples=[100]
    guard.check_and_collect()
    # 覆盖 baseline.samples 手动灌入增长序列(模拟多轮历史采样)
    baseline.samples.clear()
    baseline.record(100.0)
    baseline.record(150.0)
    baseline.record(200.0)  # 当前 samples=[100,150,200]
    # 第二次 check: fake 返回 250MB(延续增长), record(250) -> samples=[100,150,200,250]
    # 最后 3 个采样 [150,200,250] 严格单调增, 增幅 100 > 1 -> 告警
    guard.check_and_collect()
    assert len(received) >= 1
    assert "baseline_mb" in received[0]


def test_on_leak_callback_exception_swallowed() -> None:
    """on_leak 抛异常不污染主流程。"""
    from collections import deque

    baseline = MemoryBaseline(
        min_samples=2, growth_threshold_mb=1.0, samples=deque(maxlen=8)
    )
    baseline.record(100.0)
    baseline.record(200.0)
    guard, _ = _make_guard(
        [100 * 1024 * 1024],
        threshold_mb=2048,
        baseline=baseline,
        on_leak=lambda _snap: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    # 不应抛
    result = guard.check_and_collect()
    assert result.triggered is False


# ---------------------------------------------------------------- 线程安全


def test_concurrent_check_does_not_crash() -> None:
    """并发调用 check_and_collect 不崩, 结果数等于调用数。"""
    rss = 3000 * 1024 * 1024  # 超阈值, 触发 GC 分支(锁竞争)
    guard, _ = _make_guard([rss] * 200, threshold_mb=2048)
    results: list[GCCheckResult] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(20):
                results.append(guard.check_and_collect())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert errors == []
    assert len(results) == 100
    # 全部触发(超阈值)
    assert all(r.triggered for r in results)


def test_concurrent_start_stop_does_not_crash() -> None:
    """并发 start/stop 不崩。"""
    guard, _ = _make_guard([100 * 1024 * 1024], interval_sec=0.01)
    errors: list[BaseException] = []

    def starter() -> None:
        try:
            for _ in range(10):
                guard.start()
                guard.stop(timeout=1.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=starter) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert errors == []
    guard.stop()


# ---------------------------------------------------------------- 参数校验


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        GCGuard(threshold_mb=0)


def test_invalid_interval_raises() -> None:
    with pytest.raises(ValueError):
        GCGuard(threshold_mb=100, interval_sec=0)


# ---------------------------------------------------------------- set_baseline


def test_set_baseline_records_current() -> None:
    """set_baseline 把当前内存记入 baseline.baseline_mb。"""
    rss = 256 * 1024 * 1024
    guard, _ = _make_guard([rss])
    current = guard.set_baseline()
    assert current == pytest.approx(256.0, abs=0.01)
    assert guard.baseline.baseline_mb == pytest.approx(256.0, abs=0.01)


# ---------------------------------------------------------------- install_gc_guard


def test_install_gc_guard_returns_guard() -> None:
    """install_gc_guard 非 FastAPI 路径返回 GCGuard 实例。"""
    class DummyApp:
        pass

    guard = install_gc_guard(DummyApp(), threshold_mb=1024, interval_sec=30)
    assert isinstance(guard, GCGuard)
    assert guard.threshold_mb == 1024
    assert guard.interval_sec == 30


def test_install_gc_guard_fastapi_attaches_to_state() -> None:
    """FastAPI 实例: guard 挂到 app.state.gc_guard。"""
    fastapi = pytest.importorskip("fastapi")
    app = fastapi.FastAPI()
    guard = install_gc_guard(app, threshold_mb=512, interval_sec=10)
    assert isinstance(guard, GCGuard)
    assert app.state.gc_guard is guard


# ---------------------------------------------------------------- 后台循环


def test_background_loop_triggers_check() -> None:
    """短间隔后台线程能至少跑一轮 check_and_collect(通过 baseline 采样数验证)。"""
    rss = 100 * 1024 * 1024
    guard, _ = _make_guard([rss] * 50, interval_sec=0.01)
    guard.start()
    # 等待后台至少跑几轮
    for _ in range(50):
        if len(guard.baseline.samples) >= 2:
            break
        threading.Event().wait(0.02)
    guard.stop(timeout=2.0)
    assert len(guard.baseline.samples) >= 1


def test_stop_joins_thread_cleanly() -> None:
    """stop 后线程对象为 None(或非 alive)。"""
    guard, _ = _make_guard([100 * 1024 * 1024], interval_sec=0.01)
    guard.start()
    guard.stop(timeout=2.0)
    assert guard._thread is None


def test_real_gc_import_works() -> None:
    """确保 gc 模块真实可调(集成烟雾)。"""
    collected = gc.collect()
    assert isinstance(collected, int)
