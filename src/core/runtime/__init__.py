"""Runtime 子包: 长跑守护组件。

当前提供:
- `GCGuard`: 周期 GC + 内存基线泄漏检测
- `MemoryBaseline`: 滑动窗口采样器
- `install_gc_guard`: FastAPI lifespan 装配钩子
- `JSONFormatter` / `trace_context` / `install_json_logging`: 结构化 JSON 日志 + trace_id

后续可在此子包扩展 watchdog / heartbeat / metric-reporter 等守护组件。
"""

from __future__ import annotations

from .gc_guard import (
    DEFAULT_BASELINE_SAMPLES,
    DEFAULT_INTERVAL_SEC,
    DEFAULT_THRESHOLD_MB,
    DEFAULT_TREND_GROWTH_MB,
    DEFAULT_TREND_MIN_SAMPLES,
    GCCheckResult,
    GCGuard,
    MemoryBaseline,
    install_gc_guard,
)
from .structured_log import (
    JSONFormatter,
    get_trace_id,
    install_json_logging,
    trace_context,
)

__all__ = [
    "GCGuard",
    "GCCheckResult",
    "MemoryBaseline",
    "install_gc_guard",
    "DEFAULT_THRESHOLD_MB",
    "DEFAULT_INTERVAL_SEC",
    "DEFAULT_BASELINE_SAMPLES",
    "DEFAULT_TREND_MIN_SAMPLES",
    "DEFAULT_TREND_GROWTH_MB",
    "JSONFormatter",
    "trace_context",
    "get_trace_id",
    "install_json_logging",
]
