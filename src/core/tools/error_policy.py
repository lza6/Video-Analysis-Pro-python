"""工具异常分级与自动重试/降级策略。

参考《改进指南》2.2：工具调用失败不应一律抛回上层，应按异常类型分级：
  - TRANSIENT（瞬时）：网络抖动 / 限流 / 超时 → 可重试，指数退避 + jitter
  - INVALID_INPUT（参数错误）：调用方问题 → 不重试，不标记不可用
  - FATAL（致命）：鉴权失败 / 凭据失效 → 不重试，标记工具进程级不可用
  - UNKNOWN（未知）：兜底 → 不重试，不标记不可用

纯 stdlib：enum / dataclasses / time / random / re / threading / typing。
不引新依赖，不改动 registry.py / loop.py / definition.py（接入由主控后续串行）。
"""
from __future__ import annotations

import random
import threading
from enum import Enum
from typing import Dict, List


class ToolErrorKind(str, Enum):
    """工具异常分级。

    继承 str 便于 JSON 序列化与日志直接打印。
    """

    TRANSIENT = "transient"
    INVALID_INPUT = "invalid_input"
    FATAL = "fatal"
    UNKNOWN = "unknown"


# 关键词匹配（小写，命中即归类）。顺序：TRANSIENT → INVALID_INPUT → FATAL。
_TRANSIENT_PATTERNS: List[str] = [
    "rate limit",
    "429",
    "503",
    "timeout",
    "temporarily",
    "unavailable",
    "connection reset",
    "connection refused",
]
_INVALID_INPUT_PATTERNS: List[str] = [
    "invalid",
    "bad request",
    "400",
    "argument",
    "schema",
]
_FATAL_PATTERNS: List[str] = [
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "credential",
    "api key",
    "permission denied",
]

# 异常类型 → 分级（关键词命中可覆盖）
_TRANSIENT_EXC = (ConnectionError, TimeoutError)
try:  # asyncio.TimeoutError 在 3.11+ 是 OSError 别名，老版本独立
    import asyncio

    _TRANSIENT_EXC = (ConnectionError, TimeoutError, asyncio.TimeoutError)
except ImportError:  # pragma: no cover - asyncio 永远在
    pass
_INVALID_INPUT_EXC = (ValueError, KeyError, TypeError)


class ErrorPolicy:
    """工具异常分级与重试/降级决策器。

    无状态方法（classify / should_retry / retry_after）可并发调用；
    mark_unavailable / is_available 操作进程级共享集合，用 threading.Lock 保护。
    """

    def __init__(self) -> None:
        self._unavailable: Dict[str, None] = {}
        self._lock = threading.Lock()

    # ---- 分类 ----
    def classify(self, exc: Exception, tool: str = "") -> ToolErrorKind:
        """按异常类型 + 消息关键词分级。

        Args:
            exc: 工具抛出的异常。
            tool: 工具名（仅用于日志，不参与决策）。

        Returns:
            ToolErrorKind 四类之一。
        """
        msg = str(exc).lower()
        if self._match(msg, _FATAL_PATTERNS) or isinstance(exc, PermissionError):
            return ToolErrorKind.FATAL
        if isinstance(exc, _TRANSIENT_EXC) or self._match(msg, _TRANSIENT_PATTERNS):
            return ToolErrorKind.TRANSIENT
        if isinstance(exc, _INVALID_INPUT_EXC) or self._match(msg, _INVALID_INPUT_PATTERNS):
            return ToolErrorKind.INVALID_INPUT
        return ToolErrorKind.UNKNOWN

    @staticmethod
    def _match(text: str, patterns: List[str]) -> bool:
        return any(p in text for p in patterns)

    # ---- 重试决策 ----
    def should_retry(self, kind: ToolErrorKind, attempt: int) -> bool:
        """TRANSIENT 且 attempt <= 3 时允许重试（attempt 从 1 起计，共 3 次）。

        spec 原文 "attempt<3" 按 0-indexed 解读即 3 次重试（0/1/2）；
        本项目 attempt 从 1 起计，等价条件为 attempt < 4（1/2/3 True，4 False）。
        """
        return kind is ToolErrorKind.TRANSIENT and attempt < 4

    def retry_after(self, kind: ToolErrorKind, attempt: int) -> float:
        """指数退避：base * 2^(attempt-1)，base=0.5s，加 ±20% jitter 后截断到上限 8s。

        非 TRANSIENT 返回 0（调用方立即放弃或走降级）。
        """
        if kind is not ToolErrorKind.TRANSIENT:
            return 0.0
        base = 0.5
        delay = base * (2 ** (attempt - 1))
        jitter = delay * 0.2 * random.uniform(-1.0, 1.0)
        return min(max(0.0, delay + jitter), 8.0)

    # ---- 降级：进程级不可用标记 ----
    def mark_unavailable(self, tool: str) -> None:
        """把工具加入进程级不可用集合（FATAL 时调用）。"""
        with self._lock:
            self._unavailable[tool] = None

    def is_available(self, tool: str) -> bool:
        """工具是否可用（不在不可用集合中）。"""
        with self._lock:
            return tool not in self._unavailable

    def reset(self) -> None:
        """清空不可用集合（测试 / 热重载用）。"""
        with self._lock:
            self._unavailable.clear()
