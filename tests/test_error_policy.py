"""ErrorPolicy 测试（AAA 模式）。

覆盖：
  1. 四类错误（TRANSIENT / INVALID_INPUT / FATAL / UNKNOWN）分类正确
  2. TRANSIENT should_retry 第 1/2/3 次 True，第 4 次 False
  3. 非 TRANSIENT should_retry 恒 False
  4. retry_after 指数递增（0.5→1→2→4 量级）
  5. FATAL mark_unavailable 后 is_available False
  6. INVALID_INPUT 不标记不可用
  7. jitter 在 ±20% 范围内（retry_after 跑 100 次取 min/max 边界）

纯 stdlib + pytest，可独立运行：
  python -m pytest tests/test_error_policy.py -q
"""
from __future__ import annotations

import asyncio

from src.core.tools.error_policy import ErrorPolicy, ToolErrorKind


# ---------- 1. 四类错误分类 ----------


def test_classify_transient_by_type():
    # Arrange
    policy = ErrorPolicy()
    exc = ConnectionError("connection reset by peer")
    # Act
    kind = policy.classify(exc, "net_tool")
    # Assert
    assert kind is ToolErrorKind.TRANSIENT


def test_classify_transient_by_keyword():
    # Arrange
    policy = ErrorPolicy()
    exc = RuntimeError("HTTP 503 temporarily unavailable")
    # Act
    kind = policy.classify(exc, "llm_call")
    # Assert
    assert kind is ToolErrorKind.TRANSIENT


def test_classify_invalid_input_by_type_and_keyword():
    # Arrange
    policy = ErrorPolicy()
    exc = ValueError("invalid argument: foo")
    # Act
    kind = policy.classify(exc, "frame_tool")
    # Assert
    assert kind is ToolErrorKind.INVALID_INPUT


def test_classify_fatal_by_keyword():
    # Arrange
    policy = ErrorPolicy()
    exc = RuntimeError("401 unauthorized: bad api key")
    # Act
    kind = policy.classify(exc, "llm_call")
    # Assert
    assert kind is ToolErrorKind.FATAL


def test_classify_fatal_by_permission_error():
    # Arrange
    policy = ErrorPolicy()
    exc = PermissionError("permission denied")
    # Act
    kind = policy.classify(exc, "fs_tool")
    # Assert
    assert kind is ToolErrorKind.FATAL


def test_classify_unknown_for_unrelated_exception():
    # Arrange
    policy = ErrorPolicy()
    exc = RuntimeError("something weird happened")
    # Act
    kind = policy.classify(exc, "misc_tool")
    # Assert
    assert kind is ToolErrorKind.UNKNOWN


# ---------- 2/3. should_retry ----------


def test_should_retry_transient_first_three_true_fourth_false():
    # Arrange
    policy = ErrorPolicy()
    # Act / Assert
    assert policy.should_retry(ToolErrorKind.TRANSIENT, 1) is True
    assert policy.should_retry(ToolErrorKind.TRANSIENT, 2) is True
    assert policy.should_retry(ToolErrorKind.TRANSIENT, 3) is True
    assert policy.should_retry(ToolErrorKind.TRANSIENT, 4) is False


def test_should_retry_non_transient_always_false():
    # Arrange
    policy = ErrorPolicy()
    non_transient = [ToolErrorKind.INVALID_INPUT, ToolErrorKind.FATAL, ToolErrorKind.UNKNOWN]
    # Act / Assert
    for kind in non_transient:
        for attempt in range(1, 6):
            assert policy.should_retry(kind, attempt) is False, f"{kind} attempt{attempt} should not retry"


# ---------- 4. retry_after 指数递增 ----------


def test_retry_after_exponential_growth_within_bounds():
    # Arrange
    policy = ErrorPolicy()
    # Act: attempt 1..5 取延迟，断言 0.4 < base*2^(n-1) < 10（含 jitter）
    delays = [policy.retry_after(ToolErrorKind.TRANSIENT, n) for n in range(1, 6)]
    # Assert: 递增（带 jitter，允许相邻偶尔小幅波动，但总体趋势上升）
    assert all(0.4 <= d < 10.0 for d in delays), f"delays out of range: {delays}"
    assert delays[-1] > delays[0], f"no growth: {delays}"
    # 基准量级：attempt 1 ~ 0.5, attempt 4 ~ 4, attempt 5 上限 8
    assert 0.4 <= delays[0] <= 0.7, f"attempt1 not ~0.5: {delays[0]}"
    assert delays[4] <= 8.5, f"attempt5 exceeds cap+jitter: {delays[4]}"


def test_retry_after_non_transient_returns_zero():
    # Arrange
    policy = ErrorPolicy()
    # Act / Assert
    for kind in [ToolErrorKind.INVALID_INPUT, ToolErrorKind.FATAL, ToolErrorKind.UNKNOWN]:
        assert policy.retry_after(kind, 1) == 0.0


# ---------- 5/6. mark_unavailable / is_available ----------


def test_fatal_marks_tool_unavailable():
    # Arrange
    policy = ErrorPolicy()
    policy.reset()
    # Act
    policy.mark_unavailable("llm_call")
    # Assert
    assert policy.is_available("llm_call") is False
    assert policy.is_available("other_tool") is True


def test_invalid_input_does_not_mark_unavailable():
    # Arrange
    policy = ErrorPolicy()
    policy.reset()
    # Act: 即使 classify 出 INVALID_INPUT 也不应自动标记不可用
    kind = policy.classify(ValueError("invalid argument"), "frame_tool")
    # Assert
    assert kind is ToolErrorKind.INVALID_INPUT
    assert policy.is_available("frame_tool") is True


def test_reset_clears_unavailable_set():
    # Arrange
    policy = ErrorPolicy()
    policy.mark_unavailable("a")
    policy.mark_unavailable("b")
    assert policy.is_available("a") is False
    # Act
    policy.reset()
    # Assert
    assert policy.is_available("a") is True
    assert policy.is_available("b") is True


# ---------- 7. jitter ±20% ----------


def test_retry_after_jitter_within_20_percent():
    # Arrange
    policy = ErrorPolicy()
    # 基准延迟（无 jitter 的理论值）
    expected = [0.5 * (2 ** (n - 1)) for n in range(1, 6)]
    # Act: 每档跑 100 次取 min/max 边界
    for attempt, base in zip(range(1, 6), expected, strict=False):
        samples = [policy.retry_after(ToolErrorKind.TRANSIENT, attempt) for _ in range(100)]
        lo, hi = min(samples), max(samples)
        # ±20% 边界（base 已含上限 8 的截断，attempt>=5 时 base=8）
        if base >= 8.0:
            # 上限截断后 jitter 范围会偏移，但仍在 [8*0.8, 8*1.2] 附近
            assert lo >= 8.0 * 0.8 - 0.01, f"attempt{attempt} lo={lo} below 80% of cap"
            assert hi <= 8.0 * 1.2 + 0.01, f"attempt{attempt} hi={hi} above 120% of cap"
        else:
            assert lo >= base * 0.8 - 0.01, f"attempt{attempt} lo={lo} below 80% of {base}"
            assert hi <= base * 1.2 + 0.01, f"attempt{attempt} hi={hi} above 120% of {base}"


# ---------- 额外：asyncio.TimeoutError 归 TRANSIENT ----------


def test_classify_asyncio_timeout_as_transient():
    # Arrange
    policy = ErrorPolicy()
    exc = asyncio.TimeoutError()
    # Act
    kind = policy.classify(exc, "slow_tool")
    # Assert
    assert kind is ToolErrorKind.TRANSIENT


# ---------- 额外：FATAL 优先级高于 TRANSIENT（如 401 + timeout 同时出现） ----------


def test_classify_fatal_takes_precedence_over_transient_keyword():
    # Arrange: 消息同时含 "401" 和 "timeout"，应归 FATAL
    policy = ErrorPolicy()
    exc = RuntimeError("timeout: 401 unauthorized")
    # Act
    kind = policy.classify(exc, "llm_call")
    # Assert
    assert kind is ToolErrorKind.FATAL
