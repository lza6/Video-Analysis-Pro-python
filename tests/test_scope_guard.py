"""工具范围守卫(scope_guard)测试。

覆盖(契约见 batch B-P0-2):
  ① 读操作默认 allow
  ② 写操作在无 VAP_ALLOW_WRITE_* 时 ask
  ③ 危险写(delete 模式)超时 deny(ApprovalBus 超时返回 None → 拒绝)
  ④ VAP_ALLOW_WRITE_* 置 1 时放行
  ⑤ ApprovalBus request → decide → wait 全链路(含超时 false/None)
  ⑥ install_tool_guard 装配后 registry.execute 走 approval 回调
     (ask → handler True 后执行 / False 后不执行)
  ⑦ 无 handler 时(默认)原 test_agent_framework 的 execute 行为零回归

不依赖 pytest-asyncio:用 asyncio.run()/new_event_loop 驱动 async
(与 tests/test_agent_framework.py / test_sandbox.py 一致)。
"""
from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest

from src.core.tools.definition import ToolDefinition
from src.core.tools.registry import ToolCall, ToolRegistry
from src.core.tools.scope_guard import (
    DANGEROUS_WRITE_PATTERNS,
    ScopeGuard,
    VAP_ALLOW_WRITE_CUT,
    VAP_ALLOW_WRITE_DELETE,
    VAP_ALLOW_WRITE_GENERAL,
    approval_priority,
    build_approval_fn,
    build_guard_from_env,
    install_tool_guard,
)
from src.core.tools.waterfall import Ask, ToolNeedsApproval


def _run(coro):
    """同步驱动 async 测试,不依赖 pytest-asyncio(与 test_sandbox.py 一致)。"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _echo_tool(name: str = "echo") -> ToolDefinition:
    """回显工具:返回 'ran:' + value(验证 execute 是否真正执行)。"""

    async def _impl(value: str = "") -> str:
        return f"ran:{value}"

    return ToolDefinition(
        name=name, description="echo", execute_callback=_impl)


# ---------------------------------------------------------------------------
# ① 读操作默认 allow
# ---------------------------------------------------------------------------


def test_read_operations_allow() -> None:
    """读前缀(get_*/search_*/list_*/summarize_*/trace_*/point_*)全部 allow。"""
    guard = ScopeGuard(allow_write_general=False,
                       allow_write_delete=False, allow_write_cut=False)
    for name in ("get_video_meta", "search_kb", "list_frames",
                 "summarize_hits", "trace_item", "point_at_object"):
        level, _ = guard.decision(ToolCall(name=name, args={}))
        assert level == "allow", f"{name} 应 allow,实际 {level}"


def test_read_hook_returns_none() -> None:
    """读操作 pre_execute 钩子返回 None(不产生 Ask,直接放行)。"""
    guard = build_guard_from_env()
    sig = _run(guard.pre_execute(ToolCall(name="get_video_meta", args={}), None))
    assert sig is None


# ---------------------------------------------------------------------------
# ② 写操作在无 VAP_ALLOW_WRITE_* 时 ask
# ---------------------------------------------------------------------------


def test_write_operations_ask_by_default(monkeypatch) -> None:
    """写操作(无开关放行)应返回 ask,pre_execute 产生 Ask 信号。

    v10.3.1 (P0-3):send_* 升为危险写,其 Ask.priority 为
    'dangerous_write';一般写(create/update/generate)保持 'write'。
    """
    monkeypatch.delenv(VAP_ALLOW_WRITE_GENERAL, raising=False)
    monkeypatch.delenv(VAP_ALLOW_WRITE_CUT, raising=False)
    monkeypatch.delenv(VAP_ALLOW_WRITE_DELETE, raising=False)
    guard = build_guard_from_env()
    for name in ("create_job", "update_config", "generate_skill"):
        level, reason = guard.decision(ToolCall(name=name, args={}))
        assert level == "ask", f"{name} 应 ask,实际 {level}({reason})"
        sig = _run(guard.pre_execute(ToolCall(name=name, args={}), None))
        assert isinstance(sig, Ask), f"{name} 应产生 Ask"
        assert sig.priority == "write", f"{name} 一般写 priority 应为 write"


# ---------------------------------------------------------------------------
# ③ 危险写超时 deny
# ---------------------------------------------------------------------------


def test_dangerous_write_requires_approval() -> None:
    """delete_*/highlight_cut/trigger_batch/start_rtsp_monitor 必须 ask。"""
    guard = ScopeGuard(allow_write_general=False,
                       allow_write_delete=False, allow_write_cut=False)
    for name in ("delete_history", "delete_video", "highlight_cut",
                 "trigger_batch", "start_rtsp_monitor"):
        level, _ = guard.decision(ToolCall(name=name, args={}))
        assert level == "ask", f"{name} 危险写应 ask,实际 {level}"


def test_approval_timeout_denies() -> None:
    """审批超时(无人决定)默认拒绝:executed 标记不置位。"""
    from src.web.deps import ApprovalBus

    bus = ApprovalBus()
    executed: dict[str, bool] = {"ran": False}

    async def _approval_fn(call, sig) -> bool:
        return bus.wait_decision(
            bus.request_approval({"tool": call.name}), timeout=0.1)

    reg = ToolRegistry()
    reg.register(_echo_tool("delete_history"))

    async def _impl() -> str:
        executed["ran"] = True
        return "nope"

    reg.register(ToolDefinition(
        name="delete_history", description="del",
        execute_callback=_impl))
    # 覆盖同名单:直接注册 _echo 会被同名覆盖,改用单独工具
    # (上面 reg.register(_echo_tool("delete_history")) 被下面覆盖)

    guard = ScopeGuard(allow_write_general=False,
                       allow_write_delete=False, allow_write_cut=False)
    install_tool_guard(reg, approval_fn=_approval_fn)
    result = _run(reg.execute(ToolCall(name="delete_history", args={})))
    assert result.is_error()
    assert "denied" in (result.error or "").lower()
    assert executed["ran"] is False


def test_dangerous_write_never_allowed_without_delete_switch() -> None:
    """危险写即使 VAP_ALLOW_WRITE_GENERAL=1 也不放行,需 VAP_ALLOW_WRITE_DELETE。"""
    guard = ScopeGuard(allow_write_general=True,
                       allow_write_delete=False, allow_write_cut=False)
    level, _ = guard.decision(ToolCall(name="delete_history", args={}))
    assert level == "ask", "危险写不应被 general 开关放行"


# ---------------------------------------------------------------------------
# ④ VAP_ALLOW_WRITE_* 置 1 时放行
# ---------------------------------------------------------------------------


def test_env_switch_allows_general_write(monkeypatch) -> None:
    """VAP_ALLOW_WRITE_GENERAL=1 → 一般写操作 allow。"""
    monkeypatch.setenv(VAP_ALLOW_WRITE_GENERAL, "1")
    guard = build_guard_from_env()
    level, _ = guard.decision(ToolCall(name="create_job", args={}))
    assert level == "allow"


def test_env_switch_allows_delete(monkeypatch) -> None:
    """VAP_ALLOW_WRITE_DELETE=1 → 危险写 allow。"""
    monkeypatch.setenv(VAP_ALLOW_WRITE_DELETE, "1")
    guard = build_guard_from_env()
    level, _ = guard.decision(ToolCall(name="delete_history", args={}))
    assert level == "allow"


def test_env_switch_allows_cut(monkeypatch) -> None:
    """VAP_ALLOW_WRITE_CUT=1 → highlight_cut 放行。"""
    monkeypatch.setenv(VAP_ALLOW_WRITE_CUT, "1")
    monkeypatch.setenv(VAP_ALLOW_WRITE_GENERAL, "0")
    guard = build_guard_from_env()
    level, _ = guard.decision(ToolCall(name="highlight_cut", args={}))
    assert level == "allow"
    # 非剪辑写操作不受 CUT 开关影响(仍需 ask)
    level2, _ = guard.decision(ToolCall(name="update_config", args={}))
    assert level2 == "ask"


# ---------------------------------------------------------------------------
# ⑤ ApprovalBus request → decide → wait 全链路
# ---------------------------------------------------------------------------


def test_approval_bus_request_decide_wait() -> None:
    """request → wait(线程内决定)→ 返回 True;无决定超时返回 None。"""
    from src.web.deps import ApprovalBus

    bus = ApprovalBus()
    pin = bus.request_approval({"tool": "delete_history"})
    assert pin.startswith("aprv-")
    assert bus.pending_count() == 1

    # 立即查询(未决定)→ None
    assert bus.wait_decision(pin, timeout=0) is None
    # 决定后 → True
    assert bus.decide(pin, True) is True
    assert bus.wait_decision(pin, timeout=0.1) is True
    # 决定后 pin 从 pending 移除
    assert bus.pending_count() == 0
    # 重复 decide 返回 False
    assert bus.decide(pin, False) is False


def test_approval_bus_timeout_returns_none() -> None:
    """超时无人决定 → wait_decision 返回 None(安全侧默认拒绝)。"""
    from src.web.deps import ApprovalBus

    bus = ApprovalBus()
    pin = bus.request_approval({"tool": "highlight_cut"})
    assert bus.wait_decision(pin, timeout=0.05) is None
    assert bus.pending_count() == 1  # 超时不移除,仍可等


def test_approval_bus_decide_false() -> None:
    """decide(pin, False) → wait_decision 返回 False(拒绝)。"""
    from src.web.deps import ApprovalBus

    bus = ApprovalBus()
    pin = bus.request_approval({"tool": "delete_history"})
    assert bus.decide(pin, False) is True
    assert bus.wait_decision(pin, timeout=0.1) is False
    assert bus.pending_count() == 0


def test_approval_bus_ring_records_decisions() -> None:
    """decide 后 recent_decisions 记录审计条目。"""
    from src.web.deps import ApprovalBus

    bus = ApprovalBus(capacity=8)
    pin = bus.request_approval({"tool": "create_job"})
    bus.decide(pin, True)
    bus.wait_decision(pin, timeout=0.1)  # 触发记账
    recs = bus.recent_decisions()
    assert any(r["pin"] == pin and r["decided"] is True for r in recs)


def test_approval_bus_thread_safety() -> None:
    """多线程并发 request/decide/wait 不丢决定(线程安全)。"""
    from src.web.deps import ApprovalBus

    bus = ApprovalBus()
    results: list[bool] = []
    errors: list[Exception] = []

    def _worker() -> None:
        try:
            pin = bus.request_approval({"tool": "x"})
            # 另一线程决定它
            if not bus.decide(pin, True):
                raise AssertionError("decide 应首次成功")
            res = bus.wait_decision(pin, timeout=2.0)
            results.append(res is True)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert not errors, f"线程错误: {errors}"
    assert all(results) and len(results) == 16


# ---------------------------------------------------------------------------
# ⑥ install_tool_guard 装配后 registry.execute 走 approval 回调
# ---------------------------------------------------------------------------


def test_install_guard_ask_approve_then_execute() -> None:
    """写操作被 ask,handler 返回 True → 工具执行成功。"""
    reg = ToolRegistry()
    reg.register(_echo_tool("create_job"))
    log: list[str] = []
    approval_calls: list[str] = []

    async def _approval_fn(call, sig) -> bool:
        approval_calls.append(call.name)
        return True

    guard = install_tool_guard(reg, approval_fn=_approval_fn)
    assert guard is not None
    result = _run(reg.execute(ToolCall(name="create_job", args={"value": "x"},
                                       call_id="c1")))
    assert result.error is None
    assert result.output == "ran:x"
    assert approval_calls == ["create_job"]
    assert log == []


def test_install_guard_ask_deny_skips_execute() -> None:
    """handler 返回 False → 工具被拒绝,不执行。"""
    reg = ToolRegistry()
    executed: dict[str, bool] = {"ran": False}

    async def _impl() -> str:
        executed["ran"] = True
        return "nope"

    reg.register(ToolDefinition(
        name="delete_history", description="del", execute_callback=_impl))
    approval_calls: list[str] = []

    async def _approval_fn(call, sig) -> bool:
        approval_calls.append(call.name)
        return False

    install_tool_guard(reg, approval_fn=_approval_fn)
    result = _run(reg.execute(ToolCall(name="delete_history", args={})))
    assert result.is_error()
    assert "denied" in (result.error or "").lower()
    assert executed["ran"] is False
    assert approval_calls == ["delete_history"]


def test_install_guard_approval_fn_can_wait_on_bus() -> None:
    """approval_fn 经 ApprovalBus:decide True → 工具执行;False → 拒绝。"""
    from src.web.deps import ApprovalBus

    bus = ApprovalBus()

    async def _approval_fn(call, sig) -> bool:
        pin = bus.request_approval(
            {"tool": call.name, "reason": sig.prompt})
        # wait_decision 是阻塞轮询,丢到线程池避免卡事件循环;
        # 用 asyncio 事件对象通知:approval 线程 await gate 后主线程再 decide
        loop = asyncio.get_running_loop()
        gate = asyncio.Event()
        loop.call_soon_threadsafe(gate.set)
        await gate.wait()
        return bus.wait_decision(pin, timeout=0.01)

    reg = ToolRegistry()
    reg.register(_echo_tool("send_message"))
    install_tool_guard(reg, approval_fn=_approval_fn)
    result = _run(reg.execute(ToolCall(name="send_message", args={})))
    # 无人决定 → 超时返回 None → 拒绝(默认安全:超时默认 deny)
    assert result.is_error()
    assert "denied" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# ⑦ 无 handler 时(默认)原 execute 行为零回归
# ---------------------------------------------------------------------------


def test_ask_without_handler_raises_needs_approval() -> None:
    """Ask 信号 + 无 approval handler → ToolNeedsApproval(向后兼容)。"""
    reg = ToolRegistry()
    reg.register(_echo_tool("delete_history"))

    async def _ask_hook(call, defn):
        return Ask(prompt="needs approval")

    reg.waterfall.add_pre_execute(_ask_hook)
    result = _run(reg.execute(ToolCall(name="delete_history", args={})))
    assert result.is_error()
    assert "needs approval" in (result.error or "")


def test_no_guard_no_regression() -> None:
    """未装配 guard 时读/写工具均正常执行(零回归)。"""
    reg = ToolRegistry()
    reg.register(_echo_tool("get_video_meta"))
    reg.register(_echo_tool("delete_history"))
    r1 = _run(reg.execute(ToolCall(name="get_video_meta", args={})))
    r2 = _run(reg.execute(ToolCall(name="delete_history", args={})))
    assert r1.error is None and r1.output == "ran:"
    assert r2.error is None and r2.output == "ran:"


def test_install_guard_idempotent() -> None:
    """重复 install_tool_guard 不叠加 pre_execute 钩子。"""
    reg = ToolRegistry()
    reg.register(_echo_tool("create_job"))
    g1 = install_tool_guard(reg, approval_fn=None)
    g2 = install_tool_guard(reg, approval_fn=None)
    assert g1 is not None and g2 is not None
    assert sum(
        1 for _ in reg.waterfall.pre_execute
    ) == 1, "重复装配不应叠加钩子"


# ---------------------------------------------------------------------------
# 审批优先级 / 决策表辅助
# ---------------------------------------------------------------------------


def test_approval_priority_default_write() -> None:
    """Ask 缺省 priority='write'(向后兼容)。"""
    assert approval_priority(Ask(prompt="x")) == "write"
    assert approval_priority(Ask(prompt="x", priority="dangerous")) == "dangerous"


def test_classify_categories() -> None:
    """classify 分类:读 / 写 / 危险写。"""
    guard = ScopeGuard()
    assert guard.classify("get_video_meta") == "read"
    assert guard.classify("create_job") == "write"
    assert guard.classify("delete_history") == "dangerous_write"
    assert guard.classify("highlight_cut") == "dangerous_write"


# ---------------------------------------------------------------------------
# v10.3.1 (P0-3):真实落盘工具 / 任意 JS 执行的分类修正回归
# ---------------------------------------------------------------------------


def test_media_gen_disk_writers_are_ask(monkeypatch) -> None:
    """make_subtitle / make_short_video / make_voiceover 真实落盘 →
    classify=write、decision=ask(此前误归 read → allow,静默写盘)。"""
    monkeypatch.delenv(VAP_ALLOW_WRITE_GENERAL, raising=False)
    monkeypatch.delenv(VAP_ALLOW_WRITE_CUT, raising=False)
    guard = build_guard_from_env()
    for name in ("make_subtitle", "make_short_video", "make_voiceover"):
        assert guard.classify(name) == "write", \
            f"{name} 应 classify=write(P0-3),实际 {guard.classify(name)}"
        level, _ = guard.decision(ToolCall(name=name, args={}))
        assert level == "ask", f"{name} 应 ask(P0-3),实际 {level}"


def test_cdp_evaluate_is_dangerous_write(monkeypatch) -> None:
    """cdp_evaluate 可执行任意 JS → classify=dangerous_write、ask
    (此前误归 read → allow);cdp_eval_write 同样收进危险写。"""
    monkeypatch.delenv(VAP_ALLOW_WRITE_DELETE, raising=False)
    guard = build_guard_from_env()
    for name in ("cdp_evaluate", "cdp_eval_write"):
        assert guard.classify(name) == "dangerous_write", \
            f"{name} 应 classify=dangerous_write(P0-3),实际 {guard.classify(name)}"
        level, _ = guard.decision(ToolCall(name=name, args={}))
        assert level == "ask", f"{name} 应 ask(P0-3),实际 {level}"


def test_ask_priority_distinguishes_dangerous(monkeypatch) -> None:
    """Ask.priority 分级:一般写='write',危险写='dangerous_write'
    (前端据此呈现危险等级配色)。"""
    monkeypatch.delenv(VAP_ALLOW_WRITE_GENERAL, raising=False)
    monkeypatch.delenv(VAP_ALLOW_WRITE_DELETE, raising=False)
    guard = build_guard_from_env()
    sig_write = _run(guard.pre_execute(
        ToolCall(name="create_job", args={}), None))
    sig_danger = _run(guard.pre_execute(
        ToolCall(name="delete_history", args={}), None))
    assert sig_write.priority == "write"
    assert sig_danger.priority == "dangerous_write"


def test_read_tools_still_allow(monkeypatch) -> None:
    """读工具不被 P0-3 波及(get_*/search_*/cdp_list_targets 等)。"""
    guard = build_guard_from_env()
    for name in ("get_video_meta", "search_web", "cdp_list_targets",
                 "web_browser_snapshot", "web_browser_screenshot"):
        level, _ = guard.decision(ToolCall(name=name, args={}))
        assert level == "allow", f"{name} 应仍 allow,实际 {level}"


def test_install_tool_guard_wires_error_policy() -> None:
    """v10.3.1 (P0-4):install_tool_guard 默认接线 ErrorPolicy
    (此前 set_error_policy 自 v10.1 定义以来零调用方)。"""
    reg = ToolRegistry()
    reg.register(_echo_tool("echo"))
    assert reg._error_policy is None, "新 registry 应无策略"
    install_tool_guard(reg, approval_fn=_async_true)
    assert reg._error_policy is not None, \
        "P0-4:install_tool_guard 应默认装配 ErrorPolicy(重试/熔断生效)"
    # 幂等:再装一次不覆盖
    install_tool_guard(reg, approval_fn=_async_true)
    assert reg._error_policy is not None


def test_install_tool_guard_error_policy_opt_out() -> None:
    """wire_error_policy=False 可退回无策略行为(兼容旧测试)。"""
    reg = ToolRegistry()
    reg.register(_echo_tool("echo"))
    install_tool_guard(reg, approval_fn=_async_true, wire_error_policy=False)
    assert reg._error_policy is None


async def _async_true(call, sig) -> bool:
    return True


def test_install_tool_guard_wires_lock_resolver() -> None:
    """v10.3.1 (P0-4):install_tool_guard 接线 lock_resolver
    (读工具 → read 锁,写工具 → write 锁;此前零调用方)。"""
    reg = ToolRegistry()
    reg.register(_echo_tool("get_video_meta"))
    reg.register(_echo_tool("delete_history"))
    install_tool_guard(reg, approval_fn=_async_true)
    assert reg._lock_resolver is not None
    lock, mode = reg._lock_resolver("get_video_meta", {})
    assert mode == "read" and lock is not None
    lock2, mode2 = reg._lock_resolver("delete_history", {})
    assert mode2 == "write" and lock2 is not None
