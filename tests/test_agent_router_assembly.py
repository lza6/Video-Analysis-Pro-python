"""agent router react 路径装配测试(批 B-ASSEMBLE)。

覆盖(契约见 batch B-ASSEMBLE):
  ① VAP_AGENT_SUPERVISOR=0(默认)→ _build_react_agent 的 config.supervisor 为 None
  ② VAP_AGENT_SUPERVISOR=1 → supervisor 非 None
  ③ install_tool_guard 装配后 registry 走 approval(mock ApprovalBus:
     工具 ask → request → decide True 执行 / False 不执行 / 超时 deny)
  ④ SSE 审批端点 POST /api/agent/approval/{pin}/decide 与 GET pending
     用 TestClient 走通(mock ApprovalBus 单例)
  ⑤ 现有 test_agent_router_react.py 回归全绿(默认环境零回归,单独文件跑)

不依赖 pytest-asyncio:用 asyncio.new_event_loop 驱动 async(与
tests/test_scope_guard.py / test_agent_framework.py 一致)。
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402


def _run(coro):
    """同步驱动 async 测试,不依赖 pytest-asyncio。"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _make_request():
    """构造一个最小 Request 替身(_build_react_agent 只读 app.state / query_params)。"""
    class _Req:
        class _State:
            session_store = None
        app = type("App", (), {"state": _State()})
        query_params = {}
        headers = {}
    return _Req()


def _clear_backend_env(monkeypatch):
    """清掉可能影响 react 路径装配的环境变量(request/global suite 污染)。"""
    monkeypatch.delenv("VAP_AGENT_BACKEND", raising=False)
    monkeypatch.delenv("VAP_AGENT_SUPERVISOR", raising=False)
    monkeypatch.delenv("VAP_SANDBOX_ENABLED", raising=False)
    monkeypatch.delenv("VAP_ALLOW_WRITE_GENERAL", raising=False)
    monkeypatch.delenv("VAP_ALLOW_WRITE_CUT", raising=False)
    monkeypatch.delenv("VAP_ALLOW_WRITE_DELETE", raising=False)
    monkeypatch.delenv("VAP_MEMORY_LAYERED", raising=False)
    monkeypatch.delenv("VAP_SKILLS_ROSTER", raising=False)
    monkeypatch.delenv("VAP_AGENT_APPROVAL_TIMEOUT", raising=False)


# ---------------------------------------------------------------------------
# ① supervisor 默认关(零回归)
# ---------------------------------------------------------------------------


def test_supervisor_default_off_is_none(monkeypatch) -> None:
    """VAP_AGENT_SUPERVISOR 默认(0/缺省)→ config.supervisor 为 None。"""
    _clear_backend_env(monkeypatch)

    from src.web.routers import agent as mod

    agent, session, store = mod._build_react_agent(
        _make_request(), "s-test-null", None, "你好")
    assert agent.config.supervisor is None, \
        f"默认 supervisor 应为 None,实际 {agent.config.supervisor}"


def test_supervisor_default_explicit_zero_is_none(monkeypatch) -> None:
    """VAP_AGENT_SUPERVISOR=0 显式关闭 → supervisor 为 None。"""
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("VAP_AGENT_SUPERVISOR", "0")

    from src.web.routers import agent as mod

    ag, _, _ = mod._build_react_agent(
        _make_request(), "s-test-zero", None, "你好")
    assert ag.config.supervisor is None


# ---------------------------------------------------------------------------
# ② supervisor 开启
# ---------------------------------------------------------------------------


def test_supervisor_enabled_is_not_none(monkeypatch) -> None:
    """VAP_AGENT_SUPERVISOR=1 → supervisor 非 None(三件套装配)。"""
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("VAP_AGENT_SUPERVISOR", "1")

    from src.web.routers import agent as mod

    ag, _, _ = mod._build_react_agent(
        _make_request(), "s-test-on", None, "你好")
    sup = ag.config.supervisor
    assert sup is not None, "VAP_AGENT_SUPERVISOR=1 时 supervisor 应为非 None"
    assert sup.stuck is not None and sup.compressor is not None and sup.budget is not None


# ---------------------------------------------------------------------------
# ③ install_tool_guard 装配后 registry 走 approval(mock ApprovalBus)
# ---------------------------------------------------------------------------


def test_install_tool_guard_ask_approve_then_execute_with_bus() -> None:
    """工具 ask → bus.request_approval → decide True → 工具执行成功。"""
    from src.web.deps import ApprovalBus
    from src.core.tools.definition import ToolDefinition
    from src.core.tools.registry import ToolCall, ToolRegistry
    from src.core.tools.scope_guard import install_tool_guard

    bus = ApprovalBus()

    async def _approval_fn(call, sig) -> bool:
        pin = bus.request_approval(
            {"tool": call.name, "reason": sig.prompt})
        assert bus.decide(pin, True) is True
        return bool(bus.wait_decision(pin, timeout=0.1))

    async def _echo(value: str = "") -> str:
        return f"ran:{value}"

    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="create_job", description="create", execute_callback=_echo))

    install_tool_guard(reg, approval_fn=_approval_fn)
    result = _run(reg.execute(ToolCall(name="create_job", args={"value": "x"})))
    assert result.error is None, f"exec err: {result.error}"
    assert result.output == "ran:x"
    assert bus.pending_count() == 0


def test_install_tool_guard_ask_deny_skips_execute_with_bus() -> None:
    """bus.request_approval → decide False → 工具被拒绝不执行。"""
    from src.web.deps import ApprovalBus
    from src.core.tools.definition import ToolDefinition
    from src.core.tools.registry import ToolCall, ToolRegistry
    from src.core.tools.scope_guard import install_tool_guard

    bus = ApprovalBus()
    executed: dict[str, bool] = {"ran": False}

    async def _impl() -> str:
        executed["ran"] = True
        return "nope"

    async def _approval_fn(call, sig) -> bool:
        pin = bus.request_approval({"tool": call.name})
        assert bus.decide(pin, False) is True
        return bool(bus.wait_decision(pin, timeout=0.1))

    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="delete_history", description="del", execute_callback=_impl))
    install_tool_guard(reg, approval_fn=_approval_fn)
    result = _run(reg.execute(ToolCall(name="delete_history", args={})))
    assert result.is_error()
    assert "denied" in (result.error or "").lower()
    assert executed["ran"] is False


def test_install_tool_guard_ask_timeout_denies() -> None:
    """ask 超时(无人决定)→ wait_decision 返回 None → 默认拒绝执行。"""
    from src.web.deps import ApprovalBus
    from src.core.tools.definition import ToolDefinition
    from src.core.tools.registry import ToolCall, ToolRegistry
    from src.core.tools.scope_guard import install_tool_guard

    bus = ApprovalBus()
    executed: dict[str, bool] = {"ran": False}

    async def _impl() -> str:
        executed["ran"] = True
        return "nope"

    async def _approval_fn(call, sig) -> bool:
        pin = bus.request_approval({"tool": call.name})
        # 不 decide,等超时 → wait_decision 返回 None(安全侧默认拒绝)
        return bus.wait_decision(pin, timeout=0.05)

    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="send_message", description="send", execute_callback=_impl))
    install_tool_guard(reg, approval_fn=_approval_fn)
    result = _run(reg.execute(ToolCall(name="send_message", args={})))
    assert result.is_error()
    assert "denied" in (result.error or "").lower()
    assert executed["ran"] is False


# ---------------------------------------------------------------------------
# ⑤ (B-FIN-2 MAJOR-3) 分层记忆装配:VAP_MEMORY_LAYERED 默认 1=开
# ---------------------------------------------------------------------------


def test_memory_layered_default_on_not_none(monkeypatch) -> None:
    """VAP_MEMORY_LAYERED 默认(1)→ config.memory 非 None(三层记忆装配)。"""
    _clear_backend_env(monkeypatch)

    from src.web.routers import agent as mod

    ag, _, _ = mod._build_react_agent(
        _make_request(), "s-mem-on", None, "你好")
    mem = ag.config.memory
    assert mem is not None, "默认 VAP_MEMORY_LAYERED=1 时 memory 应为非 None"
    assert mem.enabled is True
    assert mem.working is not None and mem.experience is not None
    assert mem.triples is not None


def test_memory_layered_zero_is_none(monkeypatch) -> None:
    """VAP_MEMORY_LAYERED=0 显式关闭 → config.memory 为 None(零回归)。"""
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("VAP_MEMORY_LAYERED", "0")

    from src.web.routers import agent as mod

    ag, _, _ = mod._build_react_agent(
        _make_request(), "s-mem-zero", None, "你好")
    assert ag.config.memory is None, \
        "VAP_MEMORY_LAYERED=0 时 memory 应为 None"


# ---------------------------------------------------------------------------
# ⑥ (B-FIN-2 MAJOR-4) skills roster 装配:VAP_SKILLS_ROSTER=1 时换 roster
# ---------------------------------------------------------------------------


def test_skills_roster_default_off_keeps_legacy(monkeypatch) -> None:
    """VAP_SKILLS_ROSTER 默认(0)→ 构建不崩,memory 与旧行为一致(零回归)。"""
    _clear_backend_env(monkeypatch)
    monkeypatch.delenv("VAP_SKILLS_ROSTER", raising=False)

    from src.web.routers import agent as mod

    ag, _, _ = mod._build_react_agent(
        _make_request(), "s-roster-off", None, "监控分析")
    assert ag is not None and ag.config is not None


def test_skills_roster_enabled_builds(monkeypatch) -> None:
    """VAP_SKILLS_ROSTER=1 → _build_react_agent 正常构建(config 非 None)。"""
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("VAP_SKILLS_ROSTER", "1")

    from src.web.routers import agent as mod

    ag, _, _ = mod._build_react_agent(
        _make_request(), "s-roster-on", None, "帮我做个路演PPT")
    assert ag is not None and ag.config is not None


# ---------------------------------------------------------------------------
# ⑦ react 路径 SSE 审批装配:emitting approval_fn 注入 registry
# ---------------------------------------------------------------------------


def test_react_install_guard_injects_sse_approval_fn(monkeypatch) -> None:
    """_build_react_agent 装配后 registry._approval 非 None(SSE-emitting)。"""
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("VAP_AGENT_BACKEND", "react")

    from src.web.routers import agent as mod

    ag, _, _ = mod._build_react_agent(
        _make_request(), "s-sse", None, "你好")
    assert ag.tools._approval is not None, \
        "react 路径 approval_fn 应为 SSE-emitting 非 None"


# ---------------------------------------------------------------------------
# ④ SSE 审批端点(TestClient 走通, mock bus 单例)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ⑧ (B-FIN-2 MAJOR-2) 异步审批等待:wait_decision_async 不阻塞事件循环
# ---------------------------------------------------------------------------


def test_approval_wait_decision_async_allow_continue() -> None:
    """decide True → wait_decision_async 返回 True(工具继续执行)。"""
    from src.web.deps import ApprovalBus

    bus = ApprovalBus()
    pin = bus.request_approval({"tool": "create_job"})
    assert bus.decide(pin, True) is True
    assert _run(bus.wait_decision_async(pin, timeout=1.0)) is True
    assert bus.pending_count() == 0


def test_approval_wait_decision_async_deny() -> None:
    """decide False → wait_decision_async 返回 False(工具被拒)。"""
    from src.web.deps import ApprovalBus

    bus = ApprovalBus()
    pin = bus.request_approval({"tool": "delete_history"})
    assert bus.decide(pin, False) is True
    assert _run(bus.wait_decision_async(pin, timeout=1.0)) is False
    assert bus.pending_count() == 0


def test_approval_wait_decision_async_timeout_none() -> None:
    """无人决定 → 超时返回 None(安全侧默认拒绝)。"""
    from src.web.deps import ApprovalBus

    bus = ApprovalBus()
    pin = bus.request_approval({"tool": "highlight_cut"})
    assert _run(bus.wait_decision_async(pin, timeout=0.05)) is None
    assert bus.pending_count() == 1  # 超时不移除,仍可等


def test_approval_wait_decision_async_cross_thread() -> None:
    """decide 在另一线程 → wait_decision_async 仍解除(线程安全)。"""
    from src.web.deps import ApprovalBus

    bus = ApprovalBus()
    pin = bus.request_approval({"tool": "send_message"})
    import threading

    def _decide():
        import time
        time.sleep(0.05)
        bus.decide(pin, True)

    t = threading.Thread(target=_decide)
    t.start()
    decision = _run(bus.wait_decision_async(pin, timeout=2.0))
    t.join(timeout=3)
    assert decision is True


def _inject_bus(app, bus):
    """把 ApprovalBus 单例换掉(get_approval_bus 每调用都读模块级 _approval_bus)。"""
    import src.web.deps as deps
    # 直接用 setattr 替换模块级单例(无未用告警)
    setattr(deps, "_approval_bus", bus)
    app.state.test_approval_bus = bus
    return bus


@pytest.fixture
def approval_app(monkeypatch):
    """TestClient 包裹完整 app + 注入 mock bus + react backend。"""
    monkeypatch.setenv("VAP_AGENT_BACKEND", "react")
    monkeypatch.setenv("VAP_HEADLESS_TOKEN", "")
    from src.web.config import get_settings
    get_settings.cache_clear()
    try:
        from src.web.app import create_app
        from src.web.deps import ApprovalBus
        app_ = create_app()
        bus = ApprovalBus()
        _inject_bus(app_, bus)
        with TestClient(app_) as c:
            yield c, bus
    finally:
        get_settings.cache_clear()


def test_approval_decide_endpoint(approval_app) -> None:
    """POST /approval/{pin}/decide:首次决定 True,decide 幂等(重复 False)。"""
    client, bus = approval_app
    pin = bus.request_approval({"tool": "create_job", "reason": "写操作"})

    r = client.post(f"/api/agent/approval/{pin}/decide", json={"allow": True})
    assert r.status_code == 200, r.text
    assert r.json() == {"decided": True}

    # 重复决定(已决定)→ 幂等返回 False
    r2 = client.post(f"/api/agent/approval/{pin}/decide", json={"allow": False})
    assert r2.status_code == 200, r2.text
    assert r2.json() == {"decided": False}

    # wait_decision 解除,返回 True(首次决定)
    assert bus.wait_decision(pin, timeout=0.1) is True
    assert bus.pending_count() == 0


def test_approval_decide_unknown_pin(approval_app) -> None:
    """未知 pin → decide 返回 False(幂等,不 500)。"""
    client, _ = approval_app
    r = client.post("/api/agent/approval/nonexistent-pin/decide",
                    json={"allow": True})
    assert r.status_code == 200, r.text
    assert r.json() == {"decided": False}


def test_approval_pending_endpoint(approval_app) -> None:
    """GET /approval/pending:返回未决审批;真实流程(工具线程 wait + 前端 HTTP
    decide)后已决 pin 从 pending 移除。模拟完整链路:
      request_approval → (SSE 会投递) → POST decide(HTTP) → wait_decision 解除。"""
    client, bus = approval_app
    p1 = bus.request_approval({"tool": "create_job", "reason": "写操作"})
    p2 = bus.request_approval({"tool": "delete_history", "reason": "危险写"})

    r = client.get("/api/agent/approval/pending")
    assert r.status_code == 200, r.text
    pending = {p["pin"]: p for p in r.json()["pending"]}
    assert p1 in pending and p2 in pending, f"pending 缺 pin: {pending}"
    assert pending[p1]["tool"] == "create_job"
    assert pending[p2]["tool"] == "delete_history"

    # 模拟工具线程:阻塞在 wait_decision(轮询等待前端 decide)
    import threading
    holder: dict[str, object] = {}

    def _wait_remote():
        holder["decision"] = bus.wait_decision(p1, timeout=5.0)

    t = threading.Thread(target=_wait_remote)
    t.start()

    # 前端收到 SSE approval-request 后经 HTTP 决定(真实闭环)
    r3 = client.post(f"/api/agent/approval/{p1}/decide", json={"allow": True})
    assert r3.status_code == 200, r3.text
    assert r3.json() == {"decided": True}

    # 工具线程 wait_decision 解除,返回 True(允许执行)
    t.join(timeout=6)
    assert holder.get("decision") is True

    # 已决 pin 已从 pending 移除;未决的 p2 仍在
    r2 = client.get("/api/agent/approval/pending")
    pins2 = [p["pin"] for p in r2.json()["pending"]]
    assert p1 not in pins2, f"decide 后 p1 应移出 pending: {pins2}"
    assert p2 in pins2


def test_approval_endpoint_auth_required(monkeypatch, approval_app) -> None:
    """VAP_HEADLESS_TOKEN 非空时 approve 端点需 Bearer,否则 401。"""
    client, bus = approval_app
    monkeypatch.setenv("VAP_HEADLESS_TOKEN", "test-token-32chars-long-secret")
    from src.web.config import get_settings
    get_settings.cache_clear()
    # 恢复针对该 token 的 settings + bus
    p1 = bus.request_approval({"tool": "x"})
    # 无 token → 401
    r = client.post(f"/api/agent/approval/{p1}/decide", json={"allow": True})
    assert r.status_code == 401, f"期望 401 实际 {r.status_code}: {r.text}"
    get_settings.cache_clear()
    # 恢复空 token 便于后续 fixture 清理(settings.cache_clear 后重新读)
    monkeypatch.setenv("VAP_HEADLESS_TOKEN", "")
    get_settings.cache_clear()