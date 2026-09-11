"""Agent 监督层测试（批 B-P0-1）。

覆盖：
  1. 同 tool_args 重复 3 次 → stop_reason=STUCK
  2. 连续 error 循环 3 次 → stop_reason=STUCK
  3. 会话超阈值 → 压缩摘要注入 + 保留最近 N 轮（tool_call_id 配对不破裂）
  4. token 超 budget → stop_reason=BUDGET
  5. flag 默认关时行为零回归（ReactLoopAgent 用例仍绿）

不依赖 pytest-asyncio：用 asyncio.run() 在同步测试中驱动 async 代码
（与 tests/test_agent_framework.py 一致）。
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from src.core.agent import AgentConfig, ReactLoopAgent, Session, SessionEvent
from src.core.agent.session import SessionStore
from src.core.tools.registry import ToolRegistry
from src.core.agent.loop import LLMChunk, MockLLMClient
from src.core.agent.supervisor import (
    BudgetGuard,
    ContextCompressor,
    RoundSignature,
    StuckDetector,
    SUMMARY_EVENT_KIND,
    Supervisor,
    SupervisorConfig,
)
from src.core.tools import ToolCall, ToolDefinition, ToolRegistry


def _run(coro):
    """同步驱动 async 测试（与 test_agent_framework.py 的 _run 一致）。"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


# ---------------------------------------------------------------------------
# 1. StuckDetector 单元
# ---------------------------------------------------------------------------


def test_stuck_detector_same_tool_args() -> None:
    """相同 tool_args 循环（observation 有变化）连续 3 次 → same_tool_args。"""
    d = StuckDetector(repeat_threshold=3)
    sigs = [
        RoundSignature(
            actions=(("echo", '{"x": "same"}'),),
            observations=(f"echo:result-{i}",),
            has_error=False,
        )
        for i in range(3)
    ]
    assert d.observe(sigs[0]) is None
    assert d.observe(sigs[1]) is None
    tr = d.observe(sigs[2])
    assert tr is not None
    assert tr.rule == StuckDetector.RULE_TOOL_ARGS
    assert tr.count == 3


def test_stuck_detector_same_action_observation() -> None:
    """同 action 同 observation 连续 3 次 → 命中 same_action_observation。"""
    d = StuckDetector(repeat_threshold=3)
    sig = RoundSignature(
        actions=(("search", '{"q": "x"}'),),
        observations=("out:x",),
        has_error=False,
    )
    d.observe(sig)
    d.observe(sig)
    tr = d.observe(sig)
    assert tr is not None
    assert tr.rule == StuckDetector.RULE_ACTION_OBSERVATION


def test_stuck_detector_error_loop() -> None:
    """连续 error 3 次 → 命中 error_loop。"""
    d = StuckDetector(repeat_threshold=3)
    sig = RoundSignature(
        actions=(("f", "{}"),),
        observations=("Error: boom",),
        has_error=True,
    )
    d.observe(sig)
    d.observe(sig)
    tr = d.observe(sig)
    assert tr is not None
    assert tr.rule == StuckDetector.RULE_ERROR_LOOP


def test_stuck_detector_no_false_positive_on_progress() -> None:
    """有进展（observation 变化）不触发。"""
    d = StuckDetector(repeat_threshold=3)
    for i in range(3):
        sig = RoundSignature(
            actions=(("echo", f'{{"x": "{i}"}}'),),
            observations=(f"echo:{i}",),
            has_error=False,
        )
        assert d.observe(sig) is None


# ---------------------------------------------------------------------------
# 2. ContextCompressor 单元
# ---------------------------------------------------------------------------


def _mk_session(n_users: int, keep: int = 3, content_len: int = 80) -> Session:
    """造一个 Session：system + N 轮 user/assistant/tool_result。

    每轮：user(content) → assistant(tool_calls) → tool_result(content)。
    最后一轮只有 user+assistant(无 tool_calls) 模拟进行中。
    """
    s = Session("c1", system_prompt="SYS")
    for i in range(n_users):
        s.append(SessionEvent("user", time.time(),
                              {"content": f"q{i:02d}" * content_len}))
        if i < n_users - 1:
            s.append(SessionEvent("assistant", time.time(), {
                "content": "",
                "tool_calls": [{"id": f"c{i}", "name": "t", "args": {}}],
            }))
            s.append(SessionEvent("tool_result", time.time(), {
                "tool_call_id": f"c{i}", "content": f"res{i}",
            }))
        else:
            s.append(SessionEvent("assistant", time.time(),
                                  {"content": f"final{i}"}))
    return s


def test_compressor_should_compress_thresholds() -> None:
    """事件数/字符数超阈值才压缩。"""
    # 事件数：system + 2 轮 = 1 + 5 = 6 条，max_events=5 → 应压缩（> 上限）
    s7 = _mk_session(2, content_len=10)
    assert len(s7.events) == 6
    c = ContextCompressor(max_events=5, max_chars=100_000)
    assert c.should_compress(s7)
    # 字符数：内容放大（user 每条 300 字符 > max_chars=200）
    s_big = _mk_session(2, content_len=300)
    c2 = ContextCompressor(max_events=1000, max_chars=200)
    assert c2.should_compress(s_big)
    # 未超阈值不压缩
    s_small = _mk_session(2, content_len=10)
    c3 = ContextCompressor(max_events=1000, max_chars=100_000)
    assert not c3.should_compress(s_small)


def test_compressor_injects_summary_and_keeps_recent() -> None:
    """压缩后：注入摘要 system 消息 + 保留最近 N 轮完整 + tool_call_id 配对。"""
    s = _mk_session(5)  # system + 5 轮，共 5*3+1=16 事件
    c = ContextCompressor(max_events=10, keep_recent_turns=2)
    assert c.should_compress(s)

    cond = c.compress(s)
    assert cond is not None
    msgs = list(cond.messages)

    # 摘要 system 消息在最前，且含标记文本
    assert msgs[0]["role"] == "system"
    assert "历史对话摘要" in msgs[0]["content"]
    # 保留最近 keep_recent_turns 轮：最后一轮 user + assistant(final)
    tail_users = [m for m in msgs if m["role"] == "user"]
    assert tail_users[-1]["content"] == "q04" * 80
    # 最近 N 轮内的 tool_result（第 4 轮，下标索引从 0 起，即 i=3 轮）
    tool_ids = [m.get("tool_call_id") for m in msgs if m["role"] == "tool"]
    assert "c3" in tool_ids
    # 早期轮次的 tool_result（c0/c1/c2）已被摘要覆盖，不再出现在消息里
    assert "c0" not in tool_ids
    assert "c1" not in tool_ids
    assert "c2" not in tool_ids

    # 摘要事件已 append 到 session（幂等：再压缩不重复 append）
    kinds = [e.payload.get("kind") for e in s.events]
    assert SUMMARY_EVENT_KIND in kinds
    n_marker = kinds.count(SUMMARY_EVENT_KIND)
    assert n_marker == 1
    c.compress(s)
    kinds2 = [e.payload.get("kind") for e in s.events]
    assert kinds2.count(SUMMARY_EVENT_KIND) == 1


# ---------------------------------------------------------------------------
# 3. BudgetGuard 单元
# ---------------------------------------------------------------------------


def test_budget_guard_per_turn_and_total() -> None:
    """per-turn 超限先触发；累计超限也触发。"""
    b = BudgetGuard(per_turn_token_limit=100, total_token_budget=250)
    b.begin_turn()
    assert b.account(60) is None
    assert b.account(60) is not None  # 120 > 100 → per_turn
    # 超限后 per-turn 不再松绑（宁严不松）；begin_turn 重置后恢复
    assert b.account(0) is not None
    b.begin_turn()
    assert b.account(100) is None  # 累计 220 < 250
    assert b.account(100) is not None  # 累计 320 > 250 → total


def test_budget_guard_estimate_tokens() -> None:
    """估算：空串 0，非空按 ~4 字符/token。"""
    assert BudgetGuard.estimate_tokens("") == 0
    assert BudgetGuard.estimate_tokens("a") == 1
    assert BudgetGuard.estimate_tokens("a" * 8) == 2
    assert BudgetGuard.estimate_tokens("a" * 4) == 1


# ---------------------------------------------------------------------------
# 4. 监督层集成（ReactLoopAgent + Supervisor）
# ---------------------------------------------------------------------------


def _echo_registry() -> ToolRegistry:
    async def _echo(x: str = "") -> str:
        return f"echo:{x}"

    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="echo", description="echo back", execute_callback=_echo,
        input_schema={"type": "object",
                      "properties": {"x": {"type": "string"}}},
    ))
    return reg


def test_supervisor_flag_default_on_v1031(monkeypatch) -> None:
    """v10.3.1 (P0-1):feature flag 默认开 —— Supervisor.from_env() 返回实例。

    显式设 false/0/off 才返回 None(回退 v10.2 行为)。
    """
    monkeypatch.delenv("VAP_AGENT_SUPERVISOR", raising=False)
    assert Supervisor.from_env() is not None  # 默认开(v10.3.1)
    monkeypatch.setenv("VAP_AGENT_SUPERVISOR", "false")
    assert Supervisor.from_env() is None  # 显式关

    mock = MockLLMClient([
        LLMChunk(final_tool_calls=[
            {"id": "call-1", "name": "echo", "args": {"x": "hello"}},
        ]),
        LLMChunk(delta_text="final answer: echo:hello", stop_reason="stop"),
    ])
    agent = ReactLoopAgent(AgentConfig(), mock, _echo_registry())
    session = Session("s1", system_prompt="sys")
    result = _run(agent.run_turn(session, "请 echo hello"))

    assert result.stop_reason.value == "stop"
    assert result.tool_calls == 1
    assert result.steps == 2
    assert "echo:hello" in result.final_text
    types = [e.type for e in session.events]
    assert types == ["system", "user", "assistant", "tool_result", "assistant"]


def test_supervisor_stuck_same_tool_args_stops() -> None:
    """同 tool_args 重复 3 次 → STUCK。"""
    sup = Supervisor(SupervisorConfig(
        stuck_repeat_threshold=3,
        compress_max_events=1000,  # 禁用压缩干扰
        per_turn_token_limit=None,
    ))
    mock = MockLLMClient([
        LLMChunk(final_tool_calls=[
            {"id": f"c{i}", "name": "echo", "args": {"x": "same"}},
        ])
        for i in range(6)  # 第 1、2 轮不触发，第 3 轮触发
    ])
    agent = ReactLoopAgent(AgentConfig(supervisor=sup, max_steps=8),
                           mock, _echo_registry())
    session = Session("stuck1", system_prompt="sys")
    result = _run(agent.run_turn(session, "请重复 echo"))

    assert result.stop_reason.value == "stuck"
    assert result.steps == 3
    assert "stuck" in (result.error or "")
    # session 里已有 3 轮 tool_result
    n_tr = len([e for e in session.events if e.type == "tool_result"])
    assert n_tr == 3


def test_supervisor_stuck_error_loop_stops() -> None:
    """连续 error 3 次 → STUCK（error_loop 规则）。"""
    async def _boom() -> str:
        raise ValueError("boom")

    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="boom", description="boom", execute_callback=_boom))

    sup = Supervisor(SupervisorConfig(
        stuck_repeat_threshold=3,
        compress_max_events=1000,
        per_turn_token_limit=None,
    ))
    mock = MockLLMClient([
        LLMChunk(final_tool_calls=[
            {"id": f"c{i}", "name": "boom", "args": {}},
        ])
        for i in range(5)
    ])
    agent = ReactLoopAgent(AgentConfig(supervisor=sup, max_steps=8),
                           mock, reg)
    session = Session("stuck2", system_prompt="sys")
    result = _run(agent.run_turn(session, "请调用 boom"))

    assert result.stop_reason.value == "stuck"
    assert result.steps == 3
    assert "stuck" in (result.error or "")


def test_supervisor_budget_trigger() -> None:
    """每轮 LLM 请求 token 累计超 per-turn 上限 → BUDGET。

    单轮 messages（system+user）估算约 35 tokens；per_turn=30 必超。
    """
    sup = Supervisor(SupervisorConfig(
        per_turn_token_limit=30,
        enable_stuck=False,
        enable_compression=False,
    ))
    mock = MockLLMClient([
        LLMChunk(delta_text="answer", stop_reason="stop"),
    ])
    agent = ReactLoopAgent(AgentConfig(supervisor=sup), mock, _echo_registry())
    session = Session("budget1", system_prompt="sys")
    result = _run(agent.run_turn(session, "hi " * 20))

    assert result.stop_reason.value == "budget"
    assert "budget" in (result.error or "")
    assert sup.budget is not None and sup.budget.turn_spent > 30


def test_supervisor_pause_mode_continues() -> None:
    """on_stuck='pause' 时不中断，继续跑完（最终 stop）。"""
    sup = Supervisor(SupervisorConfig(
        stuck_repeat_threshold=3,
        compress_max_events=1000,
        per_turn_token_limit=None,
        on_stuck="pause",
    ))
    mock = MockLLMClient([
        LLMChunk(final_tool_calls=[
            {"id": f"c{i}", "name": "echo", "args": {"x": "same"}},
        ])
        for i in range(6)
    ] + [
        LLMChunk(delta_text="finally done", stop_reason="stop"),
    ])
    agent = ReactLoopAgent(AgentConfig(supervisor=sup, max_steps=8),
                           mock, _echo_registry())
    session = Session("pause1", system_prompt="sys")
    result = _run(agent.run_turn(session, "重复 echo"))

    assert result.stop_reason.value == "stop"
    assert "finally done" in result.final_text
    # pause 事件已写入 session 审计
    kinds = [e.payload.get("kind") for e in session.events]
    assert "stuck_paused" in kinds


# ---------------------------------------------------------------------------
# v10.3.1 (P0-6):loop 消费 cond.messages 的端到端回归
# (此前 loop 丢弃压缩结果,derive_messages 仍投影全量 → 摘要反增 token)
# ---------------------------------------------------------------------------


def test_loop_uses_condensed_messages(tmp_path) -> None:
    """压缩命中轮,LLM 收到的消息条数 < 全量投影(P0-6 核心断言)。

    用 max_events=12 的真实阈值(6 轮 × 2 事件 + system = 13 > 12),
    loop 内压缩自然命中,断言 stream 收到的 messages 是压缩集:
      1. 条数明显小于全量投影
      2. 摘要 system 消息在请求头部
    """
    store = SessionStore(str(tmp_path / "cond_loop.db"))
    session = Session("cond-e2e", system_prompt="SYS")
    for i in range(6):
        session.append(SessionEvent(
            "user", time.time() + i, {"content": f"问题 {i} " + "长" * 60}))
        session.append(SessionEvent(
            "assistant", time.time() + i + 0.1,
            {"content": f"回答 {i} " + "答" * 60}))
    store.save(session)
    full = session.derive_messages()
    assert len(full) >= 13, f"预置 6 轮应 ≥13 条消息,实际 {len(full)}"

    mock = MockLLMClient([LLMChunk(delta_text="ok", stop_reason="stop")])
    # supervisor:压缩阈值 12(6 轮历史 13 条 > 12 → 命中)
    sup = Supervisor(SupervisorConfig(
        enable_stuck=False,
        enable_compression=True,
        compress_max_events=12,
        compress_max_chars=100_000,
        keep_recent_turns=3,
        enable_budget=False,
    ))
    agent = ReactLoopAgent(AgentConfig(max_steps=2, supervisor=sup),
                           mock, ToolRegistry(), store=store)
    asyncio.run(agent.run_turn(session, "再来一问"))
    assert sup.compressor.compress_count >= 1, "压缩应已命中"

    assert mock.calls, "LLM 应被调用"
    sent = mock.calls[0]["messages"]
    assert len(sent) < len(full), \
        f"loop 应发送压缩级消息({len(sent)}) < 全量({len(full)})"
    # 摘要在头部(第一个 system 或前两条内)
    head_texts = " | ".join(m.get("content", "") for m in sent[:2])
    assert "【历史对话摘要】" in head_texts, \
        f"摘要 system 应在消息头部: {head_texts[:120]}"
