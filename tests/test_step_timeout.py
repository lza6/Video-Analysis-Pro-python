"""单步软超时测试（v10.5.0 P1-10）。

覆盖 `AgentConfig.step_timeout_sec` 的接线语义：
  - LLM 流整体超时 → TurnStopReason.ERROR（此前该字段是死配置，从未被消费）
  - 工具执行超时 → tool_result 落 Error 且轮次继续（交给 stuck 检测/用户审计）
  - None（缺省）→ 零回归（直通，不等待上限）
  - `VAP_AGENT_STEP_TIMEOUT` 环境装配解析（agent.py _step_timeout_from_env）

与项目既有风格一致：asyncio.run 同步驱动，不依赖 pytest-asyncio。
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from src.core.agent import AgentConfig, ReactLoopAgent, Session, SessionEvent
from src.core.agent.loop import LLMChunk
from src.core.tools import ToolDefinition, ToolRegistry


def _run(coro) -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


class SlowLLMClient:
    """预设脚本回放的慢速 LLM：每 chunk 前 sleep（模拟卡死/慢响应）。"""

    def __init__(self, script: List[LLMChunk], sleep: float = 0.5) -> None:
        self._script = list(script)
        self._sleep = sleep
        self._idx = 0

    async def stream(self, messages, tools) -> AsyncIterator[LLMChunk]:
        await asyncio.sleep(self._sleep)
        if self._idx >= len(self._script):
            yield LLMChunk(delta_text="(done)", stop_reason="stop")
            return
        chunk = self._script[self._idx]
        self._idx += 1
        yield chunk


def _slow_tool_registry(sleep: float = 0.5) -> ToolRegistry:
    reg = ToolRegistry()

    async def _slow(**kwargs: Any) -> str:
        await asyncio.sleep(sleep)
        return "slow-ok"

    reg.register(ToolDefinition(
        name="slow_tool",
        description="慢工具（测试用）",
        execute_callback=_slow,
    ))
    return reg


def _tool_result_content(session: Session, marker: str) -> List[str]:
    """从 session 事件里收集含 marker 的 tool_result content。"""
    out: List[str] = []
    for e in session.events:
        if e.type == "tool_result" and marker in str(e.payload.get("content", "")):
            out.append(str(e.payload.get("content", "")))
    return out


# ---------------------------------------------------------------------------
# 1) LLM 流整体超时
# ---------------------------------------------------------------------------


def test_llm_stream_timeout_stops_turn() -> None:
    """step_timeout_sec=0.1 且 LLM sleep 0.5 → 整轮 ERROR(timeout)，不悬挂。"""
    client = SlowLLMClient([LLMChunk(delta_text="never", stop_reason="stop")], sleep=0.5)
    agent = ReactLoopAgent(
        AgentConfig(step_timeout_sec=0.1, max_steps=2), client, ToolRegistry())

    import time
    t0 = time.time()
    result = _run(agent.run_turn(Session("s-tmo-1"), "hi"))
    elapsed = time.time() - t0

    assert result.stop_reason.value == "error"
    assert result.error and "timeout" in result.error.lower()
    assert elapsed < 3.0, f"超时未生效,整轮耗时 {elapsed:.2f}s"


def test_no_step_timeout_keeps_old_behavior() -> None:
    """step_timeout_sec=None（缺省）→ LLM 慢也不中断（零回归）。"""
    client = SlowLLMClient([LLMChunk(delta_text="ok", stop_reason="stop")], sleep=0.2)
    agent = ReactLoopAgent(
        AgentConfig(step_timeout_sec=None, max_steps=2), client, ToolRegistry())

    result = _run(agent.run_turn(Session("s-tmo-2"), "hi"))
    assert result.stop_reason.value == "stop"
    assert result.final_text == "ok"


# ---------------------------------------------------------------------------
# 2) 工具执行超时（Error 落库 + 轮次继续）
# ---------------------------------------------------------------------------


def test_tool_timeout_records_error_and_continues() -> None:
    """工具 sleep 0.5 而 step_timeout=0.1 → tool_result 带 tool timeout，轮次继续到 final。"""
    client = SlowLLMClient([
        LLMChunk(final_tool_calls=[{
            "id": "tc1", "name": "slow_tool", "args": {},
        }]),
        LLMChunk(delta_text="final answer", stop_reason="stop"),
    ], sleep=0.0)
    agent = ReactLoopAgent(
        AgentConfig(step_timeout_sec=0.1, max_steps=3),
        client, _slow_tool_registry(sleep=0.5))

    session = Session("s-tmo-3")
    result = _run(agent.run_turn(session, "do it"))

    assert result.stop_reason.value == "stop"
    assert result.final_text == "final answer"
    assert result.tool_calls == 1
    errs = _tool_result_content(session, "tool timeout")
    assert errs, "tool_result 未落 tool timeout Error"
    assert "0.1s" in errs[0]  # 超时时长被如实暴露给审计


def test_tool_within_budget_succeeds() -> None:
    """工具 sleep 0.05 < step_timeout=0.3 → 正常完成，无 error。"""
    client = SlowLLMClient([
        LLMChunk(final_tool_calls=[{
            "id": "tc2", "name": "slow_tool", "args": {},
        }]),
        LLMChunk(delta_text="done", stop_reason="stop"),
    ], sleep=0.0)
    agent = ReactLoopAgent(
        AgentConfig(step_timeout_sec=0.3, max_steps=3),
        client, _slow_tool_registry(sleep=0.05))

    session = Session("s-tmo-4")
    result = _run(agent.run_turn(session, "go"))

    assert result.stop_reason.value == "stop"
    assert result.tool_calls == 1
    assert not _tool_result_content(session, "tool timeout")


# ---------------------------------------------------------------------------
# 3) 环境装配解析
# ---------------------------------------------------------------------------


def test_step_timeout_env_assembly() -> None:
    """VAP_AGENT_STEP_TIMEOUT 装配：合法/空/非法/<=0 四态。"""
    from src.web.routers.agent import _step_timeout_from_env
    import os

    try:
        os.environ["VAP_AGENT_STEP_TIMEOUT"] = "1.5"
        assert _step_timeout_from_env() == 1.5

        os.environ["VAP_AGENT_STEP_TIMEOUT"] = "0"
        assert _step_timeout_from_env() is None

        os.environ["VAP_AGENT_STEP_TIMEOUT"] = "abc"
        assert _step_timeout_from_env() is None

        os.environ["VAP_AGENT_STEP_TIMEOUT"] = "  "
        assert _step_timeout_from_env() is None
    finally:
        os.environ.pop("VAP_AGENT_STEP_TIMEOUT", None)

    assert _step_timeout_from_env() is None