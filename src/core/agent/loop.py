"""ReactLoopAgent — ReAct 循环 + phase 状态机。

参考 DSH `packages/core/agent-loop/src/agent.ts:70` + codex `codex.rs:5797 run_turn`：
  - phase 状态机：idle → maintenance → running → idle
  - run_turn(session, input) → TurnResult：ReAct 循环
      assemble prompt → llm stream → tool call → repeat → stop
  - 每步从 SessionEvent log 派生消息（不维护独立消息列表）

**LLM 调用**：默认走 `LLMClient` Protocol（async stream deltas）。
真实生产用 `src/core/provider_router_client.py:ProviderRouterClient`（包装
`src/core/provider_router.py` NVIDIA 多 key），回退用 `src/core/logic.py:build_llm_client`。
两者都只读 import，不改。

工具调用解析：LLM 输出的 tool_calls 走 ToolRegistry.execute（四层 waterfall）。
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol

from src.core.agent.prompt_guard import guard_messages
from src.core.agent.session import Session, SessionEvent, SessionStore
from src.core.agent.turn import Turn, TurnHooks, TurnPhase, TurnResult, TurnStopReason
from src.core.tools.registry import ToolCall, ToolRegistry, ToolResult


class AgentPhase(str, Enum):
    """agent 顶层状态机。"""

    IDLE = "idle"
    MAINTENANCE = "maintenance"
    RUNNING = "running"


@dataclass
class AgentConfig:
    """ReactLoopAgent 配置。"""

    system_prompt: str = "You are a helpful assistant."
    max_steps: int = 8  # ReAct 循环步数上限（防失控）
    step_timeout_sec: Optional[float] = None
    auto_tool_filter: Optional[List[str]] = None  # None = 全部工具可用


class LLMClient(Protocol):
    """LLM 客户端协议（async stream deltas + 最终 tool_calls）。

    生产用 ProviderRouterClient（包 ProviderRouter），测试用 MockLLMClient。
    """

    def stream(self, messages: List[Dict[str, Any]],
               tools: List[Dict[str, Any]]) -> AsyncIterator["LLMChunk"]:
        ...


@dataclass
class LLMChunk:
    """LLM 流式 chunk。delta_text 累积，final_tool_calls 在结束时给。"""

    delta_text: str = ""
    final_tool_calls: Optional[List[Dict[str, Any]]] = None
    stop_reason: Optional[str] = None


class MockLLMClient:
    """测试用 Mock：按预设脚本回放 chunks + tool_calls。

    用法：
        mock = MockLLMClient([
            LLMChunk(final_tool_calls=[{"id":"1","name":"echo","args":{"x":"1"}}]),
            LLMChunk(delta_text="final answer", stop_reason="stop"),
        ])
    """

    def __init__(self, script: List[LLMChunk]) -> None:
        self._script = list(script)
        self._idx = 0
        self.calls: List[Dict[str, Any]] = []  # 记录每次 stream 调用入参

    async def stream(self, messages: List[Dict[str, Any]],
                     tools: List[Dict[str, Any]]) -> AsyncIterator[LLMChunk]:
        self.calls.append({"messages": messages, "tools": tools})
        if self._idx >= len(self._script):
            yield LLMChunk(delta_text="(no more script)", stop_reason="stop")
            return
        chunk = self._script[self._idx]
        self._idx += 1
        yield chunk


class ReactLoopAgent:
    """ReAct 循环 agent。

    用法：
        agent = ReactLoopAgent(config, llm_client, tool_registry)
        result = await agent.run_turn(session, "用户输入")
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        *,
        hooks: Optional[TurnHooks] = None,
        store: Optional[SessionStore] = None,
    ) -> None:
        self.config = config
        self.llm = llm_client
        self.tools = tool_registry
        self.hooks = hooks or TurnHooks()
        self.store = store  # None = 不持久化（纯内存，测试/无状态场景）
        self._phase = AgentPhase.IDLE

    @property
    def phase(self) -> AgentPhase:
        return self._phase

    @staticmethod
    def load_session(store: SessionStore, session_id: str,
                     system_prompt: Optional[str] = None) -> Session:
        """从 SessionStore 恢复 Session；不存在则建空 Session。

        崩溃恢复入口：进程重启后，调用方传 session_id + store，
        返回的 Session 已含历史 events（derive_messages 即可重建上下文）。
        若库中无此 session_id，返回全新空 Session（带 system_prompt）。
        """
        loaded = store.load(session_id)
        if loaded is not None:
            return loaded
        return Session(session_id, system_prompt=system_prompt)

    def _persist(self, session: Session) -> None:
        """把当前 Session 全量写库（WAL，崩溃可恢复）。store 为 None 时 no-op。"""
        if self.store is not None:
            self.store.save(session)

    async def run_turn(self, session: Session, input_text: str) -> TurnResult:
        """跑一个 turn：append user → 循环(llm → tool*) → stop。"""
        self._phase = AgentPhase.MAINTENANCE
        turn = Turn(
            session_id=session.session_id,
            input_text=input_text,
            hooks=self.hooks,
            ctx={"agent": self, "session": session},
        )

        # turn/start + claim input
        await turn.emit(TurnPhase.TURN_START)
        await turn.emit(TurnPhase.CLAIM_INPUT, {"input": input_text})

        # append user event
        session.append(SessionEvent(
            "user", time.time(), {"content": input_text}))
        self._persist(session)

        self._phase = AgentPhase.RUNNING
        result = TurnResult()

        try:
            while turn.step < self.config.max_steps and not turn.cancelled:
                turn.step += 1
                result.steps = turn.step

                # assemble prompt + tool schemas
                await turn.emit(TurnPhase.ASSEMBLE)
                await turn.emit(TurnPhase.PRE_STEP)
                await turn.emit(TurnPhase.STEP_START)

                messages = session.derive_messages()
                # 提示注入守卫:工具输出(role=tool)可能含"忽略之前指令"等恶意内容,
                # 交给 LLM 前先用 guard_messages 包 <tool_output> 标签 + 命中转义,
                # 防止工具返回污染系统指令(见 prompt_guard.py)。
                messages = guard_messages(messages)
                # 可选：默认注入 system（若 session 无 system event）
                tools = self.tools.schemas()
                if self.config.auto_tool_filter is not None:
                    # v10.2:to_llm_schema 改为 OpenAI 兼容嵌套结构,工具名在
                    # function.name(registry.schemas 投影后同构)。
                    tools = [t for t in tools
                             if t["function"]["name"] in self.config.auto_tool_filter]

                # request (LLM stream)
                await turn.emit(TurnPhase.REQUEST, {
                    "messages": messages, "tools": tools})

                assistant_text = ""
                final_tool_calls: List[Dict[str, Any]] = []
                try:
                    async for chunk in self.llm.stream(messages, tools):
                        if chunk.delta_text:
                            assistant_text += chunk.delta_text
                            await turn.emit(TurnPhase.ASSISTANT_STREAM, {
                                "delta": chunk.delta_text})
                        if chunk.final_tool_calls:
                            final_tool_calls = chunk.final_tool_calls
                        if chunk.stop_reason:
                            # stop_reason 在无 tool_call 时决定终止（见下方）
                            pass
                except asyncio.TimeoutError:
                    result.stop_reason = TurnStopReason.ERROR
                    result.error = "LLM stream timeout"
                    break

                # tool calls
                if final_tool_calls:
                    # assistant event 带 tool_calls（供下轮 derive）
                    session.append(SessionEvent(
                        "assistant", time.time(), {
                            "content": assistant_text or "",
                            "tool_calls": final_tool_calls,
                        }))
                    for tc in final_tool_calls:
                        await turn.emit(TurnPhase.TOOL_CALL, {"tool_call": tc})
                        call = ToolCall(
                            name=tc.get("name", ""),
                            args=tc.get("args", {}) or {},
                            call_id=tc.get("id", "") or uuid.uuid4().hex,
                        )
                        await turn.emit(TurnPhase.TOOL_PRE_EXECUTE,
                                         {"call": call})
                        await turn.emit(TurnPhase.TOOL_EXECUTE, {"call": call})
                        tr: ToolResult = await self.tools.execute(call)
                        await turn.emit(TurnPhase.TOOL_POST_EXECUTE,
                                         {"call": call, "result": tr})
                        await turn.emit(TurnPhase.TOOL_RESULT,
                                         {"call": call, "result": tr})
                        result.tool_calls += 1
                        # tool_result event
                        session.append(SessionEvent(
                            "tool_result", time.time(), {
                                "tool_call_id": call.call_id,
                                "content": _stringify_tool_output(tr),
                            }))
                    self._persist(session)  # 持久化整轮 ReAct 中间态（崩溃可续）
                    await turn.emit(TurnPhase.STEP_END)
                    continue  # 进下一轮 ReAct

                # 无 tool_call → 终答
                if assistant_text:
                    session.append(SessionEvent(
                        "assistant", time.time(), {"content": assistant_text}))
                self._persist(session)  # 持久化终答
                result.final_text = assistant_text
                result.stop_reason = TurnStopReason.STOP
                await turn.emit(TurnPhase.STEP_END)
                break
            else:
                # while 条件耗尽（max_steps）
                result.stop_reason = TurnStopReason.MAX_STEPS
                result.final_text = "(max_steps reached)"

            await turn.emit(TurnPhase.TURN_STOPPING)
            await turn.emit(TurnPhase.TURN_END)
        except asyncio.CancelledError:
            result.stop_reason = TurnStopReason.CANCELLED
            result.error = "cancelled"
            raise
        except Exception as e:  # noqa: BLE001
            result.stop_reason = TurnStopReason.ERROR
            result.error = str(e)
        finally:
            self._phase = AgentPhase.IDLE

        return result


def _stringify_tool_output(tr: ToolResult) -> str:
    """把 ToolResult.output 归一成字符串（LLM tool message content 要 str）。"""
    if tr.is_error():
        return f"Error: {tr.error}"
    out = tr.output
    if isinstance(out, (dict, list)):
        return json.dumps(out, ensure_ascii=False)
    return str(out) if out is not None else ""
