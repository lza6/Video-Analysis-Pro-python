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

**分层记忆（v10.2 P1-1）**：`AgentConfig.memory` 挂 MemoryLayeredConnector。
开启时：
  - run_turn 开始：WorkingMemory 热层注入为附加 system 段（derive_messages 之后）
  - run_turn 结束（TURN_END 前）：Experience 记录 + 热层同步 + 三元组抽取
feature flag `VAP_MEMORY_LAYERED`（默认 1=开）由主控装配时读取
（MemoryLayeredConnector.from_env）。关时 connector 为 None，行为零回归。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol

from src.core.agent.prompt_guard import guard_messages
from src.core.agent.session import Session, SessionEvent, SessionStore
from src.core.agent.supervisor import RoundSignature, Supervisor
from src.core.agent.turn import Turn, TurnHooks, TurnPhase, TurnResult, TurnStopReason
from src.core.memory.connector import MemoryLayeredConnector
from src.core.tools.registry import ToolCall, ToolRegistry, ToolResult

log = logging.getLogger("core.agent.loop")


class AgentPhase(str, Enum):
    """agent 顶层状态机。"""

    IDLE = "idle"
    MAINTENANCE = "maintenance"
    RUNNING = "running"


@dataclass
class AgentConfig:
    """ReactLoopAgent 配置。

    Attributes:
        system_prompt: 系统提示词（Session 无 system 事件时可选注入）。
        max_steps: ReAct 循环步数上限（防失控）。
        step_timeout_sec: 单步超时（None=不超时）。
        auto_tool_filter: 工具白名单（None=全部工具可用）。
        supervisor: 监督层（卡死/压缩/预算）。v10.3.1 起默认启用
            （`VAP_AGENT_SUPERVISOR=false` 显式关闭回退）。由主控装配时注入，
            见 `Supervisor.from_env()`。
    """

    system_prompt: str = "You are a helpful assistant."
    max_steps: int = 8  # ReAct 循环步数上限（防失控）
    step_timeout_sec: Optional[float] = None
    auto_tool_filter: Optional[list[str]] = None  # None = 全部工具可用
    supervisor: Optional[Supervisor] = None  # 监督层（默认 None=禁用，零回归）
    memory: Optional[MemoryLayeredConnector] = None  # 分层记忆（默认 None=禁用，零回归）


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
    def _summarize(messages: list[dict[str, Any]], budget: int = 400) -> str:
        """把一轮 LLM 请求的消息列表压缩成审计字符串（预算截断）。"""
        text = json.dumps(messages, ensure_ascii=False)
        lines = text.split("\\n")
        out: list[str] = []
        total = 0
        for ln in lines:
            total += len(ln)
            if total > budget:
                out.append("[...truncated]")
                break
            out.append(ln)
        return "\\n".join(out)

    @staticmethod
    def load_session(store: SessionStore, session_id: str,
                     system_prompt: str | None = None) -> Session:
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
        sup = self.config.supervisor
        # v10.3.1 (P0-6):本轮压缩后的消息投影(cond.messages)。压缩命中时,
        # 用它替代全量 derive_messages() —— 否则摘要事件 append 进 session
        # 后,derive_messages 仍投影全量事件,token 不降反增。
        condensed_messages: list[dict[str, Any]] | None = None

        try:
            # 预算守卫：新 turn 开始前重置 per-turn 计数
            if sup is not None and sup.budget is not None:
                sup.budget.begin_turn()

            while turn.step < self.config.max_steps and not turn.cancelled:
                turn.step += 1
                result.steps = turn.step
                # 每轮重置卡死检测样本（供本轮 tool 结果收集后构造签名）
                round_actions = []
                round_observations = []
                round_has_error = False

                # 每轮开始前：上下文压缩检查（在 ASSEMBLE 之前）
                if sup is not None and sup.compressor is not None:
                    cond = sup.compressor.compress(session)
                    if cond is not None:
                        # v10.3.1 (P0-6):消费压缩结果 —— 本轮 LLM 请求改用
                        # cond.messages(摘要 system + 最近 N 轮),替代全量
                        # derive_messages()。摘要事件同时 append 进 session
                        # (append-only 审计),tool_call_id 配对在窗口内不破裂。
                        condensed_messages = list(cond.messages)
                        log.info(
                            "supervisor: context compressed, "
                            "dropped %d events -> summary(%s)",
                            cond.dropped_events,
                            cond.summary[:80])

                # assemble prompt + tool schemas
                await turn.emit(TurnPhase.ASSEMBLE)
                await turn.emit(TurnPhase.PRE_STEP)
                await turn.emit(TurnPhase.STEP_START)

                # v10.3.1 (P0-6):压缩命中轮用压缩消息;否则全量投影。
                # 注意:事件 append-only,超阈后每轮都会命中压缩、持续走
                # 压缩投影(摘要事件幂等不叠加) —— 直至会话事件被外部清理。
                if condensed_messages is not None:
                    messages = condensed_messages
                    condensed_messages = None
                else:
                    messages = session.derive_messages()
                # 提示注入守卫:工具输出(role=tool)可能含"忽略之前指令"等恶意内容,
                # 交给 LLM 前先用 guard_messages 包 <tool_output> 标签 + 命中转义,
                # 防止工具返回污染系统指令(见 prompt_guard.py)。
                messages = guard_messages(messages)
                # 分层记忆:热层注入为附加 system 段(derive_messages 之后,
                # 作为附加 system,不污染 system_prompt)。
                if self.config.memory is not None:
                    messages = self.config.memory.inject(messages)
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
                final_tool_calls: list[dict[str, Any]] = []
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

                # 预算核算：本轮 LLM 请求 token 累计（输入+输出，估算）
                if sup is not None and sup.budget is not None:
                    trigger = sup.budget.account_request(
                        messages, assistant_text)
                    if trigger is not None:
                        result.stop_reason = TurnStopReason.BUDGET
                        result.error = trigger.describe
                        await turn.emit(TurnPhase.SUPERVISE, {
                            "kind": "budget", "trigger": trigger,
                            "messages": self._summarize(messages),
                            "assistant_text": assistant_text})
                        log.warning("supervisor: %s", trigger.describe)
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
                        # v10.3.1 (P0-1 收尾):工具事件附一句大白话解释
                        # (eli5)。只进事件流供 UI 摘要行/黑匣子人话层渲染,
                        # **不进 LLM prompt**(避免 token 膨胀)。
                        try:
                            from src.core.eli5 import explain_tool_call
                            human = explain_tool_call(
                                call.name, call.args, tr)
                        except Exception:  # noqa: BLE001 — 解释失败不阻断
                            human = ""
                        await turn.emit(TurnPhase.TOOL_POST_EXECUTE,
                                         {"call": call, "result": tr,
                                          "human": human})
                        await turn.emit(TurnPhase.TOOL_RESULT,
                                         {"call": call, "result": tr,
                                          "human": human})
                        result.tool_calls += 1
                        # tool_result event
                        session.append(SessionEvent(
                            "tool_result", time.time(), {
                                "tool_call_id": call.call_id,
                                "content": _stringify_tool_output(tr),
                            }))
                        # 监督层卡死检测样本收集
                        round_actions.append(
                            (call.name, _stable_args(call.args)))
                        round_observations.append(
                            _stringify_tool_output(tr))
                        if tr.is_error():
                            round_has_error = True
                    self._persist(session)  # 持久化整轮 ReAct 中间态（崩溃可续）
                    await turn.emit(TurnPhase.STEP_END)

                    # 监督层卡死检测（放在每轮 tool 结果之后）
                    if sup is not None and sup.stuck is not None:
                        sig = RoundSignature(
                            actions=tuple(round_actions),
                            observations=tuple(round_observations),
                            has_error=round_has_error,
                        )
                        trigger = sup.stuck.observe(sig)
                        if trigger is not None:
                            await turn.emit(TurnPhase.SUPERVISE, {
                                "kind": "stuck", "trigger": trigger})
                            if sup.config.on_stuck == "pause":
                                # 暂停提示：写入事件审计后继续（不清窗口，
                                # 下轮仍带元信息；不会无限增长，检测到头会再次命中）
                                session.append(SessionEvent(
                                    "turn", time.time(), {
                                        "kind": "stuck_paused",
                                        "detail": trigger.describe,
                                    }))
                                log.warning(
                                    "supervisor(pause): %s", trigger.describe)
                            else:
                                result.stop_reason = TurnStopReason.STUCK
                                result.error = trigger.describe
                                log.warning(
                                    "supervisor(stop): %s", trigger.describe)
                                break
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
            # 分层记忆:turn 结束时提取 Experience 并落库(幂等,同 session 同
            # intent 去重)。放在 TURN_END 之前,不改变 phase 顺序语义。
            if self.config.memory is not None:
                self.config.memory.record(session)
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


def _stable_args(args: dict[str, Any]) -> str:
    """把 tool args 归一成稳定签名（dict key 顺序无关，用于卡死比较）。"""
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(args, sort_keys=True, ensure_ascii=False,
                          default=str)
