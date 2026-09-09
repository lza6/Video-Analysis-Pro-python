"""Agent 监督层 — 卡死检测 + 上下文压缩 + 成本预算。

批 B-P0-1（指南 v2）为 ReactLoopAgent 补齐监督能力：

  1. `StuckDetector` — 参考 OpenHands stuck 场景裁剪：
       - 同 action 同 observation 连续重复 >= N 次且无进展
       - 相同 tool_args 循环无进展 >= N 次
       - 连续 error 循环 >= N 次
     触发后 loop 向 turn 写入 stuck 信息 + `stop_reason=STUCK` 中断循环
     （或按 `on_stuck="pause"` 记录提示后继续）。

  2. `ContextCompressor` — 参考 context-mode 范式：Session 事件数/字符数
     超阈值时，用窗口外历史 user 事件生成压缩摘要作为新的 system 段注入，
     保留原始 system 与最近 N 轮完整。压缩后在 session 上 append 一条带
     `SUMMARY_EVENT_KIND` 标记的 system 摘要事件（append-only 审计）。

  3. `BudgetGuard` — per-turn token 上限 + 跨 turn 累计预算，超限返回
     `BudgetTrigger`，loop 置 `stop_reason=BUDGET` 中断。

纯标准库 + 纯 asyncio（无同步阻塞 IO、无第三方依赖）。feature flag
`VAP_AGENT_SUPERVISOR`（默认关）由主控装配时读取（见 `Supervisor.from_env`），
`AgentConfig.supervisor=None` 时 ReactLoopAgent 行为零回归。
"""
from __future__ import annotations

import json as _json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.core.agent.session import Session, SessionEvent

log = logging.getLogger("core.agent.supervisor")

# 压缩摘要事件的 kind 标记：ContextCompressor 写入 session 的 system 事件。
SUMMARY_EVENT_KIND = "agent_supervisor_summary"


@dataclass(frozen=True)
class RoundSignature:
    """单轮 ReAct 的签名（卡死检测输入）。

    Attributes:
        actions: 本轮所有工具调用，每个元素 (工具名, 稳定 args JSON)。
        observations: 本轮所有 tool_result 的字符串化输出（与 actions 同序）。
        has_error: 本轮是否至少含一条 tool error。
    """

    actions: tuple[tuple[str, str], ...]
    observations: tuple[str, ...]
    has_error: bool = False


@dataclass(frozen=True)
class StuckTrigger:
    """卡死检测触发信号。"""

    rule: str  # same_action_observation | same_tool_args | error_loop
    count: int  # 触发时的重复轮数（>= repeat_threshold）
    detail: str

    @property
    def describe(self) -> str:
        return f"stuck[{self.rule}] x{self.count}: {self.detail}"


@dataclass(frozen=True)
class BudgetTrigger:
    """成本预算超限信号。"""

    reason: str  # per_turn | total
    spent: int
    limit: int

    @property
    def describe(self) -> str:
        return f"budget[{self.reason}] spent={self.spent} limit={self.limit}"


class StuckDetector:
    """卡死检测器：滚动窗口比较最近 N 轮签名。

    规则（窗口长度 = repeat_threshold，按优先级返回首个命中）：
      - `error_loop`：最近 N 轮每一轮都含 tool error。
      - `same_action_observation`：最近 N 轮 (actions, observations) 完全
        相同且 actions 非空 —— 更明确的"原地打转"。
      - `same_tool_args`：最近 N 轮 actions 完全相同（observation 可能随
        调用变化，但反复同一调参调用 = 无进展循环）。

    说明：持久化 session 里 `step/turn` 事件不会被丢弃，"无进展"由
    重复调用本身判定；sensor 只消费 tool 结果后的轮签名。
    """

    RULE_ERROR_LOOP = "error_loop"
    RULE_ACTION_OBSERVATION = "same_action_observation"
    RULE_TOOL_ARGS = "same_tool_args"

    def __init__(self, repeat_threshold: int = 3) -> None:
        if repeat_threshold < 2:
            raise ValueError("repeat_threshold 必须 >= 2")
        self.repeat_threshold = repeat_threshold
        self._window: list[RoundSignature] = []

    def observe(self, sig: RoundSignature) -> StuckTrigger | None:
        """记录一轮签名，返回命中的触发信号或 None（未触发）。"""
        self._window.append(sig)
        # 窗口只保留足够检测的量，防长期会话无限增长
        if len(self._window) > self.repeat_threshold * 4:
            self._window = self._window[-(self.repeat_threshold * 2):]
        return self._detect()

    def reset(self) -> None:
        """清空窗口（`on_stuck="pause"` 提示后继续时使用）。"""
        self._window.clear()

    def _detect(self) -> StuckTrigger | None:
        w = self._window
        n = self.repeat_threshold
        if len(w) < n:
            return None
        tail = w[-n:]
        first = tail[0]

        # 1) 连续 error 循环（error 轮同时也会满足 actions/observations 相同）
        if all(r.has_error for r in tail):
            return StuckTrigger(
                rule=self.RULE_ERROR_LOOP, count=n,
                detail=f"最近 {n} 轮工具全部报错")

        # 2) 同 action 同 observation（更明确的无进展）
        if first.actions and all(
            r.actions == first.actions
            and r.observations == first.observations
            for r in tail
        ):
            return StuckTrigger(
                rule=self.RULE_ACTION_OBSERVATION, count=n,
                detail=f"最近 {n} 轮 action+observation 完全相同"
                       f"（工具 {first.actions[0][0]}）")

        # 3) 相同 tool_args 循环（observation 可能有变化，但反复同参调用）
        if first.actions and all(r.actions == first.actions for r in tail):
            name, args = first.actions[-1]
            return StuckTrigger(
                rule=self.RULE_TOOL_ARGS, count=n,
                detail=f"最近 {n} 轮重复调用 {name}（相同 args={args[:120]}）")

        return None


@dataclass(frozen=True)
class CondensationResult:
    """一次上下文压缩的结果。"""

    summary: str  # 压缩摘要文本（作为新 system 段注入 LLM 消息）
    messages: tuple[dict[str, Any], ...]  # 压缩后的 LLM 消息列表
    dropped_events: int  # 被摘要覆盖的窗口外事件数


class ContextCompressor:
    """上下文压缩：Session 超阈值时注入历史摘要 system 段。

    范式（参考 context-mode）：
      - 把窗口外（最早部分）历史提炼成摘要，作为新的 system 消息注入
        喂给 LLM 的消息列表（真实减少 token 消耗）。
      - 保留原始 system 与最近 `keep_recent_turns` 轮完整（消息投影里
        原文保留，tool_call_id 配对不破裂）。
      - 在 session 上 append 一条带 `SUMMARY_EVENT_KIND` 标记的 system
        摘要事件（幂等：已有标记则不重复 append）。
    """

    def __init__(
        self,
        *,
        max_events: int = 50,
        max_chars: int = 40_000,
        keep_recent_turns: int = 3,
        summary_line_limit: int = 200,
    ) -> None:
        self.max_events = max_events
        self.max_chars = max_chars
        self.keep_recent_turns = max(1, keep_recent_turns)
        self.summary_line_limit = summary_line_limit
        self._compress_count = 0

    @property
    def compress_count(self) -> int:
        return self._compress_count

    def should_compress(self, session: Session) -> bool:
        """事件数或字符数是否超阈值。"""
        events = session.events
        if len(events) <= 2:  # 只有 system + user，无从压缩
            return False
        if len(events) > self.max_events:
            return True
        chars = sum(len(str(e.payload.get("content", ""))) for e in events)
        return chars > self.max_chars

    def compress(self, session: Session) -> CondensationResult | None:
        """超阈值时执行压缩；未触发或无可压缩轮次返回 None。"""
        if not self.should_compress(session):
            return None
        events = session.events
        users = [i for i, e in enumerate(events) if e.type == "user"]
        # 保留最近 keep_recent_turns 轮 user 起点的完整上下文
        if len(users) <= self.keep_recent_turns:
            return None
        start_idx = users[-self.keep_recent_turns]
        window_outer = events[:start_idx]

        summary = self._build_summary(window_outer)
        messages = self._condensed_messages(events, summary)

        # append-only 审计：摘要 system 事件，幂等（已有标记则不重复）
        has_marker = any(
            e.type == "system" and e.payload.get("kind") == SUMMARY_EVENT_KIND
            for e in events
        )
        if not has_marker:
            session.append(SessionEvent("system", time.time(), {
                "content": summary,
                "kind": SUMMARY_EVENT_KIND,
            }))
        self._compress_count += 1
        return CondensationResult(
            summary=summary,
            messages=tuple(messages),
            dropped_events=len(window_outer),
        )

    def _build_summary(self, window_outer: list[SessionEvent]) -> str:
        """把窗口外历史 user/assistant 提炼成摘要文本。

        按 user 问题归组；工具调用过程的中间轮次（assistant 带 tool_calls、
        tool_result）细节不进入摘要，只保留"问题 → 最终答复"对。
        """
        entries: list[str] = []
        current_q = ""
        for e in window_outer:
            p = e.payload
            if e.type == "user":
                if current_q:
                    entries.append(current_q)
                current_q = str(p.get("content", "")).strip()
                current_q = current_q[: self.summary_line_limit]
            elif e.type == "assistant":
                if current_q and not p.get("tool_calls"):
                    ans = str(p.get("content", "")).strip()
                    if ans:
                        current_q += " → " + ans[:80]
        if current_q:
            entries.append(current_q)
        if not entries:
            entries = ["(早期对话无 user 记录)"]
        head = "【历史对话摘要】由 Agent 监督层生成，原始事件完整保留在会话日志中："
        body = "\n".join(f"- {i + 1}. {x}" for i, x in enumerate(entries))
        return f"{head}\n{body}"

    def _condensed_messages(
        self, events: list[SessionEvent], summary: str
    ) -> list[dict[str, Any]]:
        """构建压缩后的 LLM 消息列表：摘要 system + 原始 system + 最近 N 轮。

        旧摘要标记事件被新摘要取代（避免叠加）；窗口内 system 事件归入头部。
        """
        system_msgs: list[dict[str, Any]] = [{"role": "system", "content": summary}]
        for e in events:
            if e.type != "system":
                continue
            if e.payload.get("kind") == SUMMARY_EVENT_KIND:
                continue  # 旧摘要被新摘要取代
            system_msgs.append({"role": "system", "content": e.payload.get("content", "")})

        users = [i for i, e in enumerate(events) if e.type == "user"]
        tail = events[users[-self.keep_recent_turns]:]
        tail_msgs: list[dict[str, Any]] = []
        for e in tail:
            if e.type == "system":
                continue  # 已归入头部
            msg = self._project_event(e)
            if msg is not None:
                tail_msgs.append(msg)
        return system_msgs + tail_msgs

    @staticmethod
    def _project_event(e: SessionEvent) -> dict[str, Any] | None:
        """单条事件投影成 LLM 消息（与 session.derive_messages 规则一致）。"""
        p = e.payload
        if e.type == "user":
            return {"role": "user", "content": p.get("content", "")}
        if e.type == "assistant":
            m: dict[str, Any] = {"role": "assistant"}
            if "content" in p:
                m["content"] = p["content"]
            if p.get("tool_calls"):
                m["tool_calls"] = p["tool_calls"]
            return m
        if e.type == "tool_result":
            return {
                "role": "tool",
                "tool_call_id": p.get("tool_call_id", ""),
                "content": p.get("content", ""),
            }
        # step/turn/system 元事件已在调用处处理，这里兜底返回 None
        return None


class BudgetGuard:
    """成本预算守护：per-turn token 上限 + 跨 turn 累计预算。

    token 数估算（无第三方依赖）：按 ~4 字符/token 保守估算输入+输出文本，
    偏大不偏小（宁可提前触卡预算）。真实计费可由主控传 LLM usage 覆盖。
    """

    def __init__(
        self,
        *,
        per_turn_token_limit: int | None = None,
        total_token_budget: int | None = None,
    ) -> None:
        if per_turn_token_limit is not None and per_turn_token_limit <= 0:
            raise ValueError("per_turn_token_limit 必须 > 0")
        if total_token_budget is not None and total_token_budget <= 0:
            raise ValueError("total_token_budget 必须 > 0")
        self.per_turn_token_limit = per_turn_token_limit
        self.total_token_budget = total_token_budget
        self._turn_spent = 0
        self._total_spent = 0

    @property
    def turn_spent(self) -> int:
        return self._turn_spent

    @property
    def total_spent(self) -> int:
        return self._total_spent

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """按 ~4 字符/token 估算。空串返回 0。"""
        if not text:
            return 0
        return max(1, (len(text) + 3) // 4)

    def begin_turn(self) -> None:
        """标记一个新 turn 开始（重置 per-turn 计数）。"""
        self._turn_spent = 0

    def account(self, delta_tokens: int) -> BudgetTrigger | None:
        """累计 delta tokens，返回 None（OK）或超限触发信号。"""
        if delta_tokens < 0:
            raise ValueError("delta_tokens 不能为负")
        self._turn_spent += delta_tokens
        self._total_spent += delta_tokens
        if self.per_turn_token_limit is not None \
                and self._turn_spent > self.per_turn_token_limit:
            return BudgetTrigger(
                reason="per_turn", spent=self._turn_spent,
                limit=self.per_turn_token_limit)
        if self.total_token_budget is not None \
                and self._total_spent > self.total_token_budget:
            return BudgetTrigger(
                reason="total", spent=self._total_spent,
                limit=self.total_token_budget)
        return None

    def account_request(
        self,
        messages: Sequence[dict[str, Any]],
        assistant_text: str = "",
    ) -> BudgetTrigger | None:
        """按一轮 LLM 请求估算 token 并累计（输入 messages + 输出文本）。"""
        input_text = _json.dumps(list(messages), ensure_ascii=False)
        delta = self.estimate_tokens(input_text) + self.estimate_tokens(assistant_text or "")
        return self.account(delta)


@dataclass
class SupervisorConfig:
    """监督层配置。所有开关独立可关，便于定向测试。"""

    enable_stuck: bool = True
    enable_compression: bool = True
    enable_budget: bool = True
    stuck_repeat_threshold: int = 3
    compress_max_events: int = 50
    compress_max_chars: int = 40_000
    keep_recent_turns: int = 3
    per_turn_token_limit: int | None = 50_000
    total_token_budget: int | None = 500_000
    on_stuck: str = "stop"  # "stop"=STUCK 中断 | "pause"=记录提示后继续


class Supervisor:
    """监督层组合根：装配三件套，独立开关便于定向测试。

    用法：
        supervisor = Supervisor(SupervisorConfig(...))
        agent = ReactLoopAgent(AgentConfig(supervisor=supervisor), llm, registry)
    """

    def __init__(self, config: SupervisorConfig | None = None) -> None:
        cfg = config or SupervisorConfig()
        self.config = cfg
        self.stuck: StuckDetector | None = None
        self.compressor: ContextCompressor | None = None
        self.budget: BudgetGuard | None = None
        if cfg.enable_stuck:
            self.stuck = StuckDetector(cfg.stuck_repeat_threshold)
        if cfg.enable_compression:
            self.compressor = ContextCompressor(
                max_events=cfg.compress_max_events,
                max_chars=cfg.compress_max_chars,
                keep_recent_turns=cfg.keep_recent_turns,
            )
        if cfg.enable_budget:
            self.budget = BudgetGuard(
                per_turn_token_limit=cfg.per_turn_token_limit,
                total_token_budget=cfg.total_token_budget,
            )

    @classmethod
    def from_env(cls) -> Supervisor | None:
        """按 feature flag `VAP_AGENT_SUPERVISOR`（默认关）装配。

        返回 None = 监督层禁用（AgentConfig.supervisor=None，零回归）；
        返回 Supervisor = 以默认阈值启用三件套。
        """
        flag = os.environ.get("VAP_AGENT_SUPERVISOR", "false").strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return None
        return cls()


__all__ = [
    "SUMMARY_EVENT_KIND",
    "BudgetGuard",
    "BudgetTrigger",
    "CondensationResult",
    "ContextCompressor",
    "RoundSignature",
    "StuckDetector",
    "StuckTrigger",
    "Supervisor",
    "SupervisorConfig",
]