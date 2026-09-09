"""Turn 事件链 + 钩子。

参考 DSH `packages/core/agent-loop/src/turn.ts`：一个 turn 是一系列 phase，
每个 phase 是 asyncio 钩子点（可被插件注入）。默认实现直通。

phase 顺序（简化版）：
  turn/start
  → claim input
  → assemble prompt + tool schemas
  → pre-step (reject/rewrite)
  → step/start
  → request (LLM stream)
  → assistant-stream
  → tool/call*
  → tools/pre-execute → execute → post-execute → tool/result*
  → step/end
  → supervise (监督层：卡死检测/上下文压缩/预算核算，见 supervisor.py)
  → turn-stopping
  → turn/end

每个 phase 是 async 钩子。TurnHooks 容器持有各 phase 的钩子列表。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional


class TurnPhase(str, Enum):
    """turn 的所有 phase 名（可被插件监听）。"""

    TURN_START = "turn/start"
    CLAIM_INPUT = "claim/input"
    ASSEMBLE = "assemble"
    PRE_STEP = "pre-step"
    STEP_START = "step/start"
    REQUEST = "request"
    ASSISTANT_STREAM = "assistant-stream"
    TOOL_CALL = "tool/call"
    TOOL_PRE_EXECUTE = "tools/pre-execute"
    TOOL_EXECUTE = "tools/execute"
    TOOL_POST_EXECUTE = "tools/post-execute"
    TOOL_RESULT = "tool/result"
    STEP_END = "step/end"
    SUPERVISE = "supervise"
    TURN_STOPPING = "turn-stopping"
    TURN_END = "turn/end"


class TurnStopReason(str, Enum):
    """turn 结束原因。"""

    STOP = "stop"  # LLM 输出 stop
    MAX_STEPS = "max_steps"  # 步数上限
    ERROR = "error"
    CANCELLED = "cancelled"
    STUCK = "stuck"  # 监督层卡死检测命中（同 action/observation 或同 tool_args 或 error 循环）
    BUDGET = "budget"  # 监督层成本预算超限（per-turn token 上限或累计预算）


@dataclass
class TurnResult:
    """turn 的最终不可变结果。"""

    final_text: str = ""
    stop_reason: TurnStopReason = TurnStopReason.STOP
    steps: int = 0
    tool_calls: int = 0
    error: Optional[str] = None


# phase 钩子签名：async (ctx: dict) -> Optional[dict]
# 返回 dict 会合并进 ctx（改写 phase 上下文）；返回 None 直通。
PhaseHook = Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]


@dataclass
class TurnHooks:
    """各 phase 的钩子容器。

    每个 phase 一个列表，按注册顺序执行。
    """

    hooks: Dict[str, List[PhaseHook]] = field(default_factory=dict)

    def on(self, phase: TurnPhase, hook: PhaseHook) -> None:
        self.hooks.setdefault(phase.value, []).append(hook)

    def get(self, phase: TurnPhase) -> List[PhaseHook]:
        return self.hooks.get(phase.value, [])

    async def run_phase(self, phase: TurnPhase,
                        ctx: Dict[str, Any]) -> Dict[str, Any]:
        """跑某 phase 的所有钩子，返回更新后的 ctx。"""
        for hook in self.get(phase):
            patch = await hook(ctx)
            if patch:
                ctx.update(patch)
        return ctx


@dataclass
class Turn:
    """单个 turn 的执行态。持 phase ctx + hooks + step 计数。

    实际 ReAct 循环在 loop.py 的 ReactLoopAgent.run_turn 内跑，本类只承载
    状态与 phase 派发。
    """

    session_id: str
    input_text: str
    hooks: TurnHooks = field(default_factory=TurnHooks)
    ctx: Dict[str, Any] = field(default_factory=dict)
    step: int = 0
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True

    async def emit(self, phase: TurnPhase,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """派发一个 phase，合并 extra 进 ctx。"""
        if extra:
            self.ctx.update(extra)
        return await self.hooks.run_phase(phase, self.ctx)
