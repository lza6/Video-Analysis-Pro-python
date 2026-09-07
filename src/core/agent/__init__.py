"""DSH Agent 框架 — Agent 子系统。

提供 ReactLoopAgent + Session + append-only SessionEvent log + Turn 事件链。

映射 DSH `packages/core/agent-loop` 与 codex `codex.rs:run_turn` 的概念到
Python asyncio：Cordis fiber 无 Python 等价，用 asyncio.Task + contextvars
+ AsyncExitStack 替代。
"""
from src.core.agent.session import Session, SessionEvent, SessionStore
from src.core.agent.turn import (
    Turn,
    TurnResult,
    TurnPhase,
    TurnHooks,
    TurnStopReason,
)
from src.core.agent.loop import ReactLoopAgent, AgentPhase, AgentConfig

__all__ = [
    "Session",
    "SessionEvent",
    "SessionStore",
    "Turn",
    "TurnResult",
    "TurnPhase",
    "TurnHooks",
    "TurnStopReason",
    "ReactLoopAgent",
    "AgentPhase",
    "AgentConfig",
]
