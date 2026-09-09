"""DSH Agent 框架 — 工具子系统。

提供 ToolDefinition + 四层 waterfall + 并行执行 + 桥接现有 agent_tools。

映射 DSH `packages/core/tool-host` 与 codex `tools/parallel.rs` 的概念到
Python asyncio：Cordis fiber 无 Python 等价，用 asyncio.Task + contextvars
+ AsyncExitStack 替代。
"""
from src.core.tools.definition import (
    ToolDefinition,
    ToolCallback,
    define_tool,
)
from src.core.tools.waterfall import (
    ToolWaterfall,
    Allow,
    Deny,
    Ask,
    Replace,
    Timeout,
    ToolDenied,
    ToolNeedsApproval,
)
from src.core.tools.parallel import (
    AsyncRWLock,
    ToolCallRuntime,
    ParallelExecutor,
)
from src.core.tools.registry import (
    ToolRegistry,
    ToolCall,
    ToolResult,
    ToolNotFound,
)
from src.core.tools.adapter import (
    adapt_legacy_tool,
    register_legacy_tools,
    register_media_gen_tools,
    LEGACY_TOOL_SPECS,
)

__all__ = [
    "ToolDefinition",
    "ToolCallback",
    "define_tool",
    "ToolWaterfall",
    "Allow",
    "Deny",
    "Ask",
    "Replace",
    "Timeout",
    "ToolDenied",
    "ToolNeedsApproval",
    "AsyncRWLock",
    "ToolCallRuntime",
    "ParallelExecutor",
    "ToolRegistry",
    "ToolCall",
    "ToolResult",
    "ToolNotFound",
    "adapt_legacy_tool",
    "register_legacy_tools",
    "register_media_gen_tools",
    "LEGACY_TOOL_SPECS",
]
