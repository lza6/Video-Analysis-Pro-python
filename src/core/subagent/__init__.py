"""DSH Agent 框架 — 子 agent 子系统。

SubagentDirector + 四级路由 + RoleTemplate。

参考 DSH `dsh-plugin-subagent-director/src/route-resolver.ts:137-229` +
`delegation-tool.ts:351-543`。
"""
from src.core.subagent.role_template import RoleTemplate, ToolFilter
from src.core.subagent.route_resolver import (
    resolve_route,
    RouteResolution,
)
from src.core.subagent.director import (
    SubagentDirector,
    SubagentHandle,
    SubagentMode,
)

__all__ = [
    "RoleTemplate",
    "ToolFilter",
    "resolve_route",
    "RouteResolution",
    "SubagentDirector",
    "SubagentHandle",
    "SubagentMode",
]
