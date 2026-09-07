"""角色模板。

参考 DSH `dsh-plugin-subagent-director/src/role-template.ts`：
子 agent 由角色模板定义：display_name / persona / provider / model /
reasoning_effort / tool_filter（allow/deny）。

ToolFilter 是 allow + deny 集合，决定子 agent 可用哪些工具。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass(frozen=True)
class ToolFilter:
    """工具过滤器：allow 名单 + deny 黑名单。

    语义：
      - allow 非空：只允许这些工具（白名单）
      - allow 空：允许所有，除非在 deny 里
      - deny：黑名单（即使 allow 里有也排除——deny 优先）
    """

    allow: frozenset = field(default_factory=frozenset)
    deny: frozenset = field(default_factory=frozenset)

    def permits(self, tool_name: str) -> bool:
        if tool_name in self.deny:
            return False
        if self.allow:
            return tool_name in self.allow
        return True

    @classmethod
    def from_sets(cls, allow: Optional[Set[str]] = None,
                  deny: Optional[Set[str]] = None) -> "ToolFilter":
        return cls(
            allow=frozenset(allow or ()),
            deny=frozenset(deny or ()),
        )


@dataclass(frozen=True)
class RoleTemplate:
    """子 agent 角色模板（不可变）。

    字段都可空：空表示「不指定，由路由回退到 default/inherit」。
    """

    display_name: str
    description: str = ""
    persona: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    tool_filter: ToolFilter = field(default_factory=ToolFilter)
