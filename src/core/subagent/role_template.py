"""角色模板。

参考 DSH `dsh-plugin-subagent-director/src/role-template.ts`：
子 agent 由角色模板定义：display_name / persona / provider / model /
reasoning_effort / tool_filter（allow/deny）。

ToolFilter 是 allow + deny 集合，决定子 agent 可用哪些工具。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set

# ---------------------------------------------------------------------------
# 领域角色模板（B-P2-2 电商 / PPT / 营销，v10.2）
# ---------------------------------------------------------------------------
# 角色与 SKILL.md 一一对应：
#   merchant  → ecommerce-merchant（卖家运营：计划→审批→执行）
#   shopper   → ecommerce-shopper（买家比价选品：纯读 + 本地 Mock）
#   marketer  → marketing-publish（文案生成 + Mock 分发清单）
# 写操作边界统一口径：
#   - 读工具（get_*/search_*/list_*/compare_*）白名单放行
#   - 写工具（create_*/update_*/publish_*）不在 allow 白名单 → 需 scope_guard
#     审批（Ask 信号），且当前实现仅 Mock，绝不真实调用外部平台（付费红线）。

#: 电商卖家角色：上架/改价/库存 计划→审批→执行
MERCHANT_ROLE = {
    "display_name": "merchant",
    "description": "电商卖家运营：商品上架/改价/库存管理，计划→审批→执行流程",
    "persona": (
        "你是电商卖家运营助手。所有上架/改价/库存变更都是高危写操作："
        "必须先产出运营计划并提交审批，未经人工批准绝不执行任何写操作；"
        "当前所有电商写操作仅本地 Mock，不真实调用任何电商平台。"
    ),
}

#: 电商买家角色：比价/选品（纯读）
SHOPPER_ROLE = {
    "display_name": "shopper",
    "description": "电商买家比价选品：多平台比价流程与商品评分规则，仅本地 Mock",
    "persona": (
        "你是电商买家比价选品助手。商品数据仅来自本地 Mock 商品库，"
        "绝不真实访问任何电商平台；所有结论标注基于本地 Mock 数据。"
    ),
}

#: 营销角色：文案生成 + 分发清单（发布仅 Mock）
MARKETER_ROLE = {
    "display_name": "marketer",
    "description": "营销文案生成与分发清单：公众号/小红书/微博/抖音风格模板",
    "persona": (
        "你是营销分发助手。文案草稿可自由生成；任何真实发布需 "
        "VAP_ALLOW_WRITE_PUBLISH=1 且人工审批，当前实现仅 Mock 分发清单，"
        "绝不真实调用公众号/小红书/微博/抖音接口。"
    ),
}

#: 领域角色写操作边界：create/update/delete/trigger/start/generate/send/publish
#: 均不在读工具白名单内 → 必须经 scope_guard 审批（Ask 信号）+ 人工批准。
_DOMAIN_WRITE_KEYS = (
    "create",
    "update",
    "delete",
    "trigger",
    "start",
    "generate",
    "send",
    "publish",
)


def _domain_tool_allow() -> frozenset:
    """构造领域角色读工具白名单（merchant/shopper/marketer 共用）。

    只放行读工具（get_*/search_*/list_*/compare_*）；写工具
    （create/update/delete/trigger/start/generate/send/publish）一律不在
    白名单 → 需 scope_guard 审批 + 人工批准后才执行（Ask 信号）。
    """
    allow: Set[str] = set()
    for prefix in ("get_", "search_", "list_", "summarize_", "compare_"):
        allow.add(f"{prefix}*")
    return frozenset(allow)


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

    def permits_wildcard(self, tool_name: str) -> bool:
        """通配符感知的权限判定（allow 项可含 `*` 后缀）。

        `permits` 保持精确匹配语义（向后兼容，现有调用不受影响）；
        领域角色白名单用 `get_*` / `search_*` 等前缀通配，判定走本方法。
        """
        if tool_name in self.deny:
            return False
        if self.allow:
            for pattern in self.allow:
                if pattern.endswith("*"):
                    if tool_name.startswith(pattern[:-1]):
                        return True
                elif pattern == tool_name:
                    return True
            return False
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


# ---------------------------------------------------------------------------
# 领域角色构造工厂（B-P2-2）
# ---------------------------------------------------------------------------


def build_domain_role(role_key: str, write_allow: bool = False) -> RoleTemplate:
    """按角色键构造领域 RoleTemplate（merchant / shopper / marketer）。

    Args:
        role_key: "merchant" | "shopper" | "marketer"。
        write_allow: 是否把写工具也加进白名单。默认 False=写操作不在
            白名单 → 必须经 scope_guard 审批（Ask 信号）后才执行。

    Returns:
        RoleTemplate：display_name + persona + 读工具白名单
        （write_allow=False 时）。write_allow=True 时 allow 含全部工具
        （测试 / 沙箱场景用；生产默认 False）。
    """
    table = {
        "merchant": MERCHANT_ROLE,
        "shopper": SHOPPER_ROLE,
        "marketer": MARKETER_ROLE,
    }
    spec = table.get(role_key)
    if spec is None:
        raise ValueError(
            f"unknown domain role: {role_key!r} "
            f"(available: {sorted(table)})"
        )
    allow = set(_domain_tool_allow())
    if write_allow:
        allow = set()
    return RoleTemplate(
        display_name=spec["display_name"],
        description=spec["description"],
        persona=spec["persona"],
        provider="inprocess",
        tool_filter=ToolFilter.from_sets(allow=allow),
    )
