"""四级路由解析器（纯函数）。

参考 DSH `dsh-plugin-subagent-director/src/route-resolver.ts:137-229`：
字段级独立回退，四级优先级：
  call(显式参数) > role(角色模板) > default(插件默认) > inherit(继承父 agent)

每个字段（provider/model/reasoning_effort/tool_filter）独立回退，不互相影响。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.core.subagent.role_template import RoleTemplate, ToolFilter


@dataclass(frozen=True)
class RouteResolution:
    """路由解析结果（不可变）。

    每字段已按四级回退定终值。mode 标记每字段最终来自哪一级
    （call/role/default/inherit），供审计。

    backend: 子 agent 执行后端名（默认 "inprocess"）。
    RouteResolution 由 resolve_route 产出时不变（仍为默认值），
    上层 SubagentDirector 可按 call_arg["backend"] 覆写以路由到不同后端。
    """

    provider: str
    model: str
    reasoning_effort: Optional[str]
    tool_filter: ToolFilter
    source: Dict[str, str]  # {"provider": "call", "model": "role", ...}
    backend: str = "inprocess"


def _pick(*candidates: Any) -> tuple[Any, str]:
    """按传入顺序取首个非 None/非空，返回 (value, source_label)。"""
    # candidates: (call_val, role_val, default_val, inherit_val)
    labels = ["call", "role", "default", "inherit"]
    for label, val in zip(labels, candidates):
        if val is not None and val != "" and val != "":
            return val, label
    # 全空：返回最后一个（inherit，可能本身也是 None/空）
    return candidates[-1], labels[-1]


def resolve_route(
    call_arg: Optional[Dict[str, Any]],
    role_template: RoleTemplate,
    default: Dict[str, Any],
    parent: Dict[str, Any],
) -> RouteResolution:
    """四级回退路由解析。

    Args:
        call_arg: 调用方显式参数（最高优先级）。
        role_template: 角色模板字段（次优先级）。
        default: 插件级默认（第三优先级）。
        parent: 父 agent 的路由（最低优先级，inherit）。

    Returns:
        RouteResolution：每字段最终值 + source 标记。
    """
    call_arg = call_arg or {}

    provider, p_src = _pick(
        call_arg.get("provider"),
        role_template.provider,
        default.get("provider"),
        parent.get("provider"),
    )
    model, m_src = _pick(
        call_arg.get("model"),
        role_template.model,
        default.get("model"),
        parent.get("model"),
    )
    reasoning_effort, r_src = _pick(
        call_arg.get("reasoning_effort"),
        role_template.reasoning_effort,
        default.get("reasoning_effort"),
        parent.get("reasoning_effort"),
    )

    # tool_filter：call > role > default > inherit
    tf: Optional[ToolFilter] = None
    tf_src = "inherit"
    if call_arg.get("tool_filter") is not None:
        tf = call_arg["tool_filter"]
        tf_src = "call"
    elif role_template.tool_filter and (
        role_template.tool_filter.allow or role_template.tool_filter.deny):
        tf = role_template.tool_filter
        tf_src = "role"
    elif default.get("tool_filter") is not None:
        tf = default["tool_filter"]
        tf_src = "default"
    else:
        tf = parent.get("tool_filter") or ToolFilter()
        tf_src = "inherit"

    return RouteResolution(
        provider=provider or "",
        model=model or "",
        reasoning_effort=reasoning_effort,
        tool_filter=tf,
        source={
            "provider": p_src,
            "model": m_src,
            "reasoning_effort": r_src,
            "tool_filter": tf_src,
        },
    )
