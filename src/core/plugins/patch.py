"""声明式插件 patch + 配置加载。

参考 DSH 声明式插件配置：`config/plugins.yml`（插件 id/name/config）+
`Plugin.apply(ctx, config)` 契约。

apply 返回 disposer（cleanup 闭包），插件卸载时调。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

from src.core.plugins.context import PluginContext


@dataclass
class PluginSpec:
    """声明式插件配置（不可变）。"""

    id: str
    name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    # 模块路径（"src.core.plugins.builtin.video_analysis"）或 entry point
    module: Optional[str] = None


def load_plugin_specs(path: str) -> List[PluginSpec]:
    """从 YAML 加载插件配置。

    YAML schema:
        - id: video_analysis
          name: 视频分析插件
          enabled: true
          module: src.core.plugins.builtin.video_analysis
          config:
            key: value
    """
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    specs: List[PluginSpec] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if not entry.get("id"):
            continue
        specs.append(PluginSpec(
            id=entry["id"],
            name=entry.get("name", entry["id"]),
            config=entry.get("config", {}) or {},
            enabled=entry.get("enabled", True),
            module=entry.get("module"),
        ))
    return specs


class Plugin:
    """插件契约基类。

    子类实现 apply(ctx, config) -> Optional[Callable[[], None]]。
    apply 内用 ctx.register_tool / ctx.prepend_system_prompt 等注册副作用，
    返回 disposer（或 None 表示无清理）。
    """

    spec: PluginSpec

    def __init__(self, spec: PluginSpec) -> None:
        self.spec = spec

    def apply(self, ctx: PluginContext,
              config: Optional[Dict[str, Any]] = None) -> Optional[Callable[[], None]]:
        """子类实现。默认无副作用。"""
        return None
