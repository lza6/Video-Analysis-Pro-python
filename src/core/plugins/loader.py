"""插件加载器。

发现插件（扫 plugins/ 目录 + entry_points + config/plugins.yml 声明），
调 apply(ctx, config) 入口，收集 disposer。

加载顺序：
  1. config/plugins.yml 声明的（按顺序）
  2. src/core/plugins/builtin/ 下的内置插件（约定：每个模块导出
     register(ctx) 函数 或 Plugin 子类）
  3. 第三方 entry_points（组 video_analysis.plugins）—— 仅协议，需 setuptools

unloader：dispose_all() 调所有 disposer，逆序。
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional

from src.core.plugins.context import PluginContext
from src.core.plugins.patch import Plugin, PluginSpec


@dataclass
class _LoadedPlugin:
    spec: PluginSpec
    disposer: Optional[Callable[[], None]] = None


class PluginLoader:
    """插件加载/卸载器。"""

    def __init__(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._loaded: List[_LoadedPlugin] = []

    def load_from_specs(self, specs: List[PluginSpec]) -> None:
        """按声明式 spec 列表加载。"""
        for spec in specs:
            if not spec.enabled:
                continue
            disposer = self._apply_spec(spec)
            self._loaded.append(_LoadedPlugin(spec=spec, disposer=disposer))

    def load_builtin(self, modules: List[str]) -> None:
        """加载内置插件模块（每个模块导出 register(ctx) 或 Plugin 子类）。

        Args:
            modules: 模块路径列表，如 ["src.core.plugins.builtin.video_analysis"]
        """
        for mod_path in modules:
            try:
                mod = importlib.import_module(mod_path)
            except Exception:
                continue
            # 约定 1：模块有 register(ctx) -> disposer 函数
            if hasattr(mod, "register"):
                fn = getattr(mod, "register")
                if callable(fn):
                    spec = PluginSpec(
                        id=getattr(mod, "PLUGIN_ID", mod_path.split(".")[-1]),
                        name=getattr(mod, "PLUGIN_NAME", mod_path.split(".")[-1]),
                        module=mod_path,
                    )
                    disposer = self._safe_register(fn, spec)
                    self._loaded.append(_LoadedPlugin(
                        spec=spec, disposer=disposer))
                    continue
            # 约定 2：模块有 Plugin 子类 PLUGIN_CLASS
            cls = getattr(mod, "PLUGIN_CLASS", None)
            if cls is not None and isinstance(cls, type) and issubclass(cls, Plugin):
                spec = PluginSpec(
                    id=getattr(mod, "PLUGIN_ID", mod_path.split(".")[-1]),
                    name=getattr(mod, "PLUGIN_NAME", mod_path.split(".")[-1]),
                    module=mod_path,
                )
                plugin = cls(spec)
                disposer = plugin.apply(self.ctx, spec.config) or None
                self._loaded.append(_LoadedPlugin(spec=spec, disposer=disposer))

    def _apply_spec(self, spec: PluginSpec) -> Optional[Callable[[], None]]:
        if spec.module is None:
            return None
        try:
            mod = importlib.import_module(spec.module)
        except Exception:
            return None
        # 优先 Plugin 子类
        cls = getattr(mod, "PLUGIN_CLASS", None)
        if cls is not None and isinstance(cls, type) and issubclass(cls, Plugin):
            return cls(spec).apply(self.ctx, spec.config) or None
        # 兜底：register(ctx)
        fn = getattr(mod, "register", None)
        if callable(fn):
            return self._safe_register(fn, spec)
        return None

    def _safe_register(self, fn: Callable, spec: PluginSpec) -> Optional[Callable[[], None]]:
        try:
            sig = inspect.signature(fn)
            if len(sig.parameters) == 1:
                return fn(self.ctx) or None
            # 双参：(ctx, config)
            params = list(sig.parameters.values())
            if len(params) >= 2:
                return fn(self.ctx, spec.config) or None
            return fn(self.ctx) or None
        except Exception:
            return None

    def dispose_all(self) -> None:
        """逆序调所有 disposer。"""
        for lp in reversed(self._loaded):
            if lp.disposer is not None:
                try:
                    lp.disposer()
                except Exception:
                    pass
        self._loaded.clear()

    @property
    def loaded(self) -> List[PluginSpec]:
        return [lp.spec for lp in self._loaded]
