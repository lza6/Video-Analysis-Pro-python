"""PluginContext — 插件可操作的运行时上下文。

参考 DSH `packages/core/host/src/plugin-context.ts`：插件通过 PluginContext
注册工具/子 agent/llm provider/setting/system_prompt/effect/on/inject。

Cordis fiber 无 Python 等价，用 contextvars.ContextVar 暴露「当前 active
context」，插件回调可取到。effect/on/inject 用普通 list + contextlib.AsyncExitStack
替代 fiber scope 自动回收。
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.core.tools.definition import ToolDefinition
from src.core.tools.registry import ToolRegistry


@dataclass
class PluginState:
    """单个插件在 PluginContext 内的注册态。"""

    plugin_id: str
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    disposers: List[Callable[[], None]] = field(default_factory=list)
    effects: List["PluginEffect"] = field(default_factory=list)


@dataclass
class PluginEffect:
    """插件副作用声明（供审计/卸载时清理）。"""

    kind: str  # "tool" | "system_prompt" | "setting" | "hook"
    target: str
    detail: str = ""


# 当前 active PluginContext（contextvars 替代 fiber）
_current_ctx: contextvars.ContextVar[Optional["PluginContext"]] = \
    contextvars.ContextVar("plugin_ctx", default=None)


class PluginContext:
    """插件运行时上下文。

    持有：
      - tool_registry: 工具注册中心
      - system_prompt: 系统提示词（可被插件 prepend/append）
      - settings: 插件可读写的运行时配置
      - hooks: 各 phase 钩子（注入 turn loop）
      - _plugins: 插件 state 表（按 plugin_id）

    暴露 API：
      - tools.register(def) -> disposer
      - system_prompt.prepend(text) / append(text)
      - settings.set(key, val) / get(key)
      - effect(...)
      - on(phase, hook)
    """

    def __init__(self, tool_registry: ToolRegistry,
                 system_prompt: str = "") -> None:
        self.tool_registry = tool_registry
        self._system_prompt = system_prompt
        self._prompt_parts: List[str] = [system_prompt] if system_prompt else []
        self.settings: Dict[str, Any] = {}
        self._plugins: Dict[str, PluginState] = {}
        # turn phase hooks（与 TurnHooks 对齐，但通过 ctx 注册）
        self._phase_hooks: Dict[str, List[Callable]] = {}

    # ---- system prompt ----
    def prepend_system_prompt(self, text: str, plugin_id: str = "") -> None:
        self._prompt_parts.insert(0, text)
        self._record_effect(plugin_id, "system_prompt", "prepend", text[:60])

    def append_system_prompt(self, text: str, plugin_id: str = "") -> None:
        self._prompt_parts.append(text)
        self._record_effect(plugin_id, "system_prompt", "append", text[:60])

    @property
    def system_prompt(self) -> str:
        return "\n\n".join(p for p in self._prompt_parts if p)

    # ---- tools ----
    def register_tool(self, defn: ToolDefinition,
                      plugin_id: str = "") -> Callable[[], None]:
        disposer = self.tool_registry.register(defn)
        state = self._ensure_state(plugin_id)
        state.disposers.append(disposer)
        state.effects.append(PluginEffect(
            "tool", defn.name, defn.description[:60]))
        return disposer

    # ---- settings ----
    def set_setting(self, key: str, value: Any, plugin_id: str = "") -> None:
        self.settings[key] = value
        self._record_effect(plugin_id, "setting", key, str(value)[:60])

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    # ---- hooks ----
    def on(self, phase: str, hook: Callable, plugin_id: str = "") -> None:
        self._phase_hooks.setdefault(phase, []).append(hook)
        self._record_effect(plugin_id, "hook", phase, "")

    # ---- effect 审计 ----
    def _record_effect(self, plugin_id: str, kind: str,
                       target: str, detail: str) -> None:
        if not plugin_id:
            return
        state = self._ensure_state(plugin_id)
        state.effects.append(PluginEffect(kind, target, detail))

    def _ensure_state(self, plugin_id: str) -> PluginState:
        if plugin_id not in self._plugins:
            self._plugins[plugin_id] = PluginState(
                plugin_id=plugin_id or "anonymous", name=plugin_id or "anonymous")
        return self._plugins[plugin_id]

    def get_state(self, plugin_id: str) -> Optional[PluginState]:
        return self._plugins.get(plugin_id)

    @property
    def plugins(self) -> Dict[str, PluginState]:
        return dict(self._plugins)

    # ---- contextvars active ctx ----
    def __enter__(self) -> "PluginContext":
        self._token = _current_ctx.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        _current_ctx.reset(getattr(self, "_token", None))

    @classmethod
    def current(cls) -> Optional["PluginContext"]:
        return _current_ctx.get()
