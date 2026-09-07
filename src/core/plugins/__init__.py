"""DSH Agent 框架 — 插件子系统。

PluginContext + loader + 声明式 patch。Cordis fiber 无 Python 等价，
用 asyncio.Task + contextvars.ContextVar + contextlib.AsyncExitStack 替代。
"""
from src.core.plugins.context import (
    PluginContext,
    PluginEffect,
    PluginState,
)
from src.core.plugins.patch import Plugin, PluginSpec, load_plugin_specs
from src.core.plugins.loader import PluginLoader
from src.core.plugins.builtin.video_analysis import (
    VideoAnalysisPlugin,
    register as register_video_analysis_plugin,
    PLUGIN_ID as VIDEO_ANALYSIS_PLUGIN_ID,
)

__all__ = [
    "PluginContext",
    "PluginEffect",
    "PluginState",
    "Plugin",
    "PluginSpec",
    "load_plugin_specs",
    "PluginLoader",
    "VideoAnalysisPlugin",
    "register_video_analysis_plugin",
    "VIDEO_ANALYSIS_PLUGIN_ID",
]
