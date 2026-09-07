"""内置视频分析插件。

把现有三阶段流水线（Phase 1 数据提取 / Phase 2 AI 分析 / Phase 3 媒体生成）
包装成插件，通过 PluginContext 注入 video tools + system prompt。

不重写流水线逻辑：只把 src/core/agent_tools.py 的 15 个工具桥接进来
（通过 src/core/tools/adapter.py，只读 import agent_tools）。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.core.plugins.context import PluginContext
from src.core.plugins.patch import Plugin, PluginSpec
from src.core.tools.adapter import register_legacy_tools

PLUGIN_ID = "video_analysis"
PLUGIN_NAME = "Video Analysis (三阶段流水线)"

VIDEO_SYSTEM_PROMPT = (
    "你是 TingFeng Hermes 的视频分析助手，由 听风公司 (Tingfeng) 出品。\n"
    "可用工具覆盖三阶段流水线：\n"
    "  - Phase 1 数据提取：get_video_meta / get_frame_details / ocr_frame / "
    "search_visual / search_by_image\n"
    "  - Phase 2 AI 分析：search_kb / trace_item（跨视频知识图谱）\n"
    "  - Phase 3 媒体生成：highlight_cut / point_at_object\n"
    "  - 监控/批量：scan_videos / trigger_batch / summarize_hits / "
    "start_rtsp_monitor\n"
    "  - 自治：generate_skill（按场景生成新 skill）\n"
    "调用工具前先确认输入，输出 Markdown 报告。"
)


class VideoAnalysisPlugin(Plugin):
    """视频分析插件。"""

    def apply(self, ctx: PluginContext,
              config: Optional[Dict[str, Any]] = None) -> Optional[Callable[[], None]]:
        config = config or {}
        app_getter = config.get("app_context_getter")
        if app_getter is None:
            # 无 app context 时仍可注册（工具调用时会返回友好提示）
            def _noop() -> Any:
                return None
            app_getter = config.get("app_context_getter", _noop)

        # 注入 system prompt
        ctx.prepend_system_prompt(VIDEO_SYSTEM_PROMPT, PLUGIN_ID)

        # 注册老 15 工具（桥接 agent_tools.py，只读 import）
        only = config.get("only_tools")  # 可选：只注册部分
        disposers = register_legacy_tools(ctx.tool_registry, app_getter, only=only)

        def _dispose() -> None:
            for d in disposers:
                try:
                    d()
                except Exception:
                    pass

        return _dispose


# 模块级 register 函数（loader 约定 1）
def register(ctx: PluginContext,
             config: Optional[Dict[str, Any]] = None) -> Callable[[], None]:
    spec = PluginSpec(id=PLUGIN_ID, name=PLUGIN_NAME)
    return VideoAnalysisPlugin(spec).apply(ctx, config) or (lambda: None)


PLUGIN_CLASS = VideoAnalysisPlugin
