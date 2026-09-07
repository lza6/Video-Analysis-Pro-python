"""桥接现有 `src/core/agent_tools.py` 的 15 个工具到新框架的 ToolDefinition。

**只读 import agent_tools.py，不改它**。把 create_*_tool 工厂返回的同步函数
包装成 async ToolDefinition，给新框架的 ReactLoopAgent 用。

bridge point：本模块是「新框架 ↔ 老 agent_tools」的唯一耦合点。老工具的真实
业务逻辑（OpenCV 抽帧 / Whisper / KB 检索 / moviepy 剪辑 / RTSP / video_graph）
继续在 agent_tools.py 内，新框架只负责调度与生命周期。
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional

from src.core.tools.definition import ToolDefinition

# 老 agent_tools 的工具工厂映射：name -> (factory_fn, description, input_schema)
# factory 接受 app_context_getter，返回工具回调。
# **只读 import，不修改 agent_tools.py**。
LEGACY_TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "get_video_meta",
        "factory": "create_get_video_meta_tool",
        "description": "获取当前视频元信息：文件名/时长/输出目录/帧数。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_frame_details",
        "factory": "create_get_frame_details_tool",
        "description": "按秒取视频帧详情（caption/ocr/path），无则动态抽取。",
        "input_schema": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": ["seconds"],
        },
    },
    {
        "name": "delete_history",
        "factory": "create_delete_history_tool",
        "description": "删除当前会话历史（需用户确认）。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_web",
        "factory": "create_search_web_tool",
        "description": "DuckDuckGo 文本搜索（5 条结果）。",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_visual",
        "factory": "create_visual_search_tool",
        "description": "CLIP 视觉语义搜索视频帧，返回 top3 时间点。",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "ocr_frame",
        "factory": "create_ocr_tool",
        "description": "对指定秒数帧做 PaddleOCR。",
        "input_schema": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": ["seconds"],
        },
    },
    {
        "name": "highlight_cut",
        "factory": "create_highlight_cut_tool",
        "description": "按描述自动剪辑集锦视频（jaccard 匹配 + moviepy 2.x）。",
        "input_schema": {
            "type": "object",
            "properties": {"description": {"type": "string"}},
            "required": ["description"],
        },
    },
    {
        "name": "point_at_object",
        "factory": "create_visual_grounding_tool",
        "description": "视觉定位目标物体并跳转时间轴。",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_kb",
        "factory": "create_kb_search_tool",
        "description": "跨视频知识库语义检索（带可跳转时间戳）。",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_by_image",
        "factory": "create_image_search_tool",
        "description": "图→帧跨模态搜索（截图找时刻）。",
        "input_schema": {
            "type": "object",
            "properties": {"image_path": {"type": "string"}},
            "required": ["image_path"],
        },
    },
    {
        "name": "scan_videos",
        "factory": "create_scan_videos_tool",
        "description": "扫描监控视频目录，返回视频列表+大小。",
        "input_schema": {
            "type": "object",
            "properties": {"video_dir": {"type": "string"}},
            "required": ["video_dir"],
        },
    },
    {
        "name": "trigger_batch",
        "factory": "create_batch_analyze_trigger_tool",
        "description": "触发批量监控分析（异步预填配置，用户在批量 tab 确认）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_dir": {"type": "string"},
                "item_description": {"type": "string"},
            },
        },
    },
    {
        "name": "summarize_hits",
        "factory": "create_summarize_hits_tool",
        "description": "汇总 run_store 历史命中（跨会话记忆）。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_skill",
        "factory": "create_generate_skill_tool",
        "description": "agent 自动生成新 skill（规则模板，不调付费 LLM）。",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "start_rtsp_monitor",
        "factory": "create_rtsp_monitor_tool",
        "description": "启动 RTSP 实时流监控（运动检测+VLM 判断后台运行）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "rtsp_url": {"type": "string"},
                "item_description": {"type": "string"},
            },
            "required": ["rtsp_url"],
        },
    },
    {
        "name": "trace_item",
        "factory": "create_trace_item_tool",
        "description": "跨视频追踪物品轨迹（VideoGraph 时序图遍历）。",
        "input_schema": {
            "type": "object",
            "properties": {"item_keyword": {"type": "string"}},
            "required": ["item_keyword"],
        },
    },
]


def _get_factory(factory_attr: str) -> Callable:
    """从 agent_tools 模块取工厂函数（只读 import）。"""
    import src.core.agent_tools as at

    fn = getattr(at, factory_attr, None)
    if fn is None:
        raise AttributeError(f"agent_tools.{factory_attr} 不存在")
    return fn


def adapt_legacy_tool(
    spec: Dict[str, Any],
    app_context_getter: Callable[[], Any],
) -> ToolDefinition:
    """把单个老工具 spec 包装成 ToolDefinition。

    老 factory 接受 app_context_getter，返回同步 callable。包装成 async 回调：
      - 同步调用包到 asyncio.to_thread（避免阻塞事件循环）
      - 异步直接 await
    """
    factory = _get_factory(spec["factory"])
    # 老 factory 签名有两种：
    #   - 依赖 app context: create_xxx_tool(app_context_getter) -> callable
    #   - 无依赖: create_search_web_tool() -> callable  （0 参数）
    # 用 inspect 探测参数数量，0 参直接调，1 参传 getter。
    import inspect as _inspect
    n_params = len(_inspect.signature(factory).parameters)
    if n_params == 0:
        legacy_cb = factory()
    else:
        legacy_cb = factory(app_context_getter)

    async def _wrapped(**kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(legacy_cb):
            return await legacy_cb(**kwargs)
        # 同步工具丢线程池，避免阻塞 loop（重度 OpenCV/OCR 调用尤其要）
        return await asyncio.to_thread(legacy_cb, **kwargs)

    return ToolDefinition(
        name=spec["name"],
        description=spec["description"],
        execute_callback=_wrapped,
        input_schema=spec["input_schema"],
    )


def register_legacy_tools(
    registry,
    app_context_getter: Callable[[], Any],
    only: Optional[List[str]] = None,
) -> List[Callable[[], None]]:
    """把所有老工具注册到新 registry，返回 disposer 列表。

    Args:
        registry: ToolRegistry 实例。
        app_context_getter: 老 factory 期望的 app context 提供者。
        only: 只注册这些 name（None = 全部）。
    """
    disposers: List[Callable[[], None]] = []
    for spec in LEGACY_TOOL_SPECS:
        if only and spec["name"] not in only:
            continue
        defn = adapt_legacy_tool(spec, app_context_getter)
        disposers.append(registry.register(defn))
    return disposers
