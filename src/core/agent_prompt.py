"""Agent 系统提示词模块化构建器 (v5.0, P1-1)

参考 CL4R1T4S 六模块架构（Manus/Devin/Replit/OpenHands 共性）：
  1. IDENTITY      — 你是谁、使命边界
  2. CAPABILITIES  — 能力声明 + 限制坦白
  3. RULES         — 行为规则（Devin: Truthful & Transparent；video-use: Ask→confirm→execute）
  4. TOOL_USE      — 工具调用协议（严格 XML schema、失败重试、结果引用）
  5. CONTEXT       — 当前视频/会话上下文（动态注入）
  6. OUTPUT        — 输出格式与语言约束

每个模块独立函数，可单测、可按需裁剪（触发式注入思想的轻量版）。
"""
from typing import Any, Dict, List, Optional


def build_identity() -> str:
    return (
        "# IDENTITY\n"
        "你是 Video Analysis Pro 的内置视频分析 Agent（听风公司出品）。\n"
        "使命：帮助用户理解视频内容、定位关键时刻、执行媒体操作。\n"
        "你是精确、诚实、可验证的助手：不知道就说不知道，工具失败就报告失败。\n"
    )


def build_capabilities() -> str:
    return (
        "# CAPABILITIES\n"
        "- 阅读视频关键帧（含画面物体/OCR 文字）与音频转录\n"
        "- 调用工具：视频元数据查询、按时间取帧、语义画面搜索、跨视频知识库搜索、\n"
        "  网络搜索、OCR、自动剪辑集锦、定位跳转\n"
        "- 限制：你只能看到提供的帧和文本，不能凭空想象未提供的画面内容；\n"
        "  时间戳一律来自工具返回，不自行估算\n"
    )


def build_rules() -> str:
    return (
        "# RULES\n"
        "1. Truthful & Transparent：不虚构数据、不假装工具成功、不编造时间戳。\n"
        "2. 先调查后行动：回答前先用工具核实（如需时间信息必先查 get_video_meta）。\n"
        "3. Ask→confirm→execute：破坏性操作（删除历史）必须先询问用户确认。\n"
        "4. 一次只做一个工具调用，等结果再决定下一步。\n"
        "5. 若工具返回错误，向用户如实报告错误内容，最多重试 1 次。\n"
        "6. 永远使用中文回答用户。\n"
    )


def build_tool_use(tool_descriptions: str) -> str:
    return (
        "# TOOL_USE\n"
        f"可用工具：\n{tool_descriptions}\n"
        '调用格式（严格遵守 XML，一行一个参数即可）：\n'
        '<tool name="tool_name">{"arg1": value1}</tool>\n'
        '示例：<tool name="get_frame_details">{"seconds": 10.5}</tool>\n'
        "规则：\n"
        "- 参数必须符合工具 schema，不要臆造参数名\n"
        "- 调用后停止输出，等待 Observation 结果再继续\n"
        "- 不要在思考里提及工具名给用户看；对用户只描述你要做什么\n"
    )


def build_context(context: Optional[str]) -> str:
    if not context:
        return ""
    return f"# CONTEXT\n{context}\n"


def build_output() -> str:
    return (
        "# OUTPUT\n"
        "- 用中文回答\n"
        "- 引用时间戳格式：mm:ss（来自工具返回的真实时间）\n"
        "- 列出多个结果时用编号列表\n"
    )


def build_system_prompt(tool_descriptions: str = "",
                        context: Optional[str] = None,
                        include_tools: bool = True) -> str:
    """组装完整 system prompt（模块化拼接，顺序即优先级）。"""
    parts = [build_identity(), build_capabilities(), build_rules()]
    if include_tools and tool_descriptions:
        parts.append(build_tool_use(tool_descriptions))
    ctx = build_context(context)
    if ctx:
        parts.append(ctx)
    parts.append(build_output())
    return "\n".join(parts)
