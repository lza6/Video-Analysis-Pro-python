"""ELI5（Explain Like I'm 5）—— 工具调用一句大白话解释器。

学 Manus 的 notify/ask 二分：换策略时一句解释（notify），危险操作带
risk 标签（ask 阻塞）。本模块只负责把"工具名 + 参数 + 结果"翻译成一句
用户能懂的人话，供 ThinkingWidget 摘要行与决策日志 reason 字段复用。

纯函数，无副作用，可单测。result 是 Exception 时走错误分支标"出错了"；
是 str 时取前 200 字解析关键信息（top_ts / dur / n 等）；解析失败或未知
工具退化为通用文案，绝不抛异常到调用方（UI 摘要行不能因解析崩而崩）。
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# result 字符串解析窗口：head 取前 200 字（任务约定），关键信息必在前面。
_HEAD_LIMIT = 200


def explain_tool_call(tool_name: str, args: Dict, result: object) -> str:
    """返回一句大白话解释工具调用的结果。

    分支顺序：
      1. result 是 Exception → "调用了 {tool}，出错了：{e}"
      2. 已知工具模板 → 解析 head，失败则 raise（被外层 catch 退化）
      3. 未知工具 / 解析失败 → "调用了 {tool}，返回了 {len} 字符的结果。"
    """
    args = args or {}

    # 1. Exception 分支：明确告知出错，不暴露堆栈
    if isinstance(result, Exception):
        return f"调用了 {tool_name}，出错了：{result}"

    # 非 str 统一转 str（工具返回大多是 json.dumps 的 str）
    result_str = result if isinstance(result, str) else str(result)
    head = result_str[:_HEAD_LIMIT]

    # 2. 已知工具模板
    try:
        if tool_name in ("search_visual", "search_by_image"):
            return _explain_visual_search(args, head)
        if tool_name == "get_frame_details":
            return _explain_get_frame(args)
        if tool_name == "create_highlights":
            return _explain_highlights(args, head)
        if tool_name == "search_kb":
            return _explain_search_kb(args, head)
        if tool_name == "run_ocr":
            return _explain_ocr(args, head)
        if tool_name == "search_web":
            return _explain_search_web(args, head)
        if tool_name == "get_video_meta":
            return _explain_video_meta(head)
        if tool_name == "point_and_jump":
            return _explain_point_jump(head)
        if tool_name == "delete_this_history":
            return "删除这条历史记录（需要你确认）。"
    except Exception as e:  # noqa: BLE001 — 摘要行绝不能崩，统一退化
        logger.debug("[eli5] %s 模板解析失败，退化通用文案: %s", tool_name, e)

    # 3. 未知工具 / 解析失败 退化
    return f"调用了 {tool_name}，返回了 {len(result_str)} 字符的结果。"


# ---------------------------------------------------------------- 模板

def _explain_visual_search(args: Dict, head: str) -> str:
    """search_visual / search_by_image 共用模板。

    result 行格式见 agent_tools.create_visual_search_tool：
      "时间点 12.34s (匹配度: 0.85)"  /  "时间点 12.34s (相似度: 0.85)"
    """
    query = args.get("query")
    if not query:
        img = args.get("image_path")
        query = Path(img).name if img else "指定画面"
    m = re.search(r"时间点\s*([\d.]+)\s*s.*?(?:匹配度|相似度)[:\s]*([\d.]+)", head)
    if not m:
        raise ValueError("visual_search: 未在结果中找到时间点/相似度")
    top_ts, score = m.group(1), m.group(2)
    return (f"在视频里找画面像'{query}'的时刻，"
            f"最像的是 {top_ts} 秒（相似度 {score}）。")


def _explain_get_frame(args: Dict) -> str:
    seconds = args.get("seconds", "?")
    return f"截取 {seconds} 秒那一帧，看看画面里有什么。"


def _explain_highlights(args: Dict, head: str) -> str:
    """create_highlights 模板。

    工具固定取 top3（见 agent_tools.py: scored[:3]），n=3 是代码事实。
    result 含"未找到"/"出错"时给失败文案。
    """
    desc = args.get("description", "")
    if "未找到" in head or "出错" in head or "Error" in head:
        return f"按你说的'{desc}'想剪集锦，但没找到足够的相关片段。"
    return f"按你说的'{desc}'，挑了 3 个最相关的片段拼成集锦。"


def _explain_search_kb(args: Dict, head: str) -> str:
    """search_kb 模板。每条结果以 "N. " 开头，按换行估算条数。"""
    query = args.get("query", "")
    lower = head.lower()
    if "没有匹配结果" in head or "unavailable" in lower or "knowledge base" in lower:
        n = 0
    else:
        # head 前 200 字可能截断，n 是下限估算（eli5 概览性质可接受）
        n = head.count("\n") + 1 if head.strip() else 0
    return f"在以前分析过的所有视频里搜'{query}'，找到 {n} 个相似画面。"


def _explain_ocr(args: Dict, head: str) -> str:
    """run_ocr 模板。OCR 返回拼接文本，无行数信息；用空格分词粗略估算。"""
    path = args.get("path") or f"{args.get('seconds', '?')} 秒那一帧"
    if "No text" in head or "Error" in head or not head.strip():
        n = 0
    else:
        n = max(1, len(head.split()))
    return f"对 {path} 这一帧做文字识别，认出 {n} 行字。"


def _explain_search_web(args: Dict, head: str) -> str:
    """search_web 模板。result 是 JSON list，n = len(list)。"""
    query = args.get("query", "")
    try:
        data = json.loads(head)
        n = len(data) if isinstance(data, list) else (1 if data else 0)
    except Exception:
        # JSON 截断无法解析时退化为按行数估算
        n = head.count("\n") + 1 if head.strip() else 0
    return f"网上搜'{query}'，拿到 {n} 条结果。"


def _explain_video_meta(head: str) -> str:
    """get_video_meta 模板。result 是 JSON，取 duration 字段。"""
    try:
        data = json.loads(head)
        dur = data.get("duration", 0)
    except Exception:
        m = re.search(r'"duration"[:\s]*([\d.]+)', head)
        dur = m.group(1) if m else "?"
    return f"查这个视频的基本信息：时长 {dur} 秒。"


def _explain_point_jump(head: str) -> str:
    """point_and_jump 模板。result 含 "{ts}s"，解析首个时间戳。"""
    m = re.search(r"([\d.]+)\s*s", head)
    ts = m.group(1) if m else "?"
    return f"跳到 {ts} 秒。"
