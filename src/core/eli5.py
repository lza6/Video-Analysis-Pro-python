"""ELI5（Explain Like I'm 5）—— 工具调用一句大白话解释器。

学 Manus 的 notify/ask 二分：换策略时一句解释（notify），危险操作带
risk 标签（ask 阻塞）。本模块只负责把"工具名 + 参数 + 结果"翻译成一句
用户能懂的人话，供 ThinkingWidget 摘要行与决策日志 reason 字段复用。

纯函数，无副作用，可单测。result 是 Exception 时走错误分支标"出错了"；
是 str 时取前 200 字解析关键信息（top_ts / dur / n 等）；解析失败或未知
工具退化为明确告知缺口的兜底文案，绝不抛异常到调用方（UI 摘要行不能因
解析崩而崩）。

v10.4.0（P0-5）修复两类真实缺陷
--------------------------------
1. **模板名与实际工具名不一致**：此前模板匹配 ``create_highlights`` /
   ``run_ocr`` / ``point_and_jump`` / ``delete_this_history``，而
   ``src/core/tools/adapter.py`` 注册的真实工具名是 ``highlight_cut`` /
   ``ocr_frame`` / ``point_at_object`` / ``delete_history`` —— 这 4 条模板
   从未命中过，用户看到的是"返回了 N 字符"。
2. **覆盖不完整**：32 个工具里只有 9 个有模板，``make_*`` / ``web_*`` /
   ``cdp_*`` / ``scan_videos`` 等全走兜底。现已全部覆盖，并由
   ``tests/test_eli5_coverage.py`` 从工具注册表**动态枚举**做契约守护
   —— 将来新增工具而忘记补模板时，那个测试会立刻变红。
"""
import json
import logging
import re
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

# result 字符串解析窗口：head 取前 200 字（任务约定），关键信息必在前面。
_HEAD_LIMIT = 200

#: 危险写工具 —— 文案必须显式标"危险"。
#: 与 `src/core/tools/scope_guard.py` 的 DANGEROUS_WRITE_PATTERNS 分类对齐；
#: `tests/test_eli5_coverage.py::test_eli5_risk_matches_scope_guard` 断言二者一致，
#: 防止未来分类漂移导致"危险的没标危险"。
DANGEROUS_TOOLS = frozenset({
    "delete_history",
    "highlight_cut",
    "trigger_batch",
    "start_rtsp_monitor",
    "cdp_evaluate",
    "cdp_eval_write",
})

#: 一般写工具 —— 文案提示"需要你确认后才会执行"。
#: 分类依据同上（scope_guard 的 WRITE_PATTERNS 命中且非危险写）。
WRITE_TOOLS = frozenset({
    "create_cut_clip",
    "generate_skill",
    "make_subtitle",
    "make_voiceover",
    "make_short_video",
    "web_browser_trigger",
    "web_browser_update",
})

#: 结果里出现这些信号时，说明这一步没成功（模板会附加提示）。
_FAILURE_MARKERS = (
    "未找到",
    "没有匹配",
    "出错",
    "失败",
    "Error",
    "error:",
    "No text",
    "not found",
    "Frame not found",
    "App context missing",
    "unavailable",
    "Traceback",
)

#: 兜底文案里必须出现的标记 —— 契约测试据此判断"是否走了兜底"。
_FALLBACK_MARKER = "尚未收录"


def explain_tool_call(tool_name: str, args: Dict, result: object) -> str:
    """返回一句大白话解释工具调用的结果。

    分支顺序：
      1. result 是 Exception → "调用了 {tool}，出错了：{e}"
      2. 有专属模板 → 渲染（危险/写工具自动加风险前缀）
      3. 无模板 / 渲染失败 → 明确告知缺口的兜底文案
    """
    args = args or {}

    # 1. Exception 分支：明确告知出错，不暴露堆栈
    if isinstance(result, Exception):
        return f"调用了 {tool_name}，出错了：{result}"

    # 非 str 统一转 str（工具返回大多是 json.dumps 的 str）
    result_str = result if isinstance(result, str) else str(result)
    head = result_str[:_HEAD_LIMIT]

    # 2. 专属模板
    render = _TEMPLATES.get(tool_name)
    if render is None:
        return _fallback(tool_name)

    try:
        sentence = render(args, head)
    except Exception as e:  # noqa: BLE001 — 摘要行绝不能崩，统一退化
        logger.debug("[eli5] %s 模板解析失败，退化兜底文案: %s", tool_name, e)
        return _fallback(tool_name)

    if not sentence:
        return _fallback(tool_name)

    if _looks_failed(head):
        sentence = f"{sentence}（⚠️ 这一步似乎没有成功，原始输出见日志页）"

    return _with_risk(tool_name, sentence)


# ---------------------------------------------------------------- 风险前缀


def _with_risk(tool_name: str, sentence: str) -> str:
    """按 scope_guard 的同一套分类给文案加风险前缀。"""
    if tool_name in DANGEROUS_TOOLS:
        return f"⚠️ 危险操作（需你确认）：{sentence}"
    if tool_name in WRITE_TOOLS:
        return f"需要你确认后才会执行：{sentence}"
    return sentence


def _fallback(tool_name: str) -> str:
    """无模板时的兜底：**明确告知缺口**，不再假装正常。

    旧文案是"调用了 X，返回了 N 字符的结果"——小白看到这句等于没看到。
    现在直说"尚未收录人话说明"，把缺口暴露出来（而不是静默降级）。
    """
    return f"⚠️ 尚未收录「{tool_name}」的人话说明（原始结果见日志页）。"


def _looks_failed(head: str) -> bool:
    """结果里是否出现失败信号。"""
    return any(marker in head for marker in _FAILURE_MARKERS)


def _count_items(head: str) -> Optional[int]:
    """尽力从结果里数出条目数；数不出返回 None。

    只认两类可信信号（不猜格式，宁可少说也不编造）：
      1. 明确的 "N 条 / N 个 / N 张 / N 项"；
      2. 行首编号列表（"1. " / "1) "）。
    """
    if not head.strip():
        return 0
    m = re.search(r"(\d+)\s*(?:条|个|张|项|处|段|帧)", head)
    if m:
        return int(m.group(1))
    numbered = re.findall(r"^\s*\d+[.)]\s", head, flags=re.MULTILINE)
    if numbered:
        return len(numbered)
    return None


def _clip(text: object, limit: int = 24) -> str:
    """把参数值裁成适合放进一句话的短串。"""
    s = str(text).replace("\n", " ").strip()
    return s if len(s) <= limit else s[:limit] + "…"


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
    if m:
        top_ts, score = m.group(1), m.group(2)
        return (f"在视频里找画面像「{_clip(query)}」的时刻，"
                f"最像的是 {top_ts} 秒（相似度 {score}）。")
    # 结果里没有"时间点/相似度"格式（例如 0 命中）→ 不编造分数
    n = _count_items(head)
    if n == 0:
        return f"在视频里找画面像「{_clip(query)}」的时刻，没有找到相似的片段。"
    return f"在视频里找画面像「{_clip(query)}」的时刻。"


def _explain_get_frame(args: Dict, head: str) -> str:  # noqa: ARG001 — 签名统一
    seconds = args.get("seconds", "?")
    return f"截取 {seconds} 秒那一帧，看看画面里有什么。"


def _explain_highlight_cut(args: Dict, head: str) -> str:
    """highlight_cut 模板（旧模板名 create_highlights 从未命中，P0-5 修正）。

    工具固定取 top3（见 agent_tools.py:282 `for _, f in scored[:3]`），
    成功返回 "集锦视频生成成功：<文件名>"。
    """
    desc = _clip(args.get("description", ""))
    if "未找到" in head or "出错" in head:
        return f"按你说的「{desc}」想剪集锦，但没找到足够的相关片段。"
    return f"按你说的「{desc}」，挑了 3 个最相关的片段拼成集锦视频。"


def _explain_search_kb(args: Dict, head: str) -> str:
    """search_kb 模板。每条结果以 "N. " 开头，按编号行估算条数。"""
    query = _clip(args.get("query", ""))
    lower = head.lower()
    if "没有匹配结果" in head or "unavailable" in lower:
        return f"在以前分析过的所有视频里搜「{query}」，没有找到相似画面。"
    n = _count_items(head)
    if n is None:
        return f"在以前分析过的所有视频里搜「{query}」，找到了相似画面。"
    return f"在以前分析过的所有视频里搜「{query}」，找到 {n} 个相似画面。"


def _explain_ocr_frame(args: Dict, head: str) -> str:
    """ocr_frame 模板（旧模板名 run_ocr 从未命中，且误用 args['path']）。

    adapter 里 ocr_frame 的入参是 `seconds`（无 path），P0-5 一并修正。
    结果：识别文本 / "No text detected." / "OCR Tool error: …" / "Frame not found."
    """
    seconds = args.get("seconds", "?")
    if "No text" in head or "error" in head.lower() or not head.strip():
        return f"看 {seconds} 秒那一帧有没有文字，没有认出字来。"
    return f"对 {seconds} 秒那一帧做文字识别，认出了画面里的字。"


def _explain_search_web(args: Dict, head: str) -> str:
    """search_web 模板。result 是 JSON list，n = len(list)。"""
    query = _clip(args.get("query", ""))
    try:
        data = json.loads(head)
        n: Optional[int] = len(data) if isinstance(data, list) else (1 if data else 0)
    except Exception:
        n = _count_items(head)
    if n is None:
        return f"在网上搜「{query}」。"
    return f"在网上搜「{query}」，拿到 {n} 条结果。"


def _explain_video_meta(args: Dict, head: str) -> str:  # noqa: ARG001
    """get_video_meta 模板。result 是 JSON，取 duration 字段。"""
    try:
        data = json.loads(head)
        dur = data.get("duration", 0)
    except Exception:
        m = re.search(r'"duration"[:\s]*([\d.]+)', head)
        dur = m.group(1) if m else "?"
    return f"查这个视频的基本信息：时长 {dur} 秒。"


def _explain_point_at_object(args: Dict, head: str) -> str:
    """point_at_object 模板（旧模板名 point_and_jump 从未命中，P0-5 修正）。

    结果含 "{ts}s"，解析首个时间戳。
    """
    query = _clip(args.get("query", ""))
    target = f"「{query}」" if query else "目标物体"
    m = re.search(r"([\d.]+)\s*s", head)
    ts = m.group(1) if m else None
    if ts:
        return f"在画面里定位{target}，跳到 {ts} 秒。"
    return f"在画面里定位{target}。"


def _explain_delete_history(args: Dict, head: str) -> str:  # noqa: ARG001
    """delete_history 模板。风险前缀由 _with_risk 统一加。"""
    return "删除当前会话的历史记录。"


def _explain_scan_videos(args: Dict, head: str) -> str:
    video_dir = _clip(args.get("video_dir", ""))
    n = _count_items(head)
    if n is None:
        return f"扫描监控目录「{video_dir}」，拿到了视频清单。"
    return f"扫描监控目录「{video_dir}」，找到 {n} 个视频。"


def _explain_trigger_batch(args: Dict, head: str) -> str:  # noqa: ARG001
    video_dir = _clip(args.get("video_dir", ""))
    item = args.get("item_description")
    target = f"，盯住「{_clip(item)}」" if item else ""
    return f"把批量分析准备好：目录「{video_dir}」{target}（等你到批量页确认后才真正开始跑）。"


def _explain_summarize_hits(args: Dict, head: str) -> str:  # noqa: ARG001
    n = _count_items(head)
    if n is None:
        return "汇总了历史上所有命中的记录。"
    return f"汇总了历史上所有命中的记录，共 {n} 条。"


def _explain_generate_skill(args: Dict, head: str) -> str:  # noqa: ARG001
    text = _clip(args.get("text", ""))
    return f"根据「{text}」起草了一条新技能（走本地规则模板，不调付费模型）。"


def _explain_start_rtsp_monitor(args: Dict, head: str) -> str:  # noqa: ARG001
    url = _clip(args.get("rtsp_url", ""))
    item = args.get("item_description")
    target = f"，专门找「{_clip(item)}」" if item else ""
    return f"开始盯着实时流「{url}」{target}（在后台持续跑，发现目标会提醒你）。"


def _explain_trace_item(args: Dict, head: str) -> str:
    item = _clip(args.get("item_keyword", ""))
    n = _count_items(head)
    if n is None:
        return f"跨视频追踪「{item}」出现的时间线。"
    return f"跨视频追踪「{item}」，共找到 {n} 处踪迹。"


# ---- 内容生产（media_gen）


def _explain_make_subtitle(args: Dict, head: str) -> str:  # noqa: ARG001
    fmt = args.get("fmt") or "srt"
    return f"给视频配上字幕，输出 {str(fmt).upper()} 字幕文件。"


def _explain_make_voiceover(args: Dict, head: str) -> str:
    text = _clip(args.get("text", ""))
    mock = "mock" in head.lower() or "占位" in head
    tail = "（本机没有可用的语音合成，生成的是占位音频）" if mock else ""
    return f"把「{text}」配成语音，生成音频文件{tail}。"


def _explain_make_short_video(args: Dict, head: str) -> str:  # noqa: ARG001
    extra = []
    if args.get("subtitle_path"):
        extra.append("烧入字幕")
    if args.get("bgm_path"):
        extra.append("配背景音乐")
    tail = f"（含{'、'.join(extra)}）" if extra else ""
    return f"把视频裁成 9:16 竖屏成片{tail}。"


def _explain_create_cut_clip(args: Dict, head: str) -> str:  # noqa: ARG001
    n_seg = len(args.get("segments") or []) or len(args.get("time_ranges") or [])
    n_kw = len(args.get("keywords") or [])
    if n_seg:
        return f"按你给的 {n_seg} 段时间剪出片段。"
    if n_kw:
        return f"按关键词找到并剪出相关片段（{n_kw} 个关键词）。"
    return "按你给的条件剪出片段。"


# ---- 浏览器自动化（web_auto）


def _explain_web_open(args: Dict, head: str) -> str:  # noqa: ARG001
    return f"打开网页「{_clip(args.get('url', ''))}」。"


def _explain_web_navigate(args: Dict, head: str) -> str:  # noqa: ARG001
    return f"把浏览器跳到「{_clip(args.get('url', ''))}」。"


def _explain_web_snapshot(args: Dict, head: str) -> str:  # noqa: ARG001
    return "读取当前网页的结构（供下一步点击/填写用）。"


def _explain_web_screenshot(args: Dict, head: str) -> str:  # noqa: ARG001
    return "给当前网页截了一张图。"


def _explain_web_trigger(args: Dict, head: str) -> str:  # noqa: ARG001
    sel = _clip(args.get("selector", "目标元素"))
    idx = args.get("index")
    tail = f"（第 {idx} 个）" if idx is not None else ""
    return f"点击网页上的「{sel}」{tail}。"


def _explain_web_update(args: Dict, head: str) -> str:  # noqa: ARG001
    sel = _clip(args.get("selector", "目标输入框"))
    return f"在网页的「{sel}」里填入内容。"


def _explain_web_close(args: Dict, head: str) -> str:  # noqa: ARG001
    return "关掉了浏览器。"


# ---- CDP 调试（cdp_*）


def _explain_cdp_attach(args: Dict, head: str) -> str:  # noqa: ARG001
    return f"连上浏览器的调试目标「{_clip(args.get('target_id', ''))}」。"


def _explain_cdp_list_targets(args: Dict, head: str) -> str:  # noqa: ARG001
    n = _count_items(head)
    if n is None:
        return "列出浏览器里可以调试的页面。"
    return f"列出浏览器里可以调试的页面，共 {n} 个。"


def _explain_cdp_evaluate(args: Dict, head: str) -> str:  # noqa: ARG001
    return ("在网页里执行一段只读脚本拿数据"
            "（等同于在页面上跑任意 JS，所以后端会等你确认）。")


def _explain_cdp_eval_write(args: Dict, head: str) -> str:  # noqa: ARG001
    return ("在网页里执行一段会改动页面的脚本"
            "（等同于在页面上跑任意 JS，所以后端会等你确认）。")


def _explain_cdp_close(args: Dict, head: str) -> str:  # noqa: ARG001
    return "断开了浏览器调试连接。"


# ---------------------------------------------------------------- 分发表

#: 工具名 → 渲染函数。**32 个工具全覆盖**，由契约测试动态校验。
_TEMPLATES: Dict[str, Callable[[Dict, str], str]] = {
    # legacy（16）
    "get_video_meta": _explain_video_meta,
    "get_frame_details": _explain_get_frame,
    "delete_history": _explain_delete_history,
    "search_web": _explain_search_web,
    "search_visual": _explain_visual_search,
    "search_by_image": _explain_visual_search,
    "ocr_frame": _explain_ocr_frame,
    "highlight_cut": _explain_highlight_cut,
    "point_at_object": _explain_point_at_object,
    "search_kb": _explain_search_kb,
    "scan_videos": _explain_scan_videos,
    "trigger_batch": _explain_trigger_batch,
    "summarize_hits": _explain_summarize_hits,
    "generate_skill": _explain_generate_skill,
    "start_rtsp_monitor": _explain_start_rtsp_monitor,
    "trace_item": _explain_trace_item,
    # media_gen（4）
    "make_subtitle": _explain_make_subtitle,
    "make_voiceover": _explain_make_voiceover,
    "make_short_video": _explain_make_short_video,
    "create_cut_clip": _explain_create_cut_clip,
    # web_auto（7）
    "web_browser_open": _explain_web_open,
    "web_browser_navigate": _explain_web_navigate,
    "web_browser_snapshot": _explain_web_snapshot,
    "web_browser_screenshot": _explain_web_screenshot,
    "web_browser_trigger": _explain_web_trigger,
    "web_browser_update": _explain_web_update,
    "web_browser_close": _explain_web_close,
    # cdp（5）
    "cdp_attach": _explain_cdp_attach,
    "cdp_list_targets": _explain_cdp_list_targets,
    "cdp_evaluate": _explain_cdp_evaluate,
    "cdp_eval_write": _explain_cdp_eval_write,
    "cdp_close": _explain_cdp_close,
}

__all__ = ["explain_tool_call", "DANGEROUS_TOOLS", "WRITE_TOOLS"]
