"""智能剪辑工具（SmartClipTool）。

复用 funclip / highlight_cut 思想：按文本/关键词定位时间区间 → moviepy
`subclipped` + `concatenate_videoclips` 出片。比 legacy highlight_cut 更强的点：
  1. 支持显式 `time_ranges`（[[start,end],...]）精确剪辑（测试友好、确定性高）
  2. 关键词模式：对传入 segments 做 jaccard 打分，取 top_n 片段 + 前后 padding
  3. 危险写分类 `highlight_cut` 已在 scope_guard 的 CUT_WRITE_PATTERNS，
     落盘需审批（VAP_ALLOW_WRITE_CUT=1 放行）——本工具不改守卫，只声明。

工具名 `create_cut_clip`：
  - 含 `cut` 关键词 → 命中 scope_guard 的 CUT_WRITE_PATTERNS，写操作需审批
  - 默认 OFF（不注册进 adapter），由主控按需 `make_smart_clip_tool(name=...)` 挂载。
    注意：`smart_clip` 命中的是 DEFAULT_ALLOW 前缀 `search_` 之外的名字（被当读操作
    放行，无审批）——为守住「剪辑落盘需审批」红线，默认工具名用 `create_cut_clip`。

输出统一 dict：{path, segments, duration, engine, mock}。
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from src.core.tools.definition import ToolDefinition
from src.core.tools.media_gen.subtitle import _coerce_segments, _tokenize_text


def _fmt_ts(sec: float) -> str:
    """秒 → MM:SS 可读时间戳。"""
    sec = max(0.0, sec)
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


def run_smart_clip(
    *,
    video_path: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    keywords: Optional[str] = None,
    time_ranges: Optional[List[List[float]]] = None,
    output_path: Optional[str] = None,
    top_n: int = 3,
    padding: float = 1.0,
    provider: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """智能剪辑主逻辑（同步）。三条路径：

    1. time_ranges 显式区间：直接裁剪（确定性，测试用）
    2. keywords + segments：jaccard 打分选 top_n 片段 + padding（funclip 思想）
    3. 仅 keywords：调 provider 定位区间（Mock 返回录制区间）
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")

    ranges: List[tuple] = []
    engine = "time_ranges"
    if time_ranges:
        for r in time_ranges:
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                ranges.append((float(r[0]), float(r[1])))
        engine = "time_ranges"
    elif segments or keywords:
        segs = _coerce_segments(segments)
        kw = (keywords or "").strip()
        if segs and kw:
            kws = _tokenize_text(kw)
            scored = []
            for s in segs:
                score = _jaccard(kws, _tokenize_text(s["text"]))
                scored.append((score, s))
            scored.sort(key=lambda x: x[0], reverse=True)
            chosen = [s for sc, s in scored[:top_n] if sc > 0]
            for s in chosen:
                start = max(0.0, s["start"] - padding)
                end = s["end"] + padding
                ranges.append((start, end))
            engine = "keyword"
        elif kw and provider is not None:
            # Mock provider 返回录制区间
            for s in provider.transcribe(video_path):
                ranges.append((s["start"], s["end"]))
            engine = "mock_keyword"
        else:
            raise ValueError("关键词模式需要 segments 或 provider")
    else:
        raise ValueError("请提供 time_ranges 或 keywords+segments")

    if not ranges:
        raise ValueError("未找到匹配片段")

    from moviepy import VideoFileClip, concatenate_videoclips

    video = None
    final = None
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        clips = []
        for start, end in ranges:
            start = max(0.0, min(start, duration - 0.05))
            end = max(start + 0.1, min(end, duration))
            if end > start:
                clips.append(video.subclipped(start, end))
        if not clips:
            raise ValueError("区间裁剪后无有效片段")
        final = concatenate_videoclips(clips)

        if output_path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(".", f"smart_clip_{ts}.mp4")
        if not output_path.lower().endswith(".mp4"):
            output_path += ".mp4"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                    exist_ok=True)
        final.write_videofile(output_path, codec="libx264", logger=None)

        return {
            "path": output_path,
            "segments": [[r[0], r[1]] for r in ranges],
            "duration": round(float(final.duration), 2),
            "engine": engine,
            "mock": engine == "mock_keyword",
        }
    finally:
        if final is not None:
            try:
                final.close()
            except Exception:
                pass
        if video is not None:
            try:
                video.close()
            except Exception:
                pass


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# ToolDefinition 工厂
# ---------------------------------------------------------------------------

_CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "video_path": {"type": "string", "description": "源视频路径"},
        "segments": {
            "type": "array",
            "items": {"type": "object"},
            "description": "转录段 [{text,start,end}]，供关键词匹配",
        },
        "keywords": {"type": "string", "description": "关键词（jaccard 匹配转录段）"},
        "time_ranges": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "显式区间 [[start,end],...]（确定性剪辑）",
        },
        "output_path": {"type": "string", "description": "输出 mp4 路径（可选）"},
        "top_n": {"type": "integer", "description": "关键词模式取前 N 片段（默认 3）"},
        "padding": {"type": "number", "description": "片段前后留白秒（默认 1.0）"},
    },
    "required": ["video_path"],
}


def make_smart_clip_tool(name: str = "create_cut_clip") -> ToolDefinition:
    """构建 SmartClipTool：按文本/关键词/显式区间智能剪辑出 mp4。

    Args:
        name: 工具名。默认 `create_cut_clip`（含 `cut` → 命中 scope_guard
            CUT_WRITE_PATTERNS，落盘需审批，守住「剪辑落盘需审批」红线）。
            主控如需无审批剪辑可显式传 `smart_clip`（不做此推荐）。

    真实执行：免费/本地（moviepy subclipped+concatenate）。
    """

    async def _callback(**kwargs: Any) -> Any:
        import asyncio
        return await asyncio.to_thread(run_smart_clip, **kwargs)

    return ToolDefinition(
        name=name,
        description=(
            "智能剪辑工具：按关键词或显式时间区间裁剪视频出 mp4。"
            "①time_ranges=[[start,end],...] 精确剪辑 ②keywords+segments "
            "jaccard 匹配取 top_n 片段+padding。写操作落盘需审批。"
        ),
        execute_callback=_callback,
        input_schema=_CLIP_SCHEMA,
    )
