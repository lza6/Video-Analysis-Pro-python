"""字幕全流程工具（SubtitleTool）。

输入视频 → 已有转录（faster-whisper Phase1 输出 transcript.segments，
或 ffmpeg 本地提取音频后走 provider）→ 生成 SRT / VTT。

路径：
- 免费 / 本地：已有转录段（dict/text + start/end）或本地 ffmpeg 提取音频 +
  faster-whisper（本地模型，已有）真实转录。
- 付费增强（商用 ASR / 翻译）：走 `_providers.MockProvider` 占位，绝不真调。

设计：输入 schema 兼容两种转录来源——
  1. `segments`（dict 列表，含 text/start/end）直接排版
  2. `video_path`（+ `use_local_asr`）走本地转录或 provider
输出统一 `{srt_path, vtt_path, count, engine, mock}`。
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from src.core.tools.definition import ToolDefinition

# ---------------------------------------------------------------------------
# 转录段形状（兼容 faster-whisper Segment 与 dict）
# ---------------------------------------------------------------------------


def _coerce_segments(raw: Any) -> List[Dict[str, Any]]:
    """把 faster-whisper Segment / dict / (text,start,end) 元组归一成 dict 列表。"""
    out: List[Dict[str, Any]] = []
    if raw is None:
        return out
    if isinstance(raw, dict):
        raw = [raw]
    for seg in raw:
        if hasattr(seg, "start") and hasattr(seg, "text"):
            out.append({
                "text": str(getattr(seg, "text", "")).strip(),
                "start": float(getattr(seg, "start", 0.0)),
                "end": float(getattr(seg, "end", 0.0)),
            })
        elif isinstance(seg, dict):
            text = str(seg.get("text", "")).strip()
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            if not text:
                continue
            out.append({"text": text, "start": start, "end": end})
        elif isinstance(seg, (tuple, list)) and len(seg) >= 3:
            out.append({
                "text": str(seg[0]).strip(),
                "start": float(seg[1]),
                "end": float(seg[2]),
            })
    return out


def _ts(h: float) -> str:
    """秒 → SRT 时间戳（HH:MM:SS,mmm）。"""
    h = max(0.0, h)
    ms = int(round(h * 1000))
    hh, rem = divmod(ms, 3600000)
    mm, rem = divmod(rem, 60000)
    ss, mmm = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{mmm:03d}"


def _vtt_ts(h: float) -> str:
    """秒 → VTT 时间戳（HH:MM:SS.mmm，SRT 逗号→点）。"""
    return _ts(h).replace(",", ".")


def _srt(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_ts(s['start'])} --> {_ts(s['end'])}")
        lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)


def _vtt(segments: List[Dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for i, s in enumerate(segments, 1):
        lines.append(f"{i}")
        lines.append(f"{_vtt_ts(s['start'])} --> {_vtt_ts(s['end'])}")
        lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)


def _extract_audio_ffmpeg(video_path: str, wav_path: str) -> str:
    """用本地 ffmpeg（imageio-ffmpeg 自带）从视频提取音频为 wav。免费/本地。"""
    import subprocess

    from imageio_ffmpeg import get_ffmpeg_exe

    exe = get_ffmpeg_exe()
    os.makedirs(os.path.dirname(os.path.abspath(wav_path)) or ".", exist_ok=True)
    cmd = [exe, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", wav_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 提取音频失败: {r.stderr[-400:]}")
    return wav_path


def _transcribe_local(video_path: str, wav_path: str) -> List[Dict[str, Any]]:
    """faster-whisper 本地转录（已有依赖，免费）。返回 segments dict 列表。"""
    import sys

    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        sys.exit("faster_whisper 未安装（本地 ASR 不可用）")

    wav = _extract_audio_ffmpeg(video_path, wav_path)
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _info = model.transcribe(wav, language="zh", vad_filter=True)
    out: List[Dict[str, Any]] = []
    for s in segs:
        out.append({"text": s.text.strip(), "start": s.start, "end": s.end})
    return out


def _load_transcript_json(path: str) -> List[Dict[str, Any]]:
    """读已有 transcript（Phase1 输出的 transcript.json），返回 segments。"""
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("segments", "transcript", "result"):
            if isinstance(data.get(key), list):
                return _coerce_segments(data[key])
        return _coerce_segments(data)
    return _coerce_segments(data)


def _sanitize_slug(text: str) -> str:
    """文件名安全化：非法字符 → 下划线。"""
    return re.sub(r"[^\w一-鿿-]+", "_", text.strip() or "subtitle").strip("_")


def _tokenize_text(text: str) -> set:
    """分词（中英混合）→ token 集合，供关键词 jaccard 匹配。"""
    tokens = set()
    for w in (text.lower().split()):
        w = w.strip(".,!?;:\"'()[]（）。，！？；：")
        if w:
            tokens.add(w)
    for seg in re.findall(r"[一-鿿]+", text):
        for ch in seg:
            tokens.add(ch)
    return tokens


def render_subtitle_file(
    segments: List[Dict[str, Any]],
    output_path: str,
    fmt: str = "srt",
) -> str:
    """排版 segments 为 SRT/VTT 写入文件，返回路径。

    Args:
        segments: 转录段 dict 列表（text/start/end）。
        output_path: 输出文件路径（后缀决定格式，srt/vtt）。
        fmt: 显式指定格式（srt/vtt），优先级高于后缀。
    """
    fmt = fmt.lower()
    if fmt not in ("srt", "vtt"):
        fmt = "srt"
    text = _vtt(segments) if fmt == "vtt" else _srt(segments)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


def run_subtitle(
    *,
    video_path: Optional[str] = None,
    transcript_path: Optional[str] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
    fmt: str = "srt",
    use_local_asr: bool = True,
    provider: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """字幕全流程主逻辑（同步，供 tool 回调 asyncio.to_thread 包一层）。

    优先级：显式 segments > transcript_path > video_path(+本地 ASR 或 provider)。
    """
    segs = _coerce_segments(segments)
    engine = "transcript"
    if not segs and transcript_path:
        segs = _load_transcript_json(transcript_path)
        engine = "transcript"
    if not segs and video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")
        if use_local_asr and _local_asr_available():
            wav = os.path.join(
                os.path.dirname(os.path.abspath(output_path or "subtitle.srt"))
                or ".", "_media_gen_extract.wav")
            segs = _transcribe_local(video_path, wav)
            engine = "local_whisper"
        elif provider is not None and getattr(provider, "mock", True):
            segs = provider.transcribe(video_path)
            engine = "mock"
        else:
            segs = []
            engine = "mock"

    if not segs:
        raise ValueError("无可用转录：请传 segments / transcript_path / video_path")

    out = output_path or os.path.join(
        ".", f"subtitle_{_sanitize_slug(segs[0]['text'][:12])}.{fmt}")
    if not out.lower().endswith("." + fmt):
        out = out + "." + fmt

    render_subtitle_file(segs, out, fmt=fmt)
    vtt_path = os.path.splitext(out)[0] + ".vtt"
    render_subtitle_file(segs, vtt_path, fmt="vtt")

    return {
        "srt_path": out if fmt == "srt" else out,
        "vtt_path": vtt_path,
        "count": len(segs),
        "engine": engine,
        "mock": engine == "mock",
    }


def _local_asr_available() -> bool:
    """本地 faster-whisper 是否可用（已有依赖）。"""
    try:
        __import__("faster_whisper")
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# ToolDefinition 工厂
# ---------------------------------------------------------------------------

_SUBTITLE_SCHEMA = {
    "type": "object",
    "properties": {
        "video_path": {"type": "string", "description": "视频路径（无转录时走本地 ASR）"},
        "transcript_path": {"type": "string", "description": "Phase1 transcript.json 路径"},
        "segments": {
            "type": "array",
            "items": {"type": "object"},
            "description": "转录段列表 [{text,start,end}]（优先于 transcript_path）",
        },
        "output_path": {"type": "string", "description": "输出 SRT 路径（可选，默认当前目录）"},
        "fmt": {"type": "string", "enum": ["srt", "vtt"], "description": "输出格式（默认 srt）"},
        "use_local_asr": {"type": "boolean", "description": "无转录时是否用本地 faster-whisper（默认 true）"},
    },
}


def make_subtitle_tool() -> ToolDefinition:
    """构建 SubtitleTool：字幕全流程（转录→SRT/VTT）。

    - 读操作语义：不落盘视频，只写字幕文本文件（scope_guard 视为一般写）。
    - 免费/本地路径真实执行；付费增强（商用 ASR/翻译）走 provider Mock。
    """

    async def _callback(**kwargs: Any) -> Any:
        import asyncio
        return await asyncio.to_thread(run_subtitle, **kwargs)

    return ToolDefinition(
        name="make_subtitle",
        description=(
            "字幕全流程工具：输入视频/已有转录 → 生成 SRT/VTT 字幕文件。"
            "支持 ①显式 segments（[{text,start,end}]）②transcript_path "
            "（Phase1 transcript.json）③video_path+本地 faster-whisper（免费）"
            "或 Mock provider（付费 ASR 红线不真调）。返回 srt_path/vtt_path/count。"
        ),
        execute_callback=_callback,
        input_schema=_SUBTITLE_SCHEMA,
    )
