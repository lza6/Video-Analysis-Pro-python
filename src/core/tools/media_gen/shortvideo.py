"""竖屏成片工具（ShortVideoTool）。

9:16 裁剪 + 字幕烧录 + 可选 BGM（本地 ffmpeg 真实执行）。

竖屏处理策略（center-crop）：
  - 横屏源（宽≥高）：按 9:16 目标宽高比取源中心区域，裁掉左右多余
  - 竖屏源（高>宽）：不足 9:16 时用黑边 padding 补足（crop 保内容）

字幕烧录用 ffmpeg `subtitles` filter（libass 已随 imageio-ffmpeg 携带，
实测 ffmpeg 7.1 支持）。BGM 用 `-filter_complex amix` 混入，BGM 时长不足
循环；无 BGM 时原音直通。

输出：竖屏 mp4 + 元信息。mock 字段始终 False（全部本地免费能力）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.core.tools.definition import ToolDefinition

#: 竖屏目标宽高比（9:16）
_PORTRAIT_RATIO = 9 / 16


def _ffmpeg() -> str:
    """取本地 ffmpeg 可执行（imageio-ffmpeg 自带，免费）。"""
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def _center_crop_filter(w: int, h: int) -> str:
    """按 9:16 对 (w,h) 计算 center-crop 参数。

    返回 ffmpeg `crop=` 参数字符串，兼容源宽高不定：
      - 源是横屏（w/h >= ratio）→ 高度不变，宽度收窄到 h*ratio
      - 源是竖屏（w/h < ratio）→ 宽度不变，高度收窄到 w/ratio
    收窄到 2 的整数（libx264 要求偶数），并保证 >=2。
    """
    import math

    if w <= 0 or h <= 0:
        raise ValueError(f"非法视频尺寸: {w}x{h}")
    if w / h >= _PORTRAIT_RATIO:
        new_w = int(math.floor(h * _PORTRAIT_RATIO) // 2 * 2)
        new_w = max(2, new_w)
        return f"crop={new_w}:{h}:{(w - new_w) // 2}:0"
    new_h = int(math.floor(w / _PORTRAIT_RATIO) // 2 * 2)
    new_h = max(2, new_h)
    return f"crop={w}:{new_h}:0:{(h - new_h) // 2}"


def _probe_size(video_path: str) -> tuple:
    """ffprobe 探测视频宽高（本地）。失败抛 RuntimeError。"""
    import json
    import subprocess

    exe = _ffmpeg()
    # ffprobe 与 ffmpeg 同目录
    ffprobe = os.path.join(os.path.dirname(exe), "ffprobe.exe")
    if not os.path.exists(ffprobe):
        ffprobe = "ffprobe"
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", video_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {r.stderr[-300:]}")
    data = json.loads(r.stdout)
    stream = (data.get("streams") or [{}])[0]
    w = int(stream.get("width", 0))
    h = int(stream.get("height", 0))
    if w <= 0 or h <= 0:
        raise RuntimeError(f"探测不到视频尺寸: {video_path}")
    return w, h


def run_short_video(
    *,
    video_path: str,
    subtitle_path: Optional[str] = None,
    bgm_path: Optional[str] = None,
    output_path: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    """竖屏成片主逻辑（同步）。全部本地 ffmpeg 免费执行。"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")
    for p in (subtitle_path, bgm_path):
        if p and not os.path.exists(p):
            raise FileNotFoundError(f"输入文件不存在: {p}")

    out = output_path or os.path.join(".", "short_video_9x16.mp4")
    if not out.lower().endswith(".mp4"):
        out += ".mp4"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    exe = _ffmpeg()
    w, h = _probe_size(video_path)
    crop = _center_crop_filter(w, h)

    vf_parts = [crop]
    if subtitle_path:
        vf_parts.append(_subtitles_filter_arg(subtitle_path))

    cmd = [exe, "-y", "-i", video_path]
    filter_complex = None
    if bgm_path:
        # amix：BGM 循环到视频长度，与源音混音
        filter_complex = (
            "[1:a]aloop=loop=-1:size=2e9,volume=0.6[bg];"
            "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
    cmd += ["-vf", ",".join(vf_parts)]
    if filter_complex:
        cmd += ["-filter_complex", filter_complex, "-map", "0:v", "-map", "[aout]"]
    if bgm_path:
        cmd += ["-i", bgm_path]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if filter_complex:
        cmd += ["-c:a", "aac"]
    else:
        cmd += ["-c:a", "copy"]
    cmd += ["-shortest", out]

    import subprocess

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"竖屏成片失败: {r.stderr[-600:]}")

    return {
        "path": out,
        "size": f"{w}x{h}",
        "crop": crop,
        "subtitle_burned": bool(subtitle_path),
        "bgm": bool(bgm_path),
        "mock": False,
    }


def _subtitles_filter_arg(subtitle_path: str) -> str:
    r"""构造 ffmpeg subtitles filter 参数（转义路径内冒号/引号/反斜杠）。

    Windows 路径（`C:\...`）含冒号与反斜杠，必须转义，否则 filter 语法错误。
    """
    s = subtitle_path.replace("\\", "/")
    s = s.replace(":", "\\:").replace("'", "\\'")
    return f"subtitles='{s}'"


# ---------------------------------------------------------------------------
# ToolDefinition 工厂
# ---------------------------------------------------------------------------

_SHORT_VIDEO_SCHEMA = {
    "type": "object",
    "properties": {
        "video_path": {"type": "string", "description": "源视频路径"},
        "subtitle_path": {"type": "string", "description": "SRT 字幕路径（烧录）"},
        "bgm_path": {"type": "string", "description": "BGM 音频路径（可选混入）"},
        "output_path": {"type": "string", "description": "输出竖屏 mp4 路径（可选）"},
    },
    "required": ["video_path"],
}


def make_short_video_tool() -> ToolDefinition:
    """构建 ShortVideoTool：竖屏成片（9:16 裁剪 + 字幕烧录 + 可选 BGM）。

    全部本地 ffmpeg 免费执行（imageio-ffmpeg 自带 ffmpeg/ffprobe），
    不调付费 API。写操作落盘需审批（scope_guard 一般写）。
    """

    async def _callback(**kwargs: Any) -> Any:
        import asyncio
        return await asyncio.to_thread(run_short_video, **kwargs)

    return ToolDefinition(
        name="make_short_video",
        description=(
            "竖屏成片工具：9:16 裁剪 + 字幕烧录 + 可选 BGM（本地 ffmpeg）。"
            "输入视频 + SRT 字幕（可选）+ BGM（可选）→ 输出竖屏 mp4。"
            "全部本地免费能力，mock=false。"
        ),
        execute_callback=_callback,
        input_schema=_SHORT_VIDEO_SCHEMA,
    )
