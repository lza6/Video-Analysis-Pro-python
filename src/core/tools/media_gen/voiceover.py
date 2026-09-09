"""配音工具（VoiceoverTool）。

本地 TTS（若有 pyttsx3 / edge-tts 且已装）真实执行；否则 Mock 生成占位 wav
并在返回中标注 `mock=True`。绝不真调付费 TTS API（付费红线）。

输出：wav 文件 + 时长 + mock 标记。返回 dict 统一形状：
  {wav_path, duration_sec, engine, mock}
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.core.tools.definition import ToolDefinition
from src.core.tools.media_gen._providers import MockProvider

# ---------------------------------------------------------------------------
# 本地 TTS 探测与执行
# ---------------------------------------------------------------------------


def _tts_available() -> bool:
    """本地 TTS 是否可用（pyttsx3 或 edge-tts 已安装）。"""
    for mod in ("pyttsx3", "edge_tts"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def _system_tts(text: str, wav_path: str) -> str:
    """用本地 TTS 合成 wav（真实执行）。pyttsx3 优先，edge-tts 次之。"""
    os.makedirs(os.path.dirname(os.path.abspath(wav_path)) or ".", exist_ok=True)
    try:
        import pyttsx3

        engine = pyttsx3.init()
        try:
            engine.save_to_file(text, wav_path)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass
        return wav_path
    except ImportError:
        pass

    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("无本地 TTS 可用（pyttsx3/edge-tts 均未安装）")

    import asyncio

    async def _synth() -> None:
        tts = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
        await tts.save(wav_path)

    asyncio.run(_synth())
    return wav_path


def run_voiceover(
    *,
    text: str,
    output_path: Optional[str] = None,
    provider: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """配音主逻辑（同步）。

    优先级：
      1. 显式 provider（非 mock）→ 走 provider.synthesize（付费 stub，缺实现抛错）
      2. 本地 TTS 可用 → _system_tts 真实执行
      3. 否则 → MockProvider 生成占位 wav，标注 mock
    """
    if not text or not text.strip():
        raise ValueError("text 不能为空")

    out = output_path or os.path.join(".", "voiceover_placeholder.wav")
    if not out.lower().endswith(".wav"):
        out += ".wav"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    if provider is not None and not getattr(provider, "mock", True):
        path = provider.synthesize(text, out)
        return {"wav_path": path, "duration_sec": _wav_duration(path),
                "engine": "provider", "mock": False}

    if _tts_available():
        path = _system_tts(text, out)
        return {"wav_path": path, "duration_sec": _wav_duration(path),
                "engine": "local_tts", "mock": False}

    # Mock 占位：生成正弦音 wav，标注 mock（不冒充真人语音）
    mp = MockProvider(kind="tts", duration_sec=1.0)
    path = mp.synthesize(text, out)
    return {"wav_path": path, "duration_sec": _wav_duration(path),
            "engine": "mock", "mock": True}


def _wav_duration(wav_path: str) -> float:
    """读 wav 时长（秒）。失败返回 0.0。"""
    import wave

    try:
        with wave.open(wav_path, "rb") as w:
            return w.getnframes() / max(1, w.getframerate())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# ToolDefinition 工厂
# ---------------------------------------------------------------------------

_VOICEOVER_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "要配音的文本"},
        "output_path": {"type": "string", "description": "输出 wav 路径（可选）"},
    },
    "required": ["text"],
}


def make_voiceover_tool() -> ToolDefinition:
    """构建 VoiceoverTool：文本 → wav 配音。

    - 本地 TTS（pyttsx3/edge-tts）可用 → 真实执行
    - 不可用 → Mock 占位 wav（正弦音），标注 mock 不冒充真人
    - 付费 TTS API 一律不真调（付费红线）
    """

    async def _callback(**kwargs: Any) -> Any:
        import asyncio
        return await asyncio.to_thread(run_voiceover, **kwargs)

    return ToolDefinition(
        name="make_voiceover",
        description=(
            "配音工具：文本 → wav 配音文件。本地 TTS（pyttsx3/edge-tts）可用时"
            "真实合成；否则生成 Mock 占位 wav（返回 mock=true）。付费 TTS 不真调。"
        ),
        execute_callback=_callback,
        input_schema=_VOICEOVER_SCHEMA,
    )
