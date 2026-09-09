"""媒体工具 Provider 协议与 Mock 实现。

付费红线：一切真实付费 API（商用 ASR / TTS / 视频生成）在本模块只做 **Mock 协议**，
绝不真调。免费能力（ffmpeg / moviepy 本地、已有 faster-whisper 转录、系统 TTS 若有）
在工具模块内真实执行，本模块只负责 provider 选择与占位产物。

Provider 选择：`build_provider(kind, env)` 读
  VAP_MEDIA_ASR_PROVIDER / VAP_MEDIA_TTS_PROVIDER / VAP_MEDIA_VIDEO_PROVIDER
（默认 `mock`）。非 mock 值（如真实商用 provider 名）时**没有 key 一律回落 mock**
并返回 `(provider, reason)` 的 reason 注明降级原因。真实 provider 的接入留 stub：
函数签名 + 文档，不接真实 key、不发真实请求。
"""
from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

# ---------------------------------------------------------------------------
# 环境变量
# ---------------------------------------------------------------------------

ENV_ASR_PROVIDER = "VAP_MEDIA_ASR_PROVIDER"
ENV_TTS_PROVIDER = "VAP_MEDIA_TTS_PROVIDER"
ENV_VIDEO_PROVIDER = "VAP_MEDIA_VIDEO_PROVIDER"

#: 有效 provider 值。真实商用 provider（xunfei/bilibili/runway/veo 等）预留——
#: 无 key 时 build_provider 一律回落 mock，不真调。
VALID_PROVIDERS = ("mock", "xunfei_asr", "bilibili_asr", "system_tts",
                   "azure_tts", "runway_gen", "veo_gen")

#: 需要外部 key 的 provider → 对应环境变量名（用于探测缺 key 回落 mock）
_PROVIDER_KEY_ENV: Dict[str, str] = {
    "xunfei_asr": "VAP_XUNFEI_APP_ID",
    "bilibili_asr": "VAP_BILIBILI_ASR_KEY",
    "azure_tts": "VAP_AZURE_SPEECH_KEY",
    "runway_gen": "VAP_RUNWAY_API_KEY",
    "veo_gen": "VAP_VEO_API_KEY",
}


def _provider_from_env(kind: str, env: Dict[str, str]) -> str:
    """从 env 取 provider 名（非法值回退 mock）。kind ∈ asr/tts/video。"""
    key = {"asr": ENV_ASR_PROVIDER, "tts": ENV_TTS_PROVIDER,
           "video": ENV_VIDEO_PROVIDER}[kind]
    val = (env.get(key) or "").strip().lower()
    if val in VALID_PROVIDERS:
        return val
    return "mock"


@dataclass(frozen=True)
class ProviderResult:
    """provider 选择结果：实例 + 降级原因（None=正常选择）。"""

    provider: "MockProvider"
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class ASRProvider(Protocol):
    """语音识别（字幕转录）provider 协议。

    免费路径（真实执行）：输入视频 → 提取音频 → faster-whisper 本地转录，
    产出 `list[Segment]`（text / start / end）。
    付费路径（Mock）：`transcribe(audio_path)` 返回录制好的占位字幕，
    `mock=True` 标记，绝不真调商用 ASR API。
    """

    mock: bool = False

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]: ...


class TTSProvider(Protocol):
    """文本转语音 provider 协议。

    免费路径（真实执行）：本地 TTS（pyttsx3 / edge-tts）合成 wav。
    付费路径（Mock）：`synthesize(text, wav_path)` 生成占位静音/正弦 wav，
    `mock=True` 标记，绝不真调商用 TTS API。
    """

    mock: bool = False

    def synthesize(self, text: str, wav_path: str) -> str: ...


class VideoGenProvider(Protocol):
    """视频生成 provider 协议。

    免费路径（真实执行）：moviepy / ffmpeg 本地合成。
    付费路径（Mock）：`generate(prompt, out_path)` 生成占位黑屏短视频，
    `mock=True` 标记，绝不真调商用视频生成 API。
    """

    mock: bool = False

    def generate(self, prompt: str, out_path: str) -> str: ...


# ---------------------------------------------------------------------------
# Mock 实现
# ---------------------------------------------------------------------------


class MockProvider:
    """占位 provider：生成可复用的录制/占位产物，标注 mock 不真调。

    - asr.transcribe   → 返回录制字幕（"这是 mock ASR 转录"）
    - tts.synthesize   → 生成占位 wav（PCM16 单声道 16kHz，正弦音）
    - video.generate   → 生成占位黑屏 mp4（调用方传时长）
    """

    mock = True

    def __init__(
        self,
        *,
        kind: str = "mock",
        duration_sec: float = 1.0,
        placeholder_text: str = "这是 mock ASR 转录",
    ) -> None:
        self.kind = kind
        self.duration_sec = duration_sec
        self.placeholder_text = placeholder_text

    # ---- ASR ----
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """Mock ASR：返回录制字幕，不读音频文件。"""
        dur = max(0.5, self.duration_sec)
        return [
            {"text": self.placeholder_text, "start": 0.0, "end": dur},
            {"text": "第二句 mock 字幕", "start": dur, "end": dur + 1.0},
        ]

    # ---- TTS ----
    def synthesize(self, text: str, wav_path: str) -> str:
        """Mock TTS：写占位 wav（正弦音），标注 mock 不真调。"""
        return _write_placeholder_wav(wav_path, duration_sec=self.duration_sec)

    # ---- Video ----
    def generate(self, prompt: str, out_path: str) -> str:
        """Mock 视频生成：用 ffmpeg 合成占位黑屏短视频（本地，免费）。"""
        return _write_placeholder_video(
            out_path, duration_sec=self.duration_sec, width=640, height=360)


def _write_placeholder_wav(wav_path: str, *, duration_sec: float = 1.0) -> str:
    """写一个 PCM16 单声道 16kHz 的正弦音占位 wav（本地 stdlib，无付费）。

    实际为可播放音频（正弦音），方便 Agent 直接当素材用；是否真 TTS 由
    mock 标记说明，不冒充真人语音。
    """
    import math
    import struct

    rate = 16000
    n = int(rate * max(0.1, duration_sec))
    os.makedirs(os.path.dirname(os.path.abspath(wav_path)) or ".", exist_ok=True)
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            v = int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / rate))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))
    return wav_path


def _write_placeholder_video(
    out_path: str, *, duration_sec: float = 1.0, width: int = 640, height: int = 360
) -> str:
    """用本地 ffmpeg（imageio-ffmpeg 自带）合成占位黑屏短视频。

    免费/本地路径，无付费 API。ffmpeg 不可用时抛 RuntimeError。
    """
    import subprocess

    from imageio_ffmpeg import get_ffmpeg_exe

    exe = get_ffmpeg_exe()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cmd = [
        exe, "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration_sec}:r=25",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_sec}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 占位视频合成失败: {r.stderr[-400:]}")
    return out_path


# ---------------------------------------------------------------------------
# Provider 选择
# ---------------------------------------------------------------------------


def build_provider(
    kind: str, env: Optional[Dict[str, str]] = None
) -> Tuple[MockProvider, Optional[str]]:
    """按环境变量选择 provider，返回 `(provider, reason)`。

    Args:
        kind: "asr" / "tts" / "video"。
        env: 环境变量字典（None = os.environ）。便于测试注入。

    Returns:
        (provider, reason)。reason=None 表示按配置正常选择（当前只有 mock 可用）；
        reason 非 None 表示非 mock 配置但缺 key / 不可用，已回落 mock（付费红线）。
    """
    env = os.environ if env is None else env
    name = _provider_from_env(kind, env)
    if name == "mock":
        return MockProvider(kind="mock"), None

    # 非 mock provider：缺 key 一律回落 mock（不真调付费 API）
    key_env = _PROVIDER_KEY_ENV.get(name)
    if key_env and not (env.get(key_env) or "").strip():
        return MockProvider(kind="mock"), (
            f"{name} 配置为 {name} 但缺 {key_env}，已回落 mock（付费红线）")
    # 显式设置的 provider 无 stub 实现：回退 mock 并说明
    return MockProvider(kind="mock"), (
        f"{name} 尚无 stub 实现，回落 mock（付费红线）")


# ---------------------------------------------------------------------------
# 真实 provider stub（签名 + 文档，不接真实 key，不真调）
# ---------------------------------------------------------------------------


class XunfeiASRProvider:
    """讯飞 ASR stub（付费）。接入需：app_id + api_key + api_secret。

    当前不实现：需真实付费调用。build_provider 缺 key 时自动回落 mock。
    """

    mock = False

    def __init__(self, app_id: str, api_key: str, api_secret: str) -> None:
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError("讯飞 ASR 为付费 API，未接入（Mock 红线）")


class AzureTTSProvider:
    """Azure TTS stub（付费）。接入需：speech_key + region。

    当前不实现：需真实付费调用。build_provider 缺 key 时自动回落 mock。
    """

    mock = False

    def __init__(self, speech_key: str, region: str) -> None:
        self.speech_key = speech_key
        self.region = region

    def synthesize(self, text: str, wav_path: str) -> str:  # pragma: no cover
        raise NotImplementedError("Azure TTS 为付费 API，未接入（Mock 红线）")


class RunwayVideoGenProvider:
    """Runway 视频生成 stub（付费）。接入需：api_key。

    当前不实现：需真实付费调用。build_provider 缺 key 时自动回落 mock。
    """

    mock = False

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def generate(self, prompt: str, out_path: str) -> str:  # pragma: no cover
        raise NotImplementedError("Runway 为付费 API，未接入（Mock 红线）")


# 统一 Protocol 类型别名（供 build_provider 返回 / 测试断言用）
ASR = Union[ASRProvider, MockProvider]
TTS = Union[TTSProvider, MockProvider]
VIDEO = Union[VideoGenProvider, MockProvider]

__all__ = [
    "ASRProvider",
    "TTSProvider",
    "VideoGenProvider",
    "MockProvider",
    "ProviderResult",
    "build_provider",
    "_provider_from_env",
    "ENV_ASR_PROVIDER",
    "ENV_TTS_PROVIDER",
    "ENV_VIDEO_PROVIDER",
    "VALID_PROVIDERS",
]