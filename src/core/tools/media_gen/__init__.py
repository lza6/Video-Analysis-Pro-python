"""内容生产工具集（P2-1 视频生成/剪辑 Agent 工具）。

在既有三阶段流水线（Phase1 抽帧/转录/检测 → Phase2 LLM → Phase3 媒体生成）之上，
提供四类"内容生产"工具，全部注册进 adapter 的工具面供 Agent 调用：

- subtitle  字幕全流程：输入视频 → 已有转录或本地 ASR → SRT/VTT
- clip      智能剪辑：按文本/关键词/显式区间定位 → moviepy subclipped 出片
- voiceover 配音：本地 TTS（若有）真实执行，否则 Mock 占位 wav
- shortvideo 竖屏成片：9:16 裁剪 + 字幕烧录 + 可选 BGM（本地 ffmpeg 真实执行）

付费红线：一切真实付费 API（商用 ASR/TTS/视频生成）只做 Mock 协议，不真调。
免费能力（ffmpeg/moviepy 本地、已有 faster-whisper 转录、系统 TTS 若有）可真实执行。
"""
from __future__ import annotations

from src.core.tools.media_gen._providers import (
    ASRProvider,
    MockProvider,
    TTSProvider,
    VideoGenProvider,
    build_provider,
)

__all__ = [
    "ASRProvider",
    "TTSProvider",
    "VideoGenProvider",
    "MockProvider",
    "build_provider",
]