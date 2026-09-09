"""P2-1 内容生产工具（视频生成/剪辑 Agent 工具集）测试。

覆盖：
  1. SubtitleTool 真实生成 SRT（ffmpeg 合成 1s 测试视频 + 显式 segments）
  2. SmartClipTool 按区间裁剪真实出 mp4（moviepy subclipped+concatenate）
  3. ShortVideoTool 竖屏裁剪+字幕烧录真实出 mp4（本地 ffmpeg）
  4. VoiceoverTool Mock 占位 wav（无 pyttsx3/edge-tts 环境 → mock=true）
  5. Provider 选择：默认 mock；非 mock 无 key 回落 mock（付费红线）
  6. 注册函数在临时 registry 可注册、可 disposer 卸载

本地能力探测：ffmpeg（imageio-ffmpeg 自带）与 moviepy 已在 venv 安装。
若真实路径不可用（如 moviepy 缺失）自动跳过真实断言，走 mock 说明——
本测试真实路径应全绿。
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import pytest

from src.core.tools import ToolRegistry
from src.core.tools.adapter import register_media_gen_tools
from src.core.tools.media_gen import build_provider
from src.core.tools.media_gen._providers import (
    ENV_ASR_PROVIDER,
    ENV_TTS_PROVIDER,
    ENV_VIDEO_PROVIDER,
    MockProvider,
)
from src.core.tools.media_gen.subtitle import render_subtitle_file, run_subtitle
from src.core.tools.media_gen.clip import run_smart_clip
from src.core.tools.media_gen.voiceover import run_voiceover, _tts_available
from src.core.tools.media_gen.shortvideo import (
    _center_crop_filter,
    _probe_size,
    run_short_video,
)

# ---------------------------------------------------------------------------
# 本地能力探测
# ---------------------------------------------------------------------------


def _ffmpeg() -> str:
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def _moviepy_available() -> bool:
    try:
        __import__("moviepy")
        return True
    except ImportError:
        return False


def _make_test_video(tmp: Path, seconds: float = 1.0) -> Path:
    """ffmpeg 合成 1s 测试视频（黑屏+正弦音，libx264+aac）。"""
    out = tmp / "src.mp4"
    r = subprocess.run(
        [_ffmpeg(), "-y",
         "-f", "lavfi", "-i", f"color=c=black:s=640x360:d={seconds}:r=25",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, f"ffmpeg 合成测试视频失败: {r.stderr[-300:]}"
    return out


# ---------------------------------------------------------------------------
# 1. SubtitleTool
# ---------------------------------------------------------------------------


def test_render_srt_vtt(tmp_path: Path) -> None:
    segs = [{"text": "hello world", "start": 0.0, "end": 1.0},
            {"text": "second line", "start": 1.0, "end": 2.0}]
    srt = tmp_path / "a.srt"
    vtt = tmp_path / "a.vtt"
    p = render_subtitle_file(segs, str(srt), fmt="srt")
    assert p == str(srt)
    assert srt.exists()
    assert srt.read_text(encoding="utf-8").startswith("1\n00:00:00,000 --> 00:00:01,000\nhello world")
    render_subtitle_file(segs, str(vtt), fmt="vtt")
    assert vtt.read_text(encoding="utf-8").startswith("WEBVTT")


def test_run_subtitle_segments_writes_srt_vtt(tmp_path: Path) -> None:
    segs = [{"text": "测试字幕", "start": 0.0, "end": 1.0}]
    out = tmp_path / "sub.srt"
    res = run_subtitle(segments=segs, output_path=str(out))
    assert res["count"] == 1
    assert res["engine"] == "transcript"
    assert res["mock"] is False
    assert out.exists()
    assert Path(res["vtt_path"]).exists()


def test_run_subtitle_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_subtitle(output_path=str(tmp_path / "x.srt"))


# ---------------------------------------------------------------------------
# 2. SmartClipTool（真实 moviepy）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _moviepy_available(), reason="moviepy 不可用")
def test_smart_clip_time_ranges_real(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path, seconds=2.0)
    out = tmp_path / "clip.mp4"
    res = run_smart_clip(video_path=str(video), time_ranges=[[0.0, 0.5]],
                         output_path=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert res["engine"] == "time_ranges"
    assert res["mock"] is False
    assert 0.3 <= res["duration"] <= 0.8


@pytest.mark.skipif(not _moviepy_available(), reason="moviepy 不可用")
def test_smart_clip_keyword_real(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path, seconds=1.5)
    segs = [{"text": "hello world", "start": 0.0, "end": 0.5},
            {"text": "foo bar", "start": 0.6, "end": 1.0}]
    out = tmp_path / "clip_kw.mp4"
    res = run_smart_clip(video_path=str(video), segments=segs,
                         keywords="hello", output_path=str(out), padding=0.0)
    assert out.exists()
    assert res["engine"] == "keyword"
    assert res["segments"] == [[0.0, 0.5]]


def test_smart_clip_keyword_mock_provider(tmp_path: Path) -> None:
    """无 moviepy 时走 mock provider 区间（不真调）。"""
    video = _make_test_video(tmp_path, seconds=1.0)
    mp = MockProvider(kind="asr", duration_sec=0.5)
    res = run_smart_clip(video_path=str(video), keywords="x",
                         provider=mp, output_path=str(tmp_path / "c.mp4"))
    assert res["engine"] == "mock_keyword"
    assert res["mock"] is True


def test_smart_clip_bad_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_smart_clip(video_path=str(tmp_path / "nope.mp4"), time_ranges=[[0, 1]])
    # 视频存在但既无 time_ranges 也无 keywords+segments/provider → ValueError
    video = _make_test_video(tmp_path, seconds=1.0)
    with pytest.raises(ValueError):
        run_smart_clip(video_path=str(video))

# ---------------------------------------------------------------------------
# 3. ShortVideoTool（本地 ffmpeg 真实）
# ---------------------------------------------------------------------------


def test_center_crop_filter() -> None:
    # 横屏 640x360：宽高比 1.78 > 0.5625 → 高度不变，宽收窄到 202（360*9/16=202.5→floor 偶数 202）
    f = _center_crop_filter(640, 360)
    assert f.startswith("crop=202:360:")
    # 竖屏 360x640：宽高比 0.5625 → 宽不变，高收窄到 634（360*16/9=640→640? 见下）
    f2 = _center_crop_filter(360, 640)
    assert f2.startswith("crop=360:")


def test_probe_size(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path)
    w, h = _probe_size(str(video))
    assert (w, h) == (640, 360)


def test_short_video_real(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path, seconds=1.0)
    segs = [{"text": "竖屏测试", "start": 0.0, "end": 1.0}]
    srt = tmp_path / "sub.srt"
    render_subtitle_file(segs, str(srt), fmt="srt")
    out = tmp_path / "portrait.mp4"
    res = run_short_video(video_path=str(video), subtitle_path=str(srt),
                          output_path=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert res["subtitle_burned"] is True
    assert res["mock"] is False


def test_short_video_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_short_video(video_path=str(tmp_path / "nope.mp4"))


# ---------------------------------------------------------------------------
# 4. VoiceoverTool（Mock 占位）
# ---------------------------------------------------------------------------


def test_voiceover_mock_placeholder(tmp_path: Path) -> None:
    out = tmp_path / "vo.wav"
    res = run_voiceover(text="测试配音", output_path=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert res["mock"] is True  # 无本地 TTS 环境 → mock 占位
    with wave.open(str(out), "rb") as w:
        assert w.getnframes() > 0


def test_voiceover_empty_text_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_voiceover(text="  ", output_path=str(tmp_path / "v.wav"))


@pytest.mark.skipif(_tts_available(), reason="本环境已有本地 TTS，跳过 mock 路径")
def test_voiceover_engine_mock_when_no_tts(tmp_path: Path) -> None:
    res = run_voiceover(text="x", output_path=str(tmp_path / "v.wav"))
    assert res["engine"] == "mock"


# ---------------------------------------------------------------------------
# 5. Provider 选择
# ---------------------------------------------------------------------------


def test_provider_default_mock() -> None:
    p, reason = build_provider("asr", {})
    assert isinstance(p, MockProvider)
    assert reason is None
    p2, _ = build_provider("tts", {})
    assert p2.mock is True


def test_provider_nonmock_falls_back_to_mock() -> None:
    env = {ENV_ASR_PROVIDER: "xunfei_asr"}  # 缺 key
    p, reason = build_provider("asr", env)
    assert isinstance(p, MockProvider)
    assert "回落 mock" in (reason or "")

    env2 = {ENV_VIDEO_PROVIDER: "veo_gen", "VAP_VEO_API_KEY": "fake"}
    p2, reason2 = build_provider("video", env2)
    assert isinstance(p2, MockProvider)
    assert reason2 is not None


def test_provider_invalid_falls_back_mock() -> None:
    p, _ = build_provider("asr", {ENV_ASR_PROVIDER: "bogus"})
    assert isinstance(p, MockProvider)


def test_provider_tts_system_no_key_requires() -> None:
    # system_tts 不在 _PROVIDER_KEY_ENV：无 stub 实现 → 回落 mock 并说明
    p, reason = build_provider("tts", {ENV_TTS_PROVIDER: "system_tts"})
    assert isinstance(p, MockProvider)
    assert reason is not None


# ---------------------------------------------------------------------------
# 6. 注册函数：临时 registry 可注册 + disposer 卸载
# ---------------------------------------------------------------------------


def test_register_media_gen_tools_registers_and_disposes() -> None:
    reg = ToolRegistry()
    disposers = register_media_gen_tools(reg)
    names = set(reg.list_names())
    assert {"make_subtitle", "create_cut_clip", "make_voiceover",
            "make_short_video"} <= names
    assert len(disposers) == 4

    # disposer 卸载
    for d in disposers:
        d()
    assert "create_cut_clip" not in reg.list_names()


def test_register_media_gen_tools_schemas() -> None:
    reg = ToolRegistry()
    register_media_gen_tools(reg)
    schemas = reg.schemas()
    names = {s["function"]["name"] for s in schemas}
    assert "make_subtitle" in names and "create_cut_clip" in names
    assert all(s["type"] == "function" for s in schemas)


def test_scope_guard_classifies_clip_as_cut_write() -> None:
    """create_cut_clip 必须命中 CUT_WRITE_PATTERNS（落盘需审批红线）。"""
    from src.core.tools import ToolCall
    from src.core.tools.scope_guard import ScopeGuard

    for name in ("create_cut_clip", "save_cut_video", "generate_cut_clip"):
        level, _ = ScopeGuard().decision(ToolCall(name=name, args={}))
        assert level == "ask", f"{name} 应触发审批，实际 {level}"


def test_tool_execute_through_registry(tmp_path: Path) -> None:
    """四工具注册后经 registry.execute 真实调用（无审批无守卫）。"""
    import asyncio

    reg = ToolRegistry()
    register_media_gen_tools(reg)

    async def _call_subtitle() -> None:
        out = tmp_path / "t.srt"
        from src.core.tools import ToolCall
        r = await reg.execute(ToolCall(
            name="make_subtitle",
            args={"segments": [{"text": "ok", "start": 0.0, "end": 1.0}],
                  "output_path": str(out)}))
        assert r.error is None, r.error
        assert out.exists()

    asyncio.run(_call_subtitle())


# ---------------------------------------------------------------------------
# smoke：全链路（subtitle → clip → voiceover → shortvideo 串起来）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _moviepy_available(), reason="moviepy 不可用")
def test_end_to_end_media_pipeline(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path, seconds=2.0)
    segs = [{"text": "智能剪辑", "start": 0.0, "end": 0.6},
            {"text": "竖屏成片", "start": 0.6, "end": 1.2}]

    # 1. 字幕
    srt = tmp_path / "sub.srt"
    run_subtitle(segments=segs, output_path=str(srt))

    # 2. 剪辑
    clip = tmp_path / "clip.mp4"
    run_smart_clip(video_path=str(video), segments=segs, keywords="竖屏",
                   output_path=str(clip), padding=0.0)

    # 3. 配音（mock 占位）
    vo = tmp_path / "vo.wav"
    vo_res = run_voiceover(text="内容生产工具集", output_path=str(vo))
    assert vo_res["mock"] is True

    # 4. 竖屏成片（用原视频 + 字幕烧录）
    portrait = tmp_path / "portrait.mp4"
    run_short_video(video_path=str(video), subtitle_path=str(srt),
                    output_path=str(portrait))

    assert all(p.exists() for p in (srt, clip, vo, portrait))
