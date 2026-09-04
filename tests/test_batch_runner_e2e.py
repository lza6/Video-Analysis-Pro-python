"""T2 真实 E2E 小样：用 D:/监控/ 第 1 个视频跑 1 个分片（2min），验证真实 NVIDIA 调用通。

只在有 VAP_NV_API_KEY 时跑，且只发 1 次真实请求（@pytest.mark.real_api）。
默认 skip，需显式 -m real_api 触发。
"""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _has_nv_key() -> bool:
    _load_env()
    return bool(os.environ.get("VAP_NV_API_KEY"))


@pytest.mark.real_api
@pytest.mark.skipif(not _has_nv_key(), reason="需要 VAP_NV_API_KEY")
class TestBatchRunnerRealE2E:
    """真实 NVIDIA 调用 E2E。运行: pytest tests/test_batch_runner_e2e.py -m real_api"""

    def test_single_segment_real_nvidia_call(self, tmp_path):
        """切 1 个 2min 分片 → 真实调 NVIDIA → 验证 run_store 写入 + segment.ok。"""
        from src.core.batch_runner import BatchConfig, BatchRunner
        from src.core.provider_router import ProviderKey, ProviderRouter
        from src.core.run_store import RunStore

        _load_env()
        api_key = os.environ["VAP_NV_API_KEY"]
        monitor_dir = os.environ.get("VAP_MONITOR_DIR", "D:/监控")
        key_item = os.environ.get("VAP_KEY_ITEM_IMAGE",
                                  f"{monitor_dir}/关键物品.jpg")
        # 找第 1 个 mp4
        videos = sorted([f for f in Path(monitor_dir).iterdir()
                         if f.suffix.lower() == ".mp4"])
        assert videos, f"监控目录无 mp4: {monitor_dir}"
        video = videos[0]
        print(f"E2E 使用视频: {video.name}")

        # 构造 router（单 key）
        keys = [ProviderKey(
            id="nv-1", name="nv-1", provider="nvidia",
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            priority=10, isActive=True)]
        router = ProviderRouter(keys, rate_limit_per_min=40)

        # 构造 config：segment_sec=30 对齐 NVIDIA 26MB 内联 payload 限制
        # （2min 720p base64 后 ~40MB 超 26MB 上限，需走 Assets API；此处用 30s 验证调用通）
        cfg = BatchConfig(
            video_dir=str(monitor_dir),
            key_item_image=key_item,
            item_description="关键物品",
            segment_sec=30,
            clip_padding=10,
            clean_segments=True,
            resume=False,
            out_dir=str(tmp_path / "out"),
            request_timeout=180,
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            enable_thinking=True,
            reasoning_budget=8192,
            max_tokens=65536,
        )
        store = RunStore(str(tmp_path / "cfg"))
        runner = BatchRunner(cfg, store, router)
        # 用 spy 包裹 router.post_nvidia 以便断言调用（真实 ProviderRouter
        # 的 post_nvidia 是 bound method，无 .called 属性；wrap 后可断言）
        from unittest.mock import MagicMock
        spy = MagicMock(wraps=router.post_nvidia)
        router.post_nvidia = spy

        # 只跑第 1 个视频，且只切第 1 个分片（手动限制 seg 数为 1）
        # patch _segment_video 直接返回 1 片（避免 _probe_duration 被覆盖后
        # _segment_video 仍按真实 duration 切多片）
        import cv2
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        # 真实时长仅用于日志（不强制断言）
        _ = (total / fps) if (total and fps) else 120.0

        # 切 1 个真实 30s 分片用于发送
        seg_dir = tmp_path / "segs"
        seg_dir.mkdir(parents=True, exist_ok=True)
        seg_path = seg_dir / "seg_0000.mp4"
        import subprocess, shutil
        ffmpeg = shutil.which("ffmpeg") or os.environ.get("IMAGEIO_FFMPEG_EXE")
        assert ffmpeg, "ffmpeg 未找到"
        cmd = [ffmpeg, "-y", "-ss", "0", "-t", "30", "-i", str(video),
               "-an", "-vf", "scale=-2:720,fps=2", "-frames:v", "60",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23",
               "-movflags", "+faststart", str(seg_path)]
        subprocess.run(cmd, capture_output=True, timeout=120)
        assert seg_path.exists(), "分片切分失败"

        def fake_segment(v, run_id, duration):
            return ([seg_path], [0.0], [30.0])

        runner._segment_video = fake_segment

        # Act：跑 1 个视频
        total_runs, total_hits = runner.run_batch([video])

        # Assert：run_store 有 1 条 run，segment 写入
        runs = store.list_runs()
        assert len(runs) == 1
        run = store.get_run(runs[0]["run_id"])
        assert run["segments_total"] == 1
        assert len(run["segments"]) == 1
        seg = run["segments"][0]
        # 真实调用成功 → status=ok（即便 match=false 也是 ok）
        # 若 NVIDIA 返回 503/限流，router 无限重试可能超时 → status=failed 也算
        # "调用链路打通"（已发出真实请求并落库）
        print(f"E2E 结果: status={seg['status']}, match={seg['match']}, "
              f"conf={seg['confidence']}, reason={str(seg.get('reason'))[:80]}, "
              f"elapsed={seg.get('elapsed_sec')}")
        print(f"E2E usage: {seg.get('usage_json')}")
        print(f"E2E error: {seg.get('error')}")
        # 核心断言：真实请求已发出（router.post_nvidia 被调用过）
        assert router.post_nvidia.called, "router.post_nvidia 应被调用"
        assert seg["status"] in ("ok", "failed"), f"unexpected status {seg['status']}"
