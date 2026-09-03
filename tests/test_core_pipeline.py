"""VideoProcessor 真实抽帧 / kb_indexer / ConfigManager（合成视频，tmp_path 隔离）。"""
import cv2
import numpy as np
import pytest

from src.core.logic import VideoProcessor, Frame, get_frame_metrics, get_unique_filepath
from src.core.history_manager import HistoryManager


@pytest.fixture
def sample_video(tmp_path):
    """生成 3 秒 10fps 的合成测试视频，两段颜色不同的场景。"""
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(30):
        color = (255, 0, 0) if i < 15 else (0, 255, 0)
        frame = np.full((64, 64, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class TestVideoProcessor:
    def test_extract_keyframes_basic(self, sample_video, tmp_path):
        proc = VideoProcessor(sample_video, tmp_path / "out")
        frames = proc.extract_keyframes(density=0.5)
        assert len(frames) >= 5
        assert frames[0].timestamp <= frames[-1].timestamp
        assert all(f.path.exists() for f in frames)

    def test_extract_keyframes_invalid_video(self, tmp_path):
        proc = VideoProcessor(tmp_path / "nonexistent.mp4", tmp_path / "out")
        assert proc.extract_keyframes(density=0.5) == []

    def test_frame_metrics(self, sample_video, tmp_path):
        proc = VideoProcessor(sample_video, tmp_path / "out")
        frames = proc.extract_keyframes(density=0.3)
        m = frames[0].metrics
        assert 0 <= m["brightness"] <= 255
        assert m["sharpness"] >= 0


class TestFrameUtils:
    def test_get_frame_metrics_synthetic(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        m = get_frame_metrics(img)
        assert m["brightness"] == pytest.approx(128, abs=5)

    def test_get_unique_filepath_collision(self, tmp_path):
        existing = tmp_path / "a.jpg"
        existing.write_bytes(b"x")
        result = get_unique_filepath(tmp_path, "a.jpg")
        assert result != existing
        assert result.suffix == ".jpg"


class TestKBIndexer:
    def test_index_frames_with_fake_embedder(self, sample_video, tmp_path, monkeypatch):
        from src.core.kb_indexer import index_frames
        import src.core.kb_indexer as kb

        class FakeEmbedder:
            def encode(self, paths, convert_to_tensor=False, **kw):
                out = []
                for _ in paths:
                    v = np.zeros(384, dtype=np.float32)
                    v[3] = 1.0
                    out.append(v)
                return out

        monkeypatch.setattr(kb, "get_embedder", lambda: FakeEmbedder())

        proc = VideoProcessor(sample_video, tmp_path / "out")
        frames = proc.extract_keyframes(density=0.2)
        hm = HistoryManager(str(tmp_path / "cfg"))
        sid = hm.add_session(sample_video, str(tmp_path / "out"))
        indexed = index_frames(hm, sid, "sample.mp4", str(sample_video), frames)
        assert indexed == len(frames)
        assert hm.kb_count() == len(frames)

    def test_index_frames_skips_when_no_embedder(self, sample_video, tmp_path, monkeypatch):
        from src.core.kb_indexer import index_frames
        import src.core.kb_indexer as kb
        monkeypatch.setattr(kb, "get_embedder", lambda: None)

        proc = VideoProcessor(sample_video, tmp_path / "out")
        frames = proc.extract_keyframes(density=0.2)
        hm = HistoryManager(str(tmp_path / "cfg"))
        assert index_frames(hm, "s", "v", str(sample_video), frames) == 0


class TestConfigManager:
    def test_default_config_created(self, tmp_path, monkeypatch):
        from src.utils.config_manager import ConfigurationManager
        monkeypatch.setattr("src.utils.config_manager.CONFIG_DIR", str(tmp_path))
        cm = ConfigurationManager()
        # config_manager 用模块级常量拼路径，这里通过 monkeypatch 无法完全隔离；
        # 退而验证默认返回结构
        prompts = cm.load_prompts()
        assert isinstance(prompts, list)
        assert all("name" in p and "content" in p for p in prompts)

    def test_prompts_roundtrip(self, tmp_path):
        from src.utils.config_manager import ConfigurationManager
        import src.utils.config_manager as cfg_mod
        cm = ConfigurationManager()
        original = cm.prompts_path
        cm.prompts_path = str(tmp_path / "prompts.json")
        try:
            data = [{"name": "t1", "content": "c1"}]
            cm.save_prompts(data)
            assert cm.load_prompts() == data
        finally:
            cm.prompts_path = original
