"""KnowledgeBaseRAG: index/query（mock embedder + ChromaDB，不真实调用）。"""
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.kb_rag import FrameRef, KnowledgeBaseRAG


@pytest.fixture
def manager(tmp_path):
    """真实 HistoryManager（tmp_path 隔离，ChromaDB 内存/磁盘均可）。"""
    from src.core.history_manager import HistoryManager
    return HistoryManager(str(tmp_path / "cfg"))


@dataclass
class FakeFrame:
    path: str
    timestamp: float
    vision_content: str = ""
    ocr_text: str = ""
    video_name: str = "test.mp4"
    video_path: str = "/tmp/test.mp4"


def _make_image(tmp_path: Path, name: str = "f.jpg"):
    """创建真实占位图（PIL.Image.open 需要）；返回绝对路径。"""
    from PIL import Image
    p = tmp_path / name
    Image.new("RGB", (8, 8), (0, 0, 0)).save(str(p))
    return str(p)


class TestIndexVideo:
    def test_no_frames_returns_zero(self, manager):
        rag = KnowledgeBaseRAG(manager, kilo_client=None,
                               embedder=MagicMock())
        assert rag.index_video("r1", []) == 0

    def test_no_embedder_returns_error(self, manager):
        rag = KnowledgeBaseRAG(manager, kilo_client=None, embedder=None)
        # _get_embedder 兜底走 kb_indexer.get_embedder()，patch 让它返回 None
        with patch("src.core.kb_indexer.get_embedder", return_value=None):
            result = rag.index_video("r1", [FakeFrame("p", 0.0, "x")])
        assert result == 0

    def test_indexes_frames_to_kb(self, manager, tmp_path):
        """注入 mock embedder，验证帧写入 KB（复用 add_frame_to_kb 接口）。"""
        embedder = MagicMock()
        # 返回 numpy array（HistoryManager.add_frame_to_kb 要 .tolist()）
        embedder.encode.return_value = np.ones(384, dtype=np.float32)
        rag = KnowledgeBaseRAG(manager, kilo_client=None, embedder=embedder)

        frames = [
            FakeFrame(_make_image(tmp_path, "f1.jpg"), 1.0, "a red car", "CAR"),
            FakeFrame(_make_image(tmp_path, "f2.jpg"), 2.0, "a blue sky", ""),
        ]
        n = rag.index_video("run-1", frames, transcript="whole transcript")
        assert n == 2
        # KB 计数
        assert manager.kb_count() == 2

    def test_transcript_saved_to_session_summary(self, manager, tmp_path):
        """index_video 把 transcript 写入 session summary（复用 update_session_summary）。"""
        # 需要 session 先存在（add_frame_to_kb 不创建 session）
        manager.add_session("v.mp4", "/out", summary="")
        embedder = MagicMock()
        embedder.encode.return_value = np.ones(384, dtype=np.float32)
        rag = KnowledgeBaseRAG(manager, kilo_client=None, embedder=embedder)

        # 用已存在的 session_id
        sid = manager.get_history()[0]["id"]
        rag.index_video(sid, [FakeFrame(_make_image(tmp_path), 0.0, "x")],
                        transcript="a long transcript text")
        # summary 被更新（截断 1000 字符）
        history = manager.get_history()
        assert history[0]["summary"] == "a long transcript text"

    def test_unopenable_frame_skipped(self, manager, tmp_path):
        """PIL 打不开的帧跳过，不阻塞其它帧。"""
        embedder = MagicMock()
        embedder.encode.return_value = np.ones(384, dtype=np.float32)
        rag = KnowledgeBaseRAG(manager, kilo_client=None, embedder=embedder)
        # 用一个能打开的图 + 一个不存在的路径
        good = _make_image(tmp_path, "good.jpg")
        frames = [
            FakeFrame(good, 1.0, "good frame"),
            FakeFrame("/nonexistent/x.jpg", 2.0, "bad"),
        ]
        n = rag.index_video("r1", frames)
        assert n == 1  # 只成功一张


class TestQuery:
    def test_no_embedder_returns_error(self, manager):
        rag = KnowledgeBaseRAG(manager, kilo_client=None, embedder=None)
        with patch("src.core.kb_indexer.get_embedder", return_value=None):
            out = rag.query("anything")
        assert "不可用" in out

    def test_no_hits_returns_empty_msg(self, manager, tmp_path):
        embedder = MagicMock()
        embedder.encode.return_value = np.ones(384, dtype=np.float32)
        rag = KnowledgeBaseRAG(manager, kilo_client=None, embedder=embedder)
        out = rag.query("something")
        assert "没有匹配结果" in out

    def test_no_llm_returns_raw_context(self, manager, tmp_path):
        """无 Kilo → 返回原始片段（与 search_kb 工具行为一致）。"""
        # 先索引一帧
        embedder = MagicMock()
        embedder.encode.return_value = np.ones(384, dtype=np.float32)
        rag = KnowledgeBaseRAG(manager, kilo_client=None, embedder=embedder)
        rag.index_video("r1", [FakeFrame(_make_image(tmp_path), 5.0,
                                         "a red sports car")])

        # query embedding 与 index 相同 → 必命中
        out = rag.query("red car")
        assert "test.mp4" in out
        assert "5.00s" in out
        assert "a red sports car" in out

    def test_with_llm_returns_answer(self, manager, tmp_path):
        """有 Kilo → 调用 chat，返回 LLM 回答。"""
        embedder = MagicMock()
        embedder.encode.return_value = np.ones(384, dtype=np.float32)
        kilo = MagicMock()
        kilo.chat.return_value = "这是 LLM 生成的回答"
        rag = KnowledgeBaseRAG(manager, kilo_client=kilo, embedder=embedder)
        rag.index_video("r1", [FakeFrame(_make_image(tmp_path), 5.0,
                                         "a red sports car")])

        out = rag.query("red car")
        assert out == "这是 LLM 生成的回答"
        # 校验传给 kilo.chat 的 prompt 含检索片段
        call = kilo.chat.call_args
        msgs = call.kwargs["messages"]
        assert "a red sports car" in msgs[0]["content"]
        assert "red car" in msgs[0]["content"]

    def test_llm_failure_falls_back_to_context(self, manager, tmp_path):
        """LLM 异常 → 回退原始片段，不抛。"""
        embedder = MagicMock()
        embedder.encode.return_value = np.ones(384, dtype=np.float32)
        kilo = MagicMock()
        kilo.chat.side_effect = RuntimeError("kilo down")
        rag = KnowledgeBaseRAG(manager, kilo_client=kilo, embedder=embedder)
        rag.index_video("r1", [FakeFrame(_make_image(tmp_path), 5.0, "car")])

        out = rag.query("car")
        assert "car" in out  # 原始片段兜底

    def test_run_id_filter(self, manager, tmp_path):
        """run_id 限定单视频范围。"""
        embedder = MagicMock()
        embedder.encode.return_value = np.ones(384, dtype=np.float32)
        rag = KnowledgeBaseRAG(manager, kilo_client=None, embedder=embedder)
        rag.index_video("run-A", [FakeFrame(_make_image(tmp_path, "a.jpg"), 1.0,
                                            "aaa", video_name="a.mp4")])
        rag.index_video("run-B", [FakeFrame(_make_image(tmp_path, "b.jpg"), 2.0,
                                            "bbb", video_name="b.mp4")])

        # 限定 run-B
        out_b = rag.query("query", run_id="run-B")
        assert "b.mp4" in out_b
        assert "a.mp4" not in out_b


class TestFrameRef:
    def test_defaults(self):
        f = FrameRef(path="/p", timestamp=1.0)
        assert f.vision_content == ""
        assert f.ocr_text == ""
        assert f.video_name is None
