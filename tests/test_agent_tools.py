"""ToolRegistry 与 Agent 工具。"""
import json

import pytest

from src.core.agent_tools import (ToolRegistry, create_kb_search_tool,
                                  create_get_video_meta_tool)


class TestToolRegistry:
    def test_unknown_tool_returns_error(self):
        reg = ToolRegistry()
        out = reg.execute_tool_call("nope", {})
        assert "not found" in out

    def test_tool_exception_caught(self):
        reg = ToolRegistry()

        def boom():
            raise ValueError("x")

        reg.register_tool("boom", "raises", boom)
        assert "Error executing tool boom" in reg.execute_tool_call("boom", {})

    def test_descriptions_include_schema(self):
        reg = ToolRegistry()
        reg.register_tool("t", "desc", lambda **k: "ok", {"a": "b"})
        d = reg.get_tool_descriptions()
        assert "t" in d and '"a"' in d


class TestGetVideoMeta:
    def test_no_video(self):
        tool = create_get_video_meta_tool(lambda: None)
        assert tool() == "No video loaded."

    def test_with_video(self):
        class App:
            video_path = "a.mp4"
            video_duration = 12.5
            output_dir = "out"
            frames = [1, 2, 3]

        out = json.loads(create_get_video_meta_tool(lambda: App())())
        assert out["duration"] == 12.5
        assert out["frame_count"] == 3


class TestKBSearchTool:
    def test_no_history_manager(self):
        class App:
            history_manager = None

        tool = create_kb_search_tool(lambda: App())
        assert "unavailable" in tool("red car")

    def test_with_hits(self, tmp_path, monkeypatch):
        """用确定性假 embedder 隔离：不依赖真实 CLIP 权重。"""
        import numpy as np
        from src.core.history_manager import HistoryManager
        import src.core.kb_indexer as kb_indexer

        class FakeEmbedder:
            def encode(self, texts, convert_to_tensor=False, **kw):
                # 单词 "car" → e0，其它 → e1（确定性映射，便于断言）
                if isinstance(texts, str):
                    texts = [texts]
                out = []
                for t in texts:
                    v = np.zeros(384, dtype=np.float32)
                    v[0 if "car" in t.lower() else 1] = 1.0
                    out.append(v)
                return out[0] if len(out) == 1 else out

        monkeypatch.setattr(kb_indexer, "get_embedder", lambda: FakeEmbedder())

        class App:
            history_manager = HistoryManager(str(tmp_path / "cfg"))

        App.history_manager.add_frame_to_kb(
            "s1", "car.mp4", "C:/v/car.mp4", 5.0, "a red sports car",
            np.eye(384, dtype=np.float32)[0]
        )
        tool = create_kb_search_tool(lambda: App())
        out = tool("car")
        assert "car.mp4" in out
        assert "5.00s" in out

    def test_embedder_unavailable_friendly_error(self, tmp_path, monkeypatch):
        import numpy as np
        from src.core.history_manager import HistoryManager
        import src.core.kb_indexer as kb_indexer
        monkeypatch.setattr(kb_indexer, "get_embedder", lambda: None)

        class App:
            history_manager = HistoryManager(str(tmp_path / "cfg"))

        App.history_manager.add_frame_to_kb(
            "s1", "car.mp4", "C:/v/car.mp4", 5.0, "x",
            np.eye(384, dtype=np.float32)[0]
        )
        tool = create_kb_search_tool(lambda: App())
        assert "unavailable" in tool("red car")
