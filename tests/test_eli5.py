"""eli5 工具调用解释器单测（4 测试）。

纯函数测试，无 GUI / 无 torch / 无 CLIP 依赖。
"""
from src.core.eli5 import explain_tool_call


class TestEli5VisualSearch:
    def test_search_visual_query_and_score_parsed(self):
        out = explain_tool_call(
            "search_visual",
            {"query": "红色汽车"},
            "时间点 12.34s (匹配度: 0.85)\n时间点 5.00s (匹配度: 0.72)",
        )
        assert "红色汽车" in out
        assert "12.34" in out
        assert "0.85" in out
        assert "最像的是" in out


class TestEli5UnknownTool:
    def test_unknown_tool_fallback(self):
        out = explain_tool_call("mystery_tool", {"x": 1}, "abc" * 100)
        assert "mystery_tool" in out
        assert "字符" in out


class TestEli5ExceptionSafe:
    def test_exception_result_marked_as_error(self):
        out = explain_tool_call("search_visual", {"query": "x"},
                                ValueError("网络断了"))
        assert "出错了" in out
        assert "网络断了" in out


class TestEli5VideoMeta:
    def test_get_video_meta_duration_parsed(self):
        import json
        result = json.dumps({"filename": "a.mp4", "duration": 12.5,
                             "frame_count": 3}, ensure_ascii=False)
        out = explain_tool_call("get_video_meta", {}, result)
        assert "12.5" in out
        assert "时长" in out
