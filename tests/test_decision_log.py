"""DecisionLog + eli5 决策日志核心模型测试（6 测试）。

隔离原则：
  - 不依赖 PyQt6 / torch / CLIP（纯数据模型，无 GUI）
  - tmp_path 隔离落盘文件，不污染仓库
  - reason 空串校验、不可变追加、JSON 往返、原子写 .tmp 不残留
"""
import json
from pathlib import Path

import pytest

from src.core.decision_log import DecisionEntry, DecisionLog, make_entry
from src.core.eli5 import explain_tool_call


class TestMakeEntry:
    def test_basic_fields_populated(self):
        e = make_entry("抽帧", "extract_frames", "抽了 120 帧",
                       "视频 10 分钟按 5 秒间隔抽帧，够覆盖场景切换")
        assert e.step_name == "抽帧"
        assert e.action_type == "extract_frames"
        assert e.decision == "抽了 120 帧"
        assert len(e.id) == 8  # uuid4 hex 前 8 位
        assert e.status == "ok"
        assert e.risk == "low"
        assert e.cause_id is None
        assert e.output_path is None
        assert e.duration_ms == 0.0
        assert "T" in e.timestamp  # ISO8601

    def test_reason_empty_raises(self):
        """reason 必填非空（edl.json 铁律：不许只记参数）。"""
        with pytest.raises(ValueError):
            make_entry("s", "a", "d", "")
        with pytest.raises(ValueError):
            make_entry("s", "a", "d", "   ")


class TestDecisionLogImmutable:
    def test_append_returns_new_instance_original_unchanged(self):
        e1 = make_entry("步骤1", "tool1", "决策A", "原因A")
        e2 = make_entry("步骤2", "tool2", "决策B", "原因B")
        log = DecisionLog()
        log2 = log.append(e1)
        log3 = log2.append(e2)
        # 原实例不变
        assert log.to_list() == ()
        assert len(log2.to_list()) == 1
        assert len(log3.to_list()) == 2
        # 不可变：log2 不会被 log3 的追加影响
        assert len(log2.to_list()) == 1


class TestJsonRoundtrip:
    def test_to_json_from_json_preserves_entries(self):
        e1 = make_entry("步骤1", "tool1", "决策A", "原因A",
                        cause_id=None, output_path="/tmp/x.jpg",
                        duration_ms=123.4, status="ok", risk="low")
        e2 = make_entry("步骤2", "tool2", "决策B", "原因B",
                        cause_id=e1.id, status="error", risk="high")
        log = DecisionLog().append(e1).append(e2)
        s = log.to_json()
        # pretty + 中文不转义
        assert "\n" in s
        assert "原因A" in s

        restored = DecisionLog.from_json(s)
        assert len(restored.to_list()) == 2
        r1, r2 = restored.to_list()
        assert r1.decision == "决策A"
        assert r1.output_path == "/tmp/x.jpg"
        assert r1.duration_ms == 123.4
        assert r2.cause_id == e1.id  # 跨步骤因果链保留
        assert r2.status == "error"
        assert r2.risk == "high"


class TestAtomicSave:
    def test_save_writes_file_and_no_tmp_leftover(self, tmp_path):
        e = make_entry("步骤", "tool", "决策", "原因")
        log = DecisionLog().append(e)
        target = tmp_path / "decision_log.json"
        log.save(target)
        # 目标文件存在且内容正确
        assert target.exists()
        data = json.loads(target.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["decision"] == "决策"
        # 原子写：.tmp 不残留
        assert not (tmp_path / "decision_log.json.tmp").exists()


class TestEli5Templates:
    def test_search_visual_template(self):
        """search_visual 模板：解析时间点 + 相似度。"""
        out = explain_tool_call(
            "search_visual",
            {"query": "红色汽车"},
            "时间点 12.34s (匹配度: 0.85)\n时间点 5.00s (匹配度: 0.72)",
        )
        assert "红色汽车" in out
        assert "12.34" in out
        assert "0.85" in out
        assert "最像的是" in out

    def test_unknown_tool_degrades_gracefully(self):
        """未知工具退化为通用文案，不崩。"""
        out = explain_tool_call("mystery_tool", {"x": 1}, "abc" * 100)
        assert "mystery_tool" in out
        assert "字符" in out

    def test_result_is_exception_does_not_crash(self):
        """result 是 Exception 时标"出错了"，不抛。"""
        out = explain_tool_call("search_visual", {"query": "x"},
                                ValueError("网络断了"))
        assert "出错了" in out
        assert "网络断了" in out

    def test_get_video_meta_template(self):
        """get_video_meta 模板：解析 duration。"""
        import json as _json
        result = _json.dumps({"filename": "a.mp4", "duration": 12.5,
                              "frame_count": 3}, ensure_ascii=False)
        out = explain_tool_call("get_video_meta", {}, result)
        assert "12.5" in out
        assert "时长" in out
