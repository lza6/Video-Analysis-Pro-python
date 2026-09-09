"""时序/图记忆（TripleStore）测试（v10.2 P1-1 三层记忆第 3 层）。

覆盖：
  1. record + query_subject
  2. record_experience 从 Experience 链式抽取（意图→工具→结果）
  3. query_between（时序过滤）
  4. query_object 反查
  5. 幂等 upsert（同三元组只保留最新 ts）
"""
from __future__ import annotations

from src.core.memory.triplestore import TripleStore


def test_record_and_query_subject(tmp_path):
    """record 后 query_subject 可召回。"""
    ts = TripleStore(str(tmp_path / "t.db"))
    ts.record("intent:停车场分析", "tool", "extract_frames", ts=100.0)
    ts.record("intent:停车场分析", "tool", "detect_vehicles", ts=200.0)
    rows = ts.query_subject("intent:停车场分析")
    assert [r.object for r in rows] == ["extract_frames", "detect_vehicles"]
    assert all(r.subject == "intent:停车场分析" for r in rows)


def test_record_experience_chain(tmp_path):
    """record_experience 链式抽取：意图→工具→下一工具→结果。"""
    ts = TripleStore(str(tmp_path / "t.db"))
    ts.record_experience("停车场分析", ["extract_frames", "detect_vehicles"],
                         success=True, ts=100.0)
    # 意图 → 首工具
    assert {r.object for r in ts.query_subject("停车场分析")} == {
        "extract_frames"}
    # 工具间 next_tool
    assert {r.object for r in ts.query_subject("extract_frames")} == {
        "detect_vehicles"}
    # 末工具 → result
    assert {r.object for r in ts.query_subject("detect_vehicles")} == {
        "success"}


def test_record_experience_empty_chain(tmp_path):
    """空 tool_chain 时只记意图→result。"""
    ts = TripleStore(str(tmp_path / "t.db"))
    ts.record_experience("停车场分析", [], success=False, ts=100.0)
    rows = ts.query_subject("停车场分析")
    assert len(rows) == 1
    assert rows[0].predicate == "result"
    assert rows[0].object == "failure"


def test_record_experience_empty_intent_noop(tmp_path):
    """空 intent 时 no-op（不落库）。"""
    ts = TripleStore(str(tmp_path / "t.db"))
    ts.record_experience("", ["a"], True, ts=100.0)
    assert ts.count() == 0


def test_query_between_time_filter(tmp_path):
    """query_between 按 (subject, object) + 时序窗口过滤。"""
    ts = TripleStore(str(tmp_path / "t.db"))
    for i, t in enumerate([100.0, 200.0, 300.0]):
        ts.record(f"s-{i}", "pred", "obj", ts=t)
        ts.record(f"s-{i}", "pred", "other", ts=t)
    rows = ts.query_between("s-1", "obj", start_ts=150.0, end_ts=250.0)
    assert [r.ts for r in rows] == [200.0]


def test_query_object_reverse(tmp_path):
    """query_object 按 object 反查。"""
    ts = TripleStore(str(tmp_path / "t.db"))
    ts.record("a", "rel", "共享对象", ts=100.0)
    ts.record("b", "rel", "共享对象", ts=200.0)
    rows = ts.query_object("共享对象")
    assert {r.subject for r in rows} == {"a", "b"}


def test_upsert_latest_ts_wins(tmp_path):
    """同 (subject, predicate, object) 幂等 upsert，保留最新 ts。"""
    ts = TripleStore(str(tmp_path / "t.db"))
    ts.record("s", "p", "o", ts=100.0)
    ts.record("s", "p", "o", ts=500.0)
    rows = ts.query_subject("s")
    assert len(rows) == 1  # 只有最新一条
    assert rows[0].ts == 500.0


def test_query_subject_with_time_window(tmp_path):
    """query_subject 支持时序窗口。"""
    ts = TripleStore(str(tmp_path / "t.db"))
    ts.record("s", "p1", "o1", ts=100.0)
    ts.record("s", "p2", "o2", ts=400.0)
    rows = ts.query_subject("s", start_ts=300.0)
    assert len(rows) == 1
    assert rows[0].object == "o2"
