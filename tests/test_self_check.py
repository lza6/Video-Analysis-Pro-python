"""SelfCheck: 自检闭环逻辑单测（tmp_path 隔离，不依赖真实 LLM）。

AAA 模式：Arrange 准备 RunStore+segments / Act 调 self_check 函数 /
Assert 验证灰色地带、未验证命中、报告格式、触发判定。

覆盖：
  - find_gray_zones 命中区间 + 边界闭区间 + 空库
  - find_unverified_hits match=true 低置信
  - build_self_check_report 格式
  - should_trigger_self_check 3 error 触发 / 2 error 不触发 / 5 high 触发
"""
from src.core.decision_log import DecisionLog, make_entry
from src.core.run_store import RunStore
from src.core.self_check import (
    GrayZone,
    build_self_check_report,
    find_gray_zones,
    find_unverified_hits,
    should_trigger_self_check,
)


def _make_store(tmp_path):
    return RunStore(str(tmp_path / "cfg"))


def _seed_run(store, video_path="C:/videos/cam01.mp4"):
    return store.create_run(
        video_path,
        duration_sec=3600.0,
        model="m",
        provider="openai",
        mode="surveillance",
    )


# ----------------------------------------------------------------------
# find_gray_zones：confidence 0.6-0.7 的分片被挑出
# ----------------------------------------------------------------------
def test_find_gray_zones_picks_confidence_in_range(tmp_path):
    # Arrange：3 个分片，仅 seg0 落在 0.6-0.7
    store = _make_store(tmp_path)
    run_id = _seed_run(store)
    store.add_segment(run_id, {
        "seg_idx": 0, "start_sec": 0.0, "dur_sec": 60.0,
        "status": "ok", "match": 0, "confidence": 0.65, "reason": "疑似",
    })
    store.add_segment(run_id, {
        "seg_idx": 1, "start_sec": 60.0, "dur_sec": 60.0,
        "status": "ok", "match": 1, "confidence": 0.88, "reason": "命中",
    })
    store.add_segment(run_id, {
        "seg_idx": 2, "start_sec": 120.0, "dur_sec": 60.0,
        "status": "ok", "match": 0, "confidence": 0.3, "reason": "无",
    })

    # Act
    zones = find_gray_zones(store, run_id)

    # Assert：仅 seg0，字段完整
    assert len(zones) == 1
    assert zones[0].seg_idx == 0
    assert zones[0].confidence == 0.65
    assert zones[0].reason == "疑似"
    assert zones[0].video_name == "cam01.mp4"
    assert zones[0].run_id == run_id


# ----------------------------------------------------------------------
# find_gray_zones：空库（run 不存在）返回空列表
# ----------------------------------------------------------------------
def test_find_gray_zones_empty_store_returns_empty(tmp_path):
    store = _make_store(tmp_path)
    assert find_gray_zones(store, "nonexistent-run-id") == []


# ----------------------------------------------------------------------
# find_gray_zones：边界 0.6 和 0.7 都算灰色地带（闭区间）
# ----------------------------------------------------------------------
def test_find_gray_zones_boundaries_inclusive(tmp_path):
    store = _make_store(tmp_path)
    run_id = _seed_run(store)
    store.add_segment(run_id, {
        "seg_idx": 0, "start_sec": 0.0, "dur_sec": 60.0,
        "status": "ok", "match": 0, "confidence": 0.6, "reason": "边界低",
    })
    store.add_segment(run_id, {
        "seg_idx": 1, "start_sec": 60.0, "dur_sec": 60.0,
        "status": "ok", "match": 0, "confidence": 0.7, "reason": "边界高",
    })

    zones = find_gray_zones(store, run_id)

    assert len(zones) == 2
    assert {z.seg_idx for z in zones} == {0, 1}
    assert {z.confidence for z in zones} == {0.6, 0.7}


# ----------------------------------------------------------------------
# find_unverified_hits：match=true 且 confidence<0.7 被挑出
# ----------------------------------------------------------------------
def test_find_unverified_hits_finds_match_true_low_conf(tmp_path):
    # Arrange：seg0 未验证命中；seg1 已达阈值；seg2 非命中
    store = _make_store(tmp_path)
    run_id = _seed_run(store)
    store.add_segment(run_id, {
        "seg_idx": 0, "start_sec": 0.0, "dur_sec": 60.0,
        "status": "ok", "match": 1, "confidence": 0.55, "reason": "疑似命中",
    })
    store.add_segment(run_id, {
        "seg_idx": 1, "start_sec": 60.0, "dur_sec": 60.0,
        "status": "ok", "match": 1, "confidence": 0.9, "reason": "确认命中",
    })
    store.add_segment(run_id, {
        "seg_idx": 2, "start_sec": 120.0, "dur_sec": 60.0,
        "status": "ok", "match": 0, "confidence": 0.4, "reason": "无",
    })

    hits = find_unverified_hits(store, run_id)

    assert len(hits) == 1
    assert hits[0]["seg_idx"] == 0
    assert hits[0]["confidence"] == 0.55
    assert hits[0]["video_name"] == "cam01.mp4"
    assert hits[0]["reason"] == "疑似命中"


# ----------------------------------------------------------------------
# build_self_check_report：格式含两类计数与字段
# ----------------------------------------------------------------------
def test_build_self_check_report_format():
    # Arrange
    zones = [
        GrayZone(run_id="r1", seg_idx=3, confidence=0.65,
                 reason="疑似", video_name="cam01.mp4"),
    ]
    unverified = [
        {"run_id": "r1", "seg_idx": 5, "confidence": 0.55,
         "reason": "疑似命中", "video_name": "cam01.mp4"},
    ]

    # Act
    report = build_self_check_report(zones, unverified)

    # Assert：标题、两类计数、字段都齐全
    assert "🔍 自检报告" in report
    assert "灰色地带 1 个" in report
    assert "未验证命中 1 个" in report
    assert "cam01.mp4" in report
    assert "seg3" in report
    assert "seg5" in report
    assert "conf=0.65" in report
    assert "conf=0.55" in report


# ----------------------------------------------------------------------
# should_trigger_self_check：≥3 个 error 触发
# ----------------------------------------------------------------------
def test_should_trigger_self_check_3_errors_triggers():
    log = DecisionLog()
    for i in range(3):
        log = log.append(make_entry(
            f"step{i}", "act", f"dec{i}", f"原因{i}",
            status="error", risk="low",
        ))
    assert should_trigger_self_check(log) is True


# ----------------------------------------------------------------------
# should_trigger_self_check：2 个 error 不触发
# ----------------------------------------------------------------------
def test_should_trigger_self_check_2_errors_not_trigger():
    log = DecisionLog()
    for i in range(2):
        log = log.append(make_entry(
            f"step{i}", "act", f"dec{i}", f"原因{i}",
            status="error", risk="low",
        ))
    assert should_trigger_self_check(log) is False


# ----------------------------------------------------------------------
# should_trigger_self_check：≥5 个 high risk 也触发（error 不足时的补充路径）
# ----------------------------------------------------------------------
def test_should_trigger_self_check_5_high_risk_triggers():
    log = DecisionLog()
    for i in range(5):
        log = log.append(make_entry(
            f"step{i}", "act", f"dec{i}", f"原因{i}",
            status="ok", risk="high",
        ))
    assert should_trigger_self_check(log) is True
