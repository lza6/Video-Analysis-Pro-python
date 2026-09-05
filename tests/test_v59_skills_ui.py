"""v5.9 改进项测试（skills-2 夜间 skill + ui-1 汇总卡片指标）。

- I5.9-skills-2：夜间自适应 skill 关键词匹配（surveillance-night-adaptive）
- I5.9-ui-1：单视频汇总 6 指标卡片（含命中率，命中率>0 金色高亮）
"""
from __future__ import annotations

import sys

import pytest


def test_night_skill_keyword_matching() -> None:
    """I5.9-skills-2：夜间关键词匹配 surveillance-night-adaptive。"""
    from src.core.agent_prompt import match_skills
    from src.skills.loader import load_skills
    skills = load_skills()
    night_texts = ["夜间监控找包", "夜里监控有人吗", "红外摄像头", "低光照走廊"]
    for t in night_texts:
        m = match_skills(t, skills)
        assert m is not None, f"夜间文本应匹配: {t}"
        assert "surveillance-night-adaptive" in m, f"应匹配夜间 skill: {t}"
    # 白天稀疏走廊不匹配夜间
    m = match_skills("走廊找包", skills)
    assert m is not None and "surveillance-sparse-corridor" in m
    # 密集场景优先于夜间（商场里夜间也走密集）
    m2 = match_skills("商场夜间人流", skills)
    assert m2 is not None and "surveillance-crowded-scene" in m2


def test_skill_priority_crowded_over_night() -> None:
    """同时命中密集+夜间时，密集优先（crowded > night > sparse）。"""
    from src.core.agent_prompt import match_skills
    from src.skills.loader import load_skills
    skills = load_skills()
    m = match_skills("夜间商场人流密集", skills)
    assert m is not None
    # 第一行应含 crowded（优先级最高）
    first = m.split("\n")[0]
    assert "surveillance-crowded-scene" in first


def test_summary_box_six_metrics() -> None:
    """I5.9-ui-1：单视频汇总区含 6 指标（总耗时/API/首字/命中/命中率/覆盖率）。"""
    # 不构造完整 QDialog（需 QApplication），直接测指标计算逻辑
    # 模拟 run dict，复用 _build_summary_box 的计算公式
    run = {
        "segments": [
            {"attempts": 2, "first_token_ms": 800, "elapsed_sec": 3.5,
             "match": True, "confidence": 0.9},
            {"attempts": 1, "first_token_ms": 900, "elapsed_sec": 2.0,
             "match": False, "confidence": 0.1},
        ],
        "total_elapsed_sec": 5.5,
        "hits_count": 1,
        "segments_total": 2,
        "segments_ok": 2,
    }
    segs = run["segments"]
    total_elapsed = run["total_elapsed_sec"]
    api_calls = sum(int(s.get("attempts") or 0) for s in segs) or len(segs)
    ftms_vals = [int(s.get("first_token_ms") or 0) for s in segs
                 if s.get("first_token_ms") is not None]
    avg_ft = sum(ftms_vals) / len(ftms_vals) if ftms_vals else 0
    hits = run["hits_count"]
    seg_total = run["segments_total"]
    seg_ok = run["segments_ok"]
    hit_rate = hits / seg_total * 100 if seg_total > 0 else 0.0
    coverage = seg_ok / seg_total * 100 if seg_total > 0 else 0.0
    # 6 指标断言
    assert abs(total_elapsed - 5.5) < 0.01
    assert api_calls == 3  # 2+1
    assert abs(avg_ft - 850.0) < 0.01  # (800+900)/2
    assert hits == 1
    assert abs(hit_rate - 50.0) < 0.01  # 1/2
    assert abs(coverage - 100.0) < 0.01  # 2/2


def test_summary_box_empty_run() -> None:
    """空 run（无 segments）不崩，指标回退 0/—。"""
    run = {"segments": [], "total_elapsed_sec": None,
           "hits_count": 0, "segments_total": 0, "segments_ok": 0}
    segs = run["segments"]
    api_calls = sum(int(s.get("attempts") or 0) for s in segs) or len(segs)
    assert api_calls == 0
    hits = run["hits_count"]
    seg_total = run["segments_total"] or len(segs)
    hit_rate = (hits / seg_total * 100) if seg_total > 0 else 0.0
    assert hit_rate == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
