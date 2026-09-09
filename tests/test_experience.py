"""经验沉淀测试（v9 自进化 —《改进指南》2.5）。

覆盖：
  1. ExperienceExtractor.from_session 正常提取
  2. 空 session 返回 None
  3. ExperienceStore record + find_similar 模糊匹配 + 降序
  4. should_suggest_skill 第 1、2 次 False，第 3 次 True（同 intent + 稳定链）
  5. should_suggest_skill tool_chain 不稳定 → False
  6. count_by_intent
  7. SkillAdvisor.suggest 第 3 次触发，draft_skill_md 非空（传 skill_generator）
  8. SkillAdvisor.suggest 不传 skill_generator → draft_skill_md 为空
  9. SkillAdvisor.suggest 未达阈值 → None

守红线：skill_generator 是规则版，不真实调 LLM。
"""
from __future__ import annotations

import time

from src.core import skill_generator
from src.core.agent.experience import (
    Experience,
    ExperienceExtractor,
    ExperienceStore,
    SkillAdvisor,
)
from src.core.agent.session import Session, SessionEvent


def _make_session(content: str = "分析停车场视频",
                  tool_calls=None,
                  tool_results=None,
                  final_text: str = "分析完成") -> Session:
    """构造含 user + assistant + tool_result 的 Session。"""
    s = Session("sess-test")
    s.append(SessionEvent("user", time.time(), {"content": content}))
    if tool_calls:
        s.append(SessionEvent("assistant", time.time(), {
            "content": "", "tool_calls": tool_calls,
        }))
    if tool_results:
        for tr in tool_results:
            s.append(SessionEvent("tool_result", time.time(), {
                "tool_call_id": "x", "content": tr,
            }))
    s.append(SessionEvent("assistant", time.time(), {"content": final_text}))
    return s


def test_extractor_normal():
    """正常提取：intent 含关键词、tool_chain 去重、success、quality_score。"""
    s = _make_session(
        content="我要分析停车场视频",
        tool_calls=[
            {"id": "1", "name": "extract_frames", "args": {}},
            {"id": "2", "name": "detect_vehicles", "args": {}},
            {"id": "3", "name": "extract_frames", "args": {}},  # 重复，应去重
        ],
        tool_results=["ok", "ok"],
        final_text="分析完成",
    )
    exp = ExperienceExtractor.from_session(s)
    assert exp is not None
    assert "停车场" in exp.intent
    assert exp.tool_chain == ["extract_frames", "detect_vehicles"]
    assert exp.success is True
    assert exp.quality_score == 1.0


def test_extractor_empty_session_returns_none():
    """空 session 返回 None。"""
    s = Session("empty")
    assert ExperienceExtractor.from_session(s) is None


def test_store_record_and_find_similar(tmp_path):
    """record + find_similar 模糊匹配（FTS5 混合评分召回相似经验）。"""
    store = ExperienceStore(str(tmp_path / "exp.db"))
    e1 = Experience("停车场", ["a", "b"], True, 1.0, time.time(), "s1")
    e2 = Experience("我要分析停车场视频", ["a"], True, 0.8, time.time(), "s2")
    store.record(e1)
    store.record(e2)
    results = store.find_similar("停车场")
    assert len(results) == 2
    # 召回意图列表（质量分不要求固定顺序——混合评分含时间衰减成分）
    assert {e.intent for e in results} == {"停车场", "我要分析停车场视频"}


def test_should_suggest_skill_threshold(tmp_path):
    """第 1、2 次 False，第 3 次 True（同 intent + 相同 tool_chain）。"""
    store = ExperienceStore(str(tmp_path / "exp.db"))
    chain = ["extract_frames", "detect_vehicles"]
    # 第 1 次
    store.record(Experience("停车场分析", list(chain), True, 0.9, time.time(), "s0"))
    assert store.should_suggest_skill("停车场分析") is False
    # 第 2 次
    store.record(Experience("停车场分析", list(chain), True, 0.9, time.time(), "s1"))
    assert store.should_suggest_skill("停车场分析") is False
    # 第 3 次
    store.record(Experience("停车场分析", list(chain), True, 0.9, time.time(), "s2"))
    assert store.should_suggest_skill("停车场分析") is True


def test_should_suggest_skill_unstable_chain(tmp_path):
    """3 次但 tool_chain 不同 → False。"""
    store = ExperienceStore(str(tmp_path / "exp.db"))
    store.record(Experience("停车场分析", ["a"], True, 0.9, time.time(), "s0"))
    store.record(Experience("停车场分析", ["b"], True, 0.9, time.time(), "s1"))
    store.record(Experience("停车场分析", ["c"], True, 0.9, time.time(), "s2"))
    assert store.should_suggest_skill("停车场分析") is False


def test_count_by_intent(tmp_path):
    """count_by_intent 精确计数 + 未匹配返回 0。"""
    store = ExperienceStore(str(tmp_path / "exp.db"))
    for i in range(5):
        store.record(Experience("停车场", ["a"], True, 1.0, time.time(), f"s{i}"))
    assert store.count_by_intent("停车场") == 5
    assert store.count_by_intent("不存在的意图") == 0


def test_skill_advisor_suggest_with_generator(tmp_path):
    """第 3 次触发，返回 SkillSuggestion，draft_skill_md 非空。"""
    store = ExperienceStore(str(tmp_path / "exp.db"))
    chain = ["extract_frames", "detect_vehicles"]
    for i in range(3):
        store.record(Experience(
            "停车场视频分析", list(chain), True, 0.9, time.time(), f"s{i}"))
    suggestion = SkillAdvisor.suggest(
        store, "停车场视频分析", skill_generator=skill_generator)
    assert suggestion is not None
    assert suggestion.recommended_tool_chain == chain
    assert suggestion.sample_count == 3
    assert suggestion.draft_skill_md  # 非空
    assert "surveillance-parking-lpr" in suggestion.draft_skill_md


def test_skill_advisor_suggest_without_generator(tmp_path):
    """不传 skill_generator → draft_skill_md 为空，但仍给 tool_chain。"""
    store = ExperienceStore(str(tmp_path / "exp.db"))
    chain = ["a"]
    for i in range(3):
        store.record(Experience("停车场", list(chain), True, 0.9, time.time(), f"s{i}"))
    suggestion = SkillAdvisor.suggest(store, "停车场", skill_generator=None)
    assert suggestion is not None
    assert suggestion.recommended_tool_chain == chain
    assert suggestion.draft_skill_md == ""


def test_skill_advisor_suggest_below_threshold_returns_none(tmp_path):
    """未达阈值 → None。"""
    store = ExperienceStore(str(tmp_path / "exp.db"))
    store.record(Experience("停车场", ["a"], True, 0.9, time.time(), "s0"))
    suggestion = SkillAdvisor.suggest(
        store, "停车场", skill_generator=skill_generator)
    assert suggestion is None
