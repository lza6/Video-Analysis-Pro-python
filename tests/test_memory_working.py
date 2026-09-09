"""Working Memory 热层测试（v10.2 P1-1 三层记忆第 1 层）。

覆盖：
  1. record_intent + recent 记录
  2. TTL 过期自动清除（forget_old）
  3. inject_system_messages 把热层注入为附加 system 段（不污染 system_prompt）
  4. 无事实时零注入
"""
from __future__ import annotations

from src.core.memory.working import WorkingMemory


def test_record_and_recent(tmp_path):
    """record_intent 后 recent 可召回，按 ts 倒序。"""
    wm = WorkingMemory(str(tmp_path / "wm.db"))
    wm.record_intent("s1", "停车场分析", ["extract_frames", "detect_vehicles"],
                     ts=1000.0)
    wm.record_intent("s2", "猫狗识别", ["detect_objects"], ts=2000.0)
    facts = wm.recent(now=3000.0)
    assert [f.session_id for f in facts] == ["s2", "s1"]
    assert facts[0].intent == "猫狗识别"
    assert facts[0].tool_chain == ["detect_objects"]


def test_record_empty_intent_noop(tmp_path):
    """空 intent 不落库。"""
    wm = WorkingMemory(str(tmp_path / "wm.db"))
    wm.record_intent("s1", "", ["a"])
    assert wm.recent(now=99999.0) == []


def test_ttl_expiry_autoclean(tmp_path):
    """TTL 过期自动清除：forget_old 删掉超龄事实。"""
    wm = WorkingMemory(str(tmp_path / "wm.db"), ttl_sec=100.0)
    wm.record_intent("s1", "意图A", ["a"], ts=999.0)  # 101s 龄，已过期
    wm.record_intent("s2", "意图B", ["b"], ts=1090.0)  # 10s 龄，未过期
    # now=1100，TTL=100 → cutoff=1000；ts < 1000 的删除
    deleted = wm.forget_old(now=1100.0)
    assert deleted == 1
    facts = wm.recent(now=1100.0)
    assert [f.intent for f in facts] == ["意图B"]


def test_ttl_recent_only_fresh(tmp_path):
    """recent 只返回未过期事实。"""
    wm = WorkingMemory(str(tmp_path / "wm.db"), ttl_sec=100.0)
    wm.record_intent("s1", "旧意图", ["a"], ts=100.0)
    wm.record_intent("s2", "新意图", ["b"], ts=200.0)
    facts = wm.recent(now=250.0)
    assert [f.intent for f in facts] == ["新意图"]


def test_inject_system_messages_appends_system_segment(tmp_path):
    """热层事实注入为附加 system 段（不污染 system_prompt）。"""
    wm = WorkingMemory(str(tmp_path / "wm.db"))
    wm.record_intent("s1", "停车场分析", ["extract_frames"], ts=100.0)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]
    out = wm.inject_system_messages(messages, now=200.0)
    # 返回新列表，不改原列表
    assert messages[0]["content"] == "You are a helpful assistant."
    assert len(out) == 3
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "You are a helpful assistant."  # system_prompt 原样
    assert out[1]["role"] == "system"
    assert "[Working Memory]" in out[1]["content"]
    assert "停车场分析" in out[1]["content"]
    assert out[2] == {"role": "user", "content": "hello"}


def test_inject_no_facts_returns_original_copy(tmp_path):
    """无热层事实时零注入（返回原列表副本）。"""
    wm = WorkingMemory(str(tmp_path / "wm.db"))
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    out = wm.inject_system_messages(messages, now=99999.0)
    assert out == messages
    assert out is not messages  # 副本，不可变语义


def test_inject_without_existing_system(tmp_path):
    """无 system_prompt 时注入段放最前。"""
    wm = WorkingMemory(str(tmp_path / "wm.db"))
    wm.record_intent("s1", "意图", ["t"], ts=100.0)
    out = wm.inject_system_messages([{"role": "user", "content": "u"}], now=200.0)
    assert out[0]["role"] == "system"
    assert "[Working Memory]" in out[0]["content"]
    assert out[1] == {"role": "user", "content": "u"}
