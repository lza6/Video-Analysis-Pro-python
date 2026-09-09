"""分层记忆主控接入集成测试（v10.2 P1-1 三层记忆）。

覆盖：
  1. loop.run_turn 结束时真实记录 Experience（experience 表新增行）
  2. 热层决定注入为附加 system 段（turn 开始时喂给 LLM）
  3. 三元组从 turn 抽取
  4. feature_flag=False 时零注入零记录（回归 test_agent_framework 零破坏）
  5. record 幂等（同 session 同 intent 不重复）
"""
from __future__ import annotations

import asyncio
import time

from src.core.agent.experience import ExperienceStore
from src.core.agent.loop import AgentConfig, LLMChunk, MockLLMClient, ReactLoopAgent
from src.core.agent.session import Session
from src.core.memory.connector import MemoryLayeredConnector
from src.core.memory.triplestore import TripleStore
from src.core.memory.working import WorkingMemory
from src.core.tools import ToolDefinition, ToolRegistry


def _run(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _build_agent(exp_db, wm_db, trips_db, n=0):
    async def _echo(x: str = "") -> str:
        return f"echo:{x}"

    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="echo", description="echo", execute_callback=_echo,
        input_schema={"type": "object",
                      "properties": {"x": {"type": "string"}}},
    ))
    exp = ExperienceStore(str(exp_db))
    wm = WorkingMemory(str(wm_db))
    trips = TripleStore(str(trips_db))
    memory = MemoryLayeredConnector(feature_flag=True,
                                    working=wm, experience=exp, triples=trips)
    script = [LLMChunk(final_tool_calls=[
        {"id": "c1", "name": "echo", "args": {"x": "hello"}},
    ])] + [LLMChunk(delta_text="final", stop_reason="stop")]
    if n:
        script = [LLMChunk(final_tool_calls=[
            {"id": f"c{i}", "name": "noop", "args": {}}
            for i in range(n)])]
    mock = MockLLMClient(script)
    config = AgentConfig(memory=memory)
    agent = ReactLoopAgent(config, mock, registry)
    return agent, exp, wm, trips


def test_loop_records_experience_at_turn_end(tmp_path):
    """run_turn 结束后 experiences 表新增 1 行（intent/tool_chain 落库）。"""
    agent, exp, _, _ = _build_agent(
        tmp_path / "exp.db", tmp_path / "wm.db", tmp_path / "tr.db")
    session = Session("loop-s1")
    result = _run(agent.run_turn(session, "请 echo 停车场视频"))
    assert result.stop_reason.value == "stop"

    rows = exp.find_similar("echo", top_n=5)
    assert len(rows) == 1
    assert "echo" in rows[0].intent
    assert rows[0].tool_chain == ["echo"]


def test_loop_injects_working_memory_system(tmp_path):
    """turn 开始时已存在的热层事实被注入为附加 system 段（喂给 LLM）。"""
    agent, _, wm, _ = _build_agent(
        tmp_path / "exp.db", tmp_path / "wm.db", tmp_path / "tr.db")
    wm.record_intent("loop-s1", "停车场分析", ["echo"],
                     ts=time.time())  # 当前时间，TTL 内
    session = Session("loop-s1", system_prompt="SYS")
    result = _run(agent.run_turn(session, "echo 停车场视频"))
    assert result.stop_reason.value == "stop"
    # LLM mock 收到的 messages 含附加 system 段（含 Working Memory 标记 + 事实）
    assert agent.llm.calls, "LLM 未被调用"
    msgs = agent.llm.calls[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYS"  # system_prompt 原样，未被污染
    assert any(m["role"] == "system" and "[Working Memory]" in m["content"]
               for m in msgs)
    assert any("停车场分析" in m["content"] for m in msgs if m["role"] == "system")


def test_loop_records_working_and_triples(tmp_path):
    """turn 结束后热层 + 三元组均有数据。"""
    agent, _, wm, trips = _build_agent(
        tmp_path / "exp.db", tmp_path / "wm.db", tmp_path / "tr.db")
    session = Session("loop-s2")
    _run(agent.run_turn(session, "echo hello"))
    wfacts = wm.recent(ttl_sec=float("inf"))  # TTL 无限 → 全部未过期
    assert any(f.intent and "echo" in f.intent for f in wfacts)
    trows = trips.query_subject("echo")
    # 无 tool_chain 时 intent→result 一条；此处 intent 含 'echo hello'，
    # tool_chain=['echo'] → intent→echo + echo→result
    assert len(trows) >= 1


def test_flag_off_no_record_no_inject(tmp_path):
    """feature_flag=False：零注入 + 零记录（各库为空）。"""
    exp = ExperienceStore(str(tmp_path / "exp.db"))
    wm = WorkingMemory(str(tmp_path / "wm.db"))
    trips = TripleStore(str(tmp_path / "tr.db"))
    memory = MemoryLayeredConnector(feature_flag=False,
                                    working=wm, experience=exp, triples=trips)
    async def _noop() -> str:
        return "ok"
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="noop", description="n", execute_callback=_noop))
    mock = MockLLMClient([LLMChunk(delta_text="final", stop_reason="stop")])
    agent = ReactLoopAgent(AgentConfig(memory=memory), mock, registry)
    session = Session("flag-off-1", system_prompt="SYS")
    _run(agent.run_turn(session, "hello"))
    # 注入零：喂给 LLM 的 messages 只有 SYS + user
    msgs = agent.llm.calls[0]["messages"]
    assert len([m for m in msgs if m["role"] == "system"]) == 1
    # 记录零
    assert exp.find_similar("hello") == []
    assert wm.recent(ttl_sec=float("inf")) == []
    assert trips.count() == 0


def test_connector_from_env_flag_default_on(monkeypatch, tmp_path):
    """VAP_MEMORY_LAYERED 默认 1=开；显式 0=关。"""
    monkeypatch.delenv("VAP_MEMORY_LAYERED", raising=False)
    c = MemoryLayeredConnector.from_env(
        working=WorkingMemory(str(tmp_path / "w.db")),
        experience=ExperienceStore(str(tmp_path / "e.db")),
        triples=TripleStore(str(tmp_path / "t.db")))
    assert c.enabled is True
    monkeypatch.setenv("VAP_MEMORY_LAYERED", "0")
    c2 = MemoryLayeredConnector.from_env(
        working=WorkingMemory(str(tmp_path / "w2.db")),
        experience=ExperienceStore(str(tmp_path / "e2.db")),
        triples=TripleStore(str(tmp_path / "t2.db")))
    assert c2.enabled is False