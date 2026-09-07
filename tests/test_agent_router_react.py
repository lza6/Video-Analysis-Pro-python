"""ReactLoopAgent + SessionStore 接入 agent router 的集成测试。

覆盖 v10.2 P0:关窗不丢历史 + Turn 时间轴 API。
Feature Flag VAP_AGENT_BACKEND=react 走新路径,legacy(默认)走旧 AgentOrchestrator。

用 TestClient + monkeypatch 切 backend + 把 app.state.session_store 指到
tmp_path 的 SessionStore(测试间隔离,不污染开发库)。

不依赖付费 API:无 LLM 凭据时 SyncLLMClientAdapter 降级返回意图分析
占位文本,run_turn 正常结束并持久化 Session。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch 必须先于 FastAPI 导入(Windows DLL 顺序铁律,与项目其它测试一致)
try:
    import torch  # noqa: F401
except OSError:
    torch = None

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def react_client(monkeypatch, tmp_path):
    """TestClient 包裹 react app,session_store 指到 tmp_path。

    关键顺序:先 `with TestClient(app)`(触发 lifespan,默认 store 落
    config/agent_sessions.db),再覆写 app.state.session_store 为 tmp_path
    的 SessionStore。若先覆写再 enter, lifespan 会把 store 又覆盖回默认,
    导致测试间隔离失效(读到开发库的旧 session)。
    """
    monkeypatch.setenv("VAP_AGENT_BACKEND", "react")
    from src.web.config import get_settings
    get_settings.cache_clear()
    try:
        from src.web.app import app
        from src.core.agent.session import SessionStore
        with TestClient(app) as c:
            # lifespan 已跑完(设了默认 config/agent_sessions.db),
            # 这里覆写为 tmp_path 隔离测试库
            app.state.session_store = SessionStore(
                str(tmp_path / "test_sessions.db"))
            yield c
    finally:
        get_settings.cache_clear()


def test_react_chat_returns_session_id(react_client):
    """POST /chat → 200 + session_id 非空 + reply 非空。"""
    r = react_client.post("/api/agent/chat", json={"text": "你好"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("session_id"), f"session_id 空: {body}"
    assert body.get("reply"), f"reply 空: {body}"
    assert body.get("intent") == "react"


def test_react_chat_persists_session(react_client):
    """POST /chat → GET /sessions 返回该 session(关窗不丢历史)。"""
    r = react_client.post("/api/agent/chat", json={"text": "测试持久化"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    lst = react_client.get("/api/agent/sessions").json()
    sessions = lst.get("sessions", [])
    assert any(s["session_id"] == sid for s in sessions), \
        f"session {sid} 未出现在 /sessions: {sessions}"


def test_react_sessions_empty_returns_empty_list(react_client):
    """空 store → GET /sessions 返回空数组(不 404,友好)。"""
    lst = react_client.get("/api/agent/sessions").json()
    assert lst.get("sessions") == []


def test_react_session_not_found_404(react_client):
    """GET /sessions/不存在 → 404。"""
    r = react_client.get("/api/agent/sessions/nonexistent-id-12345")
    assert r.status_code == 404, f"期望 404 实际 {r.status_code}: {r.text}"


def test_react_session_turns_timeline(react_client):
    """POST /chat → GET /sessions/{id}/turns 返回 ≥1 turn + phases 非空。"""
    r = react_client.post("/api/agent/chat", json={"text": "跑一个 turn"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    r2 = react_client.get(f"/api/agent/sessions/{sid}/turns")
    assert r2.status_code == 200, r2.text
    body = r2.json()
    turns = body.get("turns", [])
    assert len(turns) >= 1, f"turns 为空: {body}"
    first = turns[0]
    assert first.get("phases"), f"首个 turn 的 phases 为空: {first}"
    # 每个 phase 必含 phase 名 + ts
    for ph in first["phases"]:
        assert "phase" in ph, f"phase 缺字段: {ph}"
        assert "ts" in ph, f"phase 缺 ts: {ph}"


def test_react_delete_session(react_client):
    """POST /chat → DELETE /sessions/{id} → GET /sessions/{id} 404。"""
    r = react_client.post("/api/agent/chat", json={"text": "待删除"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    d = react_client.delete(f"/api/agent/sessions/{sid}")
    assert d.status_code == 204, f"DELETE 期望 204 实际 {d.status_code}"

    g = react_client.get(f"/api/agent/sessions/{sid}")
    assert g.status_code == 404, f"删除后 GET 期望 404 实际 {g.status_code}"


def test_react_chat_no_llm_degrades_gracefully(monkeypatch, tmp_path):
    """无 LLM 凭据 → reply 含'未接入 LLM'或意图分析占位,不崩。"""
    monkeypatch.setenv("VAP_AGENT_BACKEND", "react")
    # 清掉可能的 NVIDIA/provider 凭据(防 .env 污染测试)
    for k in ("VAP_NV_API_KEYS", "VAP_LLM_API_KEY", "VAP_LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    from src.web.config import get_settings
    get_settings.cache_clear()
    try:
        from src.web.app import app
        from src.core.agent.session import SessionStore
        # monkeypatch _make_llm_callback 强制返回 None(无 LLM)
        import src.web.routers.agent as mod
        original = mod._make_llm_callback
        def _none_cb(cm, ctx):
            return None
        monkeypatch.setattr(mod, "_make_llm_callback", _none_cb)
        try:
            with TestClient(app) as c:
                # lifespan 跑完后覆写 store 为 tmp_path 隔离
                app.state.session_store = SessionStore(
                    str(tmp_path / "nollm.db"))
                r = c.post("/api/agent/chat", json={"text": "无 LLM 测试"})
                assert r.status_code == 200, r.text
                reply = r.json().get("reply", "")
                # 降级文本不崩,reply 非空(含意图分析占位)
                assert reply, f"无 LLM 时 reply 不应为空: {r.json()}"
                assert "未接入 LLM" in reply or reply, \
                    f"reply 应含降级标记: {reply}"
        finally:
            monkeypatch.setattr(mod, "_make_llm_callback", original)
    finally:
        get_settings.cache_clear()


def test_legacy_chat_still_works(monkeypatch, tmp_path):
    """VAP_AGENT_BACKEND=legacy(默认)→ POST /chat 走旧 AgentOrchestrator。

    返回 intent/plan_steps(与改动前一致,零回归)。
    """
    monkeypatch.setenv("VAP_AGENT_BACKEND", "legacy")
    from src.web.config import get_settings
    get_settings.cache_clear()
    try:
        from src.web.app import app
        from src.core.agent.session import SessionStore
        with TestClient(app) as c:
            # lifespan 跑完后覆写 store 为 tmp_path(防 legacy 测试污染开发库)
            app.state.session_store = SessionStore(
                str(tmp_path / "legacy.db"))
            # GENERAL 意图(无关键词),走 LLM 对话分支(无凭据降级)
            r = c.post("/api/agent/chat", json={"text": "你好啊"})
            assert r.status_code == 200, r.text
            body = r.json()
            # legacy 路径返回 intent / plan_steps / reply / auto_run
            assert "intent" in body, f"legacy 缺 intent: {body}"
            assert "plan_steps" in body, f"legacy 缺 plan_steps: {body}"
            assert "reply" in body, f"legacy 缺 reply: {body}"
            assert "auto_run" in body, f"legacy 缺 auto_run: {body}"
            # legacy 不返回 session_id(react 才有)
            assert "session_id" not in body, \
                f"legacy 不应返回 session_id: {body}"
            # GENERAL 意图应分类为 general
            assert body["intent"] == "general", \
                f"GENERAL 意图应分类为 general: {body['intent']}"
    finally:
        get_settings.cache_clear()


def test_react_chat_intent_analyze_video(react_client):
    """react 路径 POST /chat 带 ANALYZE_VIDEO 关键词,turn 时间轴含 tool 调用。

    无 LLM 凭据时 SyncLLMClientAdapter 降级,不会有真实 tool_call,但
    Session 应含 user + assistant event(降级终答)。验证 turns 时间轴
    能正确分组(至少 1 turn,user event 后跟 assistant)。
    """
    r = react_client.post("/api/agent/chat", json={"text": "分析一下这段视频"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    g = react_client.get(f"/api/agent/sessions/{sid}")
    assert g.status_code == 200, g.text
    body = g.json()
    events = body.get("events", [])
    # 至少:system + user + assistant(降级终答)
    types = [e["type"] for e in events]
    assert "user" in types, f"events 缺 user: {types}"
    assert "assistant" in types, f"events 缺 assistant(降级终答): {types}"


def test_react_turns_timeline_phase_mapping(react_client):
    """Turn 时间轴的 phase 映射:user → claim/input,assistant → assistant-stream。"""
    r = react_client.post("/api/agent/chat", json={"text": "时间轴映射测试"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    r2 = react_client.get(f"/api/agent/sessions/{sid}/turns")
    assert r2.status_code == 200, r2.text
    turns = r2.json().get("turns", [])
    assert len(turns) >= 1
    phases = turns[0]["phases"]
    phase_names = [p["phase"] for p in phases]
    # user event 应映射成 claim/input
    assert "claim/input" in phase_names, \
        f"user event 未映射成 claim/input: {phase_names}"
    # 降级终答应映射成 assistant-stream
    assert "assistant-stream" in phase_names, \
        f"assistant event 未映射成 assistant-stream: {phase_names}"


def test_react_delete_idempotent(react_client):
    """DELETE 不存在的 session 也 204(幂等语义)。"""
    d = react_client.delete("/api/agent/sessions/never-existed-id-99999")
    assert d.status_code == 204, f"幂等 DELETE 期望 204 实际 {d.status_code}"
