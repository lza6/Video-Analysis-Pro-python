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
    """P0-2 契约:POST /chat → 200 + session_id 非空 + auto_run=True。

    v10.3.1 起 /chat 只做意图分析不跑 turn,执行移交 /run_stream
    (审批 SSE 事件由此获得消费方)。reply 为空、plan_steps 为空数组、
    auto_run=True 是前端订阅 /run_stream 的判据。
    """
    r = react_client.post("/api/agent/chat", json={"text": "你好"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("session_id"), f"session_id 空: {body}"
    assert body.get("intent") == "react"
    assert body.get("auto_run") is True, \
        f"react /chat 必须返回 auto_run=True(P0-2 审批契约): {body}"
    assert body.get("plan_steps") == [], \
        f"react /chat plan_steps 应为空数组: {body}"


def test_react_chat_persists_session(react_client):
    """P0-2 契约:POST /chat 后 session 尚未创建(执行在 /run_stream)。

    旧行为(/chat 内跑 turn 并落库)已改为 /run_stream 执行 —— 因此
    /chat 后 /sessions 不含该 session 是预期;run_stream 跑完才落库。
    """
    r = react_client.post("/api/agent/chat", json={"text": "测试持久化"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    lst = react_client.get("/api/agent/sessions").json()
    sessions = lst.get("sessions", [])
    assert not any(s["session_id"] == sid for s in sessions), \
        f"/chat 不应创建 session(执行在 /run_stream): {sessions}"


def test_react_sessions_empty_returns_empty_list(react_client):
    """空 store → GET /sessions 返回空数组(不 404,友好)。"""
    lst = react_client.get("/api/agent/sessions").json()
    assert lst.get("sessions") == []


def test_react_session_not_found_404(react_client):
    """GET /sessions/不存在 → 404。"""
    r = react_client.get("/api/agent/sessions/nonexistent-id-12345")
    assert r.status_code == 404, f"期望 404 实际 {r.status_code}: {r.text}"


def test_react_session_turns_timeline(react_client):
    """P0-2 契约:/chat 后 session 未创建,turns 端点 404(执行在 /run_stream)。

    旧行为(/chat 内跑 turn 产生 phases 时间轴)已移除 —— 时间轴分组
    逻辑由 /run_stream 执行路径的 session 覆盖(test_agent_framework)。
    """
    r = react_client.post("/api/agent/chat", json={"text": "跑一个 turn"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    r2 = react_client.get(f"/api/agent/sessions/{sid}/turns")
    assert r2.status_code == 404, \
        f"/chat 不跑 turn,session 未创建,期望 404 实际 {r2.status_code}: {r2.text}"


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
    """P0-2 契约:无 LLM 凭据 → /chat 正常返回契约(auto_run=True),不崩。

    降级文本改由 /run_stream 内 run_turn 产出(SyncLLMClientAdapter
    yield "(未接入 LLM,仅返回意图分析)"),/chat 本身不再跑 turn。
    """
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
                body = r.json()
                # P0-2:契约完整返回,不崩
                assert body.get("auto_run") is True, \
                    f"无 LLM 时也应返回 auto_run=True 契约: {body}"
                assert body.get("session_id"), f"session_id 空: {body}"
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
    """P0-2 契约:/chat 带 ANALYZE_VIDEO 关键词 → 契约返回(不跑 turn)。

    v10.3.1 起执行(含降级终答 / tool_call)移交 /run_stream;此处只验
    /chat 契约对意图类文本同样成立。
    """
    r = react_client.post("/api/agent/chat", json={"text": "分析一下这段视频"})
    assert r.status_code == 200, r.text
    body = r.json()
    sid = body["session_id"]
    assert body.get("auto_run") is True, f"P0-2 契约: {body}"

    # session 尚未创建(执行在 /run_stream),GET 404 是预期
    g = react_client.get(f"/api/agent/sessions/{sid}")
    assert g.status_code == 404, \
        f"/chat 不应创建 session(执行在 /run_stream),期望 404 实际 {g.status_code}"


def test_react_turns_timeline_phase_mapping(react_client):
    """P0-2 契约:/chat 后 session 未创建,turns 端点 404(执行在 /run_stream)。

    旧行为(/chat 内跑 turn)已移除。时间轴分组逻辑由
    test_agent_framework / run_stream 路径覆盖。
    """
    r = react_client.post("/api/agent/chat", json={"text": "时间轴映射测试"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    r2 = react_client.get(f"/api/agent/sessions/{sid}/turns")
    assert r2.status_code == 404, \
        f"/chat 不跑 turn,session 未创建,期望 404 实际 {r2.status_code}"


def test_react_delete_idempotent(react_client):
    """DELETE 不存在的 session 也 204(幂等语义)。"""
    d = react_client.delete("/api/agent/sessions/never-existed-id-99999")
    assert d.status_code == 204, f"幂等 DELETE 期望 204 实际 {d.status_code}"


def test_run_stream_text_replayed_to_session(react_client, monkeypatch):
    """Critic B-1 回归:/run_stream 的 text 必须落进 user event(非空)。

    模拟前端精确行为:POST /chat(拿 session_id)→
    GET /run_stream?session_id=…&text=…(无 LLM,降级回复)→
    session 里 user event content == text。
    """
    monkeypatch.setenv("VAP_AGENT_BACKEND", "react")
    from src.web.config import get_settings
    get_settings.cache_clear()
    try:
        from src.web.app import app
        import src.web.routers.agent as mod
        def _none_cb(cm, ctx):
            return None
        monkeypatch.setattr(mod, "_make_llm_callback", _none_cb)
        with TestClient(app) as c:
            from src.core.agent.session import SessionStore
            app.state.session_store = SessionStore(
                str(tmp_path_factory() / "text_replay.db"))
            sid = c.post("/api/agent/chat",
                         json={"text": "帮我做一个PPT演示文稿"}).json()["session_id"]
            # 消费 run_stream(降级回复,done 收尾)
            with c.stream("GET",
                          f"/api/agent/run_stream?session_id={sid}"
                          "&text=%E5%B8%AE%E6%88%91%E5%81%9A%E4%B8%80%E4%B8%AAPPT%E6%BC%94%E7%A4%BA%E6%96%87%E7%A8%BF") as resp:
                assert resp.status_code == 200
                for _ in resp.iter_text():
                    pass
            evs = c.get(f"/api/agent/sessions/{sid}").json()["events"]
            users = [e for e in evs if e["type"] == "user"]
            assert users, "应有 user event"
            assert users[-1]["payload"].get("content") == "帮我做一个PPT演示文稿", \
                f"text 必须重放进 user event(Critic B-1): {users[-1]['payload']}"
    finally:
        get_settings.cache_clear()


def tmp_path_factory():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp(prefix="vap_test_"))
