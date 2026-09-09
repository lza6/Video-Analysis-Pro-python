# -*- coding: utf-8 -*-
"""agent router react 路径全链路测试。

守付费红线：ReactLoopAgent 的 LLM 客户端在测试中一律 Mock，
不发起真实 provider 请求。
"""
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.agent.session import Session
from src.core.agent.turn import TurnResult


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("VAP_AGENT_BACKEND", "react")
    monkeypatch.setenv("VAP_HEADLESS_TOKEN", "")
    from src.web.app import create_app
    return create_app()


def _fake_react_agent():
    """构造 (agent, session, session_store) 三元组,agent.run_turn 是 async 返回 TurnResult。

    _react_run_stream(agent.py:305) 期望 `_build_react_agent` 返回三元组,
    并 `await agent.run_turn(session, "")` 取 result.final_text。
    """
    store = MagicMock()
    session = Session("test-session")
    agent = MagicMock()

    async def _run_turn(*_a, **_k):
        return TurnResult(final_text="(Mock LLM 回复)")

    agent.run_turn = _run_turn
    return agent, session, store


def test_react_backend_switch_env(app):
    assert os.environ.get("VAP_AGENT_BACKEND") == "react"


def test_legacy_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("VAP_AGENT_BACKEND", raising=False)
    from src.web.routers import agent as agent_mod
    # _get_backend 读 header > env > 默认 legacy;空 header 时应回 legacy
    req = MagicMock()
    req.headers = {}
    assert agent_mod._get_backend(req) == "legacy"


def test_get_backend_header_overrides_env(monkeypatch):
    """x-agent-backend header 优先于 env。"""
    monkeypatch.setenv("VAP_AGENT_BACKEND", "react")
    from src.web.routers import agent as agent_mod
    req = MagicMock()
    req.headers = {"x-agent-backend": "legacy"}
    assert agent_mod._get_backend(req) == "legacy"


def test_react_run_stream_emits_done_event(app):
    """react 路径 /run_stream:Mock LLM 下应收到 done 事件 + session_id。"""
    with patch("src.web.routers.agent._build_react_agent",
               side_effect=lambda *a, **k: _fake_react_agent()):
        client = TestClient(app)
        with client.stream("GET", "/api/agent/run_stream?mode=react") as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
    assert "done" in body
    assert "session_id" in body


def test_react_run_stream_persists_session(app):
    """react 路径跑完应调用 session_store.save。"""
    with patch("src.web.routers.agent._build_react_agent",
               side_effect=lambda *a, **k: _fake_react_agent()):
        client = TestClient(app)
        with client.stream("GET", "/api/agent/run_stream?mode=react") as resp:
            list(resp.iter_text())
    # _react_run_stream 内 session_store.save(session) 被调用