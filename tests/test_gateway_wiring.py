"""build_llm_client 接线 + APIGatewayClient 分隔符消除（mock，不发真实请求）。"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import src.core.logic as logic
from src.core.llm_gateway import (AnthropicBackend, GeminiBackend,
                                   OpenAIChatBackend, OpenAIResponsesBackend,
                                   GatewayRouter)


def _make_cfg(api_url, api_key="k", model="m"):
    """模拟 ConfigManager：config 是 configparser-like dict 结构。"""
    cfg = {"LastUsed": {"api_url": api_url, "api_key": api_key, "model_name": model}}
    return SimpleNamespace(config=cfg)


class _FakeResponse:
    def __init__(self, lines=None, content_type="text/event-stream", status=200):
        self._lines = lines or []
        self.headers = {"Content-Type": content_type}
        self.status_code = status
        self.text = "\n".join(l.decode() if isinstance(l, bytes) else l for l in self._lines)

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        for l in self._lines:
            if decode_unicode and isinstance(l, bytes):
                yield l.decode("utf-8")
            else:
                yield l

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestBuildLlmClientRouting:
    def test_build_llm_client_routes_anthropic(self):
        client = logic.build_llm_client(_make_cfg("https://api.anthropic.com/v1", "k", "claude-4"))
        # adapter 内部 router 解析出 AnthropicBackend
        backend = client._router._get_backend("default")
        assert isinstance(backend, AnthropicBackend)
        assert client.protocol == "anthropic"

    def test_build_llm_client_routes_openai_default(self):
        client = logic.build_llm_client(_make_cfg("https://api.deepseek.com/v1", "k", "deepseek-chat"))
        backend = client._router._get_backend("default")
        assert isinstance(backend, OpenAIChatBackend)
        assert client.protocol == "openai_chat"

    def test_build_llm_client_routes_gemini(self):
        client = logic.build_llm_client(_make_cfg("https://generativelanguage.googleapis.com/v1beta", "k", "gemini-2.0"))
        backend = client._router._get_backend("default")
        assert isinstance(backend, GeminiBackend)

    def test_build_llm_client_routes_responses(self):
        client = logic.build_llm_client(_make_cfg("https://api.openai.com/v1/responses", "k", "gpt-4o"))
        backend = client._router._get_backend("default")
        assert isinstance(backend, OpenAIResponsesBackend)

    def test_build_llm_client_missing_key_raises(self):
        with pytest.raises(ValueError, match="API key"):
            logic.build_llm_client(_make_cfg("https://x/v1", "", "m"))


class TestAPIGatewaySeparatorElimination:
    def test_apigateway_no_separator_when_messages_struct(self, monkeypatch):
        """含 '--- System Context ---' 的 str prompt 应被解析为 system+user 双消息。"""
        captured = {}

        def fake_post(url, headers=None, json=None, **kw):
            captured["payload"] = json
            return _FakeResponse(lines=[b"data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}).encode(),
                                         b"data: [DONE]"])

        client = logic.APIGatewayClient("key", "https://api.example.com/v1")
        monkeypatch.setattr("requests.post", fake_post)
        prompt = "用户的问题" + "--- System Context ---\n视频信息X\n--------------------\n"
        list(client.chat_stream("m", prompt))

        msgs = captured["payload"]["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user"], f"应为 system+user 结构，实际 {roles}"
        assert "视频信息X" in msgs[0]["content"]
        assert "用户的问题" in msgs[1]["content"]
        # 关键：分隔符不得泄漏进 user content
        assert "--- System Context ---" not in msgs[1]["content"]

    def test_apigateway_plain_str_prompt_compat(self, monkeypatch):
        """纯 str prompt（无分隔符）仍走原逻辑，单条 user 消息。"""
        captured = {}

        def fake_post(url, headers=None, json=None, **kw):
            captured["payload"] = json
            return _FakeResponse(lines=[b"data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}).encode(),
                                         b"data: [DONE]"])

        client = logic.APIGatewayClient("key", "https://api.example.com/v1")
        monkeypatch.setattr("requests.post", fake_post)
        list(client.chat_stream("m", "纯文本问题"))

        msgs = captured["payload"]["messages"]
        assert [m["role"] for m in msgs] == ["user"]
        assert msgs[0]["content"] == "纯文本问题"


class TestGatewayAdapterContract:
    def test_adapter_chat_stream_str_prompt_routes_to_backend(self, monkeypatch):
        """adapter 满足 BaseAPIClient 契约：str prompt 路由到 backend.chat_stream。"""
        import json as _json
        client = logic.build_llm_client(_make_cfg("https://api.deepseek.com/v1", "k", "deepseek-chat"))
        captured = {}

        def fake_post(url, **kw):
            captured["payload"] = kw.get("json")
            return _FakeResponse(lines=[b"data: " + _json.dumps({"choices": [{"delta": {"content": "hi"}}]}).encode(),
                                         b"data: [DONE]"])

        backend = client._router._get_backend("default")
        monkeypatch.setattr(backend._session, "post", fake_post)
        out = list(client.chat_stream("deepseek-chat", "你好"))

        assert "".join(out) == "hi"
        msgs = captured["payload"]["messages"]
        assert msgs == [{"role": "user", "content": "你好"}]

    def test_adapter_separator_prompt_uses_system_param(self, monkeypatch):
        """含分隔符的 str prompt：system 走 backend 原生 system 参数，不塞进 user content。"""
        import json as _json
        client = logic.build_llm_client(_make_cfg("https://api.deepseek.com/v1", "k", "deepseek-chat"))
        captured = {}

        def fake_post(url, **kw):
            captured["payload"] = kw.get("json")
            return _FakeResponse(lines=[b"data: " + _json.dumps({"choices": [{"delta": {"content": "ok"}}]}).encode(),
                                         b"data: [DONE]"])

        backend = client._router._get_backend("default")
        monkeypatch.setattr(backend._session, "post", fake_post)
        prompt = "问题" + "--- System Context ---\n视频元数据\n--------------------\n"
        list(client.chat_stream("deepseek-chat", prompt))

        payload = captured["payload"]
        # OpenAIChatBackend 把 system 注成 role:system 消息
        roles = [m["role"] for m in payload["messages"]]
        assert roles == ["system", "user"]
        assert "视频元数据" in payload["messages"][0]["content"]
        assert "--- System Context ---" not in payload["messages"][1]["content"]
