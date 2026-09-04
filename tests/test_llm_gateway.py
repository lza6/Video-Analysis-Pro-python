"""llm_gateway 重试/退避/协议路由 单测（mock，不发真实请求）。"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.core.llm_gateway import (AnthropicBackend, GeminiBackend,
                                  GatewayRouter, OpenAIChatBackend,
                                  build_backend, detect_protocol)


class FakeResp:
    def __init__(self, code=200, lines=None):
        self.status_code = code
        self._lines = lines or []
    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")
    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)
    def close(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class TestRetryBackoff:
    def test_429_retry_then_success(self):
        b = AnthropicBackend('k', 'https://x.io/v1', 'm', max_tokens=10)
        calls = []
        ok = FakeResp(200, ['data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}'])
        def fake_post(url, **kw):
            calls.append(1)
            return FakeResp(429) if len(calls) < 3 else ok
        with patch.object(b._session, 'post', side_effect=fake_post):
            out = ''.join(b.chat_stream([{'role': 'user', 'content': 'hi'}]))
        assert len(calls) == 3
        assert 'ok' in out

    def test_4xx_no_retry(self):
        """非 429 的 4xx 不应重试（例如 401 认证错误重试无意义）。

        chat_stream 把 HTTPError 转成错误文本 yield 给 UI（设计契约），
        所以断言：仅 1 次调用 + 错误文本包含 401。
        """
        b = AnthropicBackend('k', 'https://x.io/v1', 'm', max_tokens=10)
        calls = []
        def fake_post(url, **kw):
            calls.append(1)
            return FakeResp(401)
        with patch.object(b._session, 'post', side_effect=fake_post):
            out = ''.join(b.chat_stream([{'role': 'user', 'content': 'hi'}]))
        assert len(calls) == 1
        assert '401' in out

    def test_network_error_retry(self):
        import requests
        b = OpenAIChatBackend('k', 'https://x.io/v1', 'm', max_tokens=10)
        calls = []
        ok = FakeResp(200, ['data: {"choices":[{"delta":{"content":"hi"}}]}', 'data: [DONE]'])
        def fake_post(url, **kw):
            calls.append(1)
            if len(calls) < 2:
                raise requests.ConnectionError("reset")
            return ok
        with patch.object(b._session, 'post', side_effect=fake_post):
            out = ''.join(b.chat_stream([{'role': 'user', 'content': 'hi'}]))
        assert len(calls) == 2 and 'hi' in out


class TestProtocolDetection:
    def test_detect_gemini(self):
        assert detect_protocol('https://g.ai', 'gemini-2.0') == 'gemini'
    def test_detect_anthropic(self):
        assert detect_protocol('https://x/v1', 'claude-4') == 'anthropic'
        assert detect_protocol('https://x/v1', 'glm-5.3-flash') == 'anthropic'
    def test_detect_openai_default(self):
        assert detect_protocol('https://api.openai.com/v1', 'gpt-4o') == 'openai_chat'

    def test_build_backend_fallback(self):
        b = build_backend('unknown-proto', 'k', 'https://x/v1', 'm')
        assert isinstance(b, OpenAIChatBackend)

    def test_router_switch(self):
        providers = [
            {'id': 'a', 'base_url': 'https://a/v1', 'api_key': 'k', 'model': 'glm-5.3-flash'},
            {'id': 'b', 'base_url': 'https://b/v1', 'api_key': 'k', 'model': 'gpt-4o'},
        ]
        r = GatewayRouter(providers, current='a')
        assert r.current == 'a'
        assert r.switch('b') is True
        assert r.current == 'b'
        assert r.switch('nope') is False
        assert len(r.list_providers()) == 2


class TestThinkingTags:
    def test_openai_reasoning_content_wrapped(self):
        b = OpenAIChatBackend('k', 'https://x/v1', 'm', max_tokens=10)
        fake = FakeResp(200, ['data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}',
                              'data: {"choices":[{"delta":{"content":"answer"}}]}',
                              'data: [DONE]'])
        with patch.object(b._session, 'post', return_value=fake):
            out = ''.join(b.chat_stream([{'role': 'user', 'content': 'x'}]))
        assert '<think>thinking</think>' in out
        assert 'answer' in out
