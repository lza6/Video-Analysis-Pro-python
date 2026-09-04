"""KiloClient 多 key 轮换 + payload 结构（mock，不真实调用 Kilo）。"""
import json
from unittest.mock import patch

import requests

from src.core.kilo_provider import (DEFAULT_KILO_BASE_URL, KiloClient,
                                     KiloConfig, _parse_keys,
                                     build_kilo_config, build_kilo_client)


class _FakeResp:
    """模拟 OpenAI 兼容 SSE 流。"""
    def __init__(self, status_code=200, lines=None, json_data=None):
        self.status_code = status_code
        self._lines = lines or []
        self._json = json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def json(self):
        return self._json or {}

    def close(self):
        pass


# ---- _parse_keys ----
class TestParseKeys:
    def test_comma_separated(self):
        assert _parse_keys("k1,k2,k3") == ["k1", "k2", "k3"]

    def test_mixed_separators(self):
        assert _parse_keys("k1;k2 k3") == ["k1", "k2", "k3"]

    def test_empty_and_whitespace(self):
        assert _parse_keys("  , , ") == []
        assert _parse_keys("") == []

    def test_strips_spaces(self):
        assert _parse_keys(" k1 , k2 ") == ["k1", "k2"]


# ---- KiloConfig 规范化 ----
class TestKiloConfig:
    def test_strips_trailing_slash(self):
        cfg = KiloConfig(base_url="https://x.io/v1/", api_keys=["k"])
        assert cfg.base_url == "https://x.io/v1"

    def test_filters_empty_keys(self):
        cfg = KiloConfig(api_keys=["k1", "", "  ", "k2"])
        assert cfg.api_keys == ["k1", "k2"]

    def test_defaults(self):
        cfg = KiloConfig()
        assert cfg.base_url == DEFAULT_KILO_BASE_URL
        assert cfg.api_keys == []
        assert cfg.default_model == "openai/gpt-oss-120b:free"


# ---- build_kilo_config / client ----
class TestBuildKiloConfig:
    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("VAP_KILO_BASE_URL", "https://kilo.ai/v1")
        monkeypatch.setenv("VAP_KILO_API_KEYS", "k1,k2")
        monkeypatch.setenv("VAP_KILO_DEFAULT_MODEL", "nemotron:free")
        cfg = build_kilo_config()
        assert cfg.base_url == "https://kilo.ai/v1"
        assert cfg.api_keys == ["k1", "k2"]
        assert cfg.default_model == "nemotron:free"

    def test_no_keys_returns_none(self, monkeypatch):
        monkeypatch.delenv("VAP_KILO_API_KEYS", raising=False)
        monkeypatch.delenv("VAP_KILO_BASE_URL", raising=False)
        assert build_kilo_client() is None

    def test_with_keys_returns_client(self, monkeypatch):
        monkeypatch.setenv("VAP_KILO_API_KEYS", "k1")
        client = build_kilo_client()
        assert client is not None
        assert isinstance(client, KiloClient)


# ---- chat_stream 多 key 轮换 ----
class TestChatStreamRotation:
    def _ok_lines(self, text="hi"):
        return [
            f'data: {json.dumps({"choices":[{"delta":{"content":text}}]})}',
            'data: [DONE]',
        ]

    def test_single_key_success(self):
        cfg = KiloConfig(base_url="https://x.io/v1", api_keys=["k1"])
        client = KiloClient(cfg)
        fake = _FakeResp(200, self._ok_lines("hello"))
        with patch.object(client._session, 'post', return_value=fake) as mp:
            out = ''.join(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert "hello" in out
        # payload 结构校验
        call = mp.call_args
        payload = call.kwargs["json"]
        assert payload["model"] == "openai/gpt-oss-120b:free"
        assert payload["stream"] is True
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "hi"
        # header 含 Bearer
        headers = call.kwargs["headers"]
        assert headers["Authorization"] == "Bearer k1"

    def test_401_rotates_to_next_key(self):
        cfg = KiloConfig(base_url="https://x.io/v1", api_keys=["bad", "good"])
        client = KiloClient(cfg)
        responses = [_FakeResp(401), _FakeResp(200, self._ok_lines("ok"))]
        with patch.object(client._session, 'post', side_effect=responses) as mp:
            out = ''.join(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert "ok" in out
        assert len(mp.call_args_list) == 2
        # 第二次用第二个 key
        second_headers = mp.call_args_list[1].kwargs["headers"]
        assert second_headers["Authorization"] == "Bearer good"

    def test_429_rotates(self):
        cfg = KiloConfig(base_url="https://x.io/v1", api_keys=["k1", "k2"])
        client = KiloClient(cfg)
        responses = [_FakeResp(429), _FakeResp(200, self._ok_lines("recovered"))]
        with patch.object(client._session, 'post', side_effect=responses):
            out = ''.join(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert "recovered" in out

    def test_all_keys_fail_yields_error(self):
        cfg = KiloConfig(base_url="https://x.io/v1", api_keys=["k1", "k2"])
        client = KiloClient(cfg)
        with patch.object(client._session, 'post',
                          side_effect=[_FakeResp(401), _FakeResp(403)]):
            out = ''.join(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert "全部 key 均失败" in out

    def test_network_error_rotates(self):
        cfg = KiloConfig(base_url="https://x.io/v1", api_keys=["k1", "k2"])
        client = KiloClient(cfg)
        ok = _FakeResp(200, self._ok_lines("netok"))
        with patch.object(client._session, 'post',
                          side_effect=[requests.ConnectionError("reset"), ok]):
            out = ''.join(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert "netok" in out

    def test_no_keys_yields_error(self):
        cfg = KiloConfig(base_url="https://x.io/v1", api_keys=[])
        client = KiloClient(cfg)
        out = ''.join(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert "无可用 key" in out

    def test_system_prompt_injected(self):
        cfg = KiloConfig(base_url="https://x.io/v1", api_keys=["k1"])
        client = KiloClient(cfg)
        fake = _FakeResp(200, self._ok_lines("ok"))
        with patch.object(client._session, 'post', return_value=fake) as mp:
            ''.join(client.chat_stream([{"role": "user", "content": "q"}],
                                       system="be brief"))
        payload = mp.call_args.kwargs["json"]
        assert payload["messages"][0] == {"role": "system", "content": "be brief"}


# ---- chat 非流式 ----
class TestChatNonStream:
    def test_chat_returns_full_text(self):
        cfg = KiloConfig(base_url="https://x.io/v1", api_keys=["k1"])
        client = KiloClient(cfg)
        lines = [
            f'data: {json.dumps({"choices":[{"delta":{"content":"part1"}}]})}',
            f'data: {json.dumps({"choices":[{"delta":{"content":"part2"}}]})}',
            'data: [DONE]',
        ]
        with patch.object(client._session, 'post',
                          return_value=_FakeResp(200, lines)):
            out = client.chat([{"role": "user", "content": "hi"}])
        assert out == "part1part2"


# ---- embed ----
class TestEmbed:
    def test_embed_returns_vectors(self):
        cfg = KiloConfig(base_url="https://x.io/v1", api_keys=["k1"])
        client = KiloClient(cfg)
        fake = _FakeResp(200, json_data={
            "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
        })
        with patch.object(client._session, 'post', return_value=fake) as mp:
            embs = client.embed(["hello", "world"])
        assert embs == [[0.1, 0.2], [0.3, 0.4]]
        payload = mp.call_args.kwargs["json"]
        assert payload["input"] == ["hello", "world"]
        assert payload["model"] == "nvidia/nemotron-3-embed-1b"

    def test_embed_401_rotates(self):
        cfg = KiloConfig(base_url="https://x.io/v1", api_keys=["bad", "good"])
        client = KiloClient(cfg)
        ok = _FakeResp(200, json_data={"data": [{"embedding": [1.0]}]})
        with patch.object(client._session, 'post',
                          side_effect=[_FakeResp(401), ok]):
            embs = client.embed(["x"])
        assert embs == [[1.0]]

    def test_embed_all_fail_returns_none(self):
        cfg = KiloConfig(base_url="https://x.io/v1", api_keys=["k1"])
        client = KiloClient(cfg)
        with patch.object(client._session, 'post',
                          return_value=_FakeResp(401)):
            assert client.embed(["x"]) is None
