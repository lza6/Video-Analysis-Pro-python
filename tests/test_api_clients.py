"""APIGatewayClient.parse_endpoint 与 OllamaClient 流解析（纯逻辑，零网络）。"""
import json
import pytest

import src.core.logic as logic


class TestParseEndpoint:
    def test_bare_host_appends_v1(self):
        base, chat, models = logic.APIGatewayClient.parse_endpoint("https://api.example.com")
        assert base == "https://api.example.com/v1"
        assert chat == "https://api.example.com/v1/chat/completions"

    def test_v1_suffix_kept(self):
        base, chat, _ = logic.APIGatewayClient.parse_endpoint("http://localhost:1234/v1")
        assert chat == "http://localhost:1234/v1/chat/completions"

    def test_full_chat_endpoint(self):
        base, chat, _ = logic.APIGatewayClient.parse_endpoint("https://x.com/v1/chat/completions")
        assert chat == "https://x.com/v1/chat/completions"
        assert base == "https://x.com/v1"

    def test_hash_forces_raw_mode(self):
        base, chat, _ = logic.APIGatewayClient.parse_endpoint("https://gw.io/openai#")
        assert base == "https://gw.io/openai"
        assert chat == "https://gw.io/openai/chat/completions"

    def test_trailing_slash_stripped(self):
        base, chat, _ = logic.APIGatewayClient.parse_endpoint("https://api.example.com/")
        assert base == "https://api.example.com/v1"

    def test_whitespace_stripped(self):
        base, _, _ = logic.APIGatewayClient.parse_endpoint("  https://api.example.com  ")
        assert base.startswith("https://api.example.com")


class _FakeResponse:
    """模拟 Ollama SSE 流。"""

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.headers = {"Content-Type": "application/json"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestOllamaChatStream:
    """回归：OllamaClient 必须产出纯文本 delta，而不是原始 JSON 行。"""

    def _run(self, lines, monkeypatch):
        client = logic.OllamaClient()
        monkeypatch.setattr(client._session, "post", lambda *a, **kw: _FakeResponse(lines))
        return list(client.chat_stream("m", "hi"))

    def test_yields_plain_text_not_json(self, monkeypatch):
        line = json.dumps({"message": {"content": "你好"}, "done": False}).encode()
        out = self._run([line], monkeypatch)
        assert out == ["你好"]

    def test_multiple_lines_concatenate(self, monkeypatch):
        lines = [
            json.dumps({"message": {"content": "Hello "}}).encode(),
            json.dumps({"message": {"content": "World"}}).encode(),
        ]
        out = self._run(lines, monkeypatch)
        assert "".join(out) == "Hello World"

    def test_done_marker_skipped(self, monkeypatch):
        out = self._run([b"[DONE]"], monkeypatch)
        assert out == []

    def test_error_field_surfaced(self, monkeypatch):
        out = self._run([json.dumps({"error": "oom"}).encode()], monkeypatch)
        assert any("oom" in chunk for chunk in out)

    def test_no_json_fragment_leaks(self, monkeypatch):
        lines = [
            json.dumps({"message": {"content": "a"}}).encode(),
            json.dumps({"message": {"content": "b"}}).encode(),
        ]
        out = "".join(self._run(lines, monkeypatch))
        assert '"message"' not in out
        assert "{" not in out
