"""APIGatewayClient 流式/非流式响应解析（mock HTTP，覆盖 SSE/JSON/raw 三分支）。"""
import json

import pytest

import src.core.logic as logic


class FakeStreamResponse:
    """模拟 SSE / JSON / 纯文本响应。"""

    def __init__(self, lines, content_type="text/event-stream"):
        self._lines = lines
        self.headers = {"Content-Type": content_type}
        self.status_code = 200
        self.text = "\n".join(l.decode() if isinstance(l, bytes) else l for l in lines)

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _sse_chunk(content="", reasoning=None):
    delta = {}
    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning
    return b"data: " + json.dumps({"choices": [{"delta": delta}]}).encode()


def _sse_payload(content=""):
    """不带 data: 前缀的裸 JSON（供测试自行拼前缀）。"""
    return json.dumps({"choices": [{"delta": {"content": content}}]}).encode()


class TestAPIGatewayStreaming:
    def _run(self, lines, monkeypatch, content_type="text/event-stream"):
        client = logic.APIGatewayClient("key", "https://api.example.com/v1")
        monkeypatch.setattr(
            "requests.post", lambda *a, **kw: FakeStreamResponse(lines, content_type)
        )
        return list(client.chat_stream("m", "hi"))

    def test_sse_content_stream(self, monkeypatch):
        out = self._run([_sse_chunk("Hello "), _sse_chunk("world"), b"data: [DONE]"],
                        monkeypatch)
        assert "".join(out) == "Hello world"

    def test_sse_reasoning_wrapped_in_think(self, monkeypatch):
        out = self._run([_sse_chunk(reasoning="thinking..."), _sse_chunk("answer")],
                        monkeypatch)
        assert "<think>thinking...</think>" in "".join(out)
        assert "answer" in "".join(out)

    def test_sse_data_prefix_variants(self, monkeypatch):
        """兼容 'data: x' 与 'data:x' 两种前缀。"""
        out = self._run([b"data: " + _sse_payload("a"), b"data:" + _sse_payload("b")],
                        monkeypatch)
        assert "".join(out) == "ab"

    def test_non_streaming_json(self, monkeypatch):
        payload = json.dumps({"choices": [{"message": {"content": "full reply"}}]}).encode()
        out = self._run([payload], monkeypatch, content_type="application/json")
        assert "".join(out) == "full reply"

    def test_raw_text_fallback(self, monkeypatch):
        out = self._run(["plain body".encode()], monkeypatch, content_type="text/plain")
        assert "".join(out) == "plain body"

    def test_network_error_yields_message(self, monkeypatch):
        def boom(*a, **kw):
            raise ConnectionError("refused")
        client = logic.APIGatewayClient("key", "https://api.example.com/v1")
        monkeypatch.setattr("requests.post", boom)
        out = list(client.chat_stream("m", "hi"))
        assert any("Error" in chunk for chunk in out)

    def test_parse_endpoint_models_url(self):
        base, chat, models = logic.APIGatewayClient.parse_endpoint("https://x.io/v1")
        assert models == "https://x.io/v1/models"
