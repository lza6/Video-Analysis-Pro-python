# -*- coding: utf-8 -*-
"""ProviderRouterClient 单测：包装 ProviderRouter.post_nvidia 为 loop.LLMClient。

守付费红线：全部 Mock，不发起真实 NVIDIA 请求。
"""
import asyncio
import inspect

import pytest

from src.core.provider_router_client import ProviderRouterClient


class _FakeRouter:
    """最小可用的 ProviderRouter 双（同步 post_nvidia，与真实签名一致）。"""

    def __init__(self, choices=None, tool_calls=None, error=None):
        self._choices = choices or []
        self._tool_calls = tool_calls or []
        self._error = error
        self.last_payload = None

    def post_nvidia(self, payload, **kwargs):
        self.last_payload = dict(payload)
        if self._error is not None:
            raise self._error
        msg = {"content": "", "reasoning_content": ""}
        if self._choices:
            msg.update(self._choices[0])
        if self._tool_calls:
            msg["tool_calls"] = self._tool_calls
        return {"choices": [{"message": msg}]}


async def _collect(client, messages, tools=None):
    """消费 async generator,返回最后一个 chunk。"""
    chunk = None
    async for c in client.stream(messages, tools or []):
        chunk = c
    return chunk


def _run(client, messages, tools=None):
    return asyncio.new_event_loop().run_until_complete(
        _collect(client, messages, tools))


def test_implements_llmclient_protocol():
    """必须满足 loop.LLMClient 协议签名 stream(self, messages, tools)。"""
    client = ProviderRouterClient(_FakeRouter())
    assert hasattr(client, "stream")
    # stream 是 async generator（协议要求 AsyncIterator），不是 coroutine function
    assert inspect.isasyncgenfunction(client.stream)


def test_stream_concats_reasoning_and_content():
    """assistant chunk 含 reasoning_content + content 合并。"""
    fake = _FakeRouter(choices=[{
        "content": "最终答案",
        "reasoning_content": "思考过程",
    }])
    chunk = _run(ProviderRouterClient(fake), [{"role": "user", "content": "hi"}])
    assert chunk.delta_text == "💭思考过程\n\n最终答案"
    assert chunk.stop_reason == "stop"
    assert chunk.final_tool_calls is None


def test_stream_no_choices_returns_empty_assistant():
    chunk = _run(ProviderRouterClient(_FakeRouter()), [])
    assert chunk.delta_text == ""


def test_stream_propagates_tool_calls():
    fake = _FakeRouter(tool_calls=[{"id": "1", "name": "analyze_video",
                                    "args": {"p": 1}}])
    chunk = _run(ProviderRouterClient(fake), [])
    assert chunk.final_tool_calls == [
        {"id": "1", "name": "analyze_video", "args": {"p": 1}}]
    assert chunk.stop_reason is None


def test_stream_passes_messages_and_tools_to_router():
    """payload 应含 messages + tools。"""
    fake = _FakeRouter(choices=[{"content": "ok"}])
    _run(ProviderRouterClient(fake),
         [{"role": "user", "content": "m"}], [{"name": "t"}])
    assert fake.last_payload["messages"] == [{"role": "user", "content": "m"}]
    assert fake.last_payload["tools"] == [{"name": "t"}]


def test_stream_no_tools_omits_tools_key():
    """无 tools 时 payload 不应含 tools 字段(零回归)。"""
    fake = _FakeRouter(choices=[{"content": "ok"}])
    _run(ProviderRouterClient(fake), [{"role": "user", "content": "m"}])
    assert "tools" not in fake.last_payload


def test_stream_wraps_router_exception():
    err = RuntimeError("503 upstream")
    client = ProviderRouterClient(_FakeRouter(error=err))
    with pytest.raises(RuntimeError):
        asyncio.new_event_loop().run_until_complete(
            _collect(client, []))