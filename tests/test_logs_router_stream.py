# -*- coding: utf-8 -*-
"""logs router SSE 流式测试(/api/logs/stream)。

不通过 TestClient 的真实流式 iter_lines(同步阻塞 + 单 portal 限制),
改为验证 SSE 的真实交付机制:
  1. POST /api/logs 会把条目推给 _SUBSCRIBERS 里的每个订阅队列(stream 的投递路径)
  2. stream_logs 注册订阅队列 + 生成器能产出 SSE 格式
  3. 断开(is_disconnected=True)后 _gen 退出,finally 移除订阅者(防泄漏)

守付费红线:无外部调用,纯内存队列。
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
except OSError:
    torch = None

from fastapi.testclient import TestClient  # noqa: E402

import src.web.routers.logs as logs_mod  # noqa: E402


@pytest.fixture
def app(monkeypatch, tmp_path):
    """TestClient(app);进 TestClient 后清空缓冲/订阅者。"""
    from src.web.app import app as real_app

    with TestClient(real_app) as c:
        with logs_mod._BUFFER_LOCK:
            logs_mod._BUFFER.clear()
        with logs_mod._SUB_LOCK:
            logs_mod._SUBSCRIBERS.clear()
        yield c


def _make_request(receive_returns):
    """构造带自定义 receive 的 Request。

    receive_returns: 调用 receive 时依次返回的 scope 消息列表。
    """
    from fastapi import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/logs/stream",
        "raw_path": b"/api/logs/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }
    messages = list(receive_returns)

    async def _receive():
        if messages:
            return messages.pop(0)
        # 默认一直挂起(http.disconnect),让 is_disconnected 返回 True
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    return Request(scope, receive=_receive)


# ============================ SSE 投递 ============================


def test_post_pushes_to_subscribers(app):
    """POST /api/logs 把条目推给 _SUBSCRIBERS 中每个订阅队列(真实投递路径)。"""
    q = asyncio.Queue()
    with logs_mod._SUB_LOCK:
        logs_mod._SUBSCRIBERS.append(q)
    try:
        r = app.post("/api/logs", json={"level": "info", "message": "推送日志"})
        assert r.status_code == 200, r.text
        loop = asyncio.new_event_loop()
        try:
            entry = loop.run_until_complete(asyncio.wait_for(q.get(), timeout=2.0))
        finally:
            loop.close()
        assert entry["msg"] == "推送日志"
        assert entry["logger"] == "frontend"
        assert entry["level"] == "info"
    finally:
        with logs_mod._SUB_LOCK:
            if q in logs_mod._SUBSCRIBERS:
                logs_mod._SUBSCRIBERS.remove(q)


def test_stream_registers_subscriber(app):
    """stream_logs 打开时注册订阅队列(is_disconnected 立刻断开会清理)。"""
    req = _make_request(receive_returns=[{"type": "http.disconnect"}])

    async def _drive():
        resp = await logs_mod.stream_logs(req)
        # body_iterator 是 sse-starlette 的 AsyncIterator
        it = resp.body_iterator
        async for _ in it:
            break
        with logs_mod._SUB_LOCK:
            assert logs_mod._SUBSCRIBERS == [], "断开后订阅者未清理"

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_drive())
    finally:
        loop.close()


def test_stream_generates_sse_format(app):
    """_gen 产出 SSE 事件:event=log + data=json(与前端契约一致)。"""
    # receive 第一次返回 http.request(让 is_disconnected False,走 q.get)
    req = _make_request(receive_returns=[{"type": "http.request", "body": b"", "more_body": False}])

    async def _drive():
        resp = await logs_mod.stream_logs(req)
        # stream_logs 内部新建订阅队列并注册;取刚注册的队列塞一条日志
        with logs_mod._SUB_LOCK:
            stream_q = logs_mod._SUBSCRIBERS[-1]
        stream_q.put_nowait({"ts": 1.0, "level": "info", "logger": "frontend", "msg": "SSE格式"})
        async for chunk in resp.body_iterator:
            return chunk

    loop = asyncio.new_event_loop()
    try:
        chunk = loop.run_until_complete(_drive())
    finally:
        loop.close()

    assert chunk is not None
    raw = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
    # body_iterator 产出的是 dict {event, data}(sse-starlette 格式);校验字段而非行格式
    import json as _json
    payload = chunk if isinstance(chunk, dict) else _json.loads(raw)
    assert payload.get("event") == "log", f"缺 event=log: {payload!r}"
    assert "SSE格式" in payload.get("data", ""), f"缺日志内容: {payload!r}"


def test_stream_bridge_delivers_to_queue(app):
    """Python logging 走 _LogBridge → 订阅者队列收到(透传 trace_id)。"""
    import logging

    q = asyncio.Queue()
    with logs_mod._SUB_LOCK:
        logs_mod._SUBSCRIBERS.append(q)
    try:
        logging.getLogger("test.bridge").warning("桥接流式日志")
        loop = asyncio.new_event_loop()
        try:
            entry = loop.run_until_complete(asyncio.wait_for(q.get(), timeout=2.0))
        finally:
            loop.close()
        assert entry["msg"].endswith("桥接流式日志") or "桥接流式日志" in entry["msg"]
        assert entry["level"] == "warning"
        assert "trace_id" in entry
    finally:
        with logs_mod._SUB_LOCK:
            if q in logs_mod._SUBSCRIBERS:
                logs_mod._SUBSCRIBERS.remove(q)
