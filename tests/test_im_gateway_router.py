# -*- coding: utf-8 -*-
"""im_gateway router 测试(/api/im/gateway/*)。

覆盖 src/web/routers/im_gateway.py 的全部路由:
  - POST /api/im/gateway/start     启动网关(Mock adapter)
  - POST /api/im/gateway/stop      停止网关
  - GET  /api/im/gateway/status    状态
  - GET  /api/im/gateway/messages  查 mailbox 消息
  - POST /api/im/gateway/callback  设置回调
  - POST /api/im/gateway/inject    注入入站消息
  - POST /api/im/gateway/process   单轮驱动

守付费红线:默认 Mock adapter(WechatAdapter 等不真实连外部 IM)。
每用例重置全局 _gateway,用 tmp_path 隔离 mailbox db。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
except OSError:
    torch = None

from fastapi.testclient import TestClient  # noqa: E402

import src.web.routers.im_gateway as gw_mod  # noqa: E402
from src.core.im_gateway import IMMessage  # noqa: E402


@pytest.fixture
def app(monkeypatch, tmp_path):
    """TestClient(app) + 禁用限流 + 重置全局网关 + tmp_path mailbox。"""
    from src.web.app import app as real_app
    from src.web import security as _sec
    _sec.init_ip_limiter(0)

    # 重置进程级单例(每用例独立,防残留)
    gw_mod._gateway = None

    with TestClient(real_app) as c:
        yield c, tmp_path


def _start(app, client):
    """开一个 tmp_path 隔离的网关并 start。"""
    tmp = app[1]
    r = client.post("/api/im/gateway/start", json={
        "db_path": str(tmp / "im.db"),
        "adapters": ["wechat"],
    })
    assert r.status_code == 200, r.text
    return r.json()


# ============================ POST /start ============================


def test_start_gateway(app):
    client, _ = app
    r = client.post("/api/im/gateway/start", json={"adapters": ["wechat"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "started"
    assert body["adapters"] == ["wechat"]
    assert "不真实连 IM" in body["note"]
    # 之后 status 应 running
    st = client.get("/api/im/gateway/status").json()
    assert st["status"] in ("running", "stopped")  # Mock task 可能很快结束


def test_start_unknown_adapter(app):
    """未知 adapter → 捕获 ValueError → status=failed + error。"""
    client, _ = app
    r = client.post("/api/im/gateway/start", json={"adapters": ["whatsapp"]})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failed"
    assert "unknown adapter" in body["error"]


def test_start_empty_adapters_falls_back_wechat(app):
    """空 adapters → 默认 WechatAdapter + IMMailbox。"""
    client, _ = app
    r = client.post("/api/im/gateway/start", json={"adapters": []})
    assert r.status_code == 200
    assert r.json()["status"] == "started"


def test_start_returns_already_started(app):
    """重复 start(同进程)幂等:不报错,仍 started。"""
    client, tmp = app
    _start(app, client)
    r = client.post("/api/im/gateway/start", json={
        "db_path": str(tmp / "im2.db"), "adapters": ["wechat"],
    })
    assert r.status_code == 200
    assert r.json()["status"] == "started"


# ============================ POST /stop ============================


def test_stop_not_running(app):
    client, _ = app
    r = client.post("/api/im/gateway/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "not_running"


def test_stop_running(app):
    client, _ = app
    _start(app, client)
    r = client.post("/api/im/gateway/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "stopped"


# ============================ GET /status ============================


def test_status_not_running(app):
    client, _ = app
    r = client.get("/api/im/gateway/status")
    assert r.status_code == 200
    assert r.json()["status"] == "not_running"


def test_status_after_start(app):
    client, _ = app
    _start(app, client)
    r = client.get("/api/im/gateway/status")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("running", "stopped")
    assert "processed" in body
    assert "failed" in body
    assert "adapters" in body


# ============================ GET /messages ============================


def test_messages_empty_not_started(app):
    client, _ = app
    r = client.get("/api/im/gateway/messages")
    assert r.status_code == 200
    assert r.json()["messages"] == []


def test_messages_after_inject(app):
    client, tmp = app
    _start(app, client)
    inj = client.post("/api/im/gateway/inject", json={
        "channel": "wechat", "peer": "tester", "content": "你好",
    })
    assert inj.status_code == 200
    assert inj.json()["status"] == "ok"

    r = client.get("/api/im/gateway/messages")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert len(msgs) >= 1
    got = msgs[0]
    assert got["channel"] == "wechat"
    assert got["content"] == "你好"
    for key in ("msg_id", "peer", "status", "created_at"):
        assert key in got, f"缺字段 {key}: {got}"


# ============================ POST /callback ============================


def test_callback_not_started(app):
    client, _ = app
    r = client.post("/api/im/gateway/callback", json={"mode": "echo"})
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    assert "not started" in r.json()["error"]


def test_callback_echo_started(app):
    client, _ = app
    _start(app, client)
    r = client.post("/api/im/gateway/callback", json={"mode": "echo"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["mode"] == "echo"


def test_callback_unknown_mode(app):
    client, _ = app
    _start(app, client)
    r = client.post("/api/im/gateway/callback", json={"mode": "llm"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failed"
    assert "unknown mode" in body["error"]


# ============================ POST /inject ============================


def test_inject_not_started(app):
    client, _ = app
    r = client.post("/api/im/gateway/inject", json={
        "channel": "wechat", "peer": "x", "content": "hi",
    })
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    assert "not started" in r.json()["error"]


def test_inject_success_return_msg_id(app):
    client, _ = app
    _start(app, client)
    r = client.post("/api/im/gateway/inject", json={
        "channel": "telegram", "peer": "alice", "content": "测试",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["msg_id"]


# ============================ POST /process ============================


def test_process_not_started(app):
    client, _ = app
    r = client.post("/api/im/gateway/process")
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    assert "not started" in r.json()["error"]


def test_process_once_closed_loop(app):
    """注入消息 → process_once → 处理数 ≥0(呼应消息已被 ack)。"""
    client, _ = app
    _start(app, client)
    client.post("/api/im/gateway/inject", json={
        "channel": "wechat", "peer": "tester", "content": "回显我",
    })
    r = client.post("/api/im/gateway/process")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "processed" in body


# ============================ helpers ============================


def test_msg_to_dict_immessage():
    m = IMMessage(msg_id="m1", direction="inbound", channel="wechat",
                  peer="p", content="c", status="pending")
    d = gw_mod._msg_to_dict(m)
    assert d["msg_id"] == "m1"
    assert d["channel"] == "wechat"
    assert d["peer"] == "p"
    assert d["content"] == "c"
    assert d["status"] == "pending"
    assert "created_at" in d


def test_msg_to_dict_plain_dict():
    assert gw_mod._msg_to_dict({"msg_id": "x"}) == {"msg_id": "x"}