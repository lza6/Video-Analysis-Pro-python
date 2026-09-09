# -*- coding: utf-8 -*-
"""requests router 测试(/api/requests)。

覆盖 src/web/routers/requests.py 的三条路由:
  - GET    /api/requests            请求历史(limit/provider/since 过滤)
  - GET    /api/requests/stats      token 统计(按 provider 聚合)
  - DELETE /api/requests            清空请求日志

用真实 TestClient 调 API;store 用 tmp_path 隔离(RequestLogStore(config_dir=tmp_path))。
RequestLog 字段见 src/core/request_log.py:50。
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

import src.core.request_log as rl_mod  # noqa: E402
from src.core.request_log import RequestLog, RequestLogStore  # noqa: E402


def _entry(provider, model="m", prompt_tokens=1, completion_tokens=1,
           latency_ms=10, status_code=200, error=None, timestamp="2026-09-07T12:00:00"):
    return RequestLog(
        timestamp=timestamp, provider=provider, model=model,
        status_code=status_code, latency_ms=latency_ms,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        error=error,
    )


@pytest.fixture
def app(monkeypatch, tmp_path):
    """TestClient(app) + get_store 指向 tmp_path 隔离库。"""
    from src.web.app import app as real_app

    iso_store = RequestLogStore(config_dir=str(tmp_path))
    monkeypatch.setattr(rl_mod, "_store", iso_store)
    with TestClient(real_app) as c:
        yield c, iso_store


def _seed(store, entries: list[RequestLog]):
    for e in entries:
        store.log_request(e)
    return store


# ============================ GET /api/requests ============================


def test_list_requests_empty(app):
    client, _ = app
    r = client.get("/api/requests")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requests"] == []
    assert body["count"] == 0


def test_list_requests_with_data(app):
    client, store = app
    _seed(store, [
        _entry("openai", prompt_tokens=10, completion_tokens=20, latency_ms=100),
        _entry("ollama", status_code=500, error="timeout",
               timestamp="2026-09-07T12:01:00"),
    ])
    r = client.get("/api/requests")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    providers = {x["provider"] for x in body["requests"]}
    assert providers == {"openai", "ollama"}
    # 每行含完整字段
    row = body["requests"][0]
    for key in ("log_id", "timestamp", "provider", "model", "latency_ms",
                "prompt_tokens", "completion_tokens", "total_tokens"):
        assert key in row, f"缺字段 {key}: {row}"


def test_list_requests_provider_filter(app):
    client, store = app
    _seed(store, [
        _entry("openai", timestamp="2026-09-07T12:00:00"),
        _entry("ollama", timestamp="2026-09-07T12:01:00"),
    ])
    r = client.get("/api/requests?provider=openai")
    body = r.json()
    assert body["count"] == 1
    assert body["requests"][0]["provider"] == "openai"


def test_list_requests_limit(app):
    client, store = app
    _seed(store, [
        _entry("openai", timestamp=f"2026-09-07T12:{i:02d}:00")
        for i in range(5)
    ])
    r = client.get("/api/requests?limit=2")
    body = r.json()
    assert body["count"] == 2


def test_list_requests_limit_validation(app):
    """limit>1000 → 422(Query 约束)。"""
    client, _ = app
    r = client.get("/api/requests?limit=9999")
    assert r.status_code == 422, r.text


def test_list_requests_since_filter(app):
    """since 只返回该时刻之后的记录。"""
    client, store = app
    _seed(store, [_entry("openai", timestamp="2026-09-07T10:00:00")])
    r = client.get("/api/requests?since=2026-09-07T11:00:00")
    assert r.json()["count"] == 0, "旧记录不应被 since 命中"
    r2 = client.get("/api/requests?since=2026-09-07T09:00:00")
    assert r2.json()["count"] == 1


# ============================ GET /api/requests/stats ============================


def test_stats_empty(app):
    client, _ = app
    r = client.get("/api/requests/stats")
    assert r.status_code == 200
    assert r.json()["providers"] == {}


def test_stats_aggregates_by_provider(app):
    client, store = app
    _seed(store, [
        _entry("openai", prompt_tokens=10, completion_tokens=20, latency_ms=100),
        _entry("openai", prompt_tokens=10, completion_tokens=20, latency_ms=100,
               status_code=500, error="err"),
        _entry("ollama", prompt_tokens=5, completion_tokens=5, latency_ms=50),
    ])
    r = client.get("/api/requests/stats")
    body = r.json()
    providers = body["providers"]
    assert "openai" in providers and "ollama" in providers
    assert providers["openai"]["requests"] == 2
    assert providers["openai"]["total_tokens"] == 60
    assert providers["openai"]["error_count"] == 1
    assert providers["ollama"]["requests"] == 1
    assert providers["ollama"]["error_count"] == 0
    # 关键聚合字段齐全
    for key in ("requests", "prompt_tokens", "completion_tokens",
                "total_tokens", "avg_latency_ms", "error_count"):
        assert key in providers["openai"], f"缺字段 {key}"


def test_stats_provider_filter(app):
    client, store = app
    _seed(store, [
        _entry("openai"),
        _entry("ollama"),
    ])
    r = client.get("/api/requests/stats?provider=openai")
    providers = r.json()["providers"]
    assert "openai" in providers
    assert "ollama" not in providers


# ============================ DELETE /api/requests ============================


def test_clear_empty(app):
    client, _ = app
    r = client.delete("/api/requests")
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "deleted": 0}


def test_clear_with_data(app):
    client, store = app
    _seed(store, [
        _entry("openai"),
        _entry("ollama"),
    ])
    r = client.delete("/api/requests")
    body = r.json()
    assert body["ok"] is True
    assert body["deleted"] == 2
    # 清空后查询为空
    assert client.get("/api/requests").json()["count"] == 0


# ============================ 异常兜底分支 ============================


def test_list_requests_store_error(app, monkeypatch):
    """store.list_logs 抛异常 → 200 + error 透传(不 500)。"""
    client, store = app

    def _boom(*a, **k):
        raise RuntimeError("store boom")

    monkeypatch.setattr(store, "list_logs", _boom)
    r = client.get("/api/requests")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requests"] == []
    assert "store boom" in body["error"]


def test_stats_store_error(app, monkeypatch):
    """store.token_stats 抛异常 → 200 + providers={} + error。"""
    client, store = app

    def _boom(*a, **k):
        raise RuntimeError("stats boom")

    monkeypatch.setattr(store, "token_stats", _boom)
    r = client.get("/api/requests/stats")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["providers"] == {}
    assert "stats boom" in body["error"]


def test_clear_store_error(app, monkeypatch):
    """store.clear_all 抛异常 → 200 + ok=False + error。"""
    client, store = app

    def _boom(*a, **k):
        raise RuntimeError("clear boom")

    monkeypatch.setattr(store, "clear_all", _boom)
    r = client.delete("/api/requests")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    assert "clear boom" in body["error"]
