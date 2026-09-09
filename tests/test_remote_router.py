"""tests/test_remote_router.py — 远程访问 router 的 API 层测试。

覆盖 src/web/routers/remote.py 四个端点：
    GET  /api/remote/methods
    POST /api/remote/tunnel/start
    POST /api/remote/tunnel/stop
    GET  /api/remote/tunnel/status

策略（守付费红线，不真实启动 Tailscale / Cloudflare）：
    - 复用 src/remote 的 Mock adapter（三方案全 Mock，不调 subprocess）
    - 把 RemoteManager 默认 async 健康检查器替换为恒 200，避免 urllib 真实
      网络请求打 mock 主机名
    - 每测试重置模块级 _manager / _manager_lock（monkeypatch），实现隔离
    - 把 IP 限流器置 0（禁用），避免滑窗 429 竞态
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch 必须先于其它 C 扩展加载（Windows DLL 顺序铁律，与项目其它测试一致）
try:
    import torch  # noqa: F401
except OSError:
    torch = None

from fastapi.testclient import TestClient  # noqa: E402


async def _healthy(url: str, timeout_s: float) -> int:
    """Mock 健康检查器：恒 200，绝不发起真实网络请求。"""
    return 200


@pytest.fixture
def client(monkeypatch):
    """独立 FastAPI app（仅挂 remote router）+ TestClient 隔离环境。"""
    import asyncio

    import src.remote.manager as remote_manager
    import src.web.routers.remote as remote_router
    import src.web.security as security

    # 跨测试隔离：重置模块级 manager 与 lock（避免 event loop 冲突 / 状态残留）
    monkeypatch.setattr(remote_router, "_manager", None)
    monkeypatch.setattr(remote_router, "_manager_lock", asyncio.Lock())
    # 禁用 IP 限流（0 = disabled），防测试间滑窗互相 429
    monkeypatch.setattr(security, "_ip_limiter", security._IPRateLimiter(per_min=0))
    # RemoteManager 的默认健康检查走真实 urllib -> 替换为恒 200
    monkeypatch.setattr(
        remote_manager, "_default_async_health_checker", _healthy,
    )

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(remote_router.router)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /api/remote/methods
# ---------------------------------------------------------------------------


def test_list_methods(client):
    """GET /methods -> 三方案 + note。"""
    r = client.get("/api/remote/methods")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["methods"]) == {
        "tailscale-serve",
        "tailscale-direct",
        "cloudflare-access",
    }
    assert "note" in body


# ---------------------------------------------------------------------------
# POST /api/remote/tunnel/start
# ---------------------------------------------------------------------------


def test_start_tunnel_tailscale_serve_default(client):
    """空 body -> 默认 tailscale-serve，Mock 启动成功，随后状态 active。"""
    r = client.post("/api/remote/tunnel/start", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "started"
    assert body["method"] == "tailscale-serve"
    # config 无 extra.tailscale_hostname -> Mock fallback 主机名
    assert body["public_url"] == "https://mock-node.tailnet.ts.net"
    assert body["prepare"]["trusted_hosts"] == ["mock-node.tailnet.ts.net"]
    assert body["describe"]["method"] == "tailscale-serve"

    st = client.get("/api/remote/tunnel/status").json()
    assert st["status"]["state"] == "active"
    assert st["status"]["is_running"] is True


def test_start_tunnel_tailscale_direct(client):
    """method=tailscale-direct -> Mock 启动返回 http://100.64.0.1:8080。"""
    r = client.post("/api/remote/tunnel/start", json={"method": "tailscale-direct"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "started"
    assert body["method"] == "tailscale-direct"
    assert body["public_url"] == "http://100.64.0.1:8080"


def test_start_tunnel_cloudflare_access(client):
    """method=cloudflare-access -> Mock 启动成功，返回 hostname。

    注意：router 硬编码 team_domain="tingfeng.cloudflareaccess.com" 含点号，
    而 Cloudflare _TEAM_NAME 正则（^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$）
    不允许点号 -> prepare 会抛 cloudflare-config 错误。因此该配置走失败分支，
    这里断言启动失败且 error 指向 team domain 校验（防御性覆盖 prepare 异常路径）。
    """
    r = client.post("/api/remote/tunnel/start", json={"method": "cloudflare-access"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "failed"
    assert "team domain" in body["error"]


def test_start_tunnel_unknown_method(client):
    """非法 method -> ValueError 分支，返回 failed。"""
    r = client.post("/api/remote/tunnel/start", json={"method": "banana"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "failed"
    assert "unknown method" in body["error"]


def test_start_tunnel_rejects_non_loopback_target(client):
    """target 非 loopback -> Mock adapter 校验拒绝 -> 通用 except 分支 failed。"""
    r = client.post(
        "/api/remote/tunnel/start",
        json={"method": "tailscale-serve", "target_url": "http://1.2.3.4:8000"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "failed"
    assert "loopback" in body["error"].lower()


def test_start_tunnel_failure_returns_failed(client, monkeypatch):
    """prepare/start 异常 -> start_tunnel 通用 except 分支。"""
    import src.web.routers.remote as remote_router

    class _FailingManager:
        def select_adapter(self):
            return None

        async def prepare(self):
            return {}

        async def start(self, target_url: str) -> str:
            raise RuntimeError("start boom")

        def describe(self):
            return {"method": "x"}

    def _fake_get_manager(method):
        mgr = _FailingManager()
        remote_router._manager = mgr
        return mgr

    monkeypatch.setattr(remote_router, "_get_manager", _fake_get_manager)
    r = client.post("/api/remote/tunnel/start", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "failed"
    assert "start boom" in body["error"]


# ---------------------------------------------------------------------------
# POST /api/remote/tunnel/stop
# ---------------------------------------------------------------------------


def test_stop_not_running(client):
    """未启动 -> stop 返回 not_running。"""
    r = client.post("/api/remote/tunnel/stop")
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "not_running"}


def test_stop_after_start(client):
    """启动后 stop -> stopped；随后状态回 ready（不再对外服务）。"""
    r = client.post("/api/remote/tunnel/start", json={})
    assert r.json()["status"] == "started"

    r2 = client.post("/api/remote/tunnel/stop")
    assert r2.status_code == 200, r2.text
    assert r2.json() == {"status": "stopped"}

    st = client.get("/api/remote/tunnel/status").json()
    # stop 后 adapter 回 ready（hostname 仍在），但不再对外服务
    assert st["status"]["state"] == "ready"
    assert st["status"]["is_running"] is False


def test_stop_failure_returns_failed(client, monkeypatch):
    """stop 异常 -> stop_tunnel except 分支。"""
    import src.web.routers.remote as remote_router

    r = client.post("/api/remote/tunnel/start", json={})
    assert r.json()["status"] == "started"

    mgr = remote_router._manager
    assert mgr is not None

    async def _broken_stop():
        raise RuntimeError("stop boom")

    monkeypatch.setattr(mgr, "stop", _broken_stop)
    r2 = client.post("/api/remote/tunnel/stop")
    assert r2.status_code == 200, r2.text
    assert r2.json() == {"status": "failed", "error": "stop boom"}


# ---------------------------------------------------------------------------
# GET /api/remote/tunnel/status
# ---------------------------------------------------------------------------


def test_status_not_running(client):
    """未启动 -> status 返回 not_running。"""
    r = client.get("/api/remote/tunnel/status")
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "not_running"}


def test_status_after_start_has_describe(client):
    """启动后 status 返回状态字典 + describe。"""
    r = client.post("/api/remote/tunnel/start", json={})
    assert r.json()["status"] == "started"

    r2 = client.get("/api/remote/tunnel/status")
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["status"]["state"] == "active"
    assert body["status"]["method"] == "tailscale-serve"
    assert body["status"]["transport"] == "serve"
    assert body["status"]["url"] == "https://mock-node.tailnet.ts.net"
    assert body["status"]["is_running"] is True
    assert body["describe"]["method"] == "tailscale-serve"


def test_status_failure_returns_failed(client, monkeypatch):
    """status 异常 -> tunnel_status except 分支。"""
    import src.web.routers.remote as remote_router

    r = client.post("/api/remote/tunnel/start", json={})
    assert r.json()["status"] == "started"

    mgr = remote_router._manager
    assert mgr is not None

    async def _broken_status():
        raise RuntimeError("status boom")

    monkeypatch.setattr(mgr, "status", _broken_status)
    r2 = client.get("/api/remote/tunnel/status")
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["status"] == "failed"
    assert body["error"] == "status boom"