"""tests/test_remote_tunnel.py — 远程访问框架测试。

全部 Mock，不真实连 Tailscale / Cloudflare、不真实付费、不真实启动 subprocess。
不依赖 pytest-asyncio：用 asyncio.run() 在同步测试中驱动 async 代码。

覆盖：
    - 三 Adapter Mock start/stop/status/url
    - manager 选方案 + 启停 + 状态查询
    - 健康检查（200 / 500 / 超时）
    - 凭据从 config_manager 读（Mock keyring）
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from src.remote import (
    CredentialStore,
    RemoteConfig,
    RemoteManager,
    TunnelState,
    TunnelStatus,
)
from src.remote.adapters import (
    CloudflareAccessAdapter,
    TailscaleDirectAdapter,
    TailscaleServeAdapter,
)
from src.remote.config import RemoteMethod


def run(coro):
    """同步驱动 async 调用（不依赖 pytest-asyncio）。"""
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ts_serve_config() -> RemoteConfig:
    """Tailscale Serve Mock config。"""
    return RemoteConfig(
        enabled=True,
        method=RemoteMethod.TAILSCALE_SERVE,
        tailscale_command="tailscale",
        tailscale_serve_port=443,
        extra={"tailscale_hostname": "node.tailnet.ts.net"},
    )


@pytest.fixture
def ts_direct_config() -> RemoteConfig:
    """Tailscale Direct Mock config。"""
    return RemoteConfig(
        enabled=True,
        method=RemoteMethod.TAILSCALE_DIRECT,
        tailscale_command="tailscale",
        tailscale_ip_address="100.64.10.20",
        tailscale_direct_port=8080,
    )


@pytest.fixture
def cf_config() -> RemoteConfig:
    """Cloudflare Access Mock config。"""
    return RemoteConfig(
        enabled=True,
        method=RemoteMethod.CLOUDFLARE_ACCESS,
        cloudflared_command="cloudflared",
        cloudflare_hostname="vap.example.com",
        cloudflare_team_domain="tingfeng",
        cloudflare_audience="abc123def456",
        cloudflare_tunnel="vap-tunnel",
        cloudflare_config_path="/tmp/config.yml",
    )


@pytest.fixture
def loopback_target() -> str:
    return "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# TailscaleServeAdapter
# ---------------------------------------------------------------------------


class TestTailscaleServeAdapter:
    """Tailscale Serve Mock 生命周期。"""

    def test_prepare_returns_trusted_host(self, ts_serve_config: RemoteConfig) -> None:
        adapter = TailscaleServeAdapter(ts_serve_config)
        result = run(adapter.prepare())
        assert result["trusted_hosts"] == ["node.tailnet.ts.net"]
        status = run(adapter.status())
        assert status.state is TunnelState.READY

    def test_start_returns_https_url(
        self, ts_serve_config: RemoteConfig, loopback_target: str
    ) -> None:
        adapter = TailscaleServeAdapter(ts_serve_config)
        run(adapter.prepare())
        url = run(adapter.start(loopback_target))
        assert url == "https://node.tailnet.ts.net"
        status = run(adapter.status())
        assert status.state is TunnelState.ACTIVE
        assert status.url == url
        assert status.is_running()

    def test_start_without_prepare_raises(
        self, ts_serve_config: RemoteConfig, loopback_target: str
    ) -> None:
        adapter = TailscaleServeAdapter(ts_serve_config)
        with pytest.raises(Exception, match="prepared before"):
            run(adapter.start(loopback_target))

    def test_start_rejects_non_loopback(self, ts_serve_config: RemoteConfig) -> None:
        adapter = TailscaleServeAdapter(ts_serve_config)
        run(adapter.prepare())
        with pytest.raises(Exception, match="loopback"):
            run(adapter.start("http://1.2.3.4:8000"))

    def test_start_rejects_missing_port(self, ts_serve_config: RemoteConfig) -> None:
        adapter = TailscaleServeAdapter(ts_serve_config)
        run(adapter.prepare())
        with pytest.raises(Exception, match="port"):
            run(adapter.start("http://127.0.0.1"))

    def test_stop_clears_running_state(
        self, ts_serve_config: RemoteConfig, loopback_target: str
    ) -> None:
        adapter = TailscaleServeAdapter(ts_serve_config)
        run(adapter.prepare())
        run(adapter.start(loopback_target))
        run(adapter.stop())
        status = run(adapter.status())
        assert status.state is TunnelState.READY
        assert status.url is None
        assert not status.is_running()

    def test_unavailable_when_command_missing(self) -> None:
        adapter = TailscaleServeAdapter(RemoteConfig(method=RemoteMethod.TAILSCALE_SERVE))
        result = run(adapter.prepare())
        assert result["trusted_hosts"] == []
        status = run(adapter.status())
        assert status.state is TunnelState.UNAVAILABLE
        assert status.error is not None

    def test_start_unavailable_raises(self) -> None:
        adapter = TailscaleServeAdapter(RemoteConfig(method=RemoteMethod.TAILSCALE_SERVE))
        with pytest.raises(Exception):
            run(adapter.start("http://127.0.0.1:8000"))


# ---------------------------------------------------------------------------
# TailscaleDirectAdapter
# ---------------------------------------------------------------------------


class TestTailscaleDirectAdapter:
    """Tailscale Direct IP Mock 生命周期。"""

    def test_prepare_returns_ip_authority(self, ts_direct_config: RemoteConfig) -> None:
        adapter = TailscaleDirectAdapter(ts_direct_config)
        result = run(adapter.prepare())
        assert result["trusted_hosts"] == ["100.64.10.20:8080"]
        status = run(adapter.status())
        assert status.state is TunnelState.READY

    def test_start_returns_http_url(
        self, ts_direct_config: RemoteConfig, loopback_target: str
    ) -> None:
        adapter = TailscaleDirectAdapter(ts_direct_config)
        run(adapter.prepare())
        url = run(adapter.start(loopback_target))
        assert url == "http://100.64.10.20:8080"
        status = run(adapter.status())
        assert status.state is TunnelState.ACTIVE
        assert status.url == url

    def test_prepare_rejects_non_cgnat_ip(self) -> None:
        config = RemoteConfig(
            method=RemoteMethod.TAILSCALE_DIRECT,
            tailscale_command="tailscale",
            tailscale_ip_address="192.168.1.1",
        )
        adapter = TailscaleDirectAdapter(config)
        with pytest.raises(Exception, match="100.64.0.0/10"):
            run(adapter.prepare())

    def test_prepare_defaults_ip_when_missing(self) -> None:
        config = RemoteConfig(
            method=RemoteMethod.TAILSCALE_DIRECT,
            tailscale_command="tailscale",
            tailscale_ip_address="",
        )
        adapter = TailscaleDirectAdapter(config)
        result = run(adapter.prepare())
        # 默认 mock IP 100.64.0.1
        assert result["trusted_hosts"] == ["100.64.0.1:8080"]

    def test_stop_returns_to_ready(
        self, ts_direct_config: RemoteConfig, loopback_target: str
    ) -> None:
        adapter = TailscaleDirectAdapter(ts_direct_config)
        run(adapter.prepare())
        run(adapter.start(loopback_target))
        run(adapter.stop())
        status = run(adapter.status())
        assert status.state is TunnelState.READY
        assert status.url is None


# ---------------------------------------------------------------------------
# CloudflareAccessAdapter
# ---------------------------------------------------------------------------


class TestCloudflareAccessAdapter:
    """Cloudflare Access Mock 生命周期。"""

    def test_prepare_returns_hostname(self, cf_config: RemoteConfig) -> None:
        adapter = CloudflareAccessAdapter(cf_config)
        result = run(adapter.prepare())
        assert result["trusted_hosts"] == ["vap.example.com"]
        status = run(adapter.status())
        assert status.state is TunnelState.READY

    def test_start_returns_https_url(
        self, cf_config: RemoteConfig, loopback_target: str
    ) -> None:
        adapter = CloudflareAccessAdapter(cf_config)
        run(adapter.prepare())
        url = run(adapter.start(loopback_target))
        assert url == "https://vap.example.com"
        status = run(adapter.status())
        assert status.state is TunnelState.ACTIVE
        assert status.url == url

    def test_prepare_rejects_bad_hostname(self) -> None:
        config = RemoteConfig(
            method=RemoteMethod.CLOUDFLARE_ACCESS,
            cloudflared_command="cloudflared",
            cloudflare_hostname="UPPER-CASE-INVALID",
            cloudflare_team_domain="tingfeng",
            cloudflare_tunnel="vap-tunnel",
        )
        adapter = CloudflareAccessAdapter(config)
        with pytest.raises(Exception, match="hostname"):
            run(adapter.prepare())

    def test_prepare_rejects_bad_team(self) -> None:
        config = RemoteConfig(
            method=RemoteMethod.CLOUDFLARE_ACCESS,
            cloudflared_command="cloudflared",
            cloudflare_hostname="vap.example.com",
            cloudflare_team_domain="UPPER-CASE-INVALID",
            cloudflare_tunnel="vap-tunnel",
        )
        adapter = CloudflareAccessAdapter(config)
        with pytest.raises(Exception, match="team domain"):
            run(adapter.prepare())

    def test_prepare_rejects_bad_tunnel(self) -> None:
        config = RemoteConfig(
            method=RemoteMethod.CLOUDFLARE_ACCESS,
            cloudflared_command="cloudflared",
            cloudflare_hostname="vap.example.com",
            cloudflare_team_domain="tingfeng",
            cloudflare_tunnel="!invalid-tunnel!",
        )
        adapter = CloudflareAccessAdapter(config)
        with pytest.raises(Exception, match="tunnel"):
            run(adapter.prepare())

    def test_stop_returns_to_ready(
        self, cf_config: RemoteConfig, loopback_target: str
    ) -> None:
        adapter = CloudflareAccessAdapter(cf_config)
        run(adapter.prepare())
        run(adapter.start(loopback_target))
        run(adapter.stop())
        status = run(adapter.status())
        assert status.state is TunnelState.READY
        assert status.url is None


# ---------------------------------------------------------------------------
# RemoteManager — 选方案 / 启停 / 状态
# ---------------------------------------------------------------------------


class TestRemoteManagerLifecycle:
    """RemoteManager 编排三方案的启停与状态查询。"""

    def test_select_tailscale_serve(self, ts_serve_config: RemoteConfig) -> None:
        manager = RemoteManager(ts_serve_config)
        adapter = manager.select_adapter()
        assert isinstance(adapter, TailscaleServeAdapter)
        assert manager.method == "tailscale-serve"

    def test_select_tailscale_direct(self, ts_direct_config: RemoteConfig) -> None:
        manager = RemoteManager(ts_direct_config)
        adapter = manager.select_adapter()
        assert isinstance(adapter, TailscaleDirectAdapter)
        assert manager.method == "tailscale-direct"

    def test_select_cloudflare_access(self, cf_config: RemoteConfig) -> None:
        manager = RemoteManager(cf_config)
        adapter = manager.select_adapter()
        assert isinstance(adapter, CloudflareAccessAdapter)
        assert manager.method == "cloudflare-access"

    def test_start_stop_full_lifecycle(
        self, ts_serve_config: RemoteConfig, loopback_target: str
    ) -> None:
        async def healthy(url: str, timeout: float) -> int:
            return 200

        manager = RemoteManager(ts_serve_config, health_checker=healthy)
        url = run(manager.start(loopback_target))
        assert url == "https://node.tailnet.ts.net"
        assert manager.public_url == url
        status = run(manager.status())
        assert status.state is TunnelState.ACTIVE
        run(manager.stop())
        status = run(manager.status())
        assert status.state is TunnelState.READY
        assert manager.public_url is None

    def test_status_disabled_when_not_started(self, ts_serve_config: RemoteConfig) -> None:
        manager = RemoteManager(ts_serve_config)
        status = run(manager.status())
        assert status.state is TunnelState.DISABLED

    def test_stop_noop_when_not_started(self, ts_serve_config: RemoteConfig) -> None:
        manager = RemoteManager(ts_serve_config)
        run(manager.stop())  # 不应 raise

    def test_start_with_prepare_first(
        self, cf_config: RemoteConfig, loopback_target: str
    ) -> None:
        async def healthy(url: str, timeout: float) -> int:
            return 200

        manager = RemoteManager(cf_config, health_checker=healthy)
        result = run(manager.prepare())
        assert "vap.example.com" in result["trusted_hosts"]
        url = run(manager.start(loopback_target))
        assert url == "https://vap.example.com"


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """健康检查 200 / 500 / 超时。"""

    def test_health_check_200(self, ts_serve_config: RemoteConfig) -> None:
        calls: list[str] = []

        async def healthy(url: str, timeout: float) -> int:
            calls.append(url)
            return 200

        manager = RemoteManager(ts_serve_config, health_checker=healthy)
        manager._public_url = "https://node.tailnet.ts.net"  # type: ignore[attr-defined]
        assert run(manager.health_check()) is True
        assert calls == ["https://node.tailnet.ts.net/healthz"]

    def test_health_check_500(self, ts_serve_config: RemoteConfig) -> None:
        async def unhealthy(url: str, timeout: float) -> int:
            return 500

        manager = RemoteManager(ts_serve_config, health_checker=unhealthy)
        manager._public_url = "https://node.tailnet.ts.net"  # type: ignore[attr-defined]
        assert run(manager.health_check()) is False

    def test_health_check_timeout(self, ts_serve_config: RemoteConfig) -> None:
        async def timing_out(url: str, timeout: float) -> int:
            return 0  # 模拟超时 / 网络错误

        manager = RemoteManager(ts_serve_config, health_checker=timing_out)
        manager._public_url = "https://node.tailnet.ts.net"  # type: ignore[attr-defined]
        assert run(manager.health_check()) is False

    def test_health_check_no_url(self, ts_serve_config: RemoteConfig) -> None:
        manager = RemoteManager(ts_serve_config)
        assert run(manager.health_check()) is False

    def test_start_retries_health_check(
        self, ts_serve_config: RemoteConfig, loopback_target: str
    ) -> None:
        """健康检查失败重试 N 次但 start 仍成功返回 URL。"""
        attempts: list[int] = []

        async def flaky(url: str, timeout: float) -> int:
            attempts.append(1)
            return 500  # 永远失败

        config = RemoteConfig(
            enabled=True,
            method=RemoteMethod.TAILSCALE_SERVE,
            tailscale_command="tailscale",
            health_retries=3,
            health_interval_s=0.001,
            extra={"tailscale_hostname": "node.tailnet.ts.net"},
        )
        manager = RemoteManager(config, health_checker=flaky)
        url = run(manager.start(loopback_target))
        assert url == "https://node.tailnet.ts.net"
        assert len(attempts) == 3


# ---------------------------------------------------------------------------
# CredentialStore — 从 config_manager 读（Mock keyring）
# ---------------------------------------------------------------------------


class _MockCredentialFetcher:
    """Mock fetcher：模拟 config_manager._secure_get 行为。"""

    def __init__(self, store: dict[str, str] | None = None) -> None:
        self.store = store or {}
        self.calls: list[tuple[str, str]] = []

    def get(self, key: str, fallback: str = "") -> str:
        self.calls.append((key, fallback))
        return self.store.get(key, fallback)


class TestCredentialStore:
    """凭据读取走密钥环封装（Mock keyring）。"""

    def test_get_existing_credential(self) -> None:
        fetcher = _MockCredentialFetcher({"remote_tailscale_auth_key": "tskey-abcdef"})
        store = CredentialStore(fetcher)
        val = store.get("tailscale_auth_key")
        assert val == "tskey-abcdef"
        assert fetcher.calls == [("remote_tailscale_auth_key", "")]

    def test_get_missing_returns_fallback(self) -> None:
        fetcher = _MockCredentialFetcher({})
        store = CredentialStore(fetcher)
        val = store.get("missing_key", "default")
        assert val == "default"

    def test_has_returns_true_for_existing(self) -> None:
        fetcher = _MockCredentialFetcher({"remote_cloudflared_tunnel_token": "tok"})
        store = CredentialStore(fetcher)
        assert store.has("cloudflared_tunnel_token") is True
        assert store.has("tailscale_auth_key") is False

    def test_snapshot_exposes_only_existence(self) -> None:
        fetcher = _MockCredentialFetcher(
            {"remote_tailscale_auth_key": "tskey-abcdef"},
        )
        store = CredentialStore(fetcher)
        snap = store.snapshot(["tailscale_auth_key", "cloudflared_tunnel_token"])
        assert snap == {"tailscale_auth_key": True, "cloudflared_tunnel_token": False}
        # 不暴露真实值
        assert "tskey-abcdef" not in str(snap)

    def test_default_fetcher_delegates_to_config_manager(self) -> None:
        """默认 fetcher 应通过 patch 委托 config_manager._secure_get。"""
        with patch("src.utils.config_manager._secure_get") as mock_get:
            mock_get.return_value = "tskey-from-config-manager"
            store = CredentialStore()
            val = store.get("tailscale_auth_key")
        assert val == "tskey-from-config-manager"
        mock_get.assert_called_once_with("remote_tailscale_auth_key", "")

    def test_default_fetcher_fallback_when_config_missing(self) -> None:
        """config_manager import 失败时降级到 fallback。"""
        with patch.dict(
            "sys.modules",
            {"src.utils.config_manager": None},
        ):
            store = CredentialStore()
            val = store.get("tailscale_auth_key", "fallback")
        assert val == "fallback"


# ---------------------------------------------------------------------------
# Manager 凭据快照 / 描述
# ---------------------------------------------------------------------------


class TestManagerCredentials:
    """Manager 凭据存在性快照 + describe。"""

    def test_credential_snapshot_tailscale_serve(self, ts_serve_config: RemoteConfig) -> None:
        fetcher = _MockCredentialFetcher({"remote_tailscale_auth_key": "tskey-abcdef"})
        manager = RemoteManager(ts_serve_config, credential_store=CredentialStore(fetcher))
        snap = manager.credential_snapshot()
        assert snap == {"tailscale_auth_key": True}

    def test_credential_snapshot_cloudflare(self, cf_config: RemoteConfig) -> None:
        fetcher = _MockCredentialFetcher(
            {
                "remote_cloudflared_tunnel_token": "tok",
                "remote_cloudflare_team_domain": "tingfeng",
                # audience 缺失
            },
        )
        manager = RemoteManager(cf_config, credential_store=CredentialStore(fetcher))
        snap = manager.credential_snapshot()
        assert snap == {
            "cloudflared_tunnel_token": True,
            "cloudflare_team_domain": True,
            "cloudflare_audience": False,
        }

    def test_describe_includes_method_and_url(
        self, ts_serve_config: RemoteConfig
    ) -> None:
        manager = RemoteManager(ts_serve_config)
        info = manager.describe()
        assert info["method"] == "tailscale-serve"
        assert info["enabled"] is True
        assert info["public_url"] is None
        assert "credentials" in info


# ---------------------------------------------------------------------------
# TunnelStatus / TunnelState 单元
# ---------------------------------------------------------------------------


class TestTunnelStatusDataclass:
    """TunnelStatus 不可变 / 状态判断。"""

    def test_is_running_active(self) -> None:
        s = TunnelStatus(state=TunnelState.ACTIVE, url="https://x")
        assert s.is_running() is True
        assert s.is_error() is False

    def test_is_running_ready(self) -> None:
        s = TunnelStatus(state=TunnelState.READY, url="https://x")
        assert s.is_running() is True

    def test_is_error(self) -> None:
        s = TunnelStatus(state=TunnelState.ERROR, error="serve")
        assert s.is_error() is True
        assert s.is_running() is False

    def test_dataclass_is_frozen(self) -> None:
        s = TunnelStatus(state=TunnelState.DISABLED)
        with pytest.raises(Exception):
            s.state = TunnelState.ACTIVE  # type: ignore[misc]
