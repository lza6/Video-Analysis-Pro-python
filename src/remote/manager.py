"""RemoteManager — 远程访问方案编排器。

职责：
    - 按 RemoteConfig 选择对应 Adapter
    - prepare / start / stop / status 生命周期编排
    - 健康检查：start 后轮询 tunnel URL 健康端点

健康检查用 urllib（标准库）发 GET，不引入 requests；Mock 测试注入
health_check 回调即可。Manager 本身不真实连接外部服务，真实连接由
Adapter 负责（此处 Adapter 全 Mock）。
"""

from __future__ import annotations

import asyncio
import logging
import urllib.error
import urllib.request
from typing import Any, Awaitable, Callable

from .adapters import (
    CloudflareAccessAdapter,
    TailscaleDirectAdapter,
    TailscaleServeAdapter,
)
from .config import CredentialStore, RemoteConfig, RemoteMethod
from .tunnel import Tunnel, TunnelState, TunnelStatus

_logger = logging.getLogger(__name__)

HealthChecker = Callable[[str, float], "Awaitable[int]"]
"""健康检查回调签名：(url, timeout_s) -> HTTP status code。"""


def default_health_checker(url: str, timeout_s: float) -> int:
    """默认健康检查：urllib GET，返回 HTTP status。

    真实网络调用，但只打 tunnel URL（本地 loopback 经 tunnel 出口），
    不打外部 Tailscale / Cloudflare API。Mock 测试不调用此函数。
    """
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — tunnel URL
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0


async def _default_async_health_checker(url: str, timeout_s: float) -> int:
    """把同步 urllib 调用丢到线程池，避免阻塞事件循环。"""
    return await asyncio.to_thread(default_health_checker, url, timeout_s)


class _AdapterFactory:
    """按 RemoteConfig.method 构造 Adapter（可注入便于测试）。"""

    def __init__(self, builder: Callable[[RemoteConfig], Tunnel] | None = None) -> None:
        self._builder = builder

    def build(self, config: RemoteConfig) -> Tunnel:
        if self._builder is not None:
            return self._builder(config)
        if config.method is RemoteMethod.TAILSCALE_SERVE:
            return TailscaleServeAdapter(config)
        if config.method is RemoteMethod.TAILSCALE_DIRECT:
            return TailscaleDirectAdapter(config)
        if config.method is RemoteMethod.CLOUDFLARE_ACCESS:
            return CloudflareAccessAdapter(config)
        raise ValueError(f"unsupported remote method: {config.method!r}")


class RemoteManager:
    """远程访问编排器。

    使用：
        manager = RemoteManager(config)
        await manager.start("http://127.0.0.1:8000")
        status = await manager.status()
        await manager.stop()
    """

    def __init__(
        self,
        config: RemoteConfig,
        *,
        credential_store: CredentialStore | None = None,
        adapter_factory: _AdapterFactory | None = None,
        health_checker: HealthChecker | None = None,
    ) -> None:
        self._config = config
        self._credentials = credential_store or CredentialStore()
        self._factory = adapter_factory or _AdapterFactory()
        self._health_checker = health_checker or _default_async_health_checker
        self._tunnel: Tunnel | None = None
        self._public_url: str | None = None

    @property
    def method(self) -> str:
        """当前方案标识。"""
        return self._config.method.value

    @property
    def public_url(self) -> str | None:
        """当前公网 URL（start 后有效）。"""
        return self._public_url

    def select_adapter(self) -> Tunnel:
        """按 config 选 Adapter（不缓存，便于测试覆盖重建）。"""
        return self._factory.build(self._config)

    async def prepare(self) -> dict[str, Any]:
        """预检 Adapter，返回受信任主机清单。"""
        tunnel = self.select_adapter()
        self._tunnel = tunnel
        return await tunnel.prepare()

    async def start(self, target_url: str) -> str:
        """启动 tunnel 并健康检查，返回公网 URL。"""
        if self._tunnel is None:
            self._tunnel = self.select_adapter()
            await self._tunnel.prepare()
        url = await self._tunnel.start(target_url)
        self._public_url = url
        # 健康检查（不阻塞 start 返回；失败只记 error 状态）
        await self._health_check_loop(url)
        return url

    async def stop(self) -> None:
        """停止 tunnel。"""
        if self._tunnel is None:
            return
        await self._tunnel.stop()
        self._public_url = None

    async def status(self) -> TunnelStatus:
        """返回当前状态快照。"""
        if self._tunnel is None:
            return TunnelStatus(
                state=TunnelState.DISABLED,
                method=self.method,
                transport=self._config.method.value.split("-")[-1],
            )
        return await self._tunnel.status()

    async def health_check(self, url: str | None = None) -> bool:
        """对 tunnel URL 健康端点发 GET，200 视为健康。"""
        target = url or self._public_url
        if not target:
            return False
        path = self._config.health_check_path or "/healthz"
        full = target.rstrip("/") + path
        try:
            code = await self._health_checker(full, self._config.health_timeout_s)
        except Exception as exc:  # pragma: no cover - 防御性
            _logger.warning("health check %s failed: %s", full, exc)
            return False
        return code == 200

    async def _health_check_loop(self, url: str) -> None:
        """start 后轮询健康端点；失败最多重试 health_retries 次。"""
        for attempt in range(1, self._config.health_retries + 1):
            ok = await self.health_check(url)
            if ok:
                return
            await asyncio.sleep(self._config.health_interval_s)
        # 全部失败 → 记日志但不 raise（start 已成功，健康检查是 best-effort）
        _logger.warning(
            "tunnel %s health check failed after %d attempts",
            url,
            self._config.health_retries,
        )

    def credential_snapshot(self) -> dict[str, bool]:
        """返回凭据存在性快照（不暴露真实值）。"""
        keys = self._credential_keys()
        return self._credentials.snapshot(keys)

    def _credential_keys(self) -> list[str]:
        """按方案返回需要的凭据 key 列表。"""
        if self._config.method is RemoteMethod.TAILSCALE_SERVE:
            return ["tailscale_auth_key"]
        if self._config.method is RemoteMethod.TAILSCALE_DIRECT:
            return ["tailscale_auth_key"]
        if self._config.method is RemoteMethod.CLOUDFLARE_ACCESS:
            return [
                "cloudflared_tunnel_token",
                "cloudflare_team_domain",
                "cloudflare_audience",
            ]
        return []

    def describe(self) -> dict[str, Any]:
        """返回人可读的当前方案描述（用于 UI 状态栏）。"""
        return {
            "method": self.method,
            "enabled": self._config.enabled,
            "credentials": self.credential_snapshot(),
            "public_url": self._public_url,
        }
