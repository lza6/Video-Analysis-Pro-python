"""Cloudflare Access (named tunnel) — Mock Adapter。

模拟 `cloudflared tunnel run <name>` 的启动 / 停止，以及 Cloudflare Access
的 JWT 验证层。不真实启动 cloudflared、不真实连 Cloudflare API、不真实签发
证书。凭据（tunnel token / team domain / audience）走密钥环，Adapter 只
读 config 里的非密钥字段。

Mock 策略：
    - prepare：校验 hostname / team_domain / tunnel 字段非空，标记 ready。
    - start：置 active，返回 https://<hostname>。
    - stop：置回 disabled / ready。
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlparse

from ..config import RemoteConfig
from ..tunnel import Tunnel, TunnelError, TunnelState, TunnelStatus

# 与 Minke cloudflare.ts 同源的正则约束
_DNS_NAME = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$",
)
_TEAM_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_TUNNEL_NAME = re.compile(r"^(?=.{1,256}$)[A-Za-z0-9][A-Za-z0-9._-]*$")


class CloudflareAccessAdapter(Tunnel):
    """Cloudflare Access 方案的 Mock 实现。

    真实方案：cloudflared tunnel run <name> + Cloudflare Access JWT 验证。
    Mock 方案：模拟生命周期，凭据从 CredentialStore 读（Mock 测试注入）。
    """

    def __init__(self, config: RemoteConfig) -> None:
        self._config = config
        self._target: str | None = None
        self._running = False
        self._state: TunnelState = TunnelState.DISABLED
        self._error: str | None = None

    @property
    def method(self) -> str:
        return "cloudflare-access"

    @property
    def transport(self) -> str:
        return "access"

    async def prepare(self) -> dict[str, Any]:
        """预检：校验非密钥配置字段。"""
        if not self._config.cloudflared_command:
            self._state = TunnelState.UNAVAILABLE
            self._error = "cloudflared command not configured"
            return {"trusted_hosts": []}
        try:
            self._validate_config()
        except TunnelError as exc:
            self._state = TunnelState.ERROR
            self._error = exc.kind
            raise
        self._state = TunnelState.READY
        self._error = None
        host = self._config.cloudflare_hostname
        return {"trusted_hosts": [host] if host else []}

    async def start(self, target_url: str) -> str:
        """启动 cloudflared tunnel，返回 https://<hostname>。"""
        if not self._config.cloudflared_command:
            self._state = TunnelState.UNAVAILABLE
            self._error = "cloudflared command not configured"
            raise TunnelError("status", "cloudflared command not configured")
        if self._state not in (TunnelState.READY, TunnelState.ACTIVE):
            raise TunnelError(
                "cloudflare-access",
                "Cloudflare Access must be prepared before it starts",
            )
        _validate_loopback_target(target_url)
        await asyncio.sleep(0)
        self._target = target_url
        self._running = True
        self._state = TunnelState.ACTIVE
        self._error = None
        return f"https://{self._config.cloudflare_hostname}"

    async def stop(self) -> None:
        """停止 tunnel（Mock：清理状态）。"""
        self._running = False
        self._target = None
        if self._config.cloudflare_hostname:
            self._state = TunnelState.READY
        else:
            self._state = TunnelState.DISABLED
        self._error = None

    async def status(self) -> TunnelStatus:
        """返回当前状态快照。"""
        url = None
        if self._running and self._config.cloudflare_hostname:
            url = f"https://{self._config.cloudflare_hostname}"
        return TunnelStatus(
            state=self._state,
            url=url,
            error=self._error,
            method=self.method,
            transport=self.transport,
            metadata={
                "hostname": self._config.cloudflare_hostname,
                "team_domain": self._config.cloudflare_team_domain,
                "tunnel": self._config.cloudflare_tunnel,
                "target": self._target,
            },
        )

    def _validate_config(self) -> None:
        """校验 hostname / team / tunnel 形态（与 Minke cloudflare.ts 对齐）。"""
        host = self._config.cloudflare_hostname
        if not host or not _DNS_NAME.match(host):
            raise TunnelError(
                "cloudflare-config",
                f"invalid cloudflare hostname: {host!r}",
            )
        team = self._config.cloudflare_team_domain
        if not team or not _TEAM_NAME.match(team):
            raise TunnelError(
                "cloudflare-config",
                f"invalid cloudflare team domain: {team!r}",
            )
        tunnel = self._config.cloudflare_tunnel
        if not tunnel or not _TUNNEL_NAME.match(tunnel):
            raise TunnelError(
                "cloudflare-config",
                f"invalid cloudflare tunnel name: {tunnel!r}",
            )


def _validate_loopback_target(target_url: str) -> None:
    """校验 target 必须是 loopback URL。"""
    try:
        parsed = urlparse(target_url)
    except Exception as exc:  # pragma: no cover
        raise TunnelError("cloudflare-tunnel", f"invalid target URL: {target_url}") from exc
    if parsed.scheme not in ("http", "https"):
        raise TunnelError("cloudflare-tunnel", f"target scheme must be http/https: {target_url}")
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise TunnelError(
            "cloudflare-tunnel",
            f"target must be loopback, got hostname={parsed.hostname}",
        )
    if not parsed.port:
        raise TunnelError("cloudflare-tunnel", f"target must include explicit port: {target_url}")
