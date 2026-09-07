"""Tailscale Direct IP — Mock Adapter。

模拟 Tailscale 节点 CGNAT IPv4 直连：在 100.64.0.0/10 地址上开端口转发到
本地 loopback target。不真实启动 tailscale daemon、不真实 bind socket。

Mock 策略：
    - prepare：解析 Tailscale IP（config 提供，否则 mock 一个 100.x.x.x）。
    - start：置 active，返回 http://<ip>:<port>。
    - stop：置回 ready / disabled。
"""

from __future__ import annotations

import asyncio
import ipaddress
from typing import Any
from urllib.parse import urlparse

from ..config import RemoteConfig
from ..tunnel import Tunnel, TunnelError, TunnelState, TunnelStatus


def _is_tailscale_ipv4(hostname: str) -> bool:
    """是否为 Tailscale CGNAT IPv4（100.64.0.0/10）。"""
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    if not isinstance(addr, ipaddress.IPv4Address):
        return False
    net = ipaddress.ip_network("100.64.0.0/10")
    return addr in net


class TailscaleDirectAdapter(Tunnel):
    """Tailscale Direct IP 方案的 Mock 实现。

    真实方案：解析节点 Tailscale IPv4 → 在该 IP 上 bind TCP 端口 →
    把入站流量转发到本地 loopback target。Mock 方案：模拟生命周期。
    """

    def __init__(self, config: RemoteConfig) -> None:
        self._config = config
        self._ip: str | None = None
        self._port: int | None = None
        self._target: str | None = None
        self._running = False
        self._state: TunnelState = TunnelState.DISABLED
        self._error: str | None = None

    @property
    def method(self) -> str:
        return "tailscale-direct"

    @property
    def transport(self) -> str:
        return "direct"

    async def prepare(self) -> dict[str, Any]:
        """预检：校验配置的 IP，返回 ip:port 主机清单。"""
        if not self._config.tailscale_command:
            self._state = TunnelState.UNAVAILABLE
            self._error = "tailscale command not configured"
            return {"trusted_hosts": []}
        ip = self._config.tailscale_ip_address or "100.64.0.1"
        if not _is_tailscale_ipv4(ip):
            raise TunnelError(
                "direct-ip",
                f"configured Tailscale IP must be in 100.64.0.0/10, got {ip}",
            )
        self._ip = ip
        self._port = self._config.tailscale_direct_port
        self._state = TunnelState.READY
        self._error = None
        authority = f"{self._ip}:{self._port}"
        return {"trusted_hosts": [authority]}

    async def start(self, target_url: str) -> str:
        """启动直连转发，返回 http://<ip>:<port>。"""
        if not self._config.tailscale_command:
            self._state = TunnelState.UNAVAILABLE
            self._error = "tailscale command not configured"
            raise TunnelError("status", "tailscale command not configured")
        if self._ip is None or self._port is None:
            raise TunnelError(
                "status",
                "Tailscale direct access must be prepared before it starts",
            )
        _validate_loopback_target(target_url)
        await asyncio.sleep(0)
        self._target = target_url
        self._running = True
        self._state = TunnelState.ACTIVE
        self._error = None
        return f"http://{self._ip}:{self._port}"

    async def stop(self) -> None:
        """停止直连（Mock：清理状态）。"""
        self._running = False
        self._target = None
        if self._ip is not None and self._port is not None:
            self._state = TunnelState.READY
        else:
            self._state = TunnelState.DISABLED
        self._error = None

    async def status(self) -> TunnelStatus:
        """返回当前状态快照。"""
        url = None
        if self._running and self._ip and self._port:
            url = f"http://{self._ip}:{self._port}"
        return TunnelStatus(
            state=self._state,
            url=url,
            error=self._error,
            method=self.method,
            transport=self.transport,
            metadata={
                "ip": self._ip,
                "port": self._port,
                "target": self._target,
            },
        )


def _validate_loopback_target(target_url: str) -> None:
    """校验 target 必须是 loopback URL。"""
    try:
        parsed = urlparse(target_url)
    except Exception as exc:  # pragma: no cover
        raise TunnelError("direct-bind", f"invalid target URL: {target_url}") from exc
    if parsed.scheme not in ("http", "https"):
        raise TunnelError("direct-bind", f"target scheme must be http/https: {target_url}")
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise TunnelError(
            "direct-bind",
            f"target must be loopback, got hostname={parsed.hostname}",
        )
    if not parsed.port:
        raise TunnelError("direct-bind", f"target must include explicit port: {target_url}")
