"""Tailscale Serve over HTTPS — Mock Adapter。

模拟 `tailscale serve --https <port> <target>` 的启动 / 停止 / 状态查询，
不真实调用 tailscale CLI、不真实绑定 443、不真实签发证书。

Mock 策略：
    - prepare：检查命令是否"可用"（Mock 时命令非空即视为可用），返回
      一个稳定的 *.ts.net mock 主机名。
    - start：校验 target URL 是 loopback，置 active，返回
      `https://<hostname>` 公网 URL。
    - stop：置回 disabled / ready，清理"进程"。
    - status：返回当前 TunnelStatus。
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlparse

from ..config import RemoteConfig
from ..tunnel import Tunnel, TunnelError, TunnelState, TunnelStatus


class TailscaleServeAdapter(Tunnel):
    """Tailscale Serve 方案的 Mock 实现。

    真实方案：tailscale serve --https 443 http://127.0.0.1:<port>
    Mock 方案：模拟生命周期，不真实启动 subprocess。
    """

    def __init__(self, config: RemoteConfig) -> None:
        self._config = config
        self._hostname: str | None = None
        self._target: str | None = None
        self._running = False
        # Mock 状态机：disabled → (prepare) ready → (start) active → (stop) disabled
        self._state: TunnelState = TunnelState.DISABLED
        self._error: str | None = None

    @property
    def method(self) -> str:
        return "tailscale-serve"

    @property
    def transport(self) -> str:
        return "serve"

    async def prepare(self) -> dict[str, Any]:
        """预检：命令可用则解析 mock 主机名，否则标 unavailable。"""
        if not self._config.tailscale_command:
            self._state = TunnelState.UNAVAILABLE
            self._error = "tailscale command not configured"
            return {"trusted_hosts": []}
        # Mock 主机名：优先用 config.extra 中的 tailscale_hostname
        host = self._config.extra.get("tailscale_hostname")
        if not isinstance(host, str) or not host:
            host = "mock-node.tailnet.ts.net"
        self._hostname = host
        self._state = TunnelState.READY
        self._error = None
        return {"trusted_hosts": [host]}

    async def start(self, target_url: str) -> str:
        """启动 Serve，返回 https://<hostname>。"""
        if not self._config.tailscale_command:
            self._state = TunnelState.UNAVAILABLE
            self._error = "tailscale command not configured"
            raise TunnelError("status", "tailscale command not configured")
        if self._hostname is None:
            raise TunnelError(
                "status",
                "Tailscale Serve must be prepared before it starts",
            )
        _validate_loopback_target(target_url)
        # 模拟 tailscale serve 启动耗时
        await asyncio.sleep(0)
        self._target = target_url
        self._running = True
        self._state = TunnelState.ACTIVE
        self._error = None
        return f"https://{self._hostname}"

    async def stop(self) -> None:
        """停止 Serve（Mock：直接置 disabled）。"""
        self._running = False
        self._target = None
        if self._hostname is not None:
            self._state = TunnelState.READY
        else:
            self._state = TunnelState.DISABLED
        self._error = None

    async def status(self) -> TunnelStatus:
        """返回当前状态快照。"""
        url = f"https://{self._hostname}" if (self._running and self._hostname) else None
        return TunnelStatus(
            state=self._state,
            url=url,
            error=self._error,
            method=self.method,
            transport=self.transport,
            metadata={
                "hostname": self._hostname,
                "target": self._target,
                "serve_port": self._config.tailscale_serve_port,
            },
        )


def _validate_loopback_target(target_url: str) -> None:
    """校验 target 必须是 loopback URL（防止 Mock 也吃公网目标）。"""
    try:
        parsed = urlparse(target_url)
    except Exception as exc:  # pragma: no cover
        raise TunnelError("serve", f"invalid target URL: {target_url}") from exc
    if parsed.scheme not in ("http", "https"):
        raise TunnelError("serve", f"target scheme must be http/https: {target_url}")
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise TunnelError(
            "serve",
            f"target must be loopback, got hostname={parsed.hostname}",
        )
    if not parsed.port:
        raise TunnelError("serve", f"target must include explicit port: {target_url}")
