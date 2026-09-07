"""TingFeng Hermes — 远程访问框架。

让本地 Agent 服务可被外部网络安全触达，三方案全部 Mock 实现，
不真实连接 Tailscale / Cloudflare。参考 Minke `packages/remote-access`。

公共出口：
    - Tunnel / TunnelStatus / TunnelState  (tunnel.py)
    - RemoteManager                          (manager.py)
    - RemoteConfig / CredentialStore          (config.py)
    - 三 Adapter                              (adapters/)
"""

from __future__ import annotations

from .tunnel import Tunnel, TunnelState, TunnelStatus
from .manager import RemoteManager
from .config import RemoteConfig, CredentialStore, RemoteMethod

__all__ = [
    "Tunnel",
    "TunnelState",
    "TunnelStatus",
    "RemoteManager",
    "RemoteConfig",
    "CredentialStore",
    "RemoteMethod",
]
