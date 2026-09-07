"""远程访问 Adapter 集合（全部 Mock，不真实连外部服务）。"""

from __future__ import annotations

from .tailscale_serve import TailscaleServeAdapter
from .tailscale_direct import TailscaleDirectAdapter
from .cloudflare_access import CloudflareAccessAdapter

__all__ = [
    "TailscaleServeAdapter",
    "TailscaleDirectAdapter",
    "CloudflareAccessAdapter",
]
