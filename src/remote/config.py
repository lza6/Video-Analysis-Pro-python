"""远程访问配置 + 凭据读取。

凭据复用 `src/utils/config_manager.py` 的密钥环封装（只读 import），
不重写密钥环逻辑、不把真实 Key 写入 ini。方案选择 + 端口 + 命令路径
在此处集中，Adapter 只接收已解析好的 RemoteConfig。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class RemoteMethod(str, Enum):
    """远程访问方案选择（与 Minke REMOTE_METHODS 对齐 + 拆分 tailscale 传输）。"""

    TAILSCALE_SERVE = "tailscale-serve"
    TAILSCALE_DIRECT = "tailscale-direct"
    CLOUDFLARE_ACCESS = "cloudflare-access"


class CredentialFetcher(Protocol):
    """凭据读取协议：复用 config_manager._secure_get 的最小接口。"""

    def get(self, key: str, fallback: str = "") -> str: ...


class _DefaultCredentialFetcher:
    """默认凭据读取器：委托 config_manager._secure_get。

    只读 import：不写回、不创建 ini、不轮换密钥。
    单测可注入 Mock fetcher 覆盖。
    """

    def __init__(self) -> None:
        try:
            from src.utils.config_manager import _secure_get  # type: ignore
            self._secure_get = _secure_get  # type: ignore[assignment]
        except Exception:  # pragma: no cover - config_manager 缺失时不阻断
            self._secure_get = None  # type: ignore[assignment]

    def get(self, key: str, fallback: str = "") -> str:
        if self._secure_get is None:
            return fallback
        try:
            return self._secure_get(key, fallback)
        except Exception:
            return fallback


@dataclass(frozen=True)
class RemoteConfig:
    """远程访问方案配置（不可变）。

    所有命令路径默认空字符串，Adapter 收到空命令时直接判定 unavailable，
    不真实调用 subprocess。Mock Adapter 忽略命令是否真实存在。
    """

    enabled: bool = False
    method: RemoteMethod = RemoteMethod.TAILSCALE_SERVE
    # Tailscale
    tailscale_command: str = ""
    tailscale_ip_address: str = ""
    tailscale_serve_port: int = 443
    tailscale_direct_port: int = 8080
    # Cloudflare
    cloudflared_command: str = ""
    cloudflare_hostname: str = ""
    cloudflare_team_domain: str = ""
    cloudflare_audience: str = ""
    cloudflare_tunnel: str = ""
    cloudflare_config_path: str = ""
    cloudflare_origin_port: int = 49_321
    # 健康检查
    health_check_path: str = "/healthz"
    health_timeout_s: float = 3.0
    health_retries: int = 5
    health_interval_s: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)

    def trusted_hosts(self) -> list[str]:
        """返回受信任主机清单（prepare 阶段喂给本地服务的允许源）。"""
        if self.method == RemoteMethod.TAILSCALE_SERVE:
            host = self.extra.get("tailscale_hostname")
            return [host] if isinstance(host, str) and host else []
        if self.method == RemoteMethod.TAILSCALE_DIRECT:
            ip = self.tailscale_ip_address
            return [f"{ip}:{self.tailscale_direct_port}"] if ip else []
        if self.method == RemoteMethod.CLOUDFLARE_ACCESS:
            return [self.cloudflare_hostname] if self.cloudflare_hostname else []
        return []


class CredentialStore:
    """远程访问凭据存储（只读视图）。

    复用 config_manager 的密钥环封装，单测可注入 fetcher Mock。
    """

    # 凭据 key 前缀，避免与其它模块的 keyring 条目撞名
    PREFIX = "remote_"

    def __init__(self, fetcher: CredentialFetcher | None = None) -> None:
        self._fetcher = fetcher or _DefaultCredentialFetcher()

    def get(self, key: str, fallback: str = "") -> str:
        """读取凭据；缺失返回 fallback。"""
        full_key = f"{self.PREFIX}{key}"
        return self._fetcher.get(full_key, fallback)

    def has(self, key: str) -> bool:
        """凭据是否存在（非空）。"""
        return bool(self.get(key, ""))

    def snapshot(self, keys: list[str]) -> dict[str, bool]:
        """返回 key 是否存在的布尔字典（不暴露真实值）。"""
        return {k: self.has(k) for k in keys}


def from_env() -> RemoteConfig:
    """从环境变量构建 RemoteConfig（运行时配置，不入库）。

    真实凭据走密钥环，不在环境变量里。
    """
    method_str = os.environ.get("VAP_REMOTE_METHOD", "").strip().lower()
    method = _parse_method(method_str)

    tailscale_ip = os.environ.get("VAP_TAILSCALE_IP", "").strip()
    cloudflare_hostname = os.environ.get("VAP_CLOUDFLARE_HOSTNAME", "").strip()
    cloudflare_team = os.environ.get("VAP_CLOUDFLARE_TEAM", "").strip()

    return RemoteConfig(
        enabled=os.environ.get("VAP_REMOTE_ENABLED", "").strip().lower() in ("1", "true", "yes"),
        method=method,
        tailscale_command=os.environ.get("VAP_TAILSCALE_COMMAND", "tailscale"),
        tailscale_ip_address=tailscale_ip,
        tailscale_direct_port=int(os.environ.get("VAP_TAILSCALE_DIRECT_PORT", "8080")),
        cloudflared_command=os.environ.get("VAP_CLOUDFLARED_COMMAND", "cloudflared"),
        cloudflare_hostname=cloudflare_hostname,
        cloudflare_team_domain=cloudflare_team,
        cloudflare_audience=os.environ.get("VAP_CLOUDFLARE_AUDIENCE", "").strip(),
        cloudflare_tunnel=os.environ.get("VAP_CLOUDFLARE_TUNNEL", "").strip(),
        cloudflare_config_path=os.environ.get("VAP_CLOUDFLARE_CONFIG", "").strip(),
    )


def _parse_method(value: str) -> RemoteMethod:
    """解析方案字符串；空 / 未知回退到默认 tailscale-serve。"""
    mapping = {
        "tailscale-serve": RemoteMethod.TAILSCALE_SERVE,
        "tailscale-serve-https": RemoteMethod.TAILSCALE_SERVE,
        "tailscale-direct": RemoteMethod.TAILSCALE_DIRECT,
        "tailscale-direct-ip": RemoteMethod.TAILSCALE_DIRECT,
        "cloudflare-access": RemoteMethod.CLOUDFLARE_ACCESS,
        "cloudflare": RemoteMethod.CLOUDFLARE_ACCESS,
    }
    return mapping.get(value, RemoteMethod.TAILSCALE_SERVE)
