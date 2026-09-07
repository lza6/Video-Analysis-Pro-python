"""Tunnel 抽象基类与状态对象。

所有 Adapter 实现此接口；Mock 与真实 Adapter 共用同一契约。
参考 Minke `lifecycle.ts` 的 `RemoteAccessLifecycle`。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TunnelState(str, Enum):
    """Tunnel 运行态（与 Minke RemoteRuntimeState 对齐）。"""

    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    STARTING = "starting"
    STOPPING = "stopping"
    READY = "ready"
    ACTIVE = "active"
    ERROR = "error"


@dataclass(frozen=True)
class TunnelStatus:
    """Tunnel 一次状态快照（不可变，便于跨线程 / 事件传播）。"""

    state: TunnelState
    url: str | None = None
    error: str | None = None
    method: str = ""
    transport: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_running(self) -> bool:
        """是否对外可服务：state 为 ready/active 且有公网 url。

        READY 无 url（stop 后 / prepare 未 start）不视为对外可服务。
        """
        return self.state in (TunnelState.READY, TunnelState.ACTIVE) and bool(self.url)

    def is_error(self) -> bool:
        """是否处于错误态。"""
        return self.state == TunnelState.ERROR


class Tunnel(ABC):
    """远程访问 tunnel 抽象基类。

    生命周期：prepare(可选) → start(target) → status / health → stop。
    所有方法均为 async，方便真实 Adapter 接 subprocess / 网络轮询；
    Mock Adapter 同样 async 以保持契约一致。
    """

    @property
    @abstractmethod
    def method(self) -> str:
        """方案标识：tailscale-serve / tailscale-direct / cloudflare-access。"""

    @property
    @abstractmethod
    def transport(self) -> str:
        """传输类型：serve / direct / access。"""

    @abstractmethod
    async def prepare(self) -> dict[str, Any]:
        """预检：探测外部命令可用性、解析节点地址等，返回受信任主机清单。

        Returns:
            dict 至少含 `trusted_hosts: list[str]`。
        """

    @abstractmethod
    async def start(self, target_url: str) -> str:
        """对 target_url（本地 loopback 服务）建立 tunnel，返回公网 URL。

        Raises:
            TunnelError: 启动失败（命令不可用 / 超时 / 冲突等）。
        """

    @abstractmethod
    async def stop(self) -> None:
        """停止 tunnel，释放外部进程 / 端口。"""

    @abstractmethod
    async def status(self) -> TunnelStatus:
        """返回当前状态快照。"""


class TunnelError(RuntimeError):
    """Tunnel 操作失败。"""

    def __init__(self, kind: str, message: str, *, cause: BaseException | None = None):
        super().__init__(message)
        self.kind = kind
        self.cause = cause

    def __str__(self) -> str:  # pragma: no cover - 便于调试
        cause = f" (cause={self.cause!r})" if self.cause else ""
        return f"[{self.kind}] {self.args[0]}{cause}"
