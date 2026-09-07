"""IM adapter 抽象基类。

定义所有 IM 平台 adapter 的统一接口。真实 adapter（接微信/TG/Discord SDK）
将来实现本接口；Mock 实现用于网关开发期闭环测试，不真实连平台。

接口设计参考 Minke im-gateway 的 adapter 契约：
  - start() / stop() : 生命周期（连接 / 断开平台）
  - send(peer, text) : 发送一条出站消息
  - receive()        : 拉取一条入站消息（async，无消息时返回 None）
  - channel_name     : 平台标识（'wechat' / 'telegram' / 'discord'）
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
from collections import deque


@dataclass
class InboundEnvelope:
    """adapter.receive 返回的入站消息信封（不直接是 IMMessage）。

    gateway 拿到后调 mailbox.put_inbound(channel=..., peer=..., content=...)
    落库。adapter 不依赖 mailbox，职责仅是"从平台取消息"。
    """

    peer: str
    content: str
    raw: Dict[str, object] = field(default_factory=dict)


class IMAdapter(abc.ABC):
    """IM 平台 adapter 抽象基类。

    所有方法 async——真实 adapter 走 aiohttp/websocket 长连接；Mock
    实现直接操作内存 deque，await 时立即返回。
    """

    @property
    @abc.abstractmethod
    def channel_name(self) -> str:
        """平台标识（'wechat' / 'telegram' / 'discord'），与 mailbox.channel 对齐。"""
        raise NotImplementedError

    @abc.abstractmethod
    async def start(self) -> None:
        """启动 adapter（连平台 / 开长轮询）。Mock 只设状态标记。"""
        raise NotImplementedError

    @abc.abstractmethod
    async def stop(self) -> None:
        """停止 adapter（断连 / 取消轮询）。"""
        raise NotImplementedError

    @abc.abstractmethod
    async def send(self, peer: str, text: str) -> bool:
        """向 peer 发送一条文本。返回是否成功（Mock 永远 True）。"""
        raise NotImplementedError

    @abc.abstractmethod
    async def receive(self) -> Optional[InboundEnvelope]:
        """拉取一条入站消息。无消息返回 None（不阻塞）。

        gateway 轮询时反复调本方法。Mock 从注入的入站队列 pop。
        """
        raise NotImplementedError

    # 共享：Mock 实现用这两个 deque 记录 send / 注入 receive
    def _init_mock_queues(self) -> tuple[Deque[InboundEnvelope], List[InboundEnvelope]]:
        """Mock 实现共享的内存队列初始化辅助。

        返回 (入站注入队列, 出站记录列表)。入站用 deque（FIFO pop），
        出站用 list（保留全部历史，测试断言查）。
        """
        return deque(), []
