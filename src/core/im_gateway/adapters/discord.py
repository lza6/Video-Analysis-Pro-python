"""Discord adapter Mock 实现。

真实 DiscordAdapter 未来接入 discord.py（Bot Gateway + on_message 事件）：
  - on_message 推入 inbound 队列
  - channel.send() 发出站回复
  - 凭据（bot token）从 GatewayCipher 解密后用

本 Mock 不真实连 Discord，记到内存队列。真实凭据预算 0，标"真实凭据待验证"。
"""
from __future__ import annotations

import logging
from typing import Deque, List, Optional

from src.core.im_gateway.adapters.base import IMAdapter, InboundEnvelope

logger = logging.getLogger(__name__)


class DiscordAdapter(IMAdapter):
    """Discord adapter Mock。"""

    def __init__(self) -> None:
        self._inbound: Deque[InboundEnvelope]
        self.sent_log: List[InboundEnvelope]
        self._inbound, self.sent_log = self._init_mock_queues()
        self._started = False

    @property
    def channel_name(self) -> str:
        return "discord"

    async def start(self) -> None:
        self._started = True
        logger.info("DiscordAdapter(Mock) started")

    async def stop(self) -> None:
        self._started = False
        logger.info("DiscordAdapter(Mock) stopped")

    async def send(self, peer: str, text: str) -> bool:
        if not self._started:
            logger.warning("DiscordAdapter.send 时 adapter 未 start")
        self.sent_log.append(InboundEnvelope(peer=peer, content=text, raw={}))
        logger.info("DiscordAdapter(Mock) send -> %s: %s", peer, text[:50])
        return True

    async def receive(self) -> Optional[InboundEnvelope]:
        if not self._inbound:
            return None
        return self._inbound.popleft()

    async def inject_inbound(self, peer: str, content: str) -> None:
        """测试辅助：注入一条入站消息（模拟从 Discord 收到）。"""
        self._inbound.append(InboundEnvelope(peer=peer, content=content, raw={}))
