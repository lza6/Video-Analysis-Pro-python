"""微信 adapter Mock 实现。

不真实连微信（无真实凭据，无付费，无封号风险）。send 调用记到内存
sent_log，receive 从注入的 inbound_queue pop。测试时用 inject_inbound
灌入模拟的入站消息，验证 gateway 的 receive→lease→ack→send 闭环。

真实 WechatAdapter 未来需要：
  - 接入 itchat / wechatpy / 企业微信 SDK（个人号接口已受限，企业微信可官方）
  - 凭据（corpid + secret 或个人 token）从 GatewayCipher 解密后用
  - 真实凭据需主控授权 + 预算（本项目凭据预算 0，不真实连）
"""
from __future__ import annotations

import logging
from typing import Deque, List, Optional

from src.core.im_gateway.adapters.base import IMAdapter, InboundEnvelope

logger = logging.getLogger(__name__)


class WechatAdapter(IMAdapter):
    """微信 adapter Mock。

    用法（测试）：
        adapter = WechatAdapter()
        await adapter.start()
        await adapter.inject_inbound("wxid_xxx", "分析这段视频")
        env = await adapter.receive()
        await adapter.send("wxid_xxx", "收到")  # 记入 sent_log
    """

    def __init__(self) -> None:
        self._inbound: Deque[InboundEnvelope]
        self.sent_log: List[InboundEnvelope]
        self._inbound, self.sent_log = self._init_mock_queues()
        self._started = False

    @property
    def channel_name(self) -> str:
        return "wechat"

    async def start(self) -> None:
        self._started = True
        logger.info("WechatAdapter(Mock) started")

    async def stop(self) -> None:
        self._started = False
        logger.info("WechatAdapter(Mock) stopped")

    async def send(self, peer: str, text: str) -> bool:
        if not self._started:
            logger.warning("WechatAdapter.send 时 adapter 未 start")
        # 记录出站（用 InboundEnvelope 复用结构，peer+content+raw）
        self.sent_log.append(InboundEnvelope(peer=peer, content=text, raw={}))
        logger.info("WechatAdapter(Mock) send -> %s: %s", peer, text[:50])
        return True

    async def receive(self) -> Optional[InboundEnvelope]:
        if not self._inbound:
            return None
        return self._inbound.popleft()

    async def inject_inbound(self, peer: str, content: str) -> None:
        """测试辅助：注入一条入站消息（模拟从微信收到）。"""
        self._inbound.append(InboundEnvelope(peer=peer, content=content, raw={}))
