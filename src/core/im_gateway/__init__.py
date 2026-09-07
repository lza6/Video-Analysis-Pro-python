"""IM 网关框架（Builder-C 范围）。

让 agent 通过 IM（微信 / Telegram / Discord）触发并回复，参考 Minke 的
im-gateway 概念：SQLite mailbox + lease/ack 投递语义 + AES 凭据加密。

模块组成：
  - mailbox.py : IMMailbox（SQLite，入站/出站消息，lease/ack/超时重 lease）
  - cipher.py  : AES 加解密网关凭据（条件依赖 cryptography，降级告警）
  - adapters/  : IMAdapter 抽象基类 + Wechat/Telegram/Discord 三个 Mock 实现
  - gateway.py : IMGateway（start/stop/轮询/lease→callback→ack→send）
  - route.py   : AgentCallback Protocol + 路由（收口时主控接真 agent）

设计原则：
  - 全 Mock，不真实连 IM，不真实付费
  - Python 3.10+，asyncio，完整类型注解
  - 不碰 src/core/agent/、tools/、credentials/ 等 B/D 范围
  - config_manager.py 只读 import（凭据存储后端）
"""
from __future__ import annotations

from src.core.im_gateway.mailbox import IMMailbox, IMMessage
from src.core.im_gateway.cipher import GatewayCipher
from src.core.im_gateway.route import AgentCallback, EchoAgentCallback, route_message
from src.core.im_gateway.gateway import IMGateway
from src.core.im_gateway.adapters.base import IMAdapter
from src.core.im_gateway.adapters.wechat import WechatAdapter
from src.core.im_gateway.adapters.telegram import TelegramAdapter
from src.core.im_gateway.adapters.discord import DiscordAdapter

__all__ = [
    "IMMailbox",
    "IMMessage",
    "GatewayCipher",
    "AgentCallback",
    "EchoAgentCallback",
    "route_message",
    "IMGateway",
    "IMAdapter",
    "WechatAdapter",
    "TelegramAdapter",
    "DiscordAdapter",
]
