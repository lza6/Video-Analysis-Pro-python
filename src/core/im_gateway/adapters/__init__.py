"""IM adapter 抽象基类与实现汇总。

IMAdapter 定义统一接口（start/stop/send/receive），三个 Mock 实现：
  - WechatAdapter
  - TelegramAdapter
  - DiscordAdapter

Mock 行为：send 把出站消息记到内存 deque，receive 从注入的入站队列取
（测试时由调用方灌入），不真实连 IM 平台。
"""
from __future__ import annotations

from src.core.im_gateway.adapters.base import IMAdapter
from src.core.im_gateway.adapters.wechat import WechatAdapter
from src.core.im_gateway.adapters.telegram import TelegramAdapter
from src.core.im_gateway.adapters.discord import DiscordAdapter

__all__ = [
    "IMAdapter",
    "WechatAdapter",
    "TelegramAdapter",
    "DiscordAdapter",
]
