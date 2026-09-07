"""Agent 回调协议与消息路由。

IMGateway 不直接依赖 Builder-B 的 ReactLoopAgent 实现（B 还在并行）。
本模块定义 AgentCallback Protocol：gateway lease 一条入站消息后，调
callback.handle(text) 拿到 agent 的回复文本，再 ack + adapter.send 回复。

收口时主控把真 agent（ReactLoopAgent 或任何符合本 Protocol 的对象）
接上 gateway.register_callback 即可。本模块提供 EchoAgentCallback
（原样回显，用于测试闭环）和 route_message（路由辅助函数）。
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentCallback(Protocol):
    """agent 处理协议：输入入站文本，返回回复文本。

    runtime_checkable 让 isinstance(obj, AgentCallback) 在运行时也能校验
    （Protocol 默认只静态校验）。

    实现方（ReactLoopAgent 或任何 callable-like）只需实现 async handle。
    """

    async def handle(self, text: str) -> str:
        """处理入站文本，返回回复文本。失败应抛异常，由 gateway 标 failed。"""
        ...


class EchoAgentCallback:
    """测试用回显 agent：原样把入站文本包成 "echo: <text>" 返回。

    用于 gateway 闭环测试（不依赖真 agent）。生产由主控接真 callback。
    """

    async def handle(self, text: str) -> str:
        return f"echo: {text}"


async def route_message(callback: AgentCallback, text: str) -> str:
    """路由一条入站文本到 agent callback，返回回复。

    单独抽出函数便于测试（直接调 EchoAgentCallback.route_message 而不
    走 gateway 全流程）。gateway 内部也是调 callback.handle。
    """
    return await callback.handle(text)
