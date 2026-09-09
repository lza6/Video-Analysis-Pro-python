# -*- coding: utf-8 -*-
"""ProviderRouterClient —— 把 ProviderRouter.post_nvidia 包装成 loop.LLMClient。

loop.py:10,52 docstring 提及"生产用 ProviderRouterClient(包 ProviderRouter)",
但 v10.1.0 未落地定义。本类补齐该缺口:对外暴露与 MockLLMClient 相同的
async stream(messages, tools) -> AsyncIterator[LLMChunk] 协议。

设计(守付费 API 红线):
- 真实 provider 调用只在用户配了 VAP_NV_API_KEYS / NVIDIA key 时才发生;
  本地测试全走 Mock 的 fake router。
- ProviderRouter.post_nvidia 是同步方法,本类用 asyncio.to_thread 包一层,
  避免阻塞 ReactLoopAgent 的 asyncio 事件循环。
- tools 透传:通过 nvidia_models.build_nvidia_payload(tools=...) 塞进 payload
  顶层(OpenAI 兼容 tool calling),无 tools 时不塞(零回归)。

协议对齐:与 loop.LLMClient Protocol 一致——返回 AsyncIterator[LLMChunk],
单次 yield 汇总的 assistant chunk(非逐 token),ReactLoopAgent 逐 chunk 累积。
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from src.core.agent.loop import LLMChunk

# 与 _nvidia_chat(agent.py:813) 保持同一默认模型,避免两处漂移
DEFAULT_NV_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


class ProviderRouterClient:
    """LLMClient 协议的生产实现:把 ProviderRouter 路由到一个真实 LLM。

    Attributes:
        router: 已配置的 ProviderRouter 实例(provider_router.py:200)。
        model_id: NVIDIA 模型 id(默认读 VAP_NV_VIDEO_MODEL,与 _nvidia_chat 一致)。
    """

    def __init__(
        self,
        router: Any,
        model_id: Optional[str] = None,
    ) -> None:
        self._router = router
        self._model_id = model_id

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[LLMChunk]:
        """把 (messages, tools) 交给 ProviderRouter,产出 LLMChunk 流。

        与 MockLLMClient.stream 保持同构,ReactLoopAgent 无需感知差异。
        """
        from src.core.nvidia_models import build_nvidia_payload

        model = self._model_id or os.environ.get("VAP_NV_VIDEO_MODEL", DEFAULT_NV_MODEL)
        payload = build_nvidia_payload(
            model_id=model,
            messages=[dict(m) for m in messages],
            enable_thinking=True,
            stream=False,
            max_tokens=int(os.environ.get("VAP_NV_MAX_TOKENS", "65536")),
            tools=tools or None,
        )
        # ProviderRouter.post_nvidia 是同步阻塞(httpx 同步),丢到线程池防卡事件循环
        resp = await asyncio.to_thread(
            self._router.post_nvidia, payload, timeout=120
        )

        choices = resp.get("choices", []) if isinstance(resp, dict) else []
        text, tool_calls = "", None
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            parts = []
            if reasoning:
                parts.append(f"💭{reasoning}\n\n")
            if content:
                parts.append(content)
            text = "".join(parts)
            tool_calls = list(msg.get("tool_calls") or []) or None

        yield LLMChunk(
            delta_text=text,
            final_tool_calls=tool_calls,
            stop_reason="stop" if not tool_calls else None,
        )