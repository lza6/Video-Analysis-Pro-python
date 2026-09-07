"""In-process 子 agent 后端。

同进程 asyncio task 执行子 agent（最轻，默认）。
- foreground: 等结果文本
- background: fire-and-forget，返回 task_id，结果存内存字典供后续 poll/wait
- 默认用 MockLLMClient（loop.py 已有），避免真实付费 API（守付费红线）
- 异常隔离：子 agent 崩溃返回 error 文本，不抛到主流程

参考 DSH `dsh-plugin-subagent-director` 的 in-process 执行器概念：子 agent 不
跨进程 / 不跨 worker，直接在主 asyncio loop 内跑一个 ReactLoopAgent turn。
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from src.core.agent.loop import (
    AgentConfig,
    LLMChunk,
    MockLLMClient,
    ReactLoopAgent,
)
from src.core.agent.session import Session
from src.core.subagent.role_template import RoleTemplate
from src.core.subagent.route_resolver import RouteResolution


@dataclass
class _BgResult:
    """background 任务结果存储。"""

    status: str  # pending / done / error
    result: Optional[str] = None
    error: Optional[str] = None


class InProcessBackend:
    """同进程 asyncio task 子 agent 后端（默认最轻量）。

    Usage:
        backend = InProcessBackend()
        text = await backend.run(role, "做 X")            # foreground
        task_id = await backend.run(role, "做 Y", background=True)  # background
        result = await backend.wait(task_id)              # 后续取结果

    - foreground: 直接 await 一个 ReactLoopAgent turn
    - background: asyncio.create_task 跑同一逻辑，task_id 索引内存结果
    - 默认 LLM 走 MockLLMClient，绝不触真实付费 API（守红线）
    - 子 agent 异常被 _run_isolated 捕获，归一成 error 文本返回
    """

    def __init__(
        self,
        *,
        llm_factory: Optional[Callable[
            [RoleTemplate, Optional[RouteResolution]], Any]] = None,
        tool_registry: Any = None,
        system_prompt: str = "You are a helpful subagent.",
        max_steps: int = 8,
    ) -> None:
        self._llm_factory = llm_factory
        self._tool_registry = tool_registry
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        # background task 结果存储：task_id → _BgResult
        self._bg_results: Dict[str, _BgResult] = {}
        self._bg_tasks: Dict[str, asyncio.Task] = {}

    async def run(
        self,
        role: RoleTemplate,
        task: str,
        ctx: Optional[dict] = None,
        *,
        background: bool = False,
    ) -> str:
        """执行子 agent。

        Args:
            role: 角色模板。
            task: 任务文本。
            ctx: 上下文（含 route / subagent_id 等，director 注入）。
            background: True=fire-and-forget 返回 task_id；False=同步等结果。

        Returns:
            foreground：结果文本；background：task_id 字符串。
        """
        ctx = ctx or {}
        if background:
            task_id = uuid.uuid4().hex
            self._bg_results[task_id] = _BgResult(status="pending")
            loop = asyncio.get_running_loop()
            t = loop.create_task(self._run_isolated(role, task, ctx, task_id))
            self._bg_tasks[task_id] = t
            return task_id
        return await self._run_isolated(role, task, ctx, None)

    async def _run_isolated(
        self,
        role: RoleTemplate,
        task: str,
        ctx: dict,
        task_id: Optional[str],
    ) -> str:
        """跑子 agent，异常隔离。

        子 agent 崩溃不影响主流程：捕获异常归一成 error 文本返回。
        background 模式同时把结果写进 _bg_results 供后续 poll。
        """
        try:
            text = await self._run_agent(role, task, ctx)
            if task_id is not None:
                self._bg_results[task_id] = _BgResult(
                    status="done", result=text, error=None)
            return text
        except Exception as e:  # noqa: BLE001 — 子 agent 崩溃隔离
            err = f"subagent error: {type(e).__name__}: {e}"
            if task_id is not None:
                self._bg_results[task_id] = _BgResult(
                    status="error", result=None, error=err)
            return err

    async def _run_agent(
        self,
        role: RoleTemplate,
        task: str,
        ctx: dict,
    ) -> str:
        """构造 ReactLoopAgent 跑一个 turn，返回 final_text。"""
        route = ctx.get("route")
        llm = self._build_llm(role, route)
        registry = self._tool_registry
        if registry is None:
            # 无注入 registry → 建空 registry（子 agent 无工具也能跑 LLM 终答）
            from src.core.tools.registry import ToolRegistry
            registry = ToolRegistry()

        persona = role.persona or self._system_prompt
        cfg = AgentConfig(system_prompt=persona, max_steps=self._max_steps)
        agent = ReactLoopAgent(cfg, llm, registry)
        session = Session(uuid.uuid4().hex, system_prompt=persona)
        result = await agent.run_turn(session, task)
        if result.error:
            # run_turn 内部已捕获异常归一成 error；此处转抛给 _run_isolated 隔离
            raise RuntimeError(result.error)
        return result.final_text or ""

    def _build_llm(
        self,
        role: RoleTemplate,
        route: Optional[RouteResolution],
    ) -> Any:
        """构造 LLM 客户端。默认 MockLLMClient（守付费红线）。"""
        if self._llm_factory is not None:
            return self._llm_factory(role, route)
        # 默认 Mock：按角色回放固定文本，不触真实付费 API
        return MockLLMClient([
            LLMChunk(
                delta_text=f"[subagent:{role.display_name}] "
                           f"{role.description or 'done'}",
                stop_reason="stop",
            ),
        ])

    def poll(self, task_id: str) -> Dict[str, Any]:
        """轮询 background 任务结果（非阻塞）。"""
        r = self._bg_results.get(task_id)
        if r is None:
            return {"status": "unknown", "result": None,
                    "error": f"task_id {task_id} not found"}
        return {"status": r.status, "result": r.result, "error": r.error}

    async def wait(self, task_id: str,
                   timeout: Optional[float] = None) -> str:
        """等 background 任务完成，返回结果文本（或 error 文本）。"""
        t = self._bg_tasks.get(task_id)
        if t is None:
            return f"error: task_id {task_id} not found"
        try:
            await asyncio.wait_for(t, timeout=timeout)
        except asyncio.TimeoutError:
            return "error: timeout"
        except Exception:  # noqa: BLE001 — _run_isolated 已吞异常，这里兜底
            pass
        r = self._bg_results.get(task_id)
        if r is None:
            return "error: result not found"
        if r.status == "done":
            return r.result or ""
        if r.status == "error":
            return r.error or "error"
        return "error: still pending"
