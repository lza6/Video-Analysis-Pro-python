"""子 agent director — 启动/路由/续聊/回收。

参考 DSH `dsh-plugin-subagent-director/src/director.ts` +
`delegation-tool.ts:351-543`：

三模式：
  - foreground: await 同步等结果
  - one-shot background: asyncio.create_task，fire-and-forget
  - continuable: 保存 subagent_id，可后续 send_message 续聊

director 维护活跃子 agent 表，drain() 等所有 background 任务完成。

Backend 路由（v9.0.0+）：
  - route.backend（RouteResolution 字段，默认 "inprocess"）决定走哪个后端
  - call_arg["backend"] 可覆写（最高优先级）
  - 默认含 inprocess（同进程 ReactLoopAgent）+ mock（旧路径保留）
  - 上层可经 backend_factory 注入自定义后端
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional

from src.core.subagent.role_template import RoleTemplate
from src.core.subagent.route_resolver import RouteResolution, resolve_route
from src.core.subagent.backends.inprocess import InProcessBackend


class SubagentMode(str, Enum):
    """子 agent 启动模式。"""

    FOREGROUND = "foreground"
    BACKGROUND = "background"
    CONTINUABLE = "continuable"


@dataclass
class SubagentHandle:
    """子 agent 运行态句柄。"""

    subagent_id: str
    name: str
    role_template: RoleTemplate
    route: RouteResolution
    mode: SubagentMode
    task: Optional[asyncio.Task] = None
    result: Any = None
    error: Optional[str] = None
    # continuable 模式：保存 session_id 供续聊
    session_id: Optional[str] = None


# 子 agent 工厂：director 不直接造 agent，而是调 factory 拿到 awaitable
# 结果。factory 由上层（PluginContext）注入。
SubagentFactory = Callable[
    [RoleTemplate, RouteResolution, str, str],
    Awaitable[Any],
]


def _default_mock_backend(
    role: RoleTemplate, route: RouteResolution, sub_id: str, request: str,
) -> Awaitable[Any]:
    """默认 mock backend：直接 echo request（测试/向后兼容）。

    不触真实付费 API，不依赖 ReactLoopAgent，保证 director 旧测试零回归。
    """
    async def _run() -> str:
        await asyncio.sleep(0)
        return f"mock:{request}"
    return _run()


def _make_inprocess_backend_factory(
    backend: Optional[InProcessBackend] = None,
) -> "BackendFactory":
    """构造 inprocess backend factory：把 InProcessBackend.run 包成
    (role, route, sub_id, request) → awaitable。

    复用同一 InProcessBackend 实例（共享 background 结果字典），未注入时
    每次调用建临时实例（foreground 路径无状态）。
    """
    shared = backend

    async def _factory(
        role: RoleTemplate, route: RouteResolution,
        sub_id: str, request: str,
    ) -> str:
        bk = shared or InProcessBackend()
        ctx = {"route": route, "subagent_id": sub_id}
        return await bk.run(role, request, ctx, background=False)
    return _factory


# Backend 工厂签名：与 SubagentFactory 同形，按 route.backend 选择。
BackendFactory = Callable[
    [RoleTemplate, RouteResolution, str, str],
    Awaitable[Any],
]


class SubagentDirector:
    """子 agent 调度器。

    Usage:
        director = SubagentDirector(factory, parent_route)
        handle = director.start("researcher", request, role_template,
                                mode=SubagentMode.BACKGROUND)
        ...
        await director.drain()  # 等 background 完成

    Backend 路由：
        route.backend（RouteResolution 字段，默认 "inprocess"）决定走哪个
        backend factory。call_arg["backend"] 可覆写（最高优先级）。
        backend_factory 字典可注入自定义后端，默认含 inprocess + mock。
        旧的 factory 参数仍生效：当 route.backend == "mock" 时调旧 factory
        （零回归）；当 route.backend == "inprocess" 时调 InProcessBackend。
    """

    def __init__(
        self,
        factory: Optional[SubagentFactory] = None,
        parent_route: Optional[Dict[str, Any]] = None,
        default_route: Optional[Dict[str, Any]] = None,
        backend_factory: Optional[Dict[str, BackendFactory]] = None,
        inprocess_backend: Optional[InProcessBackend] = None,
    ) -> None:
        self._factory = factory
        self._parent_route = parent_route or {}
        self._default_route = default_route or {}
        self._active: Dict[str, SubagentHandle] = {}
        # 默认 backend 工厂表：inprocess 真后端 + mock 旧路径
        self._backend_factory: Dict[str, BackendFactory] = {
            "inprocess": _make_inprocess_backend_factory(inprocess_backend),
            "mock": _default_mock_backend,
        }
        if backend_factory:
            self._backend_factory.update(backend_factory)

    def _select_backend_coro(
        self,
        role_template: RoleTemplate,
        route: RouteResolution,
        sub_id: str,
        request: str,
        call_arg: Optional[Dict[str, Any]],
    ) -> Awaitable[Any]:
        """按 backend 路由选执行协程。

        路由优先级：
          1. call_arg["backend"]（显式，最高）
          2. route.backend（RouteResolution 字段，默认 "inprocess"）
          3. factory（旧路径，零回归兜底）

        零回归约束：factory 提供且无显式 backend 时，走 factory（旧测试不
        构造 InProcessBackend 也能跑）。factory 缺省或显式指定 backend 时，
        按 backend_factory 表路由。
        """
        explicit = (call_arg or {}).get("backend")
        name = explicit or route.backend or "inprocess"

        # 显式 backend 或无 factory 兜底 → 走 backend_factory 路由
        if explicit or self._factory is None:
            if name in self._backend_factory:
                return self._backend_factory[name](
                    role_template, route, sub_id, request)
            # 未知 backend 名：factory 兜底（若有），否则报错
            if self._factory is not None:
                return self._factory(
                    role_template, route, sub_id, request)
            raise ValueError(
                f"unknown subagent backend: {name!r} "
                f"(available: {list(self._backend_factory)})")

        # factory 提供 + 无显式 backend → 默认走 factory（零回归）
        return self._factory(role_template, route, sub_id, request)

    def start(
        self,
        name: str,
        request: str,
        role_template: RoleTemplate,
        *,
        mode: SubagentMode = SubagentMode.FOREGROUND,
        call_arg: Optional[Dict[str, Any]] = None,
    ) -> "SubagentHandle":
        """启动子 agent，返回 handle。

        - foreground: handle.task 是 awaitable，调用方 await handle.task 拿结果
        - background: task 已 create_task，fire-and-forget
        - continuable: 保存 subagent_id，session_id 可后续 send_message
        """
        route = resolve_route(
            call_arg, role_template, self._default_route, self._parent_route)
        sub_id = uuid.uuid4().hex
        handle = SubagentHandle(
            subagent_id=sub_id, name=name, role_template=role_template,
            route=route, mode=mode)

        coro = self._select_backend_coro(
            role_template, route, sub_id, request, call_arg)

        if mode == SubagentMode.FOREGROUND:
            # foreground：直接存 coroutine，调用方在 loop 内 await handle.task
            handle.task = coro  # type: ignore[assignment]
        else:
            # background / continuable: 把 task 创建推迟到调用方 loop 内。
            # 不在 start() 里 create_task/ensure_future——Python 3.14 主线程
            # 无 current loop 时会 raise。改用工厂闭包：调用方在 _run 驱动的
            # loop 内 await handle.ensure_task() 真正创建并等待 task。
            handle.task = None  # type: ignore[assignment]
            # 把 coro 存在 handle 上，drain/send_message 在 loop 内 create_task
            handle._pending_coro = self._run_and_record(handle, coro)  # type: ignore[attr-defined]
            handle.session_id = sub_id  # continuable 续聊用

        self._active[sub_id] = handle
        return handle

    async def _materialize_task(self, handle: SubagentHandle) -> asyncio.Task:
        """在当前 loop 内把 handle 的 pending coro 物化成 Task（首次 drain 触发）。"""
        if handle.task is None and getattr(handle, "_pending_coro", None) is not None:
            handle.task = asyncio.create_task(handle._pending_coro)  # type: ignore[arg-type]
            handle._pending_coro = None  # type: ignore[attr-defined]
        return handle.task  # type: ignore[return-value]

    async def _run_and_record(self, handle: SubagentHandle,
                              coro: Awaitable[Any]) -> Any:
        try:
            result = await coro
            handle.result = result
            return result
        except Exception as e:  # noqa: BLE001
            handle.error = str(e)
            raise

    async def drain(self) -> None:
        """等所有 background/continuable 子 agent 完成。

        首次 drain 时物化 pending coro 为 Task（在当前 loop 内 create_task）。
        """
        # 先物化所有 pending coroutine 为 Task
        for h in list(self._active.values()):
            await self._materialize_task(h)
        tasks = [h.task for h in self._active.values()
                 if h.task is not None and not h.task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get(self, subagent_id: str) -> Optional[SubagentHandle]:
        return self._active.get(subagent_id)

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def send_message(self, subagent_id: str, message: str) -> Any:
        """continuable 模式续聊。

        默认实现：按 handle.route.backend 再跑一次，request=message，
        复用同一 role/route。生产实现可替换为真正复用同一 Session 的
        ReactLoopAgent。
        """
        handle = self._active.get(subagent_id)
        if handle is None:
            raise KeyError(f"subagent {subagent_id} not found")
        if handle.mode != SubagentMode.CONTINUABLE:
            raise ValueError(
                f"subagent {subagent_id} is not continuable (mode={handle.mode})")
        coro = self._select_backend_coro(
            handle.role_template, handle.route, subagent_id, message, None)
        return await coro
