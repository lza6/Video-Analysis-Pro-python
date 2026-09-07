"""InProcessBackend + SubagentDirector backend 路由测试。

覆盖：
  1. InProcessBackend.run foreground → 返回文本
  2. background → 返回 task_id，后续 poll/wait 拿结果
  3. 子 agent 抛异常 → 返回 error 文本不崩溃（异常隔离）
  4. SubagentDirector 路由到 inprocess backend
  5. 并发跑 2 个子 agent → 都返回

不依赖 pytest-asyncio：用 asyncio.run() 在同步测试中驱动 async 代码
（与项目现有 tests/test_agent_framework.py 一致）。
"""
from __future__ import annotations

import asyncio

from src.core.agent.loop import LLMChunk, MockLLMClient
from src.core.subagent.backends.inprocess import InProcessBackend
from src.core.subagent.director import SubagentDirector, SubagentMode
from src.core.subagent.role_template import RoleTemplate


def _run(coro):
    """同步驱动 async 测试，不依赖 pytest-asyncio。

    显式 new_event_loop + set_event_loop：Python 3.14 上 asyncio.gather 等
    内部 API 会调 get_event_loop()，无显式 loop 时在主线程会 raise
    'no current event loop'。先 set 再 run_until_complete 避免该坑。
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


# ---------------------------------------------------------------------------
# 1. foreground 返回结果
# ---------------------------------------------------------------------------


def test_inprocess_foreground_returns_result() -> None:
    """InProcessBackend.run foreground → Mock LLM 返回文本。"""
    backend = InProcessBackend()
    role = RoleTemplate(display_name="researcher", description="查资料")

    text = _run(backend.run(role, "find X"))

    assert isinstance(text, str)
    assert text  # 非空
    assert "researcher" in text  # Mock 按角色回放


# ---------------------------------------------------------------------------
# 2. background 返回 task_id，后续 poll/wait 拿结果
# ---------------------------------------------------------------------------


def test_inprocess_background_returns_task_id() -> None:
    """background → 返回 task_id；wait() 拿最终结果。"""
    backend = InProcessBackend()
    role = RoleTemplate(display_name="planner", description="规划")

    async def _scenario() -> str:
        task_id = await backend.run(role, "do Y", background=True)
        assert isinstance(task_id, str) and len(task_id) > 0
        # poll 至少能看到 pending/done 之一
        status = backend.poll(task_id)
        assert status["status"] in ("pending", "done", "error")
        # wait 拿最终文本
        result = await backend.wait(task_id, timeout=5.0)
        assert isinstance(result, str)
        assert "planner" in result
        return task_id

    _run(_scenario())


# ---------------------------------------------------------------------------
# 3. 异常隔离：子 agent 抛异常 → 返回 error 文本不崩溃
# ---------------------------------------------------------------------------


def test_inprocess_exception_isolation() -> None:
    """子 agent 内部异常被 _run_isolated 捕获，返回 error 文本。"""
    # 用一个会抛异常的 LLM factory
    def _boom_llm_factory(role, route):
        class _BoomLLM:
            async def stream(self, messages, tools):
                raise RuntimeError("subagent boom")
                yield  # pragma: no cover  # 让 stream 是 async generator
        return _BoomLLM()

    backend = InProcessBackend(llm_factory=_boom_llm_factory)
    role = RoleTemplate(display_name="exploder")

    async def _scenario() -> str:
        task_id = await backend.run(role, "crash", background=True)
        result = await backend.wait(task_id, timeout=5.0)
        return result

    result = _run(_scenario())
    # 不应抛到调用方，返回 error 文本
    assert isinstance(result, str)
    assert "error" in result.lower() or "boom" in result.lower()


# ---------------------------------------------------------------------------
# 4. SubagentDirector 路由到 inprocess backend
# ---------------------------------------------------------------------------


def test_director_routes_to_inprocess() -> None:
    """SubagentDirector 用 call_arg["backend"]="inprocess" 路由到 InProcessBackend。

    不注入 factory（factory=None），强制走 backend_factory 路由。
    """
    backend = InProcessBackend()
    director = SubagentDirector(
        factory=None,
        inprocess_backend=backend,
    )
    role = RoleTemplate(display_name="worker", description="干活")

    handle = director.start(
        "w", "do inprocess work", role,
        mode=SubagentMode.FOREGROUND,
        call_arg={"backend": "inprocess"},
    )
    result = _run(handle.task)  # type: ignore[arg-type]
    assert isinstance(result, str)
    assert "worker" in result


def test_director_default_inprocess_when_no_factory() -> None:
    """factory=None 且无显式 backend → 默认 inprocess（RouteResolution.backend 默认值）。"""
    backend = InProcessBackend()
    director = SubagentDirector(
        factory=None,
        inprocess_backend=backend,
    )
    role = RoleTemplate(display_name="default")

    handle = director.start("d", "default path", role,
                            mode=SubagentMode.FOREGROUND)
    result = _run(handle.task)  # type: ignore[arg-type]
    assert "default" in result


# ---------------------------------------------------------------------------
# 5. 并发跑 2 个子 agent
# ---------------------------------------------------------------------------


def test_inprocess_parallel_two_subagents() -> None:
    """并发跑 2 个 inprocess 子 agent → 都返回。"""
    backend = InProcessBackend()
    role_a = RoleTemplate(display_name="alpha", description="A")
    role_b = RoleTemplate(display_name="beta", description="B")

    async def _scenario():
        # 两个 background 任务并发
        id_a = await backend.run(role_a, "task A", background=True)
        id_b = await backend.run(role_b, "task B", background=True)
        ra = await backend.wait(id_a, timeout=5.0)
        rb = await backend.wait(id_b, timeout=5.0)
        return ra, rb

    ra, rb = _run(_scenario())
    assert "alpha" in ra
    assert "beta" in rb


# ---------------------------------------------------------------------------
# 6. director + background + drain 跑 inprocess（端到端集成）
# ---------------------------------------------------------------------------


def test_director_background_inprocess_drain() -> None:
    """director background 模式 + drain 跑 inprocess backend。"""
    backend = InProcessBackend()
    director = SubagentDirector(
        factory=None,
        inprocess_backend=backend,
    )
    role = RoleTemplate(display_name="bg", description="后台任务")

    h1 = director.start("b1", "bg1", role,
                         mode=SubagentMode.BACKGROUND,
                         call_arg={"backend": "inprocess"})
    h2 = director.start("b2", "bg2", role,
                         mode=SubagentMode.BACKGROUND,
                         call_arg={"backend": "inprocess"})
    _run(director.drain())
    assert h1.result is not None
    assert h2.result is not None
    assert "bg" in str(h1.result)
    assert "bg" in str(h2.result)


# ---------------------------------------------------------------------------
# 7. 自定义 LLM factory 路径（守付费红线：用 MockLLMClient）
# ---------------------------------------------------------------------------


def test_inprocess_custom_llm_factory_uses_mock() -> None:
    """注入自定义 LLM factory 返回 MockLLMClient → 子 agent 跑通且不触真实 API。"""
    def _mock_factory(role, route):
        return MockLLMClient([
            LLMChunk(delta_text=f"custom:{role.display_name}",
                     stop_reason="stop"),
        ])

    backend = InProcessBackend(llm_factory=_mock_factory)
    role = RoleTemplate(display_name="custom-role")

    text = _run(backend.run(role, "any"))
    assert text == "custom:custom-role"
