"""工具沙箱单元测试。

策略（与 tests/test_agent_framework.py 一致）:
- 用 `asyncio.run()` 在同步测试中驱动 async 代码,不依赖 pytest-asyncio
- NullSandbox / build_sandbox 降级路径:纯 stdlib 即可测
- WindowsJobSandbox / LinuxLandlockSandbox:用 skipif 标注 pywin32 / landlock
  可用性,两种环境都跑
- 不真实跑 OS Job Object / landlock ruleset(会污染当前进程),
  只验证构造 / 协议 / 异常传播
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
from typing import Any

import pytest

from src.core.runtime.sandbox import (
    LinuxLandlockSandbox,
    NullSandbox,
    Sandbox,
    WindowsJobSandbox,
    build_sandbox,
)
from src.core.tools.definition import ToolDefinition
from src.core.tools.registry import ToolCall, ToolRegistry

# pywin32 / landlock 可用性探测(用 find_spec 避免 pyflakes unused import 告警)
_HAS_PYWIN32 = importlib.util.find_spec("win32job") is not None
_HAS_LANDLOCK = importlib.util.find_spec("landlock") is not None


def _run(coro):
    """同步驱动 async 测试,不依赖 pytest-asyncio。

    复用 tests/test_agent_framework.py 的 _run 模式:显式 new_event_loop
    + set_event_loop,Python 3.14 上 asyncio 内部 API 会调 get_event_loop(),
    无显式 loop 时会 raise。
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


# ---------------------------------------------------------------------------
# registry-sandbox 集成(本扩展新增)
# ---------------------------------------------------------------------------
def _echo_tool_def() -> ToolDefinition:
    """echo 工具:回显 args['value'],用于验证 execute 正常返回结果。"""
    async def _echo(value: int = 0) -> int:
        return value

    return ToolDefinition(
        name="echo",
        description="回显测试工具",
        execute_callback=_echo,
        input_schema={"type": "object",
                      "properties": {"value": {"type": "integer"}}},
    )


class _RecordingSandbox:
    """记录 enter/exit/run 调用次数的 Mock 沙箱(不依赖 OS API)。"""

    def __init__(self) -> None:
        self.enter_calls = 0
        self.exit_calls = 0
        self.run_calls = 0

    async def enter(self) -> None:
        self.enter_calls += 1

    async def exit(self) -> None:
        self.exit_calls += 1

    async def run(self, coro: Any) -> Any:
        self.run_calls += 1
        return await coro


class _EnterFailingSandbox:
    """enter 抛异常的 Mock 沙箱,用于验证降级直接执行。"""

    def __init__(self) -> None:
        self.exit_calls = 0

    async def enter(self) -> None:
        raise RuntimeError("sandbox enter boom")

    async def exit(self) -> None:
        self.exit_calls += 1

    async def run(self, coro: Any) -> Any:
        raise AssertionError("enter 失败时不应调用 run")


def test_registry_execute_with_sandbox_disabled() -> None:
    """默认 sandbox_enabled=False,execute 不包沙箱(零回归)。

    不注入 sandbox 或 enabled=False,工具应直接返回结果,不触任何沙箱逻辑。
    """
    reg = ToolRegistry()
    reg.register(_echo_tool_def())
    # 不调 set_sandbox:enabled 默认由 VAP_SANDBOX_ENABLED 读(测试环境为 false)
    result = _run(reg.execute(ToolCall(name="echo", args={"value": 7})))
    assert result.error is None
    assert result.output == 7


def test_registry_execute_with_sandbox_enabled() -> None:
    """enabled=True + NullSandbox,execute 正常返回结果。

    NullSandbox run 直接 await,结果不应被改变。验证沙箱包裹不破坏执行链。
    """
    reg = ToolRegistry()
    reg.register(_echo_tool_def())
    reg.set_sandbox(NullSandbox(), enabled=True)
    result = _run(reg.execute(ToolCall(name="echo", args={"value": 99})))
    assert result.error is None
    assert result.output == 99


def test_registry_sandbox_guard_called() -> None:
    """enabled=True + Mock 沙箱,验证 enter/run/exit 各调一次。"""
    reg = ToolRegistry()
    reg.register(_echo_tool_def())
    sb = _RecordingSandbox()
    reg.set_sandbox(sb, enabled=True)
    result = _run(reg.execute(ToolCall(name="echo", args={"value": 5})))
    assert result.error is None
    assert result.output == 5
    assert sb.enter_calls == 1
    assert sb.run_calls == 1
    assert sb.exit_calls == 1


def test_registry_sandbox_enter_failure_falls_back() -> None:
    """沙箱 enter 失败时降级直接执行,不阻断工具,不调 run/exit。

    enter 抛异常说明沙箱未生效,exit 也不应被调(沙箱未启动无须清理)。
    """
    reg = ToolRegistry()
    reg.register(_echo_tool_def())
    sb = _EnterFailingSandbox()
    reg.set_sandbox(sb, enabled=True)
    result = _run(reg.execute(ToolCall(name="echo", args={"value": 11})))
    assert result.error is None
    assert result.output == 11
    # enter 失败 → 沙箱未启动 → run/exit 都不应被调
    assert sb.exit_calls == 0


def test_registry_sandbox_env_var_enables_default() -> None:
    """VAP_SANDBOX_ENABLED=true 时 __init__ 默认 enabled=True。

    验证环境开关在 __init__ 时初始化(不注入沙箱实例时仍 disabled,因 sandbox=None)。
    """
    original = os.environ.get("VAP_SANDBOX_ENABLED")
    os.environ["VAP_SANDBOX_ENABLED"] = "true"
    try:
        reg = ToolRegistry()
        assert reg._sandbox_enabled is True  # noqa: SLF001 — 测试白盒验证开关
        # sandbox 实例未注入,_sandbox 为 None,execute 仍走无沙箱路径
        assert reg._sandbox is None  # noqa: SLF001
    finally:
        if original is None:
            del os.environ["VAP_SANDBOX_ENABLED"]
        else:
            os.environ["VAP_SANDBOX_ENABLED"] = original


def test_registry_sandbox_fallback_on_missing_pywin32() -> None:
    """WindowsJobSandbox 无 pywin32 时 build_sandbox 降级 NullSandbox。

    模拟无 pywin32 环境:_HAS_PYWIN32=False 时 build_sandbox('win32') 应返回
    NullSandbox。registry 用此沙箱仍可正常 execute(降级不阻断)。
    """
    if _HAS_PYWIN32:
        pytest.skip("pywin32 可用,无法验证缺失降级分支")
    sb = build_sandbox("win32")
    assert isinstance(sb, NullSandbox), "pywin32 缺失时 build_sandbox 应降级 NullSandbox"
    reg = ToolRegistry()
    reg.register(_echo_tool_def())
    reg.set_sandbox(sb, enabled=True)
    result = _run(reg.execute(ToolCall(name="echo", args={"value": 3})))
    assert result.error is None
    assert result.output == 3


def test_registry_sandbox_disabled_even_when_sandbox_set() -> None:
    """set_sandbox(sandbox, enabled=False):sandbox 实例在但不开,execute 不包。"""
    reg = ToolRegistry()
    reg.register(_echo_tool_def())
    sb = _RecordingSandbox()
    reg.set_sandbox(sb, enabled=False)  # 注入实例但关开关
    result = _run(reg.execute(ToolCall(name="echo", args={"value": 8})))
    assert result.error is None
    assert result.output == 8
    assert sb.enter_calls == 0  # 未开开关,沙箱方法不应被调
    assert sb.run_calls == 0
    assert sb.exit_calls == 0


# ---------------------------------------------------------------------------
# NullSandbox
# ---------------------------------------------------------------------------
def test_null_sandbox_returns_coro_result() -> None:
    """NullSandbox.run 直接 await coro 并返回其结果。"""

    async def coro() -> int:
        return 42

    sandbox = NullSandbox()
    result = _run(sandbox.run(coro()))
    assert result == 42


def test_null_sandbox_enter_exit_are_noop() -> None:
    """NullSandbox enter/exit 不做任何事,不抛异常。"""
    sandbox = NullSandbox()

    async def _enter_then_exit() -> None:
        await sandbox.enter()
        await sandbox.exit()

    _run(_enter_then_exit())  # 不抛即通过


# ---------------------------------------------------------------------------
# build_sandbox 选型
# ---------------------------------------------------------------------------
def test_build_sandbox_darwin_returns_null() -> None:
    """darwin 平台无原生沙箱,build_sandbox 返回 NullSandbox。"""
    sb = build_sandbox("darwin")
    assert isinstance(sb, NullSandbox)


def test_build_sandbox_unknown_platform_returns_null() -> None:
    """未知平台返回 NullSandbox。"""
    sb = build_sandbox("plan9")
    assert isinstance(sb, NullSandbox)


def test_build_sandbox_win32_returns_correct_type() -> None:
    """win32:pywin32 可用→WindowsJobSandbox;缺失→NullSandbox。两种情况都断言。"""
    sb = build_sandbox("win32")
    if _HAS_PYWIN32:
        assert isinstance(sb, WindowsJobSandbox)
    else:  # 当前 venv 就是这个分支
        assert isinstance(sb, NullSandbox)


def test_build_sandbox_linux_returns_correct_type() -> None:
    """linux:stdlib landlock 可用→LinuxLandlockSandbox;缺失→NullSandbox。"""
    sb = build_sandbox("linux")
    if _HAS_LANDLOCK:
        assert isinstance(sb, LinuxLandlockSandbox)
    else:  # 当前 venv(3.14 win32)就是 NullSandbox
        assert isinstance(sb, NullSandbox)


# ---------------------------------------------------------------------------
# WindowsJobSandbox pywin32 缺失分支
# ---------------------------------------------------------------------------
def test_windows_job_sandbox_raises_importerror_without_pywin32() -> None:
    """pywin32 缺失时 WindowsJobSandbox 构造 raise ImportError。

    pywin32 可用时跳过此用例(无法验证缺失分支)。
    """
    if _HAS_PYWIN32:
        pytest.skip("pywin32 可用,无法验证缺失分支")
    with pytest.raises(ImportError, match="pywin32"):
        WindowsJobSandbox()


# ---------------------------------------------------------------------------
# 异常传播
# ---------------------------------------------------------------------------
def test_sandbox_run_propagates_coro_exception() -> None:
    """run 包裹的协程异常必须正确传播(ValueError 不被吞)。"""
    sandbox = NullSandbox()

    async def raising() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _run(sandbox.run(raising()))


# ---------------------------------------------------------------------------
# Protocol 结构性自检
# ---------------------------------------------------------------------------
def test_null_sandbox_satisfies_sandbox_protocol() -> None:
    """NullSandbox 实现 Sandbox Protocol(结构性自检,不依赖运行时)。"""
    assert isinstance(NullSandbox(), Sandbox)


# ---------------------------------------------------------------------------
# WindowsJobSandbox 参数校验(不进 enter,避免污染当前进程)
# ---------------------------------------------------------------------------
def test_windows_job_sandbox_rejects_invalid_params() -> None:
    """memory_limit_mb <= 0 / cpu_rate 越界 raise ValueError。

    pywin32 缺失时构造本身已 ImportError,跳过参数校验。
    """
    if not _HAS_PYWIN32:
        pytest.skip("pywin32 缺失,无法构造 WindowsJobSandbox")
    with pytest.raises(ValueError):
        WindowsJobSandbox(memory_limit_mb=0)
    with pytest.raises(ValueError):
        WindowsJobSandbox(memory_limit_mb=-100)
    with pytest.raises(ValueError):
        WindowsJobSandbox(cpu_rate=0.0)
    with pytest.raises(ValueError):
        WindowsJobSandbox(cpu_rate=1.5)
