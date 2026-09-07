"""工具注册中心 + waterfall 执行调度。

参考 DSH `packages/core/tool-host/src/registry.ts`：
  - register(definition) -> disposer（卸载钩子）
  - schemas() 投影成 LLM 可见的工具 schema 列表
  - execute(call) 走四层 waterfall

disposer 是 cleanup 闭包，插件卸载时调用，从 registry 移除该工具。

v10.0.0:execute 支持可选 ErrorPolicy——TRANSIENT 异常自动重试(指数退避),
FATAL 标记工具不可用。policy=None 时退回原行为(向后兼容,零回归)。
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.core.runtime.sandbox import Sandbox
from src.core.tools.definition import ToolDefinition
from src.core.tools.error_policy import ErrorPolicy, ToolErrorKind
from src.core.tools.parallel import (
    AsyncRWLock,
    ParallelExecutor,
    ToolCallRuntime,
    _PendingCall,
)
from src.core.tools.waterfall import (
    Ask,
    Deny,
    Replace,
    Timeout,
    ToolDenied,
    ToolNeedsApproval,
    ToolWaterfall,
)

log = logging.getLogger("core.tools.registry")


class ToolNotFound(Exception):
    """调用未注册的工具。"""


@dataclass(frozen=True)
class ToolCall:
    """一次工具调用的不可变请求。"""

    name: str
    args: dict = field(default_factory=dict)
    call_id: str = ""


@dataclass(frozen=True)
class ToolResult:
    """工具执行的不可变结果。"""

    call_id: str
    name: str
    output: Any
    error: Optional[str] = None

    def is_error(self) -> bool:
        return self.error is not None


# 审批回调：Ask 信号到达时被调，返回 True/False
ApprovalFn = Callable[[ToolCall, Ask], Awaitable[bool]]
# 锁解析器：按工具名+args 返回锁策略
LockResolver = Callable[[str, dict], tuple[Optional[AsyncRWLock], Optional[str]]]


class ToolRegistry:
    """工具注册中心 + waterfall 执行调度。

    职责：
      1. 注册 ToolDefinition，返回 disposer。
      2. schemas() 投影 LLM 可见的 schema。
      3. execute(call) 穿四层 waterfall 跑工具，返回 ToolResult。
      4. 暴露 waterfall 钩子供插件注入。
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._waterfall = ToolWaterfall()
        self._executor = ParallelExecutor()
        self._approval: Optional[ApprovalFn] = None
        self._lock_resolver: Optional[LockResolver] = None
        # v10.0.0:可选工具异常分级策略。None = 退回原行为(向后兼容)。
        self._error_policy: Optional[ErrorPolicy] = None
        # v10.0.0:可选工具沙箱。默认 disabled(零回归),由 VAP_SANDBOX_ENABLED
        # 控制。enabled=True 且 sandbox 非 None 时,_execute_once 外层包
        # sandbox.guard() 包裹(单次执行进一次沙箱,ErrorPolicy 重试时每次重进)。
        self._sandbox: Optional[Sandbox] = None
        self._sandbox_enabled: bool = os.environ.get(
            "VAP_SANDBOX_ENABLED", "false").lower() == "true"

    # ---- 错误策略 ----
    def set_error_policy(self, policy: Optional[ErrorPolicy]) -> None:
        """注入工具异常分级策略。None 清除(回到向后兼容模式)。"""
        self._error_policy = policy

    # ---- 沙箱 ----
    def set_sandbox(self, sandbox: Optional[Sandbox], enabled: bool) -> None:
        """注入工具沙箱实例 + 开关。

        Args:
            sandbox: Sandbox 实例(或 None 解除)。建议由 build_sandbox() 选型。
            enabled: True 时 execute 包沙箱;False 时 _sandbox 即使非 None 也不包
                (默认,零回归)。
        """
        self._sandbox = sandbox
        self._sandbox_enabled = enabled

    # ---- 注册 ----
    def register(self, definition: ToolDefinition) -> Callable[[], None]:
        """注册工具，返回 disposer。

        disposer 调用时移除该工具。同一 name 重复注册覆盖旧的（并返回新 disposer）。
        """
        self._tools[definition.name] = definition

        def _dispose() -> None:
            # 只在仍指向同一 definition 时移除（防旧 disposer 误删新注册）
            if self._tools.get(definition.name) is definition:
                del self._tools[definition.name]

        return _dispose

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    # ---- LLM schema 投影 ----
    def schemas(self) -> List[Dict[str, Any]]:
        """投影成 LLM tool schema 列表（OpenAI function-calling 子集）。"""
        return [d.to_llm_schema() for d in self._tools.values()]

    # ---- 钩子/审批/锁 ----
    def set_approval_fn(self, fn: Optional[ApprovalFn]) -> None:
        self._approval = fn

    def set_lock_resolver(self, fn: Optional[LockResolver]) -> None:
        self._lock_resolver = fn

    @property
    def waterfall(self) -> ToolWaterfall:
        return self._waterfall

    # ---- 执行 ----
    async def execute(self, call: ToolCall) -> ToolResult:
        """穿四层 waterfall 执行单个工具调用。

        v10.0.0:若设置了 ErrorPolicy,TRANSIENT 错误自动重试(指数退避),
        FATAL 标记工具不可用。policy=None 时退回原一次性行为(向后兼容)。
        不可用工具直接返回 FATAL error,不执行。
        """
        defn = self._tools.get(call.name)
        if defn is None:
            return ToolResult(
                call_id=call.call_id, name=call.name, output=None,
                error=f"Tool '{call.name}' not found")

        # 不可用检查(FATAL 标记过)
        if self._error_policy is not None and \
                not self._error_policy.is_available(call.name):
            return ToolResult(
                call_id=call.call_id, name=call.name, output=None,
                error=f"Tool '{call.name}' is unavailable (fatal error)")

        # 无策略:一次性执行(向后兼容,零回归)
        if self._error_policy is None:
            return await self._execute_once(call, defn)

        # 有策略:TRANSIENT 重试
        attempt = 1
        while True:
            try:
                result = await self._execute_once(call, defn)
            except asyncio.TimeoutError as e:
                kind = self._error_policy.classify(e, call.name)
                if kind == ToolErrorKind.TRANSIENT and \
                        self._error_policy.should_retry(kind, attempt):
                    delay = self._error_policy.retry_after(kind, attempt)
                    log.info(
                        "工具 %s 超时(%s),%.1fs 后重试(第 %d 次)",
                        call.name, kind.value, delay, attempt)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                return ToolResult(
                    call_id=call.call_id, name=call.name, output=None,
                    error=f"Tool '{call.name}' timed out")
            except Exception as e:  # noqa: BLE001 — 策略分类决定重试/降级/放弃
                kind = self._error_policy.classify(e, call.name)
                if kind == ToolErrorKind.FATAL:
                    self._error_policy.mark_unavailable(call.name)
                    log.warning(
                        "工具 %s 命中 FATAL(%s),已标记不可用: %s",
                        call.name, kind.value, e)
                    return ToolResult(
                        call_id=call.call_id, name=call.name, output=None,
                        error=f"Tool '{call.name}' fatal: {e}",
                        )
                if kind == ToolErrorKind.TRANSIENT and \
                        self._error_policy.should_retry(kind, attempt):
                    delay = self._error_policy.retry_after(kind, attempt)
                    log.info(
                        "工具 %s 瞬时错误(%s),%.1fs 后重试(第 %d 次)",
                        call.name, kind.value, delay, attempt)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                # INVALID_INPUT / UNKNOWN / 重试耗尽:返回错误给 LLM 调整
                return ToolResult(
                    call_id=call.call_id, name=call.name, output=None,
                    error=f"{type(e).__name__}: {e}")
            else:
                return result

    async def _execute_once(self, call: ToolCall, defn: ToolDefinition) -> ToolResult:
        """单次穿四层 waterfall(不含重试逻辑)。

        v10.0.0:若 sandbox_enabled 且 self._sandbox 非 None,在 waterfall
        外层包沙箱(enter → run → exit)。沙箱 enter 失败(如 pywin32 缺失
        /assign Job 被拒)降级直接执行,不阻断工具;exit 失败仅告警不抛。
        ErrorPolicy 重试时每次都进沙箱(本方法在重试内层)。
        """

        if not (self._sandbox_enabled and self._sandbox is not None):
            return await self._execute_once_unsandboxed(call, defn)

        # per-tool opt-out:ToolDefinition 可声明 sandbox_required=False
        # 跳过沙箱(轻量工具)。未声明时 getattr 兜底 True(统一包)。
        if not getattr(defn, "sandbox_required", True):
            return await self._execute_once_unsandboxed(call, defn)

        sandbox = self._sandbox
        assert sandbox is not None  # 上面 short-circuit 保证
        try:
            await sandbox.enter()
        except Exception as exc:  # noqa: BLE001 — 沙箱装配失败降级,不阻断工具
            log.warning("沙箱 enter 失败(%s),降级直接执行: %s",
                        type(exc).__name__, exc)
            return await self._execute_once_unsandboxed(call, defn)
        try:
            return await sandbox.run(self._execute_once_unsandboxed(call, defn))
        finally:
            try:
                await sandbox.exit()
            except Exception as exc:  # noqa: BLE001 — exit 失败不阻断结果返回
                log.warning("沙箱 exit 失败(%s): %s",
                            type(exc).__name__, exc)

    async def _execute_once_unsandboxed(
        self, call: ToolCall, defn: ToolDefinition
    ) -> ToolResult:
        """单次穿四层 waterfall(不含重试 / 沙箱)。

        无 ErrorPolicy 时(向后兼容):捕获 ToolDenied/Timeout/
        ToolNeedsApproval 归一成 ToolResult.error 返回。
        有 ErrorPolicy 时:让 TimeoutError 向上抛给 execute 分级重试;
        ToolDenied/ToolNeedsApproval 仍归一(这两类不可重试)。
        """

        try:
            # 1. pre_execute
            args = dict(call.args)
            timeout_sec: Optional[float] = None
            for hook in self._waterfall.pre_execute:
                sig = await hook(call, defn)
                if isinstance(sig, Deny):
                    raise ToolDenied(sig.reason)
                if isinstance(sig, Ask):
                    if self._approval is None:
                        raise ToolNeedsApproval(
                            f"Tool {call.name} needs approval but no fn set")
                    ok = await self._approval(call, sig)
                    if not ok:
                        raise ToolDenied(sig.prompt or "approval denied")
                if isinstance(sig, Replace):
                    # Replace.value 可以是 dict（改写 args）或其它（原样透传）
                    if isinstance(sig.value, dict):
                        args = dict(sig.value)
                if isinstance(sig, Timeout):
                    timeout_sec = sig.seconds

            # 2. 锁解析
            lock: Optional[AsyncRWLock] = None
            mode: Optional[str] = None
            if self._lock_resolver is not None:
                lock, mode = self._lock_resolver(call.name, args)

            runtime = ToolCallRuntime(
                lock=lock, mode=mode, timeout_sec=timeout_sec)

            # 3. execute（真实回调，走 executor 以复用 锁/超时/异常归一）
            pending = _PendingCall(
                name=call.name, args=args,
                callback=defn.execute_callback, runtime=runtime)
            results = await self._executor.run([pending])
            raw = results[0]
            if isinstance(raw, ToolDenied):
                return ToolResult(
                    call_id=call.call_id, name=call.name, output=None,
                    error=raw.reason)

            # 4. post_execute
            final = raw
            for hook in self._waterfall.post_execute:
                sig = await hook(call, defn, final)
                if isinstance(sig, Deny):
                    raise ToolDenied(sig.reason)
                if isinstance(sig, Replace):
                    final = sig.value

            # 5. result（审计/记录钩子）
            for hook in self._waterfall.result:
                await hook(call, defn, final)

            return ToolResult(
                call_id=call.call_id, name=call.name, output=final, error=None)
        except ToolDenied as e:
            return ToolResult(
                call_id=call.call_id, name=call.name, output=None,
                error=str(e))
        except ToolNeedsApproval as e:
            return ToolResult(
                call_id=call.call_id, name=call.name, output=None,
                error=str(e))
        except asyncio.TimeoutError:
            # 无策略:归一成 error 返回(向后兼容)
            # 有策略:向上抛给 execute 分级重试(见 execute 的 except)
            if self._error_policy is None:
                return ToolResult(
                    call_id=call.call_id, name=call.name, output=None,
                    error=f"Tool '{call.name}' timed out")
            raise

