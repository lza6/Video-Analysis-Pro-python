"""工具调用范围守卫(scope_guard) + 审批装配入口。

参考 pi-guardrails / pigeon 的最小权限原则:工具调用不应一律放行,
而应按「工具名 + args」走 allow / deny / ask 三级决策表:

  - 读操作(get_* / search_* / list_* / …)         → allow(直接放行)
  - 写操作(create / update / delete / 落盘 / IM send)→ ask(弹审批)
  - 危险写操作(delete_* / highlight_cut 落盘 / 网络发送等)
      → 必须审批,且无 handler 时超时默认 deny

分级开关(环境变量,`1` = 放行,`0` / 缺省 = ask / deny):

  VAP_ALLOW_WRITE_CUT     = 放行剪辑落盘类写操作(highlight_cut 等)
  VAP_ALLOW_WRITE_GENERAL = 放行一般写操作(create / update / trigger / send 等)
  VAP_ALLOW_WRITE_DELETE  = 放行危险删除类写操作(delete_* 等)

装配:`install_tool_guard(registry, …)` 把 ScopeGuard 的 pre_execute 钩子
挂到 registry.waterfall,并注入 approval_fn / sandbox。由主控统一调用。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

from src.core.tools.registry import ApprovalFn, ToolCall, ToolRegistry
from src.core.tools.waterfall import Ask

log = logging.getLogger("core.tools.scope_guard")

# ---------------------------------------------------------------------------
# 分级开关常量(环境变量)
# ---------------------------------------------------------------------------

#: 放行「剪辑落盘」类写操作(highlight_cut 等)。`1` 放行,`0`/缺省 ask。
VAP_ALLOW_WRITE_CUT = "VAP_ALLOW_WRITE_CUT"
#: 放行「一般写操作」(create / update / trigger / send 等)。`1` 放行,`0`/缺省 ask。
VAP_ALLOW_WRITE_GENERAL = "VAP_ALLOW_WRITE_GENERAL"
#: 放行「危险删除」类写操作(delete_* 等)。`1` 放行,`0`/缺省 deny。
VAP_ALLOW_WRITE_DELETE = "VAP_ALLOW_WRITE_DELETE"

# ---------------------------------------------------------------------------
# 决策等级
# ---------------------------------------------------------------------------

#: 决策等级:allow 放行 / deny 拒绝 / ask 需审批
_ALLOW, _DENY, _ASK = "allow", "deny", "ask"


class ScopeGuard:
    """工具名 + args → allow/deny/ask 三级决策表。

    规则优先级(dangerous > write > read):
      1. `_DANGEROUS_WRITE_PATTERNS` 命中 → 危险写
         - VAP_ALLOW_WRITE_DELETE=1   → allow
         - 否则                       → ask(无 handler 时超时默认 deny)
      2. `_WRITE_PATTERNS` 命中 → 一般写
         - VAP_ALLOW_WRITE_GENERAL=1 → allow
         - VAP_ALLOW_WRITE_CUT=1 且属剪辑落盘 → allow
         - 否则                     → ask
      3. 其余(读操作 / 未知)       → allow

    `decision(call)` 返回 `(level, reason)`。`pre_execute` 钩子把
    ask 等级映射成 waterfall 的 `Ask` 信号(审批优先级带进 handler)。
    """

    def __init__(
        self,
        *,
        allow_write_cut: Optional[bool] = None,
        allow_write_general: Optional[bool] = None,
        allow_write_delete: Optional[bool] = None,
    ) -> None:
        # None → 运行时读环境变量;显式 bool → 固定(便于测试隔离)
        self._allow_write_cut = allow_write_cut
        self._allow_write_general = allow_write_general
        self._allow_write_delete = allow_write_delete

    # ---- 三级开关解析 ----
    def _switch(self, key: str, fixed: Optional[bool]) -> bool:
        if fixed is not None:
            return fixed
        return os.environ.get(key, "0").strip().lower() == "1"

    def allow_write_cut(self) -> bool:
        """是否放行剪辑落盘类写操作。"""
        return self._switch(VAP_ALLOW_WRITE_CUT, self._allow_write_cut)

    def allow_write_general(self) -> bool:
        """是否放行一般写操作。"""
        return self._switch(VAP_ALLOW_WRITE_GENERAL, self._allow_write_general)

    def allow_write_delete(self) -> bool:
        """是否放行危险删除类写操作。"""
        return self._switch(VAP_ALLOW_WRITE_DELETE, self._allow_write_delete)

    # ---- 决策表 ----
    def classify(self, name: str) -> str:
        """按工具名分类:read / write / dangerous_write。"""
        if _matches(name, DANGEROUS_WRITE_PATTERNS):
            return "dangerous_write"
        if _matches(name, WRITE_PATTERNS):
            return "write"
        return "read"

    def decision(self, call: ToolCall) -> Tuple[str, str]:
        """对一次工具调用返回 `(level, reason)`。

        level ∈ {allow, deny, ask}。deny 当前不用(危险写也走 ask),
        保留等级供调用方扩展(如显式黑名单可返回 deny)。
        """
        name = call.name
        cat = self.classify(name)
        if cat == "dangerous_write":
            if self.allow_write_delete():
                return _ALLOW, f"{name} 危险写已由 {VAP_ALLOW_WRITE_DELETE}=1 放行"
            # 剪辑落盘(hit 危险写)可由 VAP_ALLOW_WRITE_CUT 单独放行
            if self.allow_write_cut() and _matches(name, CUT_WRITE_PATTERNS):
                return _ALLOW, f"{name} 剪辑落盘已由 {VAP_ALLOW_WRITE_CUT}=1 放行"
            return _ASK, f"{name} 属危险写操作,需人工审批(超时默认拒绝)"
        if cat == "write":
            if self.allow_write_general():
                return _ALLOW, f"{name} 写操作已由 {VAP_ALLOW_WRITE_GENERAL}=1 放行"
            if self.allow_write_cut() and _matches(name, CUT_WRITE_PATTERNS):
                return _ALLOW, f"{name} 剪辑落盘已由 {VAP_ALLOW_WRITE_CUT}=1 放行"
            return _ASK, f"{name} 属写操作,需人工审批"
        return _ALLOW, f"{name} 属读操作,放行"

    # ---- waterfall 钩子 ----
    async def pre_execute(self, call: ToolCall, defn: Any) -> Optional[Ask]:
        """pre_execute 钩子:ask 等级 → Ask 信号(带审批优先级)。

        v10.3.1 (P0-3):危险写与一般写携带不同 priority,前端可据此
        呈现危险等级配色/文案(此前 priority 全链路是常量 'write')。
        Ask 是 frozen dataclass,用 dataclasses.replace 构造新实例。
        """
        level, reason = self.decision(call)
        if level == _ASK:
            cat = self.classify(call.name)
            priority = ("dangerous_write" if cat == "dangerous_write"
                        else "write")
            from dataclasses import replace as _dc_replace
            return _dc_replace(Ask(prompt=reason, default="deny"),
                               priority=priority)
        return None


# ---------------------------------------------------------------------------
# 匹配模式(工具名前缀 / 危险写关键词)
# ---------------------------------------------------------------------------

#: 读操作前缀:命中即放行(get_* / search_* / list_* / …)
DEFAULT_ALLOW_PREFIXES: Tuple[str, ...] = (
    "get_",
    "search_",
    "list_",
    "summarize_",
    "trace_",
    "point_",
)

#: 一般写操作关键词(name 包含任一即视为写操作)
WRITE_PATTERNS: Tuple[str, ...] = (
    "create",
    "update",
    "delete",
    "trigger",
    "start",
    "generate",
    "send",
    "write",
    "save",
    # v10.3.1 (P0-3):真实落盘的媒体生成工具此前因无写关键词被误归 read
    # → allow,静默写盘。make_*(make_subtitle/make_short_video/
    # make_voiceover)产物均落盘,归 write(ask 审批)。
    "make_",
)

#: 剪辑落盘子集:命中且在写操作内,受 VAP_ALLOW_WRITE_CUT 控制
CUT_WRITE_PATTERNS: Tuple[str, ...] = ("cut", "highlight_cut")

#: 危险写关键词(delete_* / 落盘 / 网络发送 / 监控启动 / 任意代码执行):
#: 命中必须审批
DANGEROUS_WRITE_PATTERNS: Tuple[str, ...] = (
    "delete",
    "highlight_cut",
    "trigger_batch",
    "start_rtsp_monitor",
    "send_",
    "_send",
    "im_send",
    # v10.3.1 (P0-3):cdp_evaluate 可在页面执行任意 JS —— 等价于任意
    # 代码执行,此前因含 "evaluate"(无写关键词)被误归 read → allow。
    # 归危险写:必须审批(超时默认 deny)。
    "cdp_evaluate",
    # 同族:cdp_eval_write 显式写(此前靠 "write" 命中归 write/ask,
    # 统一收进危险写,语义与"任意 JS"一致)。
    "cdp_eval_write",
)


def _matches(name: str, patterns: Tuple[str, ...]) -> bool:
    return any(p in name.lower() for p in patterns)


# ---------------------------------------------------------------------------
# 审批信号扩展(供 registry.execute 消费)
# ---------------------------------------------------------------------------

#: Ask 信号上携带的字段名(在 waterfall.Ask 上新增,向后兼容)
APPROVAL_PRIORITY_FIELD = "priority"


def approval_priority(sig: Ask) -> str:
    """读 Ask 信号的审批优先级,缺省 'write'。"""
    return getattr(sig, APPROVAL_PRIORITY_FIELD, "write")


# ---------------------------------------------------------------------------
# 环境装配
# ---------------------------------------------------------------------------

_DEFAULT_APPROVAL_TIMEOUT_SEC = 60.0


async def _default_approval_fn(
    call: ToolCall,
    sig: Ask,
    *,
    timeout: float = _DEFAULT_APPROVAL_TIMEOUT_SEC,
) -> bool:
    """默认审批 handler:交给 ApprovalBus 投递 + 等待用户决定。

    超时返回 False(默认拒绝)。这是「主控装配 approval_fn 时」可用的
    现成实现——ScopeGuard 钩子只负责产生 Ask 信号,审批由本函数完成。

    v10.2 (B-FIN-2 MAJOR-2):等待改用 ApprovalBus.wait_decision_async
    (asyncio.Event 通知),不再阻塞事件循环。此前直接调同步轮询版
    wait_decision,在 uvicorn 事件循环上 await 会冻结服务至多 60s,
    decide 请求进不来形成死锁。现在等待期间事件循环可正常处理前端
    POST /api/agent/approval/{pin}/decide 回调。

    注入依赖:ApprovalBus 实例经 install_tool_guard 的 approval 回调闭包
    传入(见 build_approval_fn)。
    """
    from src.web.deps import get_approval_bus

    bus = get_approval_bus()
    try:
        pin = bus.request_approval(
            {"tool": call.name, "args": call.args, "reason": sig.prompt})
    except Exception:  # noqa: BLE001 — 投递失败视为拒绝(安全侧)
        log.warning("ApprovalBus 投递失败,按拒绝处理: %s", call.name)
        return False
    decision = await bus.wait_decision_async(pin, timeout=timeout)
    return bool(decision)


def build_approval_fn(
    *, timeout: float = _DEFAULT_APPROVAL_TIMEOUT_SEC,
) -> ApprovalFn:
    """构建可注入 registry.set_approval_fn 的审批回调(走 ApprovalBus)。"""
    import functools as _ft

    return _ft.partial(_default_approval_fn, timeout=timeout)


def build_guard_from_env() -> ScopeGuard:
    """按环境变量构造 ScopeGuard(开关 None = 运行时读 env)。"""
    return ScopeGuard()


# ---------------------------------------------------------------------------
# 装配入口
# ---------------------------------------------------------------------------


def install_tool_guard(
    registry: ToolRegistry,
    *,
    approval_fn: Optional[ApprovalFn] = None,
    sandbox: Any = None,
    enabled: bool = True,
    guard: Optional[ScopeGuard] = None,
    wire_error_policy: bool = True,
) -> ScopeGuard:
    """把 ScopeGuard + approval + sandbox + error_policy 装配到 registry。

    由主控统一调用(agent.py / loop.py 不动):

      1. ScopeGuard 的 pre_execute 钩子挂到 registry.waterfall
         (决策表:读 allow / 写 ask / 危险写 ask)
      2. approval_fn 非 None → registry.set_approval_fn(…)
         None → 用默认 ApprovalBus 回调(build_approval_fn)
      3. sandbox 非 None → registry.set_sandbox(sandbox, enabled=enabled)
      4. v10.3.1 (P0-4):wire_error_policy=True → registry.set_error_policy
         (ErrorPolicy() 默认策略:TRANSIENT 指数退避重试 + FATAL 熔断)。
         此前 `set_error_policy` 自 v10.1 定义以来零调用方 —— 重试/熔断
         从未在生产生效。

    Returns:
        构造出的 ScopeGuard 实例(便于测试断言 / 后续按需调整开关)。
    """
    if guard is None:
        guard = build_guard_from_env()
    if approval_fn is None:
        approval_fn = build_approval_fn()
    if wire_error_policy and registry._error_policy is None:
        from src.core.tools.error_policy import ErrorPolicy
        registry.set_error_policy(ErrorPolicy())
    # v10.3.1 (P0-4):lock_resolver 接线 —— 按 scope_guard 分类给工具
    # 上读/写锁(读共享/写独占)。此前 set_lock_resolver 自 v10.1 定义以来
    # 零调用方,AsyncRWLock 在真实链路是摆设。锁实例按"资源名"共享:
    # 全局单锁(桌面单用户,简单正确;读工具之间共享读锁并行)。
    if registry._lock_resolver is None:
        from src.core.tools.parallel import AsyncRWLock
        _lock = AsyncRWLock()

        def _lock_resolver(name: str, args: dict):
            cat = guard.classify(name)
            if cat == "read":
                return _lock, "read"
            return _lock, "write"

        registry.set_lock_resolver(_lock_resolver)
    # 幂等:同一 guard 只挂一次钩子(重复装配不叠加)
    existing = getattr(registry.waterfall, "_scope_guard_installed", False)
    if not existing:
        registry.waterfall.add_pre_execute(guard.pre_execute)
        setattr(registry.waterfall, "_scope_guard_installed", True)
    registry.set_approval_fn(approval_fn)
    if sandbox is not None:
        registry.set_sandbox(sandbox, enabled=enabled)
    return guard


# 复用检查点:确保 import 时不隐式依赖 src.web.deps(ApprovalBus 延迟 import)
__all__ = [
    "ScopeGuard",
    "DEFAULT_ALLOW_PREFIXES",
    "WRITE_PATTERNS",
    "CUT_WRITE_PATTERNS",
    "DANGEROUS_WRITE_PATTERNS",
    "VAP_ALLOW_WRITE_CUT",
    "VAP_ALLOW_WRITE_GENERAL",
    "VAP_ALLOW_WRITE_DELETE",
    "build_guard_from_env",
    "build_approval_fn",
    "install_tool_guard",
    "approval_priority",
]
