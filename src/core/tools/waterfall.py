"""工具执行的四层 waterfall。

参考 DSH `packages/core/tool-host/src/waterfall.ts` 与 codex `tools/dispatch`：
工具调用不是单次直调，而是穿过四层钩子，每层可 allow/deny/ask/replace/timeout。

四层语义：
  - pre_execute: 调用前，可改写参数（Replace）、拒绝（Deny）、请求人工审批（Ask）。
  - execute: 真实工具回调执行（不可替换）。
  - post_execute: 拿到原始结果后，可改写结果（Replace）。
  - result: 最终化，可记录审计、metric。

每层是 asyncio 钩子点（可被插件注入）。默认实现直通。

Python 版无 Cordis fiber，用显式 asyncio.Task + dataclass 信号替代。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Union

# ---- 控制流信号（waterfall 返回值） ----


class _Signal:
    """waterfall 控制流基类。子类表示一种决策。"""


@dataclass(frozen=True)
class Allow(_Signal):
    """放行，可选改写参数。"""

    args: Optional[dict] = None
    kwargs: Optional[dict] = None


@dataclass(frozen=True)
class Deny(_Signal):
    """拒绝，附带原因。"""

    reason: str = ""


@dataclass(frozen=True)
class Ask(_Signal):
    """请求人工审批，附带 prompt 给人看。

    v10.2:追加 `priority` 字段(读操作 allow / 写操作 ask / 危险写 ask),
    scope_guard 用它把分级信息带给 approval handler。默认 "write" 保持
    既有语义(向前兼容,零回归)。
    """

    prompt: str = ""
    default: str = "deny"
    priority: str = "write"


@dataclass(frozen=True)
class Replace(_Signal):
    """替换参数或结果。"""

    value: Any = None


@dataclass(frozen=True)
class Timeout(_Signal):
    """强制超时（秒）。"""

    seconds: float = 30.0


class ToolDenied(Exception):
    """pre_execute/post_execute 显式 deny 时抛出，终止工具调用。"""

    def __init__(self, reason: str = "") -> None:
        super().__init__(f"Tool denied: {reason}" if reason else "Tool denied")
        self.reason = reason


class ToolNeedsApproval(Exception):
    """Ask 信号无人接管时抛出（默认 waterfall 直通 Ask，框架层未配审批器则报错）。"""


# 钩子签名
# pre_execute(call: ToolCall) -> Allow | Deny | Ask | Replace[dict] | Timeout | None
# post_execute(call, result) -> Replace[result] | None
# result(call, final_result) -> None
PreExecuteHook = Callable[..., Awaitable[Union[_Signal, None]]]
PostExecuteHook = Callable[..., Awaitable[Union[_Signal, None]]]
ResultHook = Callable[..., Awaitable[None]]


@dataclass
class ToolWaterfall:
    """四层 waterfall 容器。

    每层是一个 async 钩子列表，按顺序执行。None 直通。首个非 None 信号决定
    控制流。
    """

    pre_execute: list[PreExecuteHook]  # type: ignore[type-arg]
    post_execute: list[PostExecuteHook]  # type: ignore[type-arg]
    result: list[ResultHook]  # type: ignore[type-arg]

    def __init__(self) -> None:
        self.pre_execute = []
        self.post_execute = []
        self.result = []

    def add_pre_execute(self, hook: PreExecuteHook) -> None:
        self.pre_execute.append(hook)

    def add_post_execute(self, hook: PostExecuteHook) -> None:
        self.post_execute.append(hook)

    def add_result(self, hook: ResultHook) -> None:
        self.result.append(hook)
