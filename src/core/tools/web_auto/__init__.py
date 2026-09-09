"""浏览器/CDP 自动化工具包——P2-3。

`register_web_auto_tools(registry, ...)` 把 BrowserTool 与 CdpTool 的
ToolDefinition 统一注册进 registry，返回 disposer 列表。

使用方式（装配方示例，不改装配文件）：

    from src.core.tools.web_auto import register_web_auto_tools
    disposers = register_web_auto_tools(registry)

provider / CDP 控制：
  - `VAP_WEB_AUTO_PROVIDER`（`playwright` / `mock` / 其它=auto）
  - `VAP_CDP_URL`（真实 Chrome DevTools 端点，缺省 mock）
"""
from __future__ import annotations

from typing import Callable, List, Optional

from src.core.tools.web_auto.browser import (
    BrowserTool,
    build_browser_tool,
)
from src.core.tools.web_auto.cdp import (
    CDP_URL_ENV,
    CdpTool,
    build_cdp_tool,
)

__all__ = [
    "register_web_auto_tools",
    "BrowserTool",
    "CdpTool",
    "build_browser_tool",
    "build_cdp_tool",
    "CDP_URL_ENV",
]


async def _close_browser_tool(tool: BrowserTool) -> None:
    """关闭浏览器连接（不抛，工具卸载不阻断）。"""
    try:
        await tool.close()
    except Exception:  # noqa: BLE001 - 卸载清理尽力而为
        pass


async def _close_cdp_tool(tool: CdpTool) -> None:
    """关闭 CDP 连接（不抛）。"""
    try:
        await tool.close()
    except Exception:  # noqa: BLE001
        pass


def _run_or_schedule(coro: Callable[[], object]) -> Callable[[], None]:
    """把 async 关闭协程包成同步 disposer（有 running loop 则 schedule，否则 run）。"""

    def _dispose() -> None:
        import asyncio as _aio

        try:
            _aio.get_running_loop()
        except RuntimeError:
            _aio.run(coro())  # type: ignore[arg-type]
        else:
            _aio.get_running_loop().create_task(coro())  # type: ignore[arg-type]

    return _dispose


def register_web_auto_tools(
    registry,
    *,
    browser_provider: Optional[str] = None,
    cdp_url: Optional[str] = None,
    cdp_enabled: bool = True,
    browser_enabled: bool = True,
) -> List[Callable[[], None]]:
    """把浏览器（真实 Playwright 或 mock）与 CDP（真实或 mock）工具注册进 registry。

    Args:
        registry: ToolRegistry 实例。
        browser_provider: 覆盖 VAP_WEB_AUTO_PROVIDER（None = 读 env）。
        cdp_url: 覆盖 VAP_CDP_URL（None = 读 env）。
        cdp_enabled: False 跳过 CDP 工具注册（默认 True）。
        browser_enabled: False 跳过浏览器工具注册（默认 True）。

    Returns:
        disposer 列表（关闭连接 + 从 registry 移除工具）。
    """
    disposers: List[Callable[[], None]] = []
    browser_tool: Optional[BrowserTool] = None
    cdp_tool: Optional[CdpTool] = None

    if browser_enabled:
        browser_tool = build_browser_tool(provider=browser_provider)
        for defn in browser_tool.definitions():
            disposers.append(registry.register(defn))

    if cdp_enabled:
        cdp_tool = build_cdp_tool(cdp_url=cdp_url)
        for defn in cdp_tool.definitions():
            disposers.append(registry.register(defn))

    if browser_tool is not None:
        disposers.append(
            _run_or_schedule(lambda: _close_browser_tool(browser_tool)))
    if cdp_tool is not None:
        disposers.append(
            _run_or_schedule(lambda: _close_cdp_tool(cdp_tool)))

    return disposers