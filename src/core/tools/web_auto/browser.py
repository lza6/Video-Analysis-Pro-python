"""浏览器自动化（browser-use 范式，Playwright 桥接）——P2-3。

把"看 / 点 / 读网页"变成 Agent 工具：

  - `web_browser_open`      打开会话并导航（或复用现有会话）
  - `web_browser_navigate`  导航到新 URL
  - `web_browser_snapshot`  标题 + 正文文本 + 可交互元素清单（读操作）
  - `web_browser_screenshot` 截图（base64 PNG，读操作）
  - `web_browser_trigger`   点击可交互元素（写操作，scope_guard ask）
  - `web_browser_update`    向输入元素填写文本（写操作，scope_guard ask）
  - `web_browser_close`     关闭会话

Provider 解析（`VAP_WEB_AUTO_PROVIDER`）：
  - `playwright`：强制真实 Playwright（chromium headless）。启动失败 → 降级
    mock 并在结果中明确标注"mock 降级原因"（不伪造真实结果）。
  - `mock`：强制 mock（不启动浏览器），结果带 `mode: "mock"` 标注。
  - 其它 / 缺省（auto）：本机 Playwright 可用则真实；不可用则 mock 并标注。

付费红线：Playwright 走本机 chromium，不调用任何付费云端浏览器。
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional

from src.core.tools.definition import ToolDefinition

log = logging.getLogger("core.tools.web_auto.browser")

#: 浏览器 provider 环境变量：`playwright`=真实 / `mock`=强制 mock / 其它=auto。
PROVIDER_ENV = "VAP_WEB_AUTO_PROVIDER"
#: snapshot 正文最大截断字符数
_SNAPSHOT_TEXT_LIMIT = 3000
#: snapshot 可交互元素上限
_ELEMENTS_LIMIT = 60

_SIMPLE_SELECTOR_RE = r"^[a-zA-Z][a-zA-Z0-9\-_:.#\[\]=\"']*$"


def _playwright_installed() -> bool:
    """Playwright python 包是否安装（不保证浏览器二进制存在）。"""
    return importlib.util.find_spec("playwright") is not None


class _PlaywrightUnavailableError(RuntimeError):
    """Playwright 不可用（未安装 / 启动失败 / 浏览器二进制缺失）。"""


class _BaseBrowserBackend:
    """浏览器后端抽象。所有方法同步（无 IO 事件循环交互）。"""

    mode = "unknown"

    def open(self, url: str) -> Dict[str, Any]:  # pragma: no cover - 抽象
        raise NotImplementedError

    def navigate(self, url: str) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def snapshot(self) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def screenshot(self) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def click(self, selector: Optional[str], index: Optional[int]) -> Dict[str, Any]:  # noqa: E501 - pragma: no cover
        raise NotImplementedError

    def fill(self, selector: Optional[str], index: Optional[int], value: str) -> Dict[str, Any]:  # noqa: E501 - pragma: no cover
        raise NotImplementedError

    def close(self) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


class MockBrowserBackend(_BaseBrowserBackend):
    """Mock 后端：不启动真实浏览器，返回带 `mode: "mock"` 标注的占位结果。

    绝不伪造真实页面内容——snapshot 返回的文本/元素是示例骨架，
    且所有结果带 `note` 说明"mock，未读取真实页面"。
    """

    def __init__(self, note: str) -> None:
        super().__init__()
        self.mode = "mock"
        self._note = note
        self._url: Optional[str] = None
        self._title = "mock-page"
        self._body_text = "mock body text (未连真实浏览器，内容为占位)"

    def _stamp(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "mode": "mock", "note": self._note, **data}

    def open(self, url: str) -> Dict[str, Any]:
        self._url = url
        return self._stamp({"url": url, "action": "open"})

    def navigate(self, url: str) -> Dict[str, Any]:
        self._url = url
        return self._stamp({"url": url, "action": "navigate"})

    def snapshot(self) -> Dict[str, Any]:
        if self._url is None:
            return self._stamp({"error": "尚未 open/navigate 页面"})
        return self._stamp({
            "url": self._url,
            "title": self._title,
            "text": self._body_text,
            "elements": [
                {"index": 0, "tag": "a", "text": "mock-link",
                 "selector_hint": "", "visible": True},
            ],
        })

    def screenshot(self) -> Dict[str, Any]:
        return self._stamp({"image": None, "mime": None,
                           "reason": "mock 未执行截图"})

    def click(self, selector: Optional[str], index: Optional[int]) -> Dict[str, Any]:
        return self._stamp({"selector": selector, "index": index,
                           "action": "mock-click（未真实点击）"})

    def fill(self, selector: Optional[str], index: Optional[int], value: str) -> Dict[str, Any]:  # noqa: E501
        return self._stamp({"selector": selector, "index": index,
                           "field": value[:40],
                           "action": "mock-fill（未真实填写）"})

    def close(self) -> Dict[str, Any]:
        r = self._stamp({"closed": True})
        self._url = None
        return r


class PlaywrightBackend(_BaseBrowserBackend):
    """真实 Playwright chromium headless 后端。

    惰性启动：`open()` 才 launch（复用同一 instance 直到显式 close）。
    同步 API 由调用方经 `asyncio.to_thread` 包裹，避免阻塞事件循环。
    所有并发访问由 BrowserTool 内一把 threading.Lock 串行化。
    """

    def __init__(self) -> None:
        super().__init__()
        self.mode = "playwright"
        self._pw: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._url: Optional[str] = None

    # ---- 生命周期 ----
    def _ensure_started(self) -> None:
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:  # pragma: no cover
            raise _PlaywrightUnavailableError(
                f"playwright 未安装: {e}") from e
        try:
            pw = sync_playwright().start()
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
        except Exception as e:  # noqa: BLE001 - playwright 自有异常族
            try:
                pw.stop()  # type: ignore[name-defined]  # noqa: F821
            except Exception:  # noqa: BLE001
                pass
            raise _PlaywrightUnavailableError(
                f"chromium 启动失败（浏览器二进制缺失或环境不允许）: {e}") from e
        self._pw, self._browser, self._page = pw, browser, page

    def close(self) -> Dict[str, Any]:
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:  # noqa: BLE001
            pass
        self._pw = self._browser = self._page = None
        self._url = None
        return {"ok": True, "mode": "playwright", "closed": True}

    # ---- 动作 ----
    def _fallback_title(self) -> str:
        """data:/无 <title> 页面回退：取第一个 h1 文本。"""
        assert self._page is not None
        try:
            return self._page.locator("h1").first.inner_text(
                timeout=2000).strip()
        except Exception:  # noqa: BLE001
            return ""

    def open(self, url: str) -> Dict[str, Any]:
        self._ensure_started()
        assert self._page is not None
        self._page.goto(url, wait_until="load",
                        timeout=15000)
        self._url = self._page.url
        title = self._page.title()
        if not title:
            title = self._fallback_title()
        return {"ok": True, "mode": "playwright", "url": self._url,
                "title": title}

    def navigate(self, url: str) -> Dict[str, Any]:
        return self.open(url)

    def snapshot(self) -> Dict[str, Any]:
        if self._page is None:
            return {"ok": False, "mode": "playwright",
                    "error": "尚未 open 页面"}
        text = (self._page.inner_text("body") or "")[:_SNAPSHOT_TEXT_LIMIT]
        assert self._page is not None
        title = self._page.title()
        if not title:
            title = self._fallback_title()
        elements = self._page.eval_on_selector_all(
            "button, a[href], input, textarea, select, "
            "[role=button], [role=link], [contenteditable=true]",
            f"""els => {{
              const out = [];
              for (let i = 0; i < els.length && i < {_ELEMENTS_LIMIT}; i++) {{
                const el = els[i];
                const r = el.getBoundingClientRect();
                const text = (el.innerText || el.value
                    || el.getAttribute("placeholder") || el.textContent
                    || "").trim().slice(0, 80);
                const hint = el.id ? "#" + el.id
                    : (el.name ? '[name="' + el.name + '"]'
                       : (el.getAttribute("href")
                          ? 'a[href="' + el.getAttribute("href") + '"]' : ""));
                out.push({{
                  index: i,
                  tag: el.tagName.toLowerCase(),
                  text,
                  selector_hint: hint,
                  visible: r.width > 0 && r.height > 0 && el.offsetParent !== null,
                }});
              }}
              return out;
            }}""",
        )
        return {"ok": True, "mode": "playwright",
                "url": self._page.url, "title": title,
                "text": text, "elements": elements}

    def screenshot(self) -> Dict[str, Any]:
        if self._page is None:
            return {"ok": False, "mode": "playwright",
                    "error": "尚未 open 页面"}
        png = self._page.screenshot()
        return {"ok": True, "mode": "playwright",
                "image": base64.b64encode(png).decode("ascii"),
                "mime": "image/png", "format": "base64"}

    def _locator(self, selector: Optional[str], index: Optional[int]) -> Any:
        assert self._page is not None
        if selector:
            return self._page.locator(selector).first
        if index is not None:
            all_loc = self._page.locator(
                "button, a[href], input, textarea, select, "
                "[role=button], [role=link], [contenteditable=true]")
            return all_loc.nth(index)
        raise ValueError("click/fill 需要 selector 或 index 之一")

    def click(self, selector: Optional[str], index: Optional[int]) -> Dict[str, Any]:
        if self._page is None:
            return {"ok": False, "mode": "playwright", "error": "尚未 open 页面"}
        loc = self._locator(selector, index)
        loc.click(timeout=5000)
        return {"ok": True, "mode": "playwright",
                "selector": selector, "index": index,
                "url": self._page.url}

    def fill(self, selector: Optional[str], index: Optional[int], value: str) -> Dict[str, Any]:  # noqa: E501
        if self._page is None:
            return {"ok": False, "mode": "playwright", "error": "尚未 open 页面"}
        if not isinstance(value, str):
            raise ValueError("value 必须是字符串")
        loc = self._locator(selector, index)
        loc.fill(value, timeout=5000)
        return {"ok": True, "mode": "playwright",
                "selector": selector, "index": index, "field": value[:40],
                "url": self._page.url}


class BrowserTool:
    """BrowserTool 工厂：按 provider 解析产生一组 ToolDefinition。

    用法：
        tool = build_browser_tool()          # provider 从 env / 参数解析
        defs = tool.definitions()            # [ToolDefinition, ...] 待注册
        tool.close()                         # 释放真实浏览器（async）
        await tool.open(url) / snapshot() / ...  # 公开 async 方法（测试用）

    并发安全：同步 Playwright 调用经 threading.Lock 串行化 +
    asyncio.to_thread 出事件循环。
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        *,
        force_mock: bool = False,
    ) -> None:
        self._lock = threading.Lock()
        self._provider = provider or os.environ.get(PROVIDER_ENV, "")
        self._force_mock = force_mock
        self._mock_note = ""
        self._backend: _BaseBrowserBackend = self._resolve_backend()
        # 固定 worker 线程：Playwright sync API 的页面帧必须始终在同一线程
        # 活动 sync_playwright 上下文中操作，跨线程复用会抛
        # "cannot switch to a different thread"。首次真实调用时惰性创建。
        self._own_loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker: Optional["threading.Thread"] = None
        self._worker_pool: Any = None

    # ---- provider 解析 ----
    def _resolve_backend(self) -> _BaseBrowserBackend:
        if self._force_mock:
            return MockBrowserBackend("VAP_WEB_AUTO_PROVIDER=mock（或 force_mock）强制 mock")
        prov = self._provider.strip().lower()
        if prov == "mock":
            return MockBrowserBackend("VAP_WEB_AUTO_PROVIDER=mock 强制 mock（未连真实浏览器）")
        if prov == "playwright":
            if not _playwright_installed():
                return MockBrowserBackend(
                    "VAP_WEB_AUTO_PROVIDER=playwright 但 playwright 未安装 → mock 降级"
                )
            return PlaywrightBackend()
        # auto / 其它：本机可用则真实，否则 mock 并标注
        if _playwright_installed():
            return PlaywrightBackend()
        return MockBrowserBackend(
            "auto：本机未安装 playwright → mock 降级（未伪造真实结果）"
        )

    @property
    def mode(self) -> str:
        return self._backend.mode

    @property
    def mock_note(self) -> str:
        return getattr(self._backend, "_note", "")

    # ---- 底层驱动 ----
    def _exec(self, fn: Callable[..., Dict[str, Any]],
              *args: Any, **kwargs: Any) -> Dict[str, Any]:
        with self._lock:

            def _target() -> Dict[str, Any]:
                try:
                    return fn(*args, **kwargs)
                except _PlaywrightUnavailableError as e:
                    # 真实启动失败（二进制缺失等）→ 降级 mock 并明确标注
                    self._backend = MockBrowserBackend(
                        f"playwright 启动失败 → mock 降级: {e}")
                    if args and not kwargs:
                        return self._backend.open(args[0])
                    return {"ok": False, "mode": "mock",
                            "note": f"playwright 启动失败 → mock: {e}"}
                except Exception as e:  # noqa: BLE001 - 工具错误归一
                    return {"ok": False, "mode": self._backend.mode,
                            "error": f"{type(e).__name__}: {e}"}

            return _target()

    # ---- 公开 async 方法（测试直接驱动） ----
    def _wrap_thread(self, fn: Callable[..., Dict[str, Any]],
                     *args: Any, **kwargs: Any,
                     ) -> Callable[[], Dict[str, Any]]:
        """返回一个执行 `self._exec(fn, ...)` 的同步封装。

        Playwright sync API 要求页面帧始终由同一活动 sync_playwright 线程
        驱动；跨线程复用同一 browser/page 会抛
        "cannot switch to a different thread"。因此把 `_exec` 整体丢进
        一条持久专用线程（`_drive` 用 to_thread 跑封装），每次调用都复用
        该线程，保证帧始终在同一线程上。
        """
        def _guarded() -> Dict[str, Any]:
            try:
                return self._exec(fn, *args, **kwargs)
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "mode": self._backend.mode,
                        "error": f"{type(e).__name__}: {e}"}
        return _guarded

    def _drive(self, target: Callable[[], Dict[str, Any]]) -> "asyncio.coroutine":  # noqa: E501
        """保证同一 worker 线程执行：持久化 ThreadPoolExecutor(1) 复用线程。"""
        import concurrent.futures

        if self._worker_pool is None:
            # 关键：max_workers=1 且 executor 持久存活（不随调用销毁），
            # 线程 ID 保持一致，Playwright 页面帧不跨线程。
            self._worker_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="vap-webauto")

        async def _await() -> Dict[str, Any]:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(self._worker_pool, target)
            try:
                return await asyncio.wait_for(fut, timeout=30)
            except asyncio.TimeoutError:
                return {"ok": False, "mode": self._backend.mode,
                        "error": "browser 调用超时(30s)"}

        return _await()

    async def open(self, url: str) -> Dict[str, Any]:
        return await self._drive(self._wrap_thread(self._backend.open, url))

    async def navigate(self, url: str) -> Dict[str, Any]:
        return await self._drive(self._wrap_thread(self._backend.navigate, url))

    async def snapshot(self) -> Dict[str, Any]:
        return await self._drive(self._wrap_thread(self._backend.snapshot))

    async def screenshot(self) -> Dict[str, Any]:
        return await self._drive(self._wrap_thread(self._backend.screenshot))

    async def click(self, selector: Optional[str] = None,
                    index: Optional[int] = None) -> Dict[str, Any]:
        return await self._drive(self._wrap_thread(
            self._backend.click, selector, index))

    async def fill(self, value: str, selector: Optional[str] = None,
                   index: Optional[int] = None) -> Dict[str, Any]:
        return await self._drive(self._wrap_thread(
            self._backend.fill, selector, index, value))

    async def close(self) -> Dict[str, Any]:
        return await self._drive(self._wrap_thread(self._backend.close))

    # ---- ToolDefinition 装配 ----
    def definitions(self) -> List[ToolDefinition]:
        """产出 7 个工具定义。

        写操作命名带 scope_guard WRITE_PATTERNS 关键词：
          trigger/update → 分类 write → 审批 ask。
        读操作命名避开写关键词 → 分类 read → allow。
        """

        async def _open(url: str) -> Dict[str, Any]:
            return await self.open(url)

        async def _navigate(url: str) -> Dict[str, Any]:
            return await self.navigate(url)

        async def _snapshot() -> Dict[str, Any]:
            return await self.snapshot()

        async def _screenshot() -> Dict[str, Any]:
            return await self.screenshot()

        async def _trigger_click(selector: Optional[str] = None,
                                 index: Optional[int] = None) -> Dict[str, Any]:
            return await self.click(selector=selector, index=index)

        async def _update_field(value: str,
                                selector: Optional[str] = None,
                                index: Optional[int] = None) -> Dict[str, Any]:
            return await self.fill(selector=selector, index=index, value=value)

        async def _close() -> Dict[str, Any]:
            return await self.close()

        return [
            ToolDefinition(
                name="web_browser_open",
                description=(
                    "打开浏览器会话并导航到给定 URL（Playwright chromium）。"
                    "首次操作需先调用本工具。返回当前页标题。"
                ),
                execute_callback=_open,
                input_schema={"type": "object",
                              "properties": {"url": {"type": "string"}},
                              "required": ["url"]},
            ),
            ToolDefinition(
                name="web_browser_navigate",
                description="在当前浏览器会话导航到新 URL。",
                execute_callback=_navigate,
                input_schema={"type": "object",
                              "properties": {"url": {"type": "string"}},
                              "required": ["url"]},
            ),
            ToolDefinition(
                name="web_browser_snapshot",
                description=(
                    "快照当前页面：标题 + 正文文本（截断）+ 可交互元素清单"
                    "（按钮/链接/输入框，含 index 与 selector 提示）。"
                    "读操作，直接放行。"
                ),
                execute_callback=_snapshot,
                input_schema={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="web_browser_screenshot",
                description=(
                    "截取当前页面 PNG，base64 返回（data 用 image 字段还原）。"
                    "读操作，直接放行。"
                ),
                execute_callback=_screenshot,
                input_schema={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="web_browser_trigger",
                description=(
                    "点击页面上的可交互元素（写操作，需人工审批）。"
                    "selector 用 CSS 选择器，或 index 用 snapshot 里 elements 的 index。"
                ),
                execute_callback=_trigger_click,
                input_schema={"type": "object",
                              "properties": {
                                  "selector": {"type": "string"},
                                  "index": {"type": "integer"},
                              }},
            ),
            ToolDefinition(
                name="web_browser_update",
                description=(
                    "向输入元素填写文本（写操作，需人工审批）。"
                    "value 必填；目标用 selector 或 index。"
                ),
                execute_callback=_update_field,
                input_schema={"type": "object",
                              "properties": {
                                  "value": {"type": "string"},
                                  "selector": {"type": "string"},
                                  "index": {"type": "integer"},
                              },
                              "required": ["value"]},
            ),
            ToolDefinition(
                name="web_browser_close",
                description="关闭浏览器会话，释放资源。",
                execute_callback=_close,
                input_schema={"type": "object", "properties": {}},
            ),
        ]


def build_browser_tool(*, provider: Optional[str] = None,
                       force_mock: bool = False) -> BrowserTool:
    """构造 BrowserTool（provider 解析见类 docstring）。"""
    return BrowserTool(provider=provider, force_mock=force_mock)


__all__ = ["BrowserTool", "build_browser_tool", "MockBrowserBackend",
           "PlaywrightBackend", "PROVIDER_ENV"]