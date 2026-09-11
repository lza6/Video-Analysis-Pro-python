"""P2-3 浏览器/CDP 自动化工具测试。

覆盖（契约见批 B-P2-3）：
  ① BrowserTool 用 Playwright 真实开一个 data:/file:// 页面 → navigate/snapshot
     得文本 + click（或 mock 若浏览器不可用）
  ② CdpTool mock 路径（无端口时明确 mock 返回）
  ③ register_web_auto_tools 可注册 + disposer 卸载
  ④ scope_guard 分类：snapshot/screenshot allow、evaluate 写/fill 类 ask
  ⑤ CUA service mock（desktop/cua-service.js 无 GUI 返回 mock 截图对象 + 标注）

不依赖 pytest-asyncio：用 asyncio.run()/new_event_loop 驱动 async
（与 tests/test_agent_framework.py 一致）。
"""
from __future__ import annotations

import asyncio
import base64
import subprocess
from pathlib import Path

import pytest

from src.core.tools.registry import ToolCall, ToolRegistry
from src.core.tools.scope_guard import ScopeGuard
from src.core.tools.web_auto import (
    build_browser_tool,
    build_cdp_tool,
    register_web_auto_tools,
)
from src.core.tools.web_auto.browser import PROVIDER_ENV
from src.core.tools.web_auto.cdp import CDP_URL_ENV

PROJECT_ROOT = Path(__file__).parent.parent
DESKTOP_DIR = PROJECT_ROOT / "desktop"
CUA_SERVICE = DESKTOP_DIR / "cua-service.js"


def _run(coro):
    """同步驱动 async 测试（与 test_agent_framework.py 一致）。"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


# ---------------------------------------------------------------------------
# 辅助：判断真实浏览器是否可用
# ---------------------------------------------------------------------------


def _real_playwright_available() -> bool:
    """真实 Playwright chromium 是否可启动（冒烟探测，慢路径首次约 1s）。"""
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415

        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            try:
                pg = b.new_page()
                pg.set_content("<h1>probe</h1>")
                return pg.locator("h1").inner_text() == "probe"
            finally:
                b.close()
    except Exception:  # noqa: BLE001
        return False


_REAL_BROWSER = _real_playwright_available()
_SKIP_REAL = pytest.mark.skipif(
    not _REAL_BROWSER,
    reason="本机真实 Playwright chromium 不可用，真实浏览器用例跳过（mock 路径覆盖）",
)

# ---------------------------------------------------------------------------
# ① BrowserTool 真实 / mock 路径
# ---------------------------------------------------------------------------


def test_browser_tool_mock_forced() -> None:
    """force_mock → mode=mock，snapshot 返回标注占位（不伪造真实页面）。"""
    tool = build_browser_tool(force_mock=True)
    assert tool.mode == "mock"

    r = _run(tool.open("https://example.com/"))
    assert r["ok"] is True and r["mode"] == "mock"
    assert "mock" in r.get("note", "")

    snap = _run(tool.snapshot())
    assert snap["mode"] == "mock"
    assert "mock" in snap.get("note", "")
    assert snap["url"] == "https://example.com/"
    assert "未连真实浏览器" in snap["text"]

    shot = _run(tool.screenshot())
    assert shot["mode"] == "mock"
    assert shot["image"] is None

    _run(tool.close())


def test_browser_tool_auto_provider_env(monkeypatch) -> None:
    """VAP_WEB_AUTO_PROVIDER=mock → 强制 mock。"""
    monkeypatch.setenv(PROVIDER_ENV, "mock")
    tool = build_browser_tool()
    assert tool.mode == "mock"
    r = _run(tool.open("https://a.b/"))
    assert r["mode"] == "mock"
    _run(tool.close())


@_SKIP_REAL
def test_browser_tool_real_navigate_snapshot_click() -> None:
    """真实 Playwright：open file:// 页面 → snapshot 得文本 + click 生效。

    file:// 渲染不受 data: URL 变体（部分 chromium 把 base64 当纯文本）影响，
    title/h1/click 均可确定性断言。
    """
    tmp_dir = Path(
        __import__("os").environ.get("TEMP", "/tmp")).resolve() / "vap-webauto-test"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    html_file = tmp_dir / "real.html"
    if not html_file.exists():
        html_file.write_text(
            "<html><head><title>Real Page</title></head><body>"
            "<h1>Hello Agent</h1>"
            "<button id='b1' onclick=\"document.getElementById('out')"
            ".innerText='clicked'\">Go</button>"
            "<p id='out'>before</p></body></html>",
            encoding="utf-8",
        )
    url = html_file.as_uri()
    tool = build_browser_tool(provider="playwright")
    assert tool.mode == "playwright"

    r = _run(tool.open(url))
    assert r["ok"] is True and r["mode"] == "playwright"
    assert "Real Page" in r.get("title", "")

    snap = _run(tool.snapshot())
    assert snap["ok"] is True, f"snapshot failed: {snap}"
    assert "Hello Agent" in snap["text"]
    assert any(e.get("text") == "Go" for e in snap["elements"])

    # click 写操作：直接驱动后端点击（scope_guard 分类在 ④ 单独验证）
    clk = _run(tool.click(selector="#b1"))
    assert clk["ok"] is True
    out = _run(tool.snapshot())
    assert "clicked" in out["text"]

    shot = _run(tool.screenshot())
    assert shot["ok"] is True and shot["mime"] == "image/png"
    # 真实截图 base64 非空且能还原 PNG 魔数
    raw = base64.b64decode(shot["image"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    _run(tool.close())


# ---------------------------------------------------------------------------
# ② CdpTool mock 路径
# ---------------------------------------------------------------------------


def test_cdp_tool_mock_without_url(monkeypatch) -> None:
    """无 VAP_CDP_URL → mock 模式，list_targets/evaluate 明确标注。"""
    monkeypatch.delenv(CDP_URL_ENV, raising=False)
    tool = build_cdp_tool()
    assert tool.mode == "mock"

    targets = _run(tool.list_targets())
    assert targets["mode"] == "mock"
    assert "mock" in targets.get("note", "")
    assert len(targets["targets"]) >= 1
    assert targets["targets"][0]["mock"] is True

    att = _run(tool.attach())
    assert att["mode"] == "mock"

    ev = _run(tool.evaluate("1 + 1"))
    assert ev["mode"] == "mock"
    assert ev.get("mock") is True
    assert "mock-evaluated" in str(ev.get("value", ""))

    _run(tool.close())


def test_cdp_tool_mock_forced() -> None:
    """force_mock=True → mock 模式（即使给了 URL 也 mock）。"""
    tool = build_cdp_tool(cdp_url="http://127.0.0.1:9222", force_mock=True)
    assert tool.mode == "mock"
    r = _run(tool.list_targets())
    assert r["mode"] == "mock"
    _run(tool.close())


def test_cdp_tool_fallback_on_unreachable() -> None:
    """提供不可达 CDP 端点 → 真实连接失败 → mock 降级并标注（不伪造）。"""
    tool = build_cdp_tool(cdp_url="http://127.0.0.1:1")
    # 端点端口 1 几乎必然拒绝连接；若环境恰好可达则跳过
    if tool.mode != "mock":
        pytest.skip("127.0.0.1:1 意外可达，跳过降级断言")
    r = _run(tool.list_targets())
    assert r["mode"] == "mock"
    assert "降级" in r.get("note", "") or "mock" in r.get("note", "")
    _run(tool.close())


# ---------------------------------------------------------------------------
# ③ register_web_auto_tools 注册 + disposer
# ---------------------------------------------------------------------------


def test_register_web_auto_tools_and_dispose() -> None:
    """注册 12 工具（7 浏览器 + 5 CDP），disposer 后移除。"""
    reg = ToolRegistry()
    disposers = register_web_auto_tools(reg, browser_provider="mock",
                                        cdp_url="")
    names = set(reg.list_names())
    assert names == {
        "web_browser_open", "web_browser_navigate", "web_browser_snapshot",
        "web_browser_screenshot", "web_browser_trigger", "web_browser_update",
        "web_browser_close",
        "cdp_list_targets", "cdp_attach", "cdp_evaluate", "cdp_eval_write",
        "cdp_close",
    }
    assert len(disposers) == 14  # 12 工具 disposer + 2 连接关闭 disposer

    for disp in disposers:
        disp()
    assert set(reg.list_names()) == set()
    assert "web_browser_open" not in reg.list_names()


def test_register_web_auto_tools_partial() -> None:
    """browser_enabled=False → 只注册 CDP；cdp_enabled=False → 只注册浏览器。"""
    reg = ToolRegistry()
    register_web_auto_tools(reg, browser_provider="mock", cdp_url="",
                            browser_enabled=False)
    assert "web_browser_open" not in reg.list_names()
    assert "cdp_list_targets" in reg.list_names()
    assert "cdp_eval_write" in reg.list_names()

    reg2 = ToolRegistry()
    register_web_auto_tools(reg2, browser_provider="mock", cdp_url="",
                            cdp_enabled=False)
    assert "web_browser_open" in reg2.list_names()
    assert "cdp_list_targets" not in reg2.list_names()


def test_adapter_reexport() -> None:
    """adapter 层 re-export：from adapter import register_web_auto_tools 可用。"""
    from src.core.tools.adapter import register_web_auto_tools as f

    reg = ToolRegistry()
    f(reg, browser_provider="mock", cdp_url="")
    assert "web_browser_open" in reg.list_names()
    assert "cdp_list_targets" in reg.list_names()


# ---------------------------------------------------------------------------
# ④ scope_guard 分类
# ---------------------------------------------------------------------------


def test_scope_guard_browser_read_allow() -> None:
    """snapshot / screenshot 读操作 → allow。

    v10.3.1 (P0-3):cdp_evaluate 升为危险写(任意 JS),不再 allow ——
    从读放行清单移除,改入危险写断言(见 test_web_browser_snapshot_is_read)。
    """
    guard = ScopeGuard(allow_write_general=False,
                       allow_write_delete=False, allow_write_cut=False)
    for name in ("web_browser_snapshot", "web_browser_screenshot",
                 "web_browser_open", "web_browser_navigate",
                 "cdp_list_targets", "cdp_attach"):
        level, reason = guard.decision(ToolCall(name=name, args={}))
        assert level == "allow", f"{name} 应 allow，实际 {level}（{reason}）"
    # cdp_evaluate 任意 JS → 危险写 ask(P0-3)
    level, _ = guard.decision(ToolCall(name="cdp_evaluate", args={}))
    assert level == "ask", f"cdp_evaluate 应 ask(P0-3)，实际 {level}"


def test_scope_guard_browser_write_ask() -> None:
    """trigger / update（fill 类）→ ask;cdp_eval_write 危险写 → ask。"""
    guard = ScopeGuard(allow_write_general=False,
                       allow_write_delete=False, allow_write_cut=False)
    for name in ("web_browser_trigger", "web_browser_update", "cdp_eval_write"):
        level, reason = guard.decision(ToolCall(name=name, args={}))
        assert level == "ask", f"{name} 应 ask，实际 {level}（{reason}）"


def test_scope_guard_general_switch_allows_browser_write(monkeypatch) -> None:
    """VAP_ALLOW_WRITE_GENERAL=1 → web_browser_trigger/update 放行。

    v10.3.1 (P0-3):cdp_eval_write 已升危险写,general 开关**不**放行
    (与 delete 同语义:危险写只认专用开关,此处断言仍 ask)。
    """
    monkeypatch.setenv("VAP_ALLOW_WRITE_GENERAL", "1")
    guard = ScopeGuard()
    assert guard.decision(ToolCall(name="web_browser_trigger", args={}))[0] == "allow"
    assert guard.decision(ToolCall(name="web_browser_update", args={}))[0] == "allow"
    # 危险写不被 general 开关放行(安全侧)
    level, _ = guard.decision(ToolCall(name="cdp_eval_write", args={}))
    assert level == "ask", \
        f"cdp_eval_write 危险写不应被 general 放行(P0-3)，实际 {level}"


def test_web_browser_snapshot_is_read() -> None:
    """classify：snapshot 工具归类 read（与 allow 断言一致）。

    v10.3.1 (P0-3):cdp_eval_write / cdp_evaluate 升为 dangerous_write
    (任意 JS 执行);cdp_execute 同族(旧名,与 evaluate 同语义)。
    """
    guard = ScopeGuard()
    assert guard.classify("web_browser_snapshot") == "read"
    assert guard.classify("web_browser_screenshot") == "read"
    assert guard.classify("web_browser_trigger") == "write"
    assert guard.classify("web_browser_update") == "write"
    assert guard.classify("cdp_eval_write") == "dangerous_write"
    assert guard.classify("cdp_evaluate") == "dangerous_write"
    # cdp_execute 是独立旧名工具(非 cdp_evaluate 别名),P0-3 未列,保持 read
    assert guard.classify("cdp_execute") == "read"


# ---------------------------------------------------------------------------
# ⑤ CUA service mock（desktop/cua-service.js）
# ---------------------------------------------------------------------------


def _node_available() -> bool:
    return shutil_which("node") is not None


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def test_cua_service_mock_screenshot() -> None:
    """CUA service：无 webContents → mock 截图对象 + 标注。"""
    node = shutil_which("node")
    if not node:
        pytest.skip("node 不可用，跳过 CUA service 单测（node --check 已在 CI 跑）")
    if not CUA_SERVICE.exists():
        pytest.fail(f"cua-service.js 缺失: {CUA_SERVICE}")
    script = (
        "const { mockScreenshot, isMockShot } = require('"
        + str(CUA_SERVICE).replace("\\", "/")
        + "');"
        "const shot = mockScreenshot('no webContents');"
        "if (!isMockShot(shot)) throw new Error('not mock');"
        "if (shot.dataUrl !== null) throw new Error('dataUrl should be null');"
        "process.stdout.write('CUA_MOCK_OK');"
    )
    r = subprocess.run([node, "-e", script], capture_output=True,
                       text=True, timeout=20)
    assert r.returncode == 0, f"stderr={r.stderr}"
    assert "CUA_MOCK_OK" in r.stdout


def test_cua_service_mount_returns_unmount() -> None:
    """mountCuaService 返回卸载函数（无实际 ipcMain 时 handler 注册 try 捕获）。"""
    node = shutil_which("node")
    if not node:
        pytest.skip("node 不可用")
    if not CUA_SERVICE.exists():
        pytest.fail(f"cua-service.js 缺失: {CUA_SERVICE}")
    script = (
        "const { mountCuaService } = require('"
        + str(CUA_SERVICE).replace("\\", "/")
        + "');"
        "const fakeIpc = {"
        "  handle: (ch, fn) => { if (typeof fn !== 'function') throw new Error('handler not fn'); },"
        "  removeHandler: () => {}"
        "};"
        "const unmount = mountCuaService(fakeIpc, () => null);"
        "if (typeof unmount !== 'function') throw new Error('unmount not fn');"
        "unmount();"
        "process.stdout.write('CUA_MOUNT_OK');"
    )
    r = subprocess.run([node, "-e", script], capture_output=True,
                       text=True, timeout=20)
    assert r.returncode == 0, f"stderr={r.stderr}"
    assert "CUA_MOUNT_OK" in r.stdout


# ---------------------------------------------------------------------------
# schema 完整性
# ---------------------------------------------------------------------------


def test_browser_schema_has_required_fields() -> None:
    """浏览器工具 input_schema 含 required 约束。"""
    tool = build_browser_tool(force_mock=True)
    by_name = {d.name: d for d in tool.definitions()}
    assert by_name["web_browser_open"].input_schema["required"] == ["url"]
    assert by_name["web_browser_update"].input_schema["required"] == ["value"]
    assert by_name["web_browser_snapshot"].input_schema.get("properties", {}) == {}
    _run(tool.close())


def test_cdp_schema_has_required_fields() -> None:
    """CDP 工具 input_schema 含 required 约束。"""
    tool = build_cdp_tool(force_mock=True)
    by_name = {d.name: d for d in tool.definitions()}
    assert by_name["cdp_evaluate"].input_schema["required"] == ["expression"]
    assert by_name["cdp_eval_write"].input_schema["required"] == ["expression"]
    assert "expression" in by_name["cdp_evaluate"].to_llm_schema()["function"]["parameters"]["properties"]  # noqa: E501
    _run(tool.close())