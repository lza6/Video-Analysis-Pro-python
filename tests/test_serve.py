"""tests/test_serve.py — serve.py 启动入口测试。

覆盖范围:
  - _port_available / _find_available_port 端口探测逻辑(用真实 socket bind)
  - run_server 远程监听安全守卫(非 loopback 必须配 token)
  - run_server 端口被占自动找下一个可用端口
  - main 命令行参数解析(--port / --no-browser)

策略:
  - 用 unittest.mock.patch 拦截 uvicorn.run,不真实起服务(避免阻塞)
  - 用真实 socket bind 占端口(端口是系统资源,与 tmp 无关)
  - patch webbrowser.open 不打开真实浏览器
  - patch _try_build_frontend 跳过前端构建(避免外部 npm 依赖)
  - patch app.FRONTEND_DIST 指向存在的 tmp 目录,跳过产物缺失分支
  - 每条用例清 settings lru_cache,确保 env 驱动的分支生效
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.web.config import get_settings  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reset_settings_cache() -> None:
    """清 get_settings lru_cache,确保 env 变化后重新构造。"""
    get_settings.cache_clear()  # type: ignore[attr-defined]


def _grab_free_port() -> int:
    """让 OS 分配一个空闲端口(socket 关闭后短暂可用)。

    用 0 绑定让内核选端口,返回后立即关闭。竞态极小概率(本地工具场景可接受)。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hold_port(port: int | None = None) -> tuple[socket.socket, int]:
    """以独占方式占住一个端口,防止 SO_REUSEADDR 再 bind。

    Windows 上 SO_REUSEADDR 允许多 socket 绑同端口,SO_EXCLUSIVEADDRUSE 才能
    真正独占。serve._port_available 用 SO_REUSEADDR 探测,被占端口应返回 False。

    全量合跑时进程内大量 socket 开合会占用大量动态端口。bind(0) 让 OS 从动态
    端口池(Windows 默认 49152-65535)分配;若动态池被占满,bind(0) 会不断返回
    已占用端口,导致本函数重试耗尽。因此:
      1. 优先用 bind(0) 拿 OS 分配端口 + EXCLUSIVEADDRUSE 持有 + 反向探测;
      2. 反向探测确认 _port_available 返回 False,否则换端口重试;
      3. 动态池满时(bind(0) 返回已占用端口),改用低位段 1024-9000 搜索可用
         端口兜底(该段与动态池不重叠,几乎不可能被占满)。
    显式指定端口时不换号(bind 失败/探测失败都抛异常)。
    """
    from src.web.serve import _port_available

    def _try_hold(p: int) -> socket.socket | None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            s.bind(("127.0.0.1", p))
            s.listen(1)
        except OSError:
            s.close()
            return None
        if _port_available("127.0.0.1", p):
            s.close()
            return None
        return s

    if port is not None:
        s = _try_hold(port)
        if s is None:
            raise RuntimeError(f"端口 {port} 无法可靠独占(可能被占或 TIME_WAIT 复用)")
        return s, port

    # 动态池优先
    for _ in range(20):
        s = _try_hold(0)
        if s is not None:
            return s, s.getsockname()[1]
    # 动态池满 → 低位段兜底(1024-9000)
    for p in range(1024, 9000):
        s = _try_hold(p)
        if s is not None:
            return s, p
    raise RuntimeError("无法找到可可靠独占的端口(动态池与低位段均不可用)")


def _stub_uvicorn(calls: list) -> None:
    """patch uvicorn.run 为 stub,记录调用(守卫放行才到这)。"""
    def _fake_run(app, **kwargs):  # noqa: ANN001
        calls.append({
            "app": app,
            "host": kwargs.get("host"),
            "port": kwargs.get("port"),
        })

    # patch 模块级 import 的 uvicorn.run(用 import uvicorn + uvicorn.run 调用)
    import uvicorn  # type: ignore
    uvicorn.run = _fake_run  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """每条用例:清 settings 缓存 + 干净 env + 跳过端口/前端/浏览器。"""
    _reset_settings_cache()
    for k in ("VAP_HOST", "VAP_HEADLESS_TOKEN", "VAP_PORT",
             "VAP_NO_BROWSER", "VAP_ANALYZE_CONCURRENCY"):
        monkeypatch.delenv(k, raising=False)
    # 不打开真实浏览器
    monkeypatch.setattr("src.web.serve.webbrowser.open", lambda url: None)
    # 跳过前端自动构建(避免 npm 依赖)
    monkeypatch.setattr("src.web.serve._try_build_frontend", lambda: None)
    # patch app 模块 FRONTEND_DIST 指向存在的 tmp 目录,跳过产物缺失分支
    fake_dist = tmp_path / "frontend-stub"
    fake_dist.mkdir()
    monkeypatch.setattr("src.web.app.FRONTEND_DIST", fake_dist)
    yield
    _reset_settings_cache()


# ---------------------------------------------------------------------------
# _port_available
# ---------------------------------------------------------------------------

class TestPortAvailable:
    """_port_available: 端口探测逻辑。"""

    def test_returns_true_for_free_port(self):
        # Arrange: OS 分配一个空闲端口
        port = _grab_free_port()

        # Act
        from src.web.serve import _port_available

        result = _port_available("127.0.0.1", port)

        # Assert
        assert result is True

    def test_returns_false_for_occupied_port(self):
        # Arrange: 用 SO_EXCLUSIVEADDRUSE 独占端口(Windows 上 SO_REUSEADDR
        # 允许多 socket 绑同端口,必须用 EXCLUSIVEADDRUSE 才能真正占住)
        holder, port = _hold_port()
        try:
            from src.web.serve import _port_available

            # Act: 同端口再探测应不可用
            result = _port_available("127.0.0.1", port)
        finally:
            holder.close()

        # Assert
        assert result is False, f"端口 {port} 已被占,应返回 False"


# ---------------------------------------------------------------------------
# _find_available_port
# ---------------------------------------------------------------------------

class TestFindAvailablePort:
    """_find_available_port: 从 preferred 起向上找可用端口。"""

    def test_returns_preferred_when_free(self):
        # Arrange: 选一个空闲端口作为 preferred
        preferred = _grab_free_port()

        from src.web.serve import _find_available_port

        # Act
        result = _find_available_port("127.0.0.1", preferred)

        # Assert
        assert result == preferred

    def test_finds_next_when_preferred_occupied(self):
        # Arrange: 用 SO_EXCLUSIVEADDRUSE 独占 preferred 端口
        holder, occupied = _hold_port()
        try:
            from src.web.serve import _find_available_port

            # Act: preferred 被占,应返回 preferred+1 或更大的可用端口
            result = _find_available_port("127.0.0.1", occupied, max_tries=20)
        finally:
            holder.close()

        # Assert
        assert result != 0, "应找到下一个可用端口,而非返回 0"
        assert result > occupied, "返回端口应大于被占的 preferred"

    def test_returns_zero_when_all_occupied(self):
        # Arrange: 连续独占 20 个端口(preferred..preferred+19)全占住。
        # preferred 选一个较高值避免与系统端口冲突。
        base = 50000
        holders = []
        try:
            for offset in range(20):
                try:
                    s, _ = _hold_port(base + offset)
                    holders.append(s)
                except OSError:
                    # 该端口已被占,跳过(最终全部不可用即符合"全占"语义)
                    continue

            from src.web.serve import _find_available_port

            # Act: max_tries=20,base..base+19 全不可用
            result = _find_available_port("127.0.0.1", base, max_tries=20)
        finally:
            for s in holders:
                s.close()

        # Assert
        assert result == 0, "20 个端口全被占应返回 0"


# ---------------------------------------------------------------------------
# run_server — 远程监听安全守卫
# ---------------------------------------------------------------------------

class TestRunServerGuard:
    """run_server: 对外监听必须配 token。"""

    def test_rejects_non_loopback_without_token(self, monkeypatch):
        # Arrange: host=0.0.0.0 + 无 token → 拒绝(exit 1),不进 uvicorn
        monkeypatch.setenv("VAP_HOST", "0.0.0.0")
        calls: list = []
        _stub_uvicorn(calls)
        from src.web.serve import run_server

        # Act
        rc = run_server(open_browser=False)

        # Assert
        assert rc == 1, "对外监听 + 无 token 必须拒绝启动"
        assert calls == [], "拒绝时不应进入 uvicorn.run"

    def test_accepts_loopback_without_token(self, monkeypatch):
        # Arrange: host=127.0.0.1 + 无 token → 放行(纯本地桌面体验)
        monkeypatch.setenv("VAP_HOST", "127.0.0.1")
        calls: list = []
        _stub_uvicorn(calls)
        # 确保端口可用:用 OS 分配的空闲端口
        port = _grab_free_port()
        monkeypatch.setenv("VAP_PORT", str(port))
        from src.web.serve import run_server

        # Act
        rc = run_server(open_browser=False)

        # Assert
        assert rc == 0
        assert calls, "守卫放行后应进入 uvicorn.run"
        assert calls[0]["host"] == "127.0.0.1"

    def test_accepts_non_loopback_with_token(self, monkeypatch):
        # Arrange: host=0.0.0.0 + token(>=32 字符) → 放行
        monkeypatch.setenv("VAP_HOST", "0.0.0.0")
        monkeypatch.setenv("VAP_HEADLESS_TOKEN", "x" * 32)
        calls: list = []
        _stub_uvicorn(calls)
        port = _grab_free_port()
        monkeypatch.setenv("VAP_PORT", str(port))
        from src.web.serve import run_server

        # Act
        rc = run_server(open_browser=False)

        # Assert
        assert rc == 0
        assert calls and calls[0]["host"] == "0.0.0.0"


# ---------------------------------------------------------------------------
# run_server — 端口被占自动找下一个
# ---------------------------------------------------------------------------

class TestRunServerPortFallback:
    """run_server: preferred 端口被占时自动跳到下一个可用端口。"""

    def test_port_occupied_finds_next(self, monkeypatch):
        # Arrange: 用 SO_EXCLUSIVEADDRUSE 独占 preferred 端口
        holder, occupied = _hold_port()
        try:
            monkeypatch.setenv("VAP_HOST", "127.0.0.1")
            monkeypatch.setenv("VAP_PORT", str(occupied))
            calls: list = []
            _stub_uvicorn(calls)
            from src.web.serve import run_server

            # Act: preferred 被占,应自动找下一个可用端口
            rc = run_server(open_browser=False)
        finally:
            holder.close()

        # Assert
        assert rc == 0
        assert calls, "应进入 uvicorn.run"
        assert calls[0]["port"] != occupied, "端口被占应跳到下一个可用端口"
        assert calls[0]["port"] > occupied, "下一个端口应大于被占端口"


# ---------------------------------------------------------------------------
# main — 命令行参数解析
# ---------------------------------------------------------------------------

class TestMain:
    """main(): 命令行参数解析 + 转发到 run_server。"""

    def test_parses_args(self, monkeypatch):
        # Arrange: mock sys.argv,捕获 run_server 调用(通过 patch run_server)
        monkeypatch.setattr(sys, "argv", ["serve.py", "--port", "8000"])
        captured = {}

        def _fake_run_server(host=None, port=None, open_browser=True, reload=False):
            captured.update(host=host, port=port,
                           open_browser=open_browser, reload=reload)
            return 0

        monkeypatch.setattr("src.web.serve.run_server", _fake_run_server)
        from src.web.serve import main

        # Act: main 内部 raise SystemExit(rc),需捕获
        with pytest.raises(SystemExit) as exc_info:
            main()

        # Assert
        assert exc_info.value.code == 0
        assert captured["port"] == 8000
        assert captured["open_browser"] is True

    def test_no_browser_flag(self, monkeypatch):
        # Arrange: --no-browser → open_browser=False
        monkeypatch.setattr(sys, "argv", ["serve.py", "--no-browser"])
        captured = {}

        def _fake_run_server(host=None, port=None, open_browser=True, reload=False):
            captured["open_browser"] = open_browser
            return 0

        monkeypatch.setattr("src.web.serve.run_server", _fake_run_server)
        from src.web.serve import main

        # Act
        with pytest.raises(SystemExit):
            main()

        # Assert
        assert captured["open_browser"] is False
