"""tests/test_serve_security_guard.py — serve.py 远程监听安全守卫测试。

验证用户启动日志实证的安全洞修复:
  - 默认 host=127.0.0.1(纯本地) + 无 token → 允许启动(保留桌面无 token 体验)
  - host=0.0.0.0(对外暴露) + 无 token → 拒绝启动(exit 1)
  - host=0.0.0.0 + 配 token → 允许启动

纯逻辑层测试:不真实起 uvicorn(会阻塞),只测 run_server 的守卫分支返回码。
通过 monkeypatch uvicorn.run 为 stub,验证守卫在 uvicorn.run 之前的短路。

策略:
  - monkeypatch uvicorn.run → 记录被调用即可(守卫放行才到这一步)
  - monkeypatch 端口探测/前端构建,避免外部依赖
  - 用 VAP_HOST / VAP_HEADLESS_TOKEN 环境变量驱动分支
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 清掉 lru_cache,确保每次用例的 env 生效
from src.web.config import get_settings  # noqa: E402


def _reset_settings_cache() -> None:
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """每条用例:清 settings 缓存 + 干净 env(只保留本用例设的)。"""
    _reset_settings_cache()
    # 清掉可能影响分支的 env
    for k in ("VAP_HOST", "VAP_HEADLESS_TOKEN", "VAP_PORT",
             "VAP_NO_BROWSER", "VAP_ANALYZE_CONCURRENCY"):
        monkeypatch.delenv(k, raising=False)
    # 不打开浏览器,不真实构建前端
    monkeypatch.setenv("VAP_NO_BROWSER", "1")
    _stub_port_and_frontend(monkeypatch, tmp_path)
    yield
    _reset_settings_cache()


def _stub_uvicorn(monkeypatch, calls: list) -> None:
    """把 uvicorn.run 换成 stub,记录调用(守卫放行才会到这)。"""
    import uvicorn  # type: ignore

    def _fake_run(app, **kwargs):  # noqa: ANN001
        calls.append({"app": app, "host": kwargs.get("host"), "port": kwargs.get("port")})

    monkeypatch.setattr(uvicorn, "run", _fake_run)


def _stub_port_and_frontend(monkeypatch, tmp_path) -> None:
    """绕过端口探测 + 前端构建(避免外部依赖/阻塞)。

    serve.run_server 内部 `from .app import FRONTEND_DIST` 读的是 app 模块的
    绑定,所以 patch app 模块的 FRONTEND_DIST 指向一个存在的 tmp 目录,
    让两次 `is_dir()` 都为真,跳过自动构建分支。
    """
    import src.web.serve as serve_mod
    import src.web.app as app_mod

    monkeypatch.setattr(serve_mod, "_port_available", lambda host, port: True)
    monkeypatch.setattr(serve_mod, "_try_build_frontend", lambda: None)
    # app.py 的 FRONTEND_DIST 是模块级常量,patch 它指向存在的 tmp 目录
    fake_dist = tmp_path / "frontend-stub"
    fake_dist.mkdir()
    monkeypatch.setattr(app_mod, "FRONTEND_DIST", fake_dist)


class TestRemoteListenGuard:
    """对外监听(非 loopback)必须配 token。"""

    def test_loopback_no_token_allows_start(self, monkeypatch):
        # 127.0.0.1 + 无 token → 放行(纯本地桌面体验)
        monkeypatch.setenv("VAP_HOST", "127.0.0.1")
        _stub_uvicorn(monkeypatch, calls := [])
        from src.web.serve import run_server

        rc = run_server(open_browser=False)
        assert rc == 0
        assert calls, "守卫放行后应进入 uvicorn.run"
        assert calls[0]["host"] == "127.0.0.1"

    def test_all_interfaces_no_token_rejects(self, monkeypatch):
        # 0.0.0.0 + 无 token → 拒绝(exit 1),不进 uvicorn.run
        monkeypatch.setenv("VAP_HOST", "0.0.0.0")
        calls: list = []
        _stub_uvicorn(monkeypatch, calls)
        from src.web.serve import run_server

        rc = run_server(open_browser=False)
        assert rc == 1, "对外监听 + 无 token 必须拒绝启动"
        assert calls == [], "拒绝时不应进入 uvicorn.run"

    def test_all_interfaces_with_token_allows_start(self, monkeypatch):
        # 0.0.0.0 + 配 token(>=32 字符) → 放行
        monkeypatch.setenv("VAP_HOST", "0.0.0.0")
        monkeypatch.setenv("VAP_HEADLESS_TOKEN", "x" * 32)
        _stub_uvicorn(monkeypatch, calls := [])
        from src.web.serve import run_server

        rc = run_server(open_browser=False)
        assert rc == 0
        assert calls and calls[0]["host"] == "0.0.0.0"

    def test_localhost_treated_as_loopback(self, monkeypatch):
        # localhost 也算 loopback,无 token 放行
        monkeypatch.setenv("VAP_HOST", "localhost")
        _stub_uvicorn(monkeypatch, calls := [])
        from src.web.serve import run_server

        rc = run_server(open_browser=False)
        assert rc == 0
