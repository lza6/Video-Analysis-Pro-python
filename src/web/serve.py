"""Web 服务启动入口(替代原 src/ui/main_window.run_main)。

用法:
  python -m src.web.serve [--port 8000] [--no-browser]

行为:
  1. 探测端口可用性(被占则报错退出,不静默换端口)
  2. 后台线程等待服务就绪后自动打开浏览器
  3. 运行 uvicorn(阻塞)
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
import webbrowser

log = logging.getLogger("web.serve")


def _port_available(host: str, port: int) -> bool:
    """检查端口是否可绑定。127.0.0.1 与 0.0.0.0 语义不同,统一试绑定。"""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((probe_host, port))
            return True
        except OSError:
            return False


def _wait_and_open(host: str, port: int, timeout: float = 30.0) -> None:
    """轮询 /api/health 直到服务就绪,然后打开浏览器。

    用探测 URL 而非固定 localhost:服务绑 0.0.0.0 时浏览器仍走 127.0.0.1。
    """
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{probe_host}:{port}/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            import urllib.request
            with urllib.request.urlopen(f"{url}api/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.4)
    else:
        log.warning(f"服务在 {timeout}s 内未就绪,仍尝试打开浏览器")

    log.info(f"打开浏览器: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        log.warning(f"自动打开浏览器失败({e}),请手动访问 {url}")


def _try_build_frontend() -> None:
    """首次启动自动构建前端(next build → out/)。

    需 node/npm 在 PATH。失败只告警不阻断(API 仍可用,前端 404 可后续手动构建)。
    """
    import shutil
    import subprocess

    npm = shutil.which("npm")
    if npm is None:
        log.info("未找到 npm,跳过前端自动构建(需手动安装 Node.js 后 cd webapp && npm install && npm run build)")
        return

    from .app import FRONTEND_DIST
    webapp_dir = FRONTEND_DIST.parent  # .../webapp
    log.info(f"首次启动:自动构建前端到 {FRONTEND_DIST}(可能需 1-3 分钟)…")
    # Windows 默认 GBK 编码会导致读 npm 输出 UnicodeDecodeError,统一用 utf-8
    import subprocess as _sp
    enc_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        # 1) 装依赖(若 node_modules 不存在)
        if not (webapp_dir / "node_modules").is_dir():
            log.info("安装前端依赖(npm install)…")
            r = _sp.run(
                [npm, "install"], cwd=str(webapp_dir),
                capture_output=True, timeout=600, encoding="utf-8", errors="replace",
                env=enc_env,
            )
            if r.returncode != 0:
                log.warning(f"npm install 失败(exit={r.returncode}): {(r.stderr or '')[-300:]}")
                return

        # 2) 构建
        log.info("构建前端(npm run build)…")
        r = _sp.run(
            [npm, "run", "build"], cwd=str(webapp_dir),
            capture_output=True, timeout=600, encoding="utf-8", errors="replace",
            env=enc_env,
        )
        if r.returncode != 0:
            log.warning(f"npm run build 失败(exit={r.returncode}): {(r.stderr or '')[-300:]}")
            return
        if FRONTEND_DIST.is_dir():
            log.info(f"前端构建完成: {FRONTEND_DIST}")
        else:
            log.warning(f"构建看似成功但 {FRONTEND_DIST} 仍不存在,检查 next.config output:export")
    except _sp.TimeoutExpired:
        log.warning("前端构建超时(>10min),跳过。请手动 cd webapp && npm run build")
    except Exception as e:
        log.warning(f"前端自动构建异常(不阻断 API): {e}")


def run_server(host: str | None = None, port: int | None = None,
               open_browser: bool = True, reload: bool = False) -> int:
    """启动 Web 服务。返回进程退出码。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
    )

    from .config import get_settings
    settings = get_settings()
    host = host or os.environ.get("VAP_HOST") or settings.host
    port = port or settings.port

    if not _port_available(host, port):
        log.error(
            f"端口 {port} 已被占用。请关闭占用进程,或设置环境变量 "
            f"VAP_PORT=<其他端口> 后重试。"
        )
        return 1

    # 前端产物缺失时尝试自动构建(首次启动或产物被清时)
    from .app import FRONTEND_DIST
    if not FRONTEND_DIST.is_dir():
        _try_build_frontend()
    if not FRONTEND_DIST.is_dir():
        log.warning(
            f"未找到前端构建产物 {FRONTEND_DIST},仅提供 API(浏览器打开会 404)。"
            f"请手动执行: cd webapp && npm install && npm run build"
        )

    if open_browser:
        threading.Thread(
            target=_wait_and_open, args=(host, port), daemon=True, name="browser-opener"
        ).start()

    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    log.info(f"Video Analysis Pro Web UI: http://{probe_host}:{port}")

    import uvicorn
    try:
        uvicorn.run(
            "src.web.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        log.info("收到中断信号,服务退出")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Video Analysis Pro Web UI 服务")
    parser.add_argument("--host", default=None, help="监听地址(默认取 VAP_HOST/配置)")
    parser.add_argument("--port", type=int, default=None, help="端口(默认取 VAP_PORT/配置)")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()
    raise SystemExit(
        run_server(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            reload=args.reload,
        )
    )


if __name__ == "__main__":
    main()
