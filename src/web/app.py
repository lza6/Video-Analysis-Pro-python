"""FastAPI 应用实例 + lifespan + 路由挂载 + 静态资源。

启动: uvicorn src.web.app:app --host 127.0.0.1 --port 8000
开发: 与 Next.js dev(:3000)并行,前端 rewrite /api → :8000
生产: next build 后由本服务挂载 standalone 静态(见 _mount_frontend)
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import get_settings, warn_on_weak_token
from .deps import get_config_manager, get_job_store, web_jobs_root
from .routers import (
    agent, analyze, batch, config, decisions,
    health, im_gateway, logs, media, metrics, models,
    providers, remote, requests, skills, surveillance,
)
from .routers.analyze import init_analyze_semaphore
from .security import register_content_root
from .services.analyzer_service import AnalyzerService

log = logging.getLogger("web.app")

# 前端构建产物目录(next build standalone 输出)。不存在则不挂载(开发态)。
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "webapp" / "out"


def _load_dotenv() -> None:
    """加载项目根 .env 到 os.environ(供 ProviderRouter 读多 key)。

    不依赖 python-dotenv 第三方包:手动解析,与 load_from_env 的解析逻辑一致。
    已存在的环境变量不覆盖(尊重用户显式设置)。
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception as e:
        log.warning(f"加载 .env 失败: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动: 配置日志、加载 .env、初始化信号量、注入 AnalyzerService(持主 loop)。

    v10.0.0:启动 GCGuard(7×24 监控防 OOM,超阈值 gc.collect + 内存基线告警)。
    """
    settings = get_settings()
    # v10.0.0:结构化 JSON 日志 + trace_id 贯穿(便于前端 /api/logs 按 trace 过滤)。
    # 失败回退纯文本 basicConfig(防御性,不阻断启动)。
    try:
        from src.core.runtime.structured_log import install_json_logging
        install_json_logging(level=logging.INFO)
    except Exception as e:  # noqa: BLE001
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
        )
        log.warning("结构化日志初始化失败,回退纯文本: %s", e)

    # 加载 .env(供 ProviderRouter 的 load_from_env 读 VAP_NV_API_KEYS 多 key 路由)
    _load_dotenv()
    warn_on_weak_token()
    log.info("TingFeng Hermes Web 后端启动")

    init_analyze_semaphore(settings.analyze_concurrency)

    # 注入 AnalyzerService:持有当前(主)event loop,供后台线程 call_soon_threadsafe
    loop = asyncio.get_running_loop()
    cm = get_config_manager()
    app.state.analyzer_service = AnalyzerService(get_job_store(), cm, loop)

    # v10.0.0:SessionStore 单例(ReactLoopAgent 持久化 Session 用,崩溃可恢复)。
    # 延迟 import 避免启动期强依赖;失败不阻断 Web 服务(降级为内存 Session)。
    try:
        from src.core.agent.session import SessionStore
        app.state.session_store = SessionStore()
        log.info("SessionStore 已初始化: %s", app.state.session_store.db_path)
    except Exception as e:  # noqa: BLE001
        log.warning("SessionStore 初始化失败(降级内存 Session): %s", e)
        app.state.session_store = None

    # 注册内容根(frames 目录在 job 创建时动态注册,这里注册全局根)
    register_content_root("web_jobs", web_jobs_root())

    # 安装日志桥:把 src.core logger 转发到 /api/logs 环形缓冲 + SSE
    from .routers.logs import install_log_bridge
    install_log_bridge()

    # GCGuard:进程级长跑守护。阈值由 VAP_GC_THRESHOLD_MB 覆盖(默认 2GB),
    # 采样间隔 VAP_GC_INTERVAL_SEC(默认 60s)。失败不阻断启动(防御性)。
    from src.core.runtime.gc_guard import install_gc_guard
    try:
        gc_guard = install_gc_guard(
            app,
            threshold_mb=float(os.environ.get("VAP_GC_THRESHOLD_MB", "2048")),
            interval_sec=float(os.environ.get("VAP_GC_INTERVAL_SEC", "60")),
        )
        app.state.gc_guard = gc_guard
        gc_guard.set_baseline()
        gc_guard.start()
        log.info(
            "GCGuard 已启动: threshold=%.0fMB interval=%.1fs",
            gc_guard.threshold_mb, gc_guard.interval_sec,
        )
    except Exception as e:  # noqa: BLE001 — 守护失败不阻断 Web 服务
        log.warning("GCGuard 启动失败(已跳过,不阻断服务): %s", e)

    yield

    # 优雅停 GCGuard
    gc_guard = getattr(app.state, "gc_guard", None)
    if gc_guard is not None:
        try:
            gc_guard.stop()
        except Exception:  # noqa: BLE001
            log.warning("GCGuard 停止异常,已忽略")
    log.info("TingFeng Hermes Web 后端关闭")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="TingFeng Hermes — Web API",
        version="10.2.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # CORS:本地开发态 Next.js :3000 跨域调 :8000;生产同源可关
    if settings.web_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.web_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(metrics.router)
    app.include_router(media.router)
    app.include_router(models.router)
    app.include_router(agent.router)
    app.include_router(config.router)
    # v9.x:多 provider 预设 (cc-switch 风格,与 config 的 LastUsed 互补)
    app.include_router(providers.router)
    app.include_router(logs.router)
    app.include_router(decisions.router)
    app.include_router(skills.router)
    app.include_router(batch.router)
    app.include_router(surveillance.router)
    # v9.0.0:IM 网关 + 远程访问(Mock adapter,真实凭据预算 0)
    app.include_router(im_gateway.router)
    app.include_router(remote.router)
    # F7:LLM 请求日志 + token 消耗统计(/api/requests, /api/requests/stats)
    app.include_router(requests.router)

    # 生产:挂载前端静态产物(若存在)
    _mount_frontend(app)

    return app


def _mount_frontend(app: FastAPI) -> None:
    """挂载前端构建产物。开发态(webapp 跑 next dev)时目录不存在,跳过。"""
    if not FRONTEND_DIST.is_dir():
        return
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    log.info(f"前端静态已挂载: {FRONTEND_DIST}")


app = create_app()
