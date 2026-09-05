"""依赖注入(Depends providers)。

集中提供 Settings / ConfigurationManager / 能力矩阵 / JobStore,
便于路由声明依赖、便于测试替换。
"""
from __future__ import annotations

import functools
import importlib
import logging
import os
import shutil
from pathlib import Path

from src.utils.config_manager import ConfigurationManager
from src.utils.constants import CACHE_DIR
from src.core.logic import (
    CLIP_AVAILABLE,
    NVIDIA_GPU_AVAILABLE,
    ADVANCED_FEATURES_AVAILABLE,
    FFMPEG_AVAILABLE,
)

from .job_store import JobStore

log = logging.getLogger("web.deps")

# 单例 JobStore(进程内,线程安全)
_job_store: JobStore | None = None


def get_job_store() -> JobStore:
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store


@functools.lru_cache(maxsize=1)
def _config_manager() -> ConfigurationManager:
    cm = ConfigurationManager()
    cm.load_main_config()
    return cm


def get_config_manager() -> ConfigurationManager:
    return _config_manager()


def capability_matrix() -> dict:
    """能力矩阵。复用 headless.py 的语义,/api/health 必须永远 <50ms。

    Ollama 探测留给 /api/analyze 真正调用时(/healthz 不探,避免阻塞)。
    """
    return {
        "clip_semantic": CLIP_AVAILABLE,
        "nvidia_gpu": NVIDIA_GPU_AVAILABLE,
        "advanced_media": ADVANCED_FEATURES_AVAILABLE,
        "ffmpeg": FFMPEG_AVAILABLE,
        "ocr": _module_available("paddleocr"),
        "llm_backend": "unknown",
    }


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def web_jobs_root() -> Path:
    """所有 Web 分析作业的工作目录根。注册为 'frames' 内容根。

    放在 CACHE_DIR 下(CACHE_DIR = "cache",已 gitignore),跨重启可清理。
    """
    root = Path(CACHE_DIR) / "web_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def disk_free_gb() -> float:
    try:
        usage = shutil.disk_usage(os.getcwd())
        return round(usage.free / 1024**3, 1)
    except Exception:
        return -1.0
