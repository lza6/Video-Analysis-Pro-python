"""GET /api/health — 能力矩阵 + 磁盘余量 + keyring 状态。

复用 headless.py 的 /healthz 语义,必须永远 <50ms 返回(不探 Ollama)。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import capability_matrix, disk_free_gb
from ..schemas import HealthResponse
from ..security import require_auth

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse, dependencies=[Depends(require_auth)])
def health() -> HealthResponse:
    from src.utils.config_manager import is_keyring_available
    return HealthResponse(
        status="ok",
        capabilities=capability_matrix(),
        disk_free_gb=disk_free_gb(),
        keyring_available=is_keyring_available(),
    )
