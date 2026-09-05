"""安全工具:Bearer Token 鉴权 + IP 滑动窗口限流 + 路径消毒。

从 src/server/headless.py 迁移并泛化,供所有 Web 路由复用。
"""
from __future__ import annotations

import hmac
import logging
import time
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings, get_settings

log = logging.getLogger("web.security")

# 复用 headless.py 的视频扩展名白名单,防 .exe/.py 写入
ALLOWED_VIDEO_EXTS = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".wmv", ".ts"}
)

# 帧图片扩展名白名单(静态资源挂载用)
ALLOWED_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})

# 媒体产物(GIF/Clips)扩展名
ALLOWED_MEDIA_EXTS = frozenset({".gif", ".mp4", ".webm", ".mov"})

# 配置层:白名单扩展 → 对应目录
# 启动时由 app.py lifespan 注入实际路径
_CONTENT_ROOTS: dict[str, Path] = {}


def register_content_root(category: str, path: Path) -> None:
    """注册某个内容类别(如 frames/gifs/clips)的根目录。

    路径消毒时只允许在已注册根目录内取文件,且扩展名必须匹配类别白名单。
    """
    _CONTENT_ROOTS[category] = path.resolve()


_bearer_scheme = HTTPBearer(auto_error=False)


def _check_token(got: str, expected: str) -> bool:
    """常量时间比较 token,防时序攻击;非 ASCII 兜底 False。"""
    try:
        return hmac.compare_digest(got, expected)
    except TypeError:
        return False


async def require_auth(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> str:
    """Bearer Token 依赖。token 为空时鉴权关闭(向后兼容)。

    用法: router 传 dependencies=[Depends(require_auth)] 或单个端点 Depends。
    返回 "ok"(供未来加用户体系)。
    """
    expected = settings.headless_token
    if not expected:
        return "anonymous"  # 鉴权关闭

    if creds is None or creds.scheme.lower() != "bearer":
        # 401 不打印收到的 token,只记 mismatch
        log.warning("unauthorized: missing or non-bearer scheme")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"Connection": "close"},
            detail={"error": "unauthorized"},
        )

    if not _check_token(creds.credentials, expected):
        log.warning("unauthorized: token mismatch")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"Connection": "close"},
            detail={"error": "unauthorized"},
        )
    return "ok"


# ============================ IP 滑动窗口限流 ============================

# 复用 headless.py 的 _ip_request_times 语义,改为类封装便于测试
class _IPRateLimiter:
    def __init__(self, per_min: int, window: float = 60.0) -> None:
        self.per_min = per_min
        self.window = window
        self._times: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def is_limited(self, client_ip: str) -> bool:
        if self.per_min <= 0:
            return False  # 0 = 禁用
        now = time.time()
        with self._lock:
            dq = self._times[client_ip]
            # 滑出过期
            while dq and now - dq[0] > self.window:
                dq.popleft()
            if len(dq) >= self.per_min:
                return True
            dq.append(now)
            return False


# 进程内全局实例;设置在 app.py lifespan 时初始化(知道 settings.ip_rate_limit_per_min)
_ip_limiter: Optional[_IPRateLimiter] = None


def init_ip_limiter(per_min: int) -> None:
    global _ip_limiter
    _ip_limiter = _IPRateLimiter(per_min=per_min)


def _client_ip(request: Request) -> str:
    """提取客户端 IP,优先信任 X-Forwarded-For 首段(反代后场景)。"""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",", 1)[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def require_rate_limit(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """IP 限流依赖。超限返回 429。

    注:默认对写操作(/api/analyze, /api/agent/chat, /api/models/download)
    生效,轻量 GET 不挂此依赖。
    """
    if _ip_limiter is None:
        init_ip_limiter(settings.ip_rate_limit_per_min)
    ip = _client_ip(request)
    if _ip_limiter.is_limited(ip):
        log.warning(f"rate limited: {ip} 超过 {_ip_limiter.per_min} req/min")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={
                "Connection": "close",
                "Retry-After": "10",
            },
            detail={"error": "rate limit exceeded, retry later"},
        )


# ============================ 路径消毒 ============================


def resolve_within_root(root: Path, raw: str, allowed_exts: frozenset[str]) -> Path:
    """把用户传入的相对路径消毒成 root 内的安全绝对路径(纯函数,无全局状态)。

    防御目标:
      - 路径遍历: ../etc/passwd / ..\\..\\windows\\system32
      - 绝对路径注入: /etc/passwd / C:\\Windows\\system32
      - 非白名单扩展名: .exe / .py / .php 读取

    策略:
      1. PurePosixPath 规范化,过滤掉任何 "." / ".." / 空段
      2. 重组后与 root join,resolve + relative_to 校验最终落在 root 内
      3. 扩展名必须在白名单
    """
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "empty path"},
        )

    parts = [p for p in PurePosixPath(raw).parts if p not in (".", "..", "")]
    if not parts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid path"},
        )

    rel = Path(*parts)
    root = root.resolve()
    full = (root / rel).resolve()

    try:
        full.relative_to(root)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "path escapes content root"},
        )

    if full.suffix.lower() not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"file type not allowed: {full.suffix}"},
        )

    return full


def sanitize_relative_path(raw: str, category: str, allowed_exts: frozenset[str]) -> Path:
    """按已注册类别消毒(用于全局静态挂载,如 web_jobs 根)。"""
    root = _CONTENT_ROOTS.get(category)
    if root is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"unknown content category: {category}"},
        )
    return resolve_within_root(root, raw, allowed_exts)


def sanitize_upload_filename(raw: str) -> str:
    """把 multipart 上传的 filename 消毒成安全基名。

    复用 headless.py 的逻辑:取 basename 防逃逸,扩展名非白名单则改名。
    """
    base = Path(raw).name or "upload.mp4"
    if Path(base).suffix.lower() not in ALLOWED_VIDEO_EXTS:
        base = "upload.mp4"
    return base
