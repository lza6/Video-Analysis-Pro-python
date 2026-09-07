"""Web 后端配置(pydantic-settings)。

复用原 headless.py 的环境变量约定(VAP_PORT / VAP_HEADLESS_TOKEN /
VAP_MAX_UPLOAD_MB / VAP_ANALYZE_CONCURRENCY / VAP_IP_RATE_LIMIT_PER_MIN),
新增 VAP_WEB_CORS_ORIGINS 控制 CORS(本地默认允许 Next.js dev :3000)。

所有配置集中此处,路由通过 Depends(get_settings) 注入,便于测试替换。
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行时配置。优先级:环境变量 > .env > 默认值。

    与 src/server/headless.py 保持同一变量名,迁移用户零配置。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VAP_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 服务 ---
    port: int = Field(8000, description="Web 服务端口")
    # 安全默认:127.0.0.1 仅 loopback,纯本地桌面/开发免 token。
    # 对外暴露(0.0.0.0 / 局域网 IP)时 serve.py 守卫强制要求 VAP_HEADLESS_TOKEN,
    # 防"监听所有网卡 + 鉴权关闭"并存(用户日志实证的安全洞)。
    host: str = Field("127.0.0.1", description="监听地址;对外暴露需配 token")

    # --- 鉴权(可选)---
    # 空 = 鉴权关闭(仅本地用安全);非空 = Bearer Token,建议 >=32 字符随机串
    headless_token: str = Field("", description="可选 Bearer Token;空=禁用")
    # 弱 Token 警告阈值(不阻断,只日志)
    min_token_length: int = 16

    # --- 上传/并发/限流 ---
    max_upload_mb: int = 512
    analyze_concurrency: int = 1
    ip_rate_limit_per_min: int = 10

    # --- CORS ---
    # 本地开发:Next.js dev :3000 需跨域调 :8000;生产同源不需 CORS
    web_cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # --- Whisper/Ollama(透传到 src/core)---
    ollama_num_ctx: int = 4096


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例 Settings,lru_cache 保证进程内只构造一次。"""
    return Settings()


def warn_on_weak_token() -> None:
    """启动时探测弱 Token(不阻断),复用 headless.py 的告警语义。"""
    import logging
    log = logging.getLogger("web.config")
    s = get_settings()
    if not s.headless_token:
        log.warning(
            "Web auth: DISABLED (VAP_HEADLESS_TOKEN 未配置,/api/** 任何人可调,"
            "仅本地用安全)"
        )
        return
    if len(s.headless_token) < s.min_token_length:
        log.warning(
            f"⚠️ VAP_HEADLESS_TOKEN 长度 {len(s.headless_token)} < {s.min_token_length},"
            f"强度不足,建议用 >=32 字符随机串(可用 "
            f"python -c \"import secrets;print(secrets.token_urlsafe(32))\" 生成)"
        )
    else:
        log.info(f"Web auth: enabled (token length={len(s.headless_token)})")
