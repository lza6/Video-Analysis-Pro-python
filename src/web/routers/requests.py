"""LLM 请求日志路由 (F7)。

  GET  /api/requests          → 请求历史(分页 + provider 过滤 + since 起始时刻)
  GET  /api/requests/stats    → token 统计(按 provider 聚合)
  DELETE /api/requests        → 清空请求日志(重置)

复用 src.core.request_log.RequestLogStore 单例(get_store)。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from ..security import require_auth
from ...core.request_log import get_store

log = logging.getLogger("web.requests")

router = APIRouter(prefix="/api/requests", tags=["requests"])


@router.get("", dependencies=[Depends(require_auth)])
def list_requests(
    limit: int = Query(100, ge=1, le=1000, description="最多返回 N 条"),
    provider: str | None = Query(None, description="过滤指定 provider"),
    since: str | None = Query(
        None, description="ISO8601 起始时刻(含),只返回该时刻之后的记录"),
) -> dict:
    """请求历史(按时间倒序)。"""
    try:
        store = get_store()
        items = store.list_logs(limit=limit, provider=provider, since=since)
        return {"requests": items, "count": len(items)}
    except Exception as e:  # noqa: BLE001
        log.warning(f"list requests failed: {e}")
        return {"requests": [], "count": 0, "error": str(e)}


@router.get("/stats", dependencies=[Depends(require_auth)])
def token_stats(
    provider: str | None = Query(None, description="只统计指定 provider"),
    since: str | None = Query(
        None, description="ISO8601 起始时刻(含),只统计该时刻之后的记录"),
) -> dict:
    """token 用量统计(按 provider 聚合)。

    返回 {providers: {provider: {requests, prompt_tokens,
        completion_tokens, total_tokens, avg_latency_ms, error_count}}}。
    """
    try:
        store = get_store()
        stats = store.token_stats(provider=provider, since=since)
        return {"providers": stats}
    except Exception as e:  # noqa: BLE001
        log.warning(f"token stats failed: {e}")
        return {"providers": {}, "error": str(e)}


@router.delete("", dependencies=[Depends(require_auth)])
def clear_requests() -> dict:
    """清空所有请求日志。返回删除的行数。"""
    try:
        store = get_store()
        count = store.clear_all()
        return {"ok": True, "deleted": count}
    except Exception as e:  # noqa: BLE001
        log.warning(f"clear requests failed: {e}")
        return {"ok": False, "error": str(e)}
