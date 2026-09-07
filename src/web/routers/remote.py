"""远程访问 router(Tailscale/Cloudflare tunnel)。

把 src/remote/ 的 RemoteManager 暴露为 Web API:
  - GET  /api/remote/methods       列可用方案
  - POST /api/remote/tunnel/start 启动 tunnel(Mock,不真实连 Tailscale/CF)
  - POST /api/remote/tunnel/stop   停止 tunnel
  - GET  /api/remote/tunnel/status 查询状态

真实外部服务预算 0:默认 Mock adapter,不真实连 Tailscale/Cloudflare。
真实化需用户授权凭据后替换 adapter 实现。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.remote import RemoteConfig, RemoteManager, RemoteMethod
from ..security import require_auth, require_rate_limit

log = logging.getLogger("web.routers.remote")

router = APIRouter(prefix="/api/remote", tags=["remote-access"])


class TunnelStartRequest(BaseModel):
    """启动 tunnel 请求。

    method: tailscale-serve / tailscale-direct / cloudflare-access
    target_url: 要暴露的本地后端 URL(默认 http://127.0.0.1:8000)
    """

    method: str = "tailscale-serve"
    target_url: str = "http://127.0.0.1:8000"


_manager: Optional[RemoteManager] = None
_manager_lock = asyncio.Lock()


def _get_manager(method: RemoteMethod) -> RemoteManager:
    """按 method 建一个 RemoteManager(config.method 决定 adapter)。

    Mock 模式:tailscale_command 留空时 adapter 会标 UNAVAILABLE。
    给 Mock 一个非空 command 让 prepare/start 走通(返回 mock URL)。
    """
    global _manager
    config = RemoteConfig(
        enabled=True,
        method=method,
        tailscale_command="tailscale",  # Mock 占位,adapter 不真实调用 CLI
        tailscale_ip_address="100.64.0.1",
        cloudflared_command="cloudflared",
        cloudflare_hostname="hermes.example.com",
        cloudflare_team_domain="tingfeng.cloudflareaccess.com",
    )
    _manager = RemoteManager(config)
    return _manager


@router.get("/methods", dependencies=[Depends(require_auth)])
async def list_methods() -> dict[str, Any]:
    """列可用 tunnel 方案。"""
    return {
        "methods": [m.value for m in RemoteMethod],
        "note": "Mock adapter,不真实连 Tailscale/CF;真实化需授权凭据",
    }


@router.post("/tunnel/start",
             dependencies=[Depends(require_auth), Depends(require_rate_limit)])
async def start_tunnel(req: TunnelStartRequest) -> dict[str, Any]:
    async with _manager_lock:
        try:
            try:
                method = RemoteMethod(req.method)
            except ValueError:
                return {"status": "failed", "error": f"unknown method: {req.method}"}
            mgr = _get_manager(method)
            # RemoteManager 真实流程:select_adapter()→prepare()→start(target_url)
            # prepare/start 是 async coroutine(不是同步函数),直接 await
            mgr.select_adapter()
            prepare_info = await mgr.prepare()
            public_url = await mgr.start(req.target_url)
            return {
                "status": "started",
                "method": req.method,
                "target_url": req.target_url,
                "public_url": public_url,
                "prepare": prepare_info,
                "describe": mgr.describe(),
                "note": "Mock,public_url 非真实;真实化需 Tailscale/CF 凭据",
            }
        except Exception as e:
            log.warning(f"tunnel 启动失败: {e}")
            return {"status": "failed", "error": str(e)}


@router.post("/tunnel/stop",
             dependencies=[Depends(require_auth), Depends(require_rate_limit)])
async def stop_tunnel() -> dict[str, Any]:
    async with _manager_lock:
        if _manager is None:
            return {"status": "not_running"}
        try:
            await _manager.stop()
            return {"status": "stopped"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}


@router.get("/tunnel/status", dependencies=[Depends(require_auth)])
async def tunnel_status() -> dict[str, Any]:
    if _manager is None:
        return {"status": "not_running"}
    try:
        st = await _manager.status()
        # TunnelStatus 是 frozen dataclass(非 pydantic),手动转 dict
        status_dict = {
            "state": st.state.value,
            "url": st.url,
            "error": st.error,
            "method": st.method,
            "transport": st.transport,
            "is_running": st.is_running(),
            "metadata": st.metadata,
        }
        return {
            "status": status_dict,
            "describe": _manager.describe(),
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}
