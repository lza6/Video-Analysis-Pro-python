"""IM 网关 router(微信/TG/Discord 作为 agent 远程入口)。

把 src/core/im_gateway/ 的 IMGateway 暴露为 Web API:
  - POST /api/im/gateway/start   启动网关(Mock adapter,不真实连 IM)
  - POST /api/im/gateway/stop    停止网关
  - GET  /api/im/gateway/status   查询状态
  - GET  /api/im/gateway/messages 查 mailbox 消息
  - POST /api/im/gateway/process  单轮驱动(Mock 闭环测试用,不真实连 IM)

真实 IM 凭据预算 0:默认 Mock adapter,不真实连微信/TG/Discord。
真实化需用户授权凭据后替换 adapter 实现。
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.core.im_gateway import (
    EchoAgentCallback,
    IMGateway,
    IMMailbox,
    IMMessage,
)
from src.core.im_gateway.adapters import (
    DiscordAdapter,
    TelegramAdapter,
    WechatAdapter,
)
from ..security import require_auth, require_rate_limit

log = logging.getLogger("web.routers.im_gateway")

router = APIRouter(prefix="/api/im", tags=["im-gateway"])


class GatewayStartRequest(BaseModel):
    """启动网关请求。

    adapters: 启用哪些 IM 渠道(list,默认 ["wechat"])。
    master_key: AES 主密钥(可选,不传则不加密存盘)。
    db_path: mailbox SQLite 路径(可选,默认 config/im_gateway.db)。
    """

    adapters: list[str] = ["wechat"]
    master_key: Optional[str] = None
    db_path: Optional[str] = None


class CallbackSetRequest(BaseModel):
    """设置 AgentCallback(生产接真 agent)。当前仅支持 echo。"""

    mode: str = "echo"


class InjectInboundRequest(BaseModel):
    """Mock 注入入站消息(Mock adapter 闭环测试用)。"""

    channel: str = "wechat"
    peer: str = "tester"
    content: str = ""


# 进程级单例(本地工具单进程够用)
_gateway: Optional[IMGateway] = None
_gateway_lock = asyncio.Lock()


_ADAPTER_FACTORIES = {
    "wechat": lambda: WechatAdapter(),
    "telegram": lambda: TelegramAdapter(),
    "discord": lambda: DiscordAdapter(),
}


def _get_or_create_gateway(req: GatewayStartRequest) -> IMGateway:
    global _gateway
    if _gateway is None:
        # IMMailbox 用 config_dir + db_filename(非 db_path)——修复 2026-09-08
        if req.db_path:
            db = Path(req.db_path)
            mailbox = IMMailbox(config_dir=str(db.parent), db_filename=db.name)
        else:
            mailbox = IMMailbox()
        adapter_objs = []
        for name in req.adapters:
            factory = _ADAPTER_FACTORIES.get(name)
            if factory is None:
                raise ValueError(f"unknown adapter: {name}")
            adapter_objs.append(factory())
        if not adapter_objs:
            adapter_objs = [WechatAdapter()]
        _gateway = IMGateway(
            mailbox=mailbox,
            adapters=adapter_objs,
            callback=EchoAgentCallback(),
        )
    return _gateway


@router.post("/gateway/start",
             dependencies=[Depends(require_auth), Depends(require_rate_limit)])
async def start_gateway(req: GatewayStartRequest) -> dict[str, Any]:
    """启动 IM 网关(Mock adapter,不真实连 IM)。

    默认 EchoAgentCallback(回显),生产由主控在 lifespan 换成 ReactLoopAgent。
    """
    async with _gateway_lock:
        try:
            gw = _get_or_create_gateway(req)
            await gw.start()
            return {
                "status": "started",
                "adapters": req.adapters,
                "note": "Mock adapter,不真实连 IM;真实化需授权凭据",
            }
        except Exception as e:
            log.warning(f"IM 网关启动失败: {e}")
            return {"status": "failed", "error": str(e)}


@router.post("/gateway/stop",
             dependencies=[Depends(require_auth), Depends(require_rate_limit)])
async def stop_gateway() -> dict[str, Any]:
    async with _gateway_lock:
        if _gateway is None:
            return {"status": "not_running"}
        try:
            await _gateway.stop()
            return {"status": "stopped"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}


@router.get("/gateway/status", dependencies=[Depends(require_auth)])
async def gateway_status() -> dict[str, Any]:
    if _gateway is None:
        return {"status": "not_running"}
    running = _gateway._task is not None and not _gateway._task.done()
    return {
        "status": "running" if running else "stopped",
        "adapters": list(_gateway._adapters.keys()),
        "processed": _gateway.processed_count,
        "failed": _gateway.failed_count,
    }


@router.get("/gateway/messages", dependencies=[Depends(require_auth)])
async def list_messages(limit: int = 50) -> dict[str, Any]:
    """查 mailbox pending 消息(lease/ack 投递语义)。"""
    if _gateway is None:
        return {"messages": []}
    try:
        msgs = _gateway.mailbox.list_pending(limit=limit)
        return {"messages": [_msg_to_dict(m) for m in msgs]}
    except Exception as e:
        return {"messages": [], "error": str(e)}


@router.post("/gateway/callback",
             dependencies=[Depends(require_auth), Depends(require_rate_limit)])
async def set_callback(req: CallbackSetRequest) -> dict[str, Any]:
    """设置 AgentCallback(当前仅 echo;真 agent 接入由 lifespan 注入)。"""
    if _gateway is None:
        return {"status": "failed", "error": "gateway not started"}
    if req.mode == "echo":
        _gateway.register_callback(EchoAgentCallback())
        return {"status": "ok", "mode": "echo"}
    return {"status": "failed", "error": f"unknown mode: {req.mode}"}


@router.post("/gateway/inject",
             dependencies=[Depends(require_auth), Depends(require_rate_limit)])
async def inject_inbound(req: InjectInboundRequest) -> dict[str, Any]:
    """Mock 注入入站消息(Mock adapter 闭环测试用,不真实连 IM)。"""
    if _gateway is None:
        return {"status": "failed", "error": "gateway not started"}
    try:
        # mailbox.put_inbound 接关键字参数,内部建 IMMessage(msg_id 自动生成)
        msg_id = _gateway.mailbox.put_inbound(
            channel=req.channel, peer=req.peer, content=req.content
        )
        return {"status": "ok", "msg_id": msg_id}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@router.post("/gateway/process",
             dependencies=[Depends(require_auth), Depends(require_rate_limit)])
async def process_once() -> dict[str, Any]:
    """单轮驱动 gateway(Mock 闭环测试,不真实连 IM)。"""
    if _gateway is None:
        return {"status": "failed", "error": "gateway not started"}
    try:
        n = await _gateway.process_once()
        return {"status": "ok", "processed": n}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def _msg_to_dict(m: Any) -> dict[str, Any]:
    """把 IMMessage 转前端 dict。"""
    if isinstance(m, IMMessage):
        return {
            "msg_id": m.msg_id,
            "channel": m.channel,
            "peer": m.peer,
            "content": m.content,
            "status": m.status,
            "created_at": m.created_at,
        }
    if isinstance(m, dict):
        return m
    return {"raw": str(m)}
