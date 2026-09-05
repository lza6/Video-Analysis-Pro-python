"""决策日志路由。

  GET    /api/decisions          → 当前内存决策列表
  POST   /api/decisions          → 追加一条 DecisionEntry(Agent run_plan 每步调)
  GET    /api/decisions/export   → 导出全部为 JSON
  DELETE /api/decisions          → 清空(重置单例)

复用 src/core/decision_log.DecisionLog + DecisionEntry + make_entry。
字段映射修正:make_entry 用 step_name/action_type/decision/reason/args_json;
DecisionLog.append 返回新实例(不可变),需接住返回值重置单例。
"""
from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..security import require_auth

log = logging.getLogger("web.decisions")

router = APIRouter(prefix="/api/decisions", tags=["decisions"])

# 进程级单例(线程安全)。DecisionLog 是 frozen 不可变,append 返回新实例,
# 用 _log_instance 持有最新实例。
_log_instance = None
_lock = threading.Lock()


def _get_log():
    global _log_instance
    if _log_instance is None:
        with _lock:
            if _log_instance is None:
                try:
                    from src.core.decision_log import DecisionLog
                    _log_instance = DecisionLog()
                except Exception as e:
                    log.warning(f"DecisionLog 不可用: {e}")
                    _log_instance = DecisionLog() if _can_import() else []
    return _log_instance


def _can_import() -> bool:
    try:
        import importlib
        return importlib.util.find_spec("src.core.decision_log") is not None
    except Exception:
        return False


class DecisionIn(BaseModel):
    step_name: str = ""
    action_type: str = ""
    decision: str = ""
    reason: str = ""
    cause_id: str | None = None
    output_path: str | None = None
    duration_ms: float = 0.0
    status: str = "ok"
    risk: str = "low"
    args_json: str | None = None


@router.get("", dependencies=[Depends(require_auth)])
def list_decisions(limit: int = 200) -> dict:
    dl = _get_log()
    if isinstance(dl, list):
        return {"decisions": list(dl)[-limit:], "available": False}
    try:
        items = list(getattr(dl, "entries", ()) or ())[-limit:]
        return {
            "decisions": [_entry_to_dict(e) for e in items],
            "available": True,
            "count": len(items),
        }
    except Exception as e:
        log.warning(f"list decisions failed: {e}")
        return {"decisions": [], "available": False, "error": str(e)}


@router.post("", dependencies=[Depends(require_auth)])
def append_decision(d: DecisionIn) -> dict:
    """追加一条决策。make_entry 构造 + append 返回新实例(接住重置单例)。"""
    global _log_instance
    dl = _get_log()
    try:
        from src.core.decision_log import make_entry, DecisionLog
        entry = make_entry(
            step_name=d.step_name,
            action_type=d.action_type,
            decision=d.decision,
            reason=d.reason or "(无)",
            cause_id=d.cause_id,
            output_path=d.output_path,
            duration_ms=d.duration_ms,
            status=d.status,
            risk=d.risk,
            args_json=d.args_json,
        )
        with _lock:
            if isinstance(dl, DecisionLog):
                _log_instance = dl.append(entry)
            else:
                dl.append(d.model_dump())
        return {"ok": True}
    except Exception as e:
        log.warning(f"append failed, fallback to dict: {e}")
        if isinstance(dl, list):
            dl.append(d.model_dump())
        return {"ok": True, "fallback": True}


@router.get("/export", dependencies=[Depends(require_auth)])
def export_decisions() -> dict:
    dl = _get_log()
    if isinstance(dl, list):
        return {"decisions": list(dl)}
    try:
        items = list(getattr(dl, "entries", ()) or ())
        return {"decisions": [_entry_to_dict(e) for e in items]}
    except Exception as e:
        return {"decisions": [], "error": str(e)}


@router.delete("", dependencies=[Depends(require_auth)])
def clear_decisions() -> dict:
    """清空(DecisionLog 不可变,重置单例为新空实例)。"""
    global _log_instance
    with _lock:
        if isinstance(_log_instance, list):
            _log_instance = []
        else:
            try:
                from src.core.decision_log import DecisionLog
                _log_instance = DecisionLog()
            except Exception:
                _log_instance = []
    return {"ok": True}


def _entry_to_dict(e) -> dict:
    if isinstance(e, dict):
        return e
    return {
        "id": getattr(e, "id", ""),
        "timestamp": getattr(e, "timestamp", ""),
        "step_name": getattr(e, "step_name", ""),
        "action_type": getattr(e, "action_type", ""),
        "decision": getattr(e, "decision", ""),
        "reason": getattr(e, "reason", ""),
        "cause_id": getattr(e, "cause_id", ""),
        "output_path": getattr(e, "output_path", ""),
        "duration_ms": getattr(e, "duration_ms", 0),
        "status": getattr(e, "status", ""),
        "risk": getattr(e, "risk", ""),
        "args_json": getattr(e, "args_json", ""),
    }
