"""多 Provider 预设路由 (cc-switch 风格)。

  GET    /api/providers            → 列所有 preset (api_key 不回传,只 has_key)
  POST   /api/providers            → 新增 preset (api_key 走 keyring)
  GET    /api/providers/active     → 当前活跃 preset
  GET    /api/providers/{id}       → 单个 preset 详情
  PUT    /api/providers/{id}       → 更新 preset (api_key 可选,给则更新 keyring)
  DELETE /api/providers/{id}       → 删除 preset (含 keyring 清理)
  POST   /api/providers/{id}/activate → 设为活跃
  POST   /api/providers/{id}/rotate  → 轮换 api_key (写新值,返回 fingerprint)
  POST   /api/providers/{id}/revoke  → 吊销 api_key (keyring 值追加 revoked 后缀)

凭据安全:api_key 永不回传明文 (GET 只返回 has_key 布尔);
POST/PUT/rotate 接收明文后立即走 keyring,不落 DB;
rotate 只回传 fingerprint (sha256 前 8 位),绝不回传明文 key。
"""
from __future__ import annotations

import hashlib
import logging
import threading

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..security import require_auth

log = logging.getLogger("web.providers")

router = APIRouter(prefix="/api/providers", tags=["providers"])

# 进程级单例 store (线程安全)。SQLite 短连接 + WAL,多线程读写安全。
_store = None
_lock = threading.Lock()


def _get_store():
    """惰性构造单例 ProviderPresetStore (避免 import 时建 DB)。"""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                from src.core.provider_preset import ProviderPresetStore
                _store = ProviderPresetStore()
    return _store


def _preset_to_public(p) -> dict:
    """preset → 公开 dict (剥离 api_key_ref / 明文 key,只暴露 has_key)。

    返回字段:id/name/provider_type/base_url/model/enabled/created_at/
    is_active/has_key。api_key_ref 与明文 key 永不出现。
    """
    s = _get_store()
    return {
        "id": p.id,
        "name": p.name,
        "provider_type": p.provider_type,
        "base_url": p.base_url,
        "model": p.model,
        "enabled": p.enabled,
        "created_at": p.created_at,
        "is_active": p.is_active,
        "has_key": s.has_api_key(p.id),
    }


# ============================ Pydantic 模型 ============================

class PresetIn(BaseModel):
    """新增 preset 输入。api_key 可选 (Ollama 本地不需)。"""
    name: str = Field(..., min_length=1, max_length=64)
    provider_type: str = Field(
        "custom", description="openai/anthropic/gemini/ollama/custom"
    )
    base_url: str = ""
    model: str = ""
    api_key: str = Field("", description="明文 key,仅写入用,不回传")
    enabled: bool = True


class PresetUpdate(BaseModel):
    """更新 preset 输入。所有字段可选 (部分更新)。api_key 给非空才更新 keyring。"""
    name: str | None = None
    provider_type: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = Field(
        None, description="给非空则更新 keyring;None/空 不动"
    )
    enabled: bool | None = None


class RotateRequest(BaseModel):
    """轮换 api_key 输入。new_key 至少 8 字符防弱 key。

    明文 key 仅用于写入 keyring,响应中绝不回传 (只回 fingerprint)。
    """
    new_key: str = Field(..., min_length=8, description="新 api_key 明文,至少 8 字符")


# ============================ 路由 ============================

@router.get("", dependencies=[Depends(require_auth)])
def list_providers() -> list[dict]:
    """列所有 preset (按 created_at 升序)。api_key 不回传。"""
    s = _get_store()
    return [_preset_to_public(p) for p in s.list_presets()]


@router.post("", dependencies=[Depends(require_auth)])
def create_provider(req: PresetIn) -> dict:
    """新增 preset。api_key 走 keyring,DB 不落明文。"""
    s = _get_store()
    from src.core.provider_preset import ProviderPreset
    p = ProviderPreset(
        id="",
        name=req.name,
        provider_type=req.provider_type,
        base_url=req.base_url,
        model=req.model,
        api_key_ref="",
        enabled=req.enabled,
    )
    try:
        created = s.add_preset(p)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    # api_key 给非空才存 keyring (Ollama 本地可不给)
    if req.api_key:
        if not s.set_api_key(created.id, req.api_key):
            log.warning(f"keyring 写入失败,api_key 未持久化 (preset={created.id})")
    return _preset_to_public(created)


@router.get("/active", dependencies=[Depends(require_auth)])
def get_active_provider() -> dict:
    """当前活跃 preset。无活跃返回 is_active=False 的空壳。"""
    s = _get_store()
    p = s.get_active()
    if p is None:
        return {"id": None, "name": "", "is_active": False, "has_key": False}
    return _preset_to_public(p)


@router.get("/{preset_id}", dependencies=[Depends(require_auth)])
def get_provider(preset_id: str) -> dict:
    """单个 preset 详情。api_key 不回传。"""
    s = _get_store()
    p = s.get_preset(preset_id)
    if p is None:
        raise HTTPException(status_code=404, detail="preset not found")
    return _preset_to_public(p)


@router.put("/{preset_id}", dependencies=[Depends(require_auth)])
def update_provider(preset_id: str, req: PresetUpdate) -> dict:
    """更新 preset。api_key 给非空则更新 keyring。"""
    s = _get_store()
    from src.core.provider_preset import ProviderPreset
    p = ProviderPreset(
        id=preset_id,
        name=req.name,
        provider_type=req.provider_type,
        base_url=req.base_url,
        model=req.model,
        enabled=req.enabled,
    )
    try:
        updated = s.update_preset(preset_id, p)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail="preset not found")
    if req.api_key:
        if not s.set_api_key(preset_id, req.api_key):
            log.warning(f"keyring 写入失败,api_key 未更新 (preset={preset_id})")
    return _preset_to_public(updated)


@router.delete("/{preset_id}", dependencies=[Depends(require_auth)])
def delete_provider(preset_id: str) -> dict:
    """删 preset + best-effort 清 keyring 中的 key。"""
    s = _get_store()
    if not s.delete_preset(preset_id):
        raise HTTPException(status_code=404, detail="preset not found")
    return {"ok": True, "id": preset_id}


@router.post("/{preset_id}/activate", dependencies=[Depends(require_auth)])
def activate_provider(preset_id: str) -> dict:
    """设为活跃 (同时只一个)。"""
    s = _get_store()
    p = s.set_active(preset_id)
    if p is None:
        raise HTTPException(status_code=404, detail="preset not found")
    return _preset_to_public(p)


# ============================ 凭据轮换 / 吊销 ============================
# KeyRotator / CredentialAudit / CredentialResolver 惰性构造 (避免 import 时建 DB),
# 进程内复用单例 (rotation 操作频率低,不必每次新建)。审计 DB 落
# config/cred_audit.db (已 gitignore)。KeyRotator 通过 _parse_key 拆
# "provider:key_name",而 preset 的 api_key_ref 是 "vap_provider:{id}"
# 完整字符串,直接作为 key_name 传入会被拆成 (provider="vap_provider",
# kn="{id}"),与 set_api_key 写入 keyring 时的 service/username 一致
# (KeyringCredentialSource 用 f"{provider}:{key_name}" 作 keyring username,
# 即 "vap_provider:{id}"),故 rotate/revoke 与 set_api_key 操作同一 keyring 条目。

_rotator = None
_audit = None
_resolver = None
_rotator_lock = threading.Lock()


def _get_rotator():
    """惰性构造单例 KeyRotator + CredentialAudit。

    CredentialAudit 落 config/cred_audit.db;KeyRotator 持有 audit。
    CredentialResolver 由 _get_resolver() 单独构造。
    """
    global _audit, _rotator
    if _rotator is None:
        with _rotator_lock:
            if _rotator is None:
                from src.core.credentials.audit import CredentialAudit
                from src.core.credentials.rotation import KeyRotator
                _audit = CredentialAudit()
                _rotator = KeyRotator(audit=_audit)
    return _rotator


def _get_resolver():
    """惰性构造单例 CredentialResolver (env > keyring > ini 三源分层)。"""
    global _resolver
    if _resolver is None:
        with _rotator_lock:
            if _resolver is None:
                from src.core.credentials.key import CredentialResolver
                _resolver = CredentialResolver.default()
    return _resolver


def _preset_key_name(preset_id: str) -> str:
    """preset 在 keyring 中的 key 名 (与 _key_ref 一致, vap_provider:{id})。

    rotation._parse_key 会按 ":" 拆成 (provider="vap_provider", kn="{id}"),
    KeyringCredentialSource 用 f"{provider}:{key_name}" 作 keyring username
    查找,即 "vap_provider:{id}",与 set_api_key 写入的条目相同。
    """
    return f"vap_provider:{preset_id}"


def _fingerprint(plaintext: str) -> str:
    """新 key 的 sha256 前 8 位 (用于响应中代替明文回显,绝不回传明文)。"""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()[:8]


@router.post("/{preset_id}/rotate", dependencies=[Depends(require_auth)])
def rotate_provider_key(preset_id: str, req: RotateRequest) -> dict:
    """轮换 preset 的 api_key:写新值到 keyring + 记 audit。

    接收明文 new_key (仅写入用),调 KeyRotator.rotate 写入 keyring,
    返回 fingerprint (sha256 前 8 位),绝不回传明文 key。
    preset 不存在返回 404;rotation 异常返回 400。
    """
    s = _get_store()
    if s.get_preset(preset_id) is None:
        raise HTTPException(status_code=404, detail="preset not found")
    resolver = _get_resolver()
    rotator = _get_rotator()
    key_name = _preset_key_name(preset_id)
    try:
        result = rotator.rotate(
            key_name=key_name, new_value=req.new_key, resolver=resolver,
        )
    except Exception as e:
        log.warning(f"rotate 失败 (preset={preset_id}): {e}")
        raise HTTPException(status_code=400, detail=f"rotate failed: {e}")
    if not result.new_key_set:
        raise HTTPException(
            status_code=400,
            detail="rotate failed: new key not persisted (keyring unavailable)",
        )
    return {
        "ok": True,
        "key_name": key_name,
        "old_revoked": result.old_revoked,
        "new_key_set": result.new_key_set,
        "fingerprint": _fingerprint(req.new_key),
    }


@router.post("/{preset_id}/revoke", dependencies=[Depends(require_auth)])
def revoke_provider_key(preset_id: str) -> dict:
    """吊销 preset 的 api_key:在 keyring 值后追加 _revoked_<ts> 后缀。

    无 body。调 KeyRotator.revoke 标记旧 key revoked。
    幂等:已吊销再吊销仍返回 200 (ok=True 表示标记成功或已处于吊销态)。
    preset 不存在返回 404。
    """
    s = _get_store()
    if s.get_preset(preset_id) is None:
        raise HTTPException(status_code=404, detail="preset not found")
    resolver = _get_resolver()
    rotator = _get_rotator()
    key_name = _preset_key_name(preset_id)
    try:
        ok = rotator.revoke(key_name=key_name, resolver=resolver)
    except Exception as e:
        log.warning(f"revoke 失败 (preset={preset_id}): {e}")
        raise HTTPException(status_code=400, detail=f"revoke failed: {e}")
    # 幂等:ok=True 表示本次成功追加后缀;ok=False 可能是已吊销或值不存在。
    # 对已吊销再吊销,视为幂等成功 (返回 ok=True, revoked=True)。
    try:
        already = rotator.is_revoked(key_name, resolver)
    except Exception:
        already = False
    return {
        "ok": True,
        "key_name": key_name,
        "revoked": True if already else ok,
    }
