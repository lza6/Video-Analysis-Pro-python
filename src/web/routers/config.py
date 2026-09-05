"""配置路由(Provider / API 预设 / 提示词模板)。

  GET  /api/config              → 读 LastUsed(provider/api_url/model,api_key 不回传)
  PUT  /api/config              → 更新 provider 配置(api_key 走 keyring)
  GET  /api/config/presets      → API 预设列表
  POST /api/config/presets      → 新增/更新预设
  DELETE /api/config/presets/{id} → 删除预设
  GET  /api/config/prompts     → 提示词模板列表
  PUT  /api/config/prompts      → 更新模板
  POST /api/config/test        → 测 Provider 活性(不真实调付费 API,只 list_models)

凭据安全:api_key 优先 keyring,降级 ini(并告警)。不回传明文 key。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..deps import get_config_manager
from ..security import require_auth

log = logging.getLogger("web.config")

router = APIRouter(prefix="/api/config", tags=["config"])


class ProviderConfig(BaseModel):
    client_type: int = 1
    api_url: str = ""
    api_key: str = ""  # 仅 PUT 时接收;GET 不回传
    model_name: str = ""


@router.get("", dependencies=[Depends(require_auth)])
def get_config() -> dict:
    """读配置。api_key 永不回传明文(只回传 has_key 布尔)。"""
    cm = get_config_manager()
    cfg = cm.config
    has_key = False
    # keyring 优先:即便 ini 是标记位,也按真实 keyring 是否有值回传 has_key
    try:
        from src.utils.config_manager import _secure_get
        has_key = bool(_secure_get("api_key", ""))
    except Exception:
        pass
    # NVIDIA 多 key 路由(.env VAP_NV_API_KEYS):provider=nvidia 时
    # has_key 以 env 有值为准;暴露 key 数量(非 key 本身)
    nv_keys = _load_nv_keys_from_env()
    return {
        "client_type": int(_get(cfg, "LastUsed", "client_type", "1") or "1"),
        "api_url": _get(cfg, "LastUsed", "api_url", ""),
        "model_name": _get(cfg, "LastUsed", "model_name", ""),
        "has_key": has_key or bool(nv_keys),
        "keyring_available": _keyring_ok(),
        "nvidia_keys": len(nv_keys),
    }


@router.put("", dependencies=[Depends(require_auth)])
def update_config(cfg_in: ProviderConfig) -> dict:
    """更新 provider 配置。api_key 走 keyring(失败降级 ini + 告警)。"""
    cm = get_config_manager()
    cm.update_config("LastUsed", "client_type", str(cfg_in.client_type))
    cm.update_config("LastUsed", "api_url", cfg_in.api_url)
    cm.update_config("LastUsed", "model_name", cfg_in.model_name)
    if cfg_in.api_key:
        try:
            from src.utils.config_manager import _secure_set
            _secure_set("api_key", cfg_in.api_key)
            # ini 里只留标记位,不落明文
            cm.update_config("LastUsed", "api_key", "__keyring__")
        except Exception as e:
            log.warning(f"keyring 写入失败,降级明文存 ini: {e}")
            cm.update_config("LastUsed", "api_key", cfg_in.api_key)
    return {"ok": True, "has_key": bool(cfg_in.api_key)}


class TestProviderRequest(BaseModel):
    api_url: str = ""
    api_key: str = ""
    model: str = ""
    provider: str = Field("", description="nvidia 走 ProviderRouter 多 key 路由;空=普通单 key")


@router.post("/test", dependencies=[Depends(require_auth)])
def test_provider(req: TestProviderRequest) -> dict:
    """测 Provider 活性:nvidia 走多 key 路由探活;其他走 list_models。

    红线:真实付费 chat 调用预算为 0。list_models / nvidia 的 models 端点
    不计费或极低,是探测连通性的标准做法。
    """
    # nvidia 多 key 路由:.env 已配 VAP_NV_API_KEYS 时优先走它
    nv_keys = _load_nv_keys_from_env()
    if req.provider == "nvidia" or (not req.provider and nv_keys):
        return _test_nvidia_router(nv_keys, req.model)

    if not req.api_url or not req.api_key:
        raise HTTPException(status_code=400, detail={"error": "api_url 和 api_key 必填"})
    try:
        from src.core.logic import APIGatewayClient
        client = APIGatewayClient(req.api_key, req.api_url)
        models = client.list_models()
        return {"ok": True, "models": models, "count": len(models), "via": "list_models"}
    except Exception as e:
        return {"ok": False, "error": str(e), "models": [], "count": 0}


def _test_nvidia_router(nv_keys: list, model: str) -> dict:
    """用 ProviderRouter 探活:发一次 models 端点请求,不真实 chat。"""
    if not nv_keys:
        return {"ok": False, "error": "无 nvidia key(配 .env VAP_NV_API_KEYS)", "via": "router"}
    try:
        import requests
        from src.core.provider_router import ProviderRouter, load_router_config_from_env
        cfg = load_router_config_from_env()
        router = ProviderRouter(nv_keys, **cfg)
        key = router.select_key(provider="nvidia")
        if key is None:
            return {"ok": False, "error": "无可用 nvidia key(全部失效/backoff)", "via": "router"}
        # 探活:GET models 端点(只读,不计费)
        s = requests.Session()
        s.trust_env = False
        url = f"{key.base_url.rstrip('/')}/models"
        r = s.get(url, headers={"Authorization": f"Bearer {key.api_key}"}, timeout=10)
        router.record_result(key.id, r.status_code)
        if r.ok:
            data = r.json()
            models = [m.get("id") for m in data.get("data", [])] if isinstance(data, dict) else []
            return {
                "ok": True,
                "models": models,
                "count": len(models),
                "key_used": key.name,
                "total_keys": len(nv_keys),
                "via": "router",
            }
        return {"ok": False, "error": f"HTTP {r.status_code}", "key_used": key.name, "via": "router"}
    except Exception as e:
        return {"ok": False, "error": str(e), "via": "router"}


# ============================ 预设 ============================

@router.get("/presets", dependencies=[Depends(require_auth)])
def list_presets() -> list:
    return get_config_manager().load_api_presets()


class PresetIn(BaseModel):
    name: str
    api_url: str
    model: str = ""
    notes: str = ""


@router.post("/presets", dependencies=[Depends(require_auth)])
def save_preset(p: PresetIn) -> dict:
    cm = get_config_manager()
    presets = cm.load_api_presets()
    # upsert by name
    presets = [x for x in presets if x.get("name") != p.name]
    presets.append(p.model_dump())
    cm.save_api_presets(presets)
    return {"ok": True, "count": len(presets)}


@router.delete("/presets/{name}", dependencies=[Depends(require_auth)])
def delete_preset(name: str) -> dict:
    cm = get_config_manager()
    presets = [x for x in cm.load_api_presets() if x.get("name") != name]
    cm.save_api_presets(presets)
    return {"ok": True, "count": len(presets)}


# ============================ 提示词模板 ============================

@router.get("/prompts", dependencies=[Depends(require_auth)])
def list_prompts() -> list:
    return get_config_manager().load_prompts()


class PromptIn(BaseModel):
    name: str
    content: str


@router.put("/prompts", dependencies=[Depends(require_auth)])
def save_prompt(p: PromptIn) -> dict:
    cm = get_config_manager()
    prompts = cm.load_prompts()
    prompts = [x for x in prompts if x.get("name") != p.name]
    prompts.append(p.model_dump())
    cm.save_prompts(prompts)
    return {"ok": True, "count": len(prompts)}


# ============================ helpers ============================

def _get(cfg, section: str, key: str, fallback: str = "") -> str:
    try:
        return str(cfg.get(section, key, fallback=fallback) or "")
    except Exception:
        return fallback


def _keyring_ok() -> bool:
    try:
        from src.utils.config_manager import is_keyring_available
        return bool(is_keyring_available())
    except Exception:
        return False


def _load_nv_keys_from_env() -> list:
    """从 .env 读 VAP_NV_API_KEYS(逗号分隔多 key),返回 key 列表。

    ProviderRouter 走 .env 多 key 路由(nvidia),不走 ini/keyring。
    env_path 默认项目根 .env;进程环境变量优先(load_from_env 内部处理)。
    """
    try:
        from src.core.provider_router import load_from_env
        keys = load_from_env()  # 已按进程 env 优先合并
        return [k for k in keys if k.provider == "nvidia"]
    except Exception as e:
        log.debug(f"load nv keys failed: {e}")
        return []
