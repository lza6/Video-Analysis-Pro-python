"""Skills 管理路由。

  GET    /api/skills            → 已加载 skills 列表(含启用状态)
  POST   /api/skills/{name}/toggle → 启用/禁用切换(写 config/skills_state.json)
  POST   /api/skills/generate    → skill_generator 自动生成(返回生成日志)
  GET    /api/skills/state       → 全部启用状态 map

复用 src.skills(load_skills / set_enabled_state) + src/core/skill_generator。
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..security import require_auth

log = logging.getLogger("web.skills")

router = APIRouter(prefix="/api/skills", tags=["skills"])


def _load_skills():
    try:
        from src.skills import load_skills  # type: ignore
        return load_skills()
    except Exception as e:
        log.warning(f"load_skills 不可用: {e}")
        return ()


def _skill_to_dict(s) -> dict:
    return {
        "name": getattr(s, "name", str(s)),
        "description": getattr(s, "description", ""),
        "triggers": list(getattr(s, "triggers", ()) or ()),
        "path": str(getattr(s, "path", "")) or "",
        "enabled": bool(getattr(s, "enabled", True)),
    }


@router.get("", dependencies=[Depends(require_auth)])
def list_skills() -> dict:
    skills = _load_skills()
    return {
        "skills": [_skill_to_dict(s) for s in skills],
        "count": len(skills),
    }


class ToggleReq(BaseModel):
    enabled: bool


@router.post("/{name}/toggle", dependencies=[Depends(require_auth)])
def toggle_skill(name: str, req: ToggleReq) -> dict:
    """切换 skill 启用状态。"""
    try:
        from src.skills import set_enabled_state  # type: ignore
        set_enabled_state(name, req.enabled)
        return {"ok": True, "name": name, "enabled": req.enabled}
    except Exception as e:
        log.warning(f"toggle {name} failed: {e}")
        return {"ok": False, "error": str(e)}


class GenerateReq(BaseModel):
    """skill_generator 输入:text 为场景描述文本(parking/face/fire/vehicle)。"""
    text: str = "停车场监控"
    overwrite: bool = False


@router.post("/generate", dependencies=[Depends(require_auth)])
def generate_skill(req: GenerateReq) -> dict:
    """自动生成 skill(skill_generator.generate_skill)。

    真实签名:generate_skill(text, skills_dir: Path, overwrite=False) -> dict
    红线:纯规则模板,不真实调付费 LLM。
    """
    try:
        from src.core.skill_generator import generate_skill as gen
        from src.utils.constants import CONFIG_DIR
        skills_dir = Path(CONFIG_DIR) / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        result = gen(req.text, skills_dir, overwrite=req.overwrite)
        if isinstance(result, dict) and result.get("ok"):
            return {
                "ok": True,
                "scene": result.get("scene", ""),
                "skill_name": result.get("skill_name", ""),
                "path": result.get("path", ""),
            }
        return {"ok": False, "error": str(result.get("error", "生成失败")) if isinstance(result, dict) else str(result)}
    except Exception as e:
        log.warning(f"generate skill failed: {e}")
        return {"ok": False, "error": str(e)}
