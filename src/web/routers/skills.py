"""Skills 管理路由。

  GET    /api/skills            → 已加载 skills 列表(含启用状态)
  POST   /api/skills/{name}/toggle → 启用/禁用切换(写 config/skills_state.json)
  POST   /api/skills/generate    → skill_generator 自动生成(返回生成日志)
  GET    /api/skills/state       → 全部启用状态 map
  POST   /api/skills/ratchet     → 对指定 skill 重跑评分棘轮(new/improved/rejected)
  POST   /api/skills/distill     → 对 ExperienceStore 聚合 >=3 次 intent 蒸馏草稿

复用 src.skills(load_skills / set_enabled_state) + src/core/skill_generator。
v10.2 (B-FIN-2 MAJOR-4):ratchet / distill 两个端点把 src/skills/scoring 与
distiller 从「模块层」接到 HTTP 生产入口,skills 闭环有真实调用方。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

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


# ---------------------------------------------------------------------------
# v10.2 (B-FIN-2 MAJOR-4):skills 闭环生产入口 —— 评分棘轮 + 经验蒸馏
# ---------------------------------------------------------------------------


class RatchetReq(BaseModel):
    """棘轮输入:skill_name 必须是已加载 skill 的 name(防止任意路径)."""

    skill_name: str


@router.post("/ratchet", dependencies=[Depends(require_auth)])
def ratchet_skill(req: RatchetReq) -> dict:
    """对指定 skill 重跑 scoring.apply_ratchet(棘轮:只留改进,回滚退步)。

    返回 {"ok": True, "skill", "score", "ratchet": "new"|"improved"|"rejected"}。
    skill_name 必须命中已加载 skill(返回其真实路径),未命中返回 404 语义的
    ok=False。纯规则无需付费 LLM。
    """
    try:
        import src.skills.scoring as scoring

        # 防止路径注入:先在已加载 skills 里按 name 定位真实 skill 目录
        name = req.skill_name.strip()
        if not name:
            return {"ok": False, "error": "skill_name 不能为空"}
        target: Optional[Path] = None
        for s in _load_skills():
            if s.name == name:
                p = Path(str(getattr(s, "path", "")))
                if p.is_file():
                    target = p.parent
                break
        if target is None:
            return {"ok": False, "error": f"skill {name!r} 未加载", "skill": name}
        result = scoring.apply_ratchet(target)
        return {"ok": True, **result}
    except Exception as e:
        log.warning(f"ratchet skill failed: {e}")
        return {"ok": False, "error": str(e)}


class DistillReq(BaseModel):
    """蒸馏输入:intent 选填——填则只看该 intent;空则取全局最稳定 intent。"""

    intent: str = ""
    min_count: int = 3
    skills_dir: str = ""


@router.post("/distill", dependencies=[Depends(require_auth)])
def distill_skill(req: DistillReq) -> dict:
    """从 ExperienceStore 聚合经验蒸馏 skill 草稿(B-FIN-2 MAJOR-4)。

    逻辑:读 ExperienceStore(默认 config/agent_experiences.db)全部经验 →
    DraftPipeline.distill(>=3 次 + 稳定 tool_chain)→ 返回草稿 markdown 与
    intent/occurrences/chain。未达阈值返回 ok=False + reason,不 500。
    纯规则(复用 skill_generator 场景模板),红线:不调付费 LLM。
    """
    try:
        from src.core.agent.experience import ExperienceStore
        from src.skills.distiller import DraftPipeline
        from src.utils.constants import CONFIG_DIR

        # ExperienceStore() 默认 db 路径是相对 'config/agent_experiences.db'
        # (非 CONFIG_DIR 拼接)。先 resolve 到项目根 config 保证语义一致。
        store = ExperienceStore(Path(CONFIG_DIR) / "agent_experiences.db")
        experiences = store.find_all()
        if not experiences:
            return {"ok": False, "reason": "无经验记录(config/agent_experiences.db 为空)"}
        pipeline = DraftPipeline(min_count=max(1, int(req.min_count)))
        skill_dir = Path(req.skills_dir) if req.skills_dir else Path(CONFIG_DIR) / "skills"
        if req.intent.strip():
            # 过滤只看该 intent 的经验(防止其它 intent 抢占最稳定)
            filtered = [e for e in experiences if req.intent.strip() in str(e.intent)]
            if not filtered:
                return {"ok": False, "reason": f"intent {req.intent!r} 无经验记录"}
            experiences = filtered
        draft = pipeline.distill(experiences, skills_dir=skill_dir) if experiences else None
        if draft is None:
            return {"ok": False, "reason": "未达蒸馏阈值(>=3 次 + 稳定 tool_chain)"}
        return {
            "ok": True,
            "intent": draft.intent,
            "occurrences": draft.occurrences,
            "chain": list(draft.chain),
            "skill_name": draft.draft.name,
            "markdown": draft.markdown,
        }
    except Exception as e:
        log.warning(f"distill skill failed: {e}")
        return {"ok": False, "error": str(e)}
