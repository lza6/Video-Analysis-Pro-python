"""Skill 自动生成器（v7.0 指南 4.3）。

agent 遇到现有 skill 不合用的场景时，自己写一个新 SKILL.md：
1. 分析视频元数据 + 抽样几帧判断场景
2. 选最接近的现有 skill 作为模板
3. 调 LLM 生成新 SKILL.md（描述场景/算法/参数）—— 付费 API 红线：
   不真实调付费 LLM，用规则模板生成（用户给预算后可接 LLM）
4. 用户确认后存 config/skills/<name>/ 下，下次自动匹配

纯逻辑（不依赖真实 LLM 调用），由 agent_orchestrator / skills_manager_tab 调用。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 内置模板：按场景关键词选最近 skill 作为模板
_SCENE_TEMPLATES = {
    "parking": {
        "name": "surveillance-parking-lpr",
        "description": "停车场车牌识别监控，车牌检测+OCR，追踪车辆进出",
        "algorithm": "YOLO 车牌检测 + PaddleOCR 识别 + 时序进出统计",
        "triggers": "停车场,车牌,lpr,停车,车辆",
    },
    "face": {
        "name": "surveillance-face-recognition",
        "description": "人脸识别监控，face detection + embedding 比对",
        "algorithm": "face_detection + face_embedding + 比对已知人脸库",
        "triggers": "人脸,face,识别,人员",
    },
    "fire": {
        "name": "surveillance-fire-detection",
        "description": "火灾烟雾检测监控，火焰/烟雾颜色+运动特征",
        "algorithm": "火焰颜色 HSV 阈值 + 烟雾扩散运动检测",
        "triggers": "火灾,火焰,烟雾,fire,smoke",
    },
    "vehicle": {
        "name": "surveillance-vehicle-counting",
        "description": "车辆计数监控，路口车流统计",
        "algorithm": "YOLO 车辆检测 + 虚拟线计数 + 方向判断",
        "triggers": "车辆,车流,计数,traffic,vehicle",
    },
}


@dataclass
class SkillDraft:
    """agent 生成的新 skill 草稿（用户确认前）。"""
    name: str
    description: str
    triggers: str
    algorithm: str
    parameters: str = ""
    when_to_use: str = ""
    when_not_to_use: str = ""
    fallback: str = "无对应依赖时降级为纯帧差分，不崩"


def detect_scene(text: str) -> Optional[str]:
    """从用户描述识别场景关键词，返回模板 key 或 None。

    纯规则匹配（无 LLM），覆盖停车场/人脸/火灾/车辆四类常见新场景。
    """
    if not text:
        return None
    lower = text.lower()
    if any(k in lower for k in ("停车场", "车牌", "lpr", "停车", "parking")):
        return "parking"
    if any(k in lower for k in ("人脸", "face", "识别人员", "面部")):
        return "face"
    if any(k in lower for k in ("火灾", "火焰", "烟雾", "fire", "smoke", "着火")):
        return "fire"
    if any(k in lower for k in ("车辆计数", "车流", "traffic", "vehicle counting")):
        return "vehicle"
    return None


def draft_skill_from_scene(scene_key: str, user_text: str = "") -> Optional[SkillDraft]:
    """按场景模板生成 skill 草稿。未知场景返回 None。"""
    tpl = _SCENE_TEMPLATES.get(scene_key)
    if tpl is None:
        return None
    return SkillDraft(
        name=tpl["name"],
        description=tpl["description"],
        triggers=tpl["triggers"],
        algorithm=tpl["algorithm"],
        when_to_use=f"用户说{tpl['description'].split('，')[0]}等场景",
        when_not_to_use="稀疏走廊/楼梯口（长时间无人）→ 用 surveillance-sparse-corridor",
    )


def render_skill_md(draft: SkillDraft) -> str:
    """把 SkillDraft 渲染成 SKILL.md 内容（frontmatter + 正文）。"""
    return f"""---
name: {draft.name}
description: {draft.description}
triggers: {draft.triggers}
---

# {draft.name}

## 适用场景
{draft.when_to_use or draft.description}

## 算法
{draft.algorithm}

## 参数
{draft.parameters or "（沿用 surveillance-sparse-corridor 默认：sample_fps=1.0, day_threshold=15, night_threshold=6）"}

## 何时用
{draft.when_to_use}

## 何时不该用
{draft.when_not_to_use}

## 降级行为
{draft.fallback}
"""


def save_skill(draft: SkillDraft, skills_dir: Path,
               overwrite: bool = False) -> Path:
    """把草稿存为 config/skills/<name>/SKILL.md。返回路径。

    overwrite=False 时若目录已存在抛 FileExistsError（用户需先确认覆盖）。
    """
    skill_dir = skills_dir / draft.name
    if skill_dir.exists() and not overwrite:
        raise FileExistsError(
            f"skill 目录已存在：{skill_dir}（需 overwrite=True 确认覆盖）")
    skill_dir.mkdir(parents=True, exist_ok=True)
    md_path = skill_dir / "SKILL.md"
    md_path.write_text(render_skill_md(draft), encoding="utf-8")
    logger.info(f"[skill_gen] 已生成新 skill：{md_path}")
    return md_path


def generate_skill(text: str, skills_dir: Path,
                   overwrite: bool = False) -> Optional[dict]:
    """端到端：文本 → 场景识别 → 草稿 → 存盘。返回结果 dict 或 None。

    不真实调付费 LLM（红线）：用规则模板生成。用户给预算后可扩展
    draft_skill_from_scene 调 LLM 补全 algorithm/parameters。
    """
    scene = detect_scene(text)
    if scene is None:
        logger.info(f"[skill_gen] 未识别场景关键词：{text[:50]}")
        return None
    draft = draft_skill_from_scene(scene, text)
    if draft is None:
        return None
    try:
        md_path = save_skill(draft, skills_dir, overwrite=overwrite)
        return {
            "ok": True,
            "scene": scene,
            "skill_name": draft.name,
            "path": str(md_path),
            "draft": draft,
        }
    except FileExistsError as e:
        return {"ok": False, "error": str(e), "skill_name": draft.name}
