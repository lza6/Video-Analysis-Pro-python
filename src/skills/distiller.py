"""Skill 蒸馏器 — 从 ExperienceStore 聚合形成 skill 草稿 (closed-loop).

Provides two entry points:
  1. ``DraftPipeline.distill`` — full pipeline: read the stable tool-chain work-
     steps from ExperienceStore and let skill_generator's scene template render
     the final Markdown. Pure rules, no paid LLM (same red line as
     skill_generator). Returns ExperienceDraft (or None when threshold not met).
  2. ``derive_workflow_steps`` — pure rule-based induction of workflow steps
     from a tool-chain (input/output-frame variable tuples -> steps).

Used by distributor/pipeline code and exposed for test isolation (a dict of
Experience dataclasses keeps this module free of SQLite for unit tests).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import src.core.skill_generator as sg  # reused: detect_scene/draft_skill_from_scene/render_skill_md

log = logging.getLogger(__name__)

# 每脚本仅匹配意图的最长步骤模板，保证确定性
_STEP_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "FramesAndAudio": (
        ("extract_frames", "抽帧采样"),
        ("transcribe", "提取语音转录"),
        ("analyze_frames", "逐帧分析"),
        ("generate_summary", "生成摘要"),
    ),
    "VideoOnly": (
        ("extract_frames", "抽帧采样"),
        ("analyze_frames", "逐帧分析"),
        ("generate_summary", "生成摘要"),
    ),
    "AudioOnly": (
        ("transcribe", "提取语音转录"),
        ("analyze_frames", "逐帧分析"),
        ("generate_summary", "生成摘要"),
    ),
}


def derive_workflow_steps(tool_chain: Sequence[str]) -> List[str]:
    """从 tool_chain 归纳工作流步骤（规则化，无 LLM）。

    用「至少 3 步的模板」依次匹配；无模板命中时对每一步做动词规约
    （去掉请求 frame 的写法、下划线转空格）。
    """
    chain = list(tool_chain)
    if not chain:
        return []
    for name, steps in _STEP_TEMPLATES.items():
        if len(steps) >= 3:
            if _matches_steps(chain, steps):
                return [desc for _, desc in steps]
    return [_verbalize_step(t) for t in chain]


def _matches_steps(chain: Sequence[str], steps: Sequence[tuple[str, str]]) -> bool:
    """chain 是否是 steps 的子序列（核心步骤按序出现即可）。"""
    it = iter(chain)
    return all(any(s == token for token in it) for s, _ in steps)


def _verbalize_step(name: str) -> str:
    """把工具名转换为中文步骤描述。"""
    v = name.replace("_", " ")
    verb = v.lower()
    if verb.startswith("search"):
        v = "搜索/检索 " + v[len("search"):].strip()
    elif verb.startswith("detect"):
        v = "检测 " + v[len("detect"):].strip()
    elif verb.startswith("count"):
        v = "计数 " + v[len("count"):].strip()
    else:
        v = v if v else name
    return v


@dataclass(frozen=True)
class ExperienceDraft:
    """蒸馏草稿：聚合经验 + 配置好的 skill 草稿。"""

    intent: str
    occurrences: int
    chain: tuple[str, ...]
    draft: sg.SkillDraft
    markdown: str


@dataclass(frozen=True)
class DistillTemplateOption:
    """一个已匹配场景模板的中间结果（供 pipeline 阶段排序）。"""

    scene: str
    name: str
    description: str


class DraftPipeline:
    """从 Experience 集合蒸馏 skill 草稿 (naive, no network)."""

    def __init__(self, min_count: int = 3) -> None:
        self.min_count = int(min_count)

    @staticmethod
    def _extract_vertex_case(e: object) -> Optional[Dict[str, object]]:
        """把 Experience 规范化为 dict（兼容 dataclass Experience 或 plain dict）。"""
        if hasattr(e, "intent") and hasattr(e, "tool_chain"):
            try:
                return {
                    "intent": str(getattr(e, "intent")),
                    "tool_chain": [str(x) for x in getattr(e, "tool_chain")],
                }
            except Exception:  # noqa: BLE001
                return None
        if isinstance(e, dict):
            return e
        return None

    @staticmethod
    def _key_chain(chain: Sequence[str]) -> tuple[str, ...]:
        return tuple(chain)

    def distill(
        self,
        experiences: Sequence[object],
        *,
        skills_dir: Optional[Path] = None,
        existing: Sequence[sg.SkillDraft] = (),
    ) -> Optional[ExperienceDraft]:
        """从给定经验集合提取稳定 intent (>= min_count) 与稳定 tool-chain，
        得到蒸馏草稿。

        - intent 命中 skill_generator.detect_scene 时复用既有场景模板
          (draft_skill_from_scene + render_skill_md)；
        - 否则对 tool_chain 归纳工作流步骤自建草稿。

        返回 None 表示未达阈值或没有稳定链。
        """
        groups: dict[str, dict[str, object]] = {}
        for e in experiences:
            norm = self._extract_vertex_case(e)
            if norm is None:
                continue
            intent = str(norm.get("intent") or "").strip().lower()
            chain = tuple(str(t) for t in norm.get("tool_chain") or ())
            if not intent or not chain:
                continue
            group = groups.setdefault(intent, {"count": 0, "chains": [], "chain": None})
            group["count"] = int(group["count"]) + 1
            chains = group["chains"]
            if isinstance(chains, list) and self._key_chain(chain) not in chains:
                chains.append(self._key_chain(chain))

        candidates: list[ExperienceDraft] = []

        for intent, group in groups.items():
            count = int(group["count"])
            if count < self.min_count:
                continue
            chains = group["chains"]
            if isinstance(chains, list) and len(chains) == 1:
                chain = list(chains[0])
            else:
                # 不稳定链跳过（避免把多意图捏成一条）
                continue
            scene = sg.detect_scene(intent)
            if scene is not None:
                draft = sg.draft_skill_from_scene(scene, intent)
                if draft is None:
                    continue
                steps = derive_workflow_steps(chain)
                rendered = sg.render_skill_md(draft)
                candidates.append(ExperienceDraft(
                    intent=intent, occurrences=count,
                    chain=tuple(chain), draft=draft, markdown=rendered))
            else:
                if not chain:
                    continue
                # 规则化自建草稿（非模板场景）
                steps = derive_workflow_steps(chain)
                draft = sg.SkillDraft(
                    name=make_skill_name(intent, steps),
                    description=make_skill_description(intent, steps),
                    triggers=",".join(_top_triggers(intent)),
                    algorithm="; ".join(steps) if steps else "规则工作流",
                    parameters="",
                    when_to_use=make_skill_description(intent, steps),
                    when_not_to_use="无对应依赖时降级为纯帧差分，不崩",
                    fallback="无对应依赖时降级为纯帧差分，不崩",
                )
                rendered = sg.render_skill_md(draft)
                candidates.append(ExperienceDraft(
                    intent=intent, occurrences=count,
                    chain=tuple(chain), draft=draft, markdown=rendered))

        if not candidates:
            return None
        # 确定性优先：次数降序 → intent 字典序（测试稳定）
        candidates.sort(key=lambda c: (-c.occurrences, c.intent))
        return candidates[0]


def make_skill_name(intent: str, steps: Sequence[str]) -> str:
    """由 intent + 归纳步骤生成 slug 化 skill 名（纯 stdlib）。"""
    import re
    slug = re.sub(r"[^a-zA-Z0-9一-鿿_-]+", "-", intent.strip().lower())
    slug = slug.strip("-").strip()
    if not slug:
        slug = "experience-skill"
    if len(slug) > 40:
        slug = slug[:40].rstrip("-").rstrip()
    verb = (steps[0] if steps else "").replace(" ", "-")
    if verb and verb not in slug:
        slug = f"{slug}-{verb}"
    return slug


def make_skill_description(intent: str, steps: Sequence[str]) -> str:
    """由 intent + 步骤生成描述（中文风格，沿用指南命名）。"""
    if steps:
        core = "；".join(steps[:5])
        desc = f"根据用户意图「{intent}」归纳的工作流：{core}"
    else:
        desc = f"根据用户意图「{intent}」归纳的工作流"
    return desc[:200]


def _top_triggers(intent: str) -> List[str]:
    """把 intent 拆成逗号分隔的触发词（去重去空）。"""
    out: List[str] = []
    for chunk in intent.replace("/", ",").split(","):
        chunk = chunk.strip()
        if chunk and chunk not in out:
            out.append(chunk)
    return out[:4]


# ---------------------------------------------------------------------------
# 便捷入口：从 ExperienceStore 聚合（复用 + 新增便捷函数）
# ---------------------------------------------------------------------------
def load_grouped_experiences(store: object) -> Dict[str, Dict[str, object]]:
    """从 ExperienceStore 读出全部经验，按 intent 分组聚合。

    Return: {intent: {"count": int, "chains": list[tuple], "occurrences": int}}
    (order-agnostic for tests).
    """
    items: List[object] = []
    try:
        for candidate in store.find_all():  # ExperienceStore.find_all 已存在
            items.append(candidate)
    except AttributeError:
        try:
            items = store.read_all()
        except AttributeError:
            items = []
    groups: Dict[str, Dict[str, object]] = {}
    for e in items:
        intent = getattr(e, "intent", "")
        chain = tuple(getattr(e, "tool_chain", ()) or ())
        intent = str(intent).strip().lower()
        chain = tuple(str(x) for x in chain)
        if not intent or not chain:
            continue
        g = groups.setdefault(intent, {"count": 0, "chains": []})
        g["count"] = int(g["count"]) + 1
        if chain not in g["chains"]:
            g["chains"].append(chain)
    return groups


def group_experiences(experiences: Sequence[object]) -> Dict[str, Dict[str, object]]:
    """形态无关分组（dict 或 Experience dataclass 均可）。"""
    groups: Dict[str, Dict[str, object]] = {}
    for e in experiences:
        norm = DraftPipeline._extract_vertex_case(e)
        if norm is None:
            continue
        intent = str(norm.get("intent") or "").strip().lower()
        chain = tuple(str(t) for t in norm.get("tool_chain") or ())
        if not intent or not chain:
            continue
        g = groups.setdefault(intent, {"count": 0, "chains": []})
        g["count"] = int(g["count"]) + 1
        if chain not in g["chains"]:
            g["chains"].append(chain)
    return groups


def aggregate_workflow_steps(experiences: Sequence[object]) -> Dict[str, List[str]]:
    """返回 {intent: [步骤描述...]}，仅保留稳定链（≥3 次 + 唯一链）。"""
    out: Dict[str, List[str]] = {}
    groups = group_experiences(experiences)
    for intent, g in groups.items():
        count = int(g["count"])
        chains = g["chains"]
        if count < 3 or len(chains) != 1:
            continue
        chain = list(chains[0])
        if not chain:
            continue
        steps = derive_workflow_steps(chain)
        if steps:
            out[intent] = steps
    return out