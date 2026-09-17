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
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import src.core.skill_generator as sg  # reused: detect_scene/draft_skill_from_scene/render_skill_md

log = logging.getLogger(__name__)

#: v10.4.0 (P0-1)：准入闸门开关。沿用既有 VAP_SKILLS_AUTODISTILL
#: （默认 1=开）。设 0 → 完全跳过 validator，行为与 v10.3.1 一致（零回归）。
_ENV_AUTODISTILL = "VAP_SKILLS_AUTODISTILL"


def _admission_enabled() -> bool:
    """准入闸门是否启用（默认启用）。"""
    return os.environ.get(_ENV_AUTODISTILL, "1").strip().lower() != "0"

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
    """蒸馏草稿：聚合经验 + 配置好的 skill 草稿。

    v10.4.0 (P0-1)：新增准入结果字段。``admitted=False`` 表示 validator 判定
    该草稿不应进入棘轮（例如与现有 skill 主题重复），**但草稿本身仍然返回**
    —— 不静默丢弃，让用户在 UI 上看到“为什么被拒”。
    """

    intent: str
    occurrences: int
    chain: tuple[str, ...]
    draft: sg.SkillDraft
    markdown: str
    #: validator 是否放行（未启用闸门时恒为 True）
    admitted: bool = True
    #: 准入报告（未启用闸门时为 None）；结构见 _build_validation_report
    validation_report: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class _AdmissionCandidate:
    """把 ExperienceDraft 适配成 validator 期望的形状。

    WHY：``SkillValidator.predictability`` 用 candidate 的 ``tool_chain`` 与历史经验的
    ``tool_chain`` 比复现率；而 ``SkillDraft.algorithm`` 是**中文步骤描述**
    （如“抽帧采样”），历史里的 ``tool_chain`` 是**原始工具名**（如
    ``extract_frames``）—— 两者不同域，直接传 SkillDraft 会让复现率恒为 0，
    把每一条草稿都误杀。这里显式提供原始 chain，让比较在同域进行。

    另注：``validator._get_tool_steps`` 对带 ``chain`` 属性的对象会先解到 ``.draft``，
    因此这里特意**不叫** ``chain``，避免被它重新解包。
    """

    name: str
    description: str
    triggers: tuple[str, ...]
    tool_chain: tuple[str, ...]


def _build_validation_report(result: Any) -> Dict[str, Any]:
    """把 ValidationResult 转成可 JSON 序列化的准入报告。"""
    return {
        "admitted": bool(getattr(result, "pass_", False)),
        "checks": dict(getattr(result, "check_results", {}) or {}),
        "scores": dict(getattr(result, "scores", {}) or {}),
        "reasons": list(getattr(result, "reasons", []) or []),
    }


def _admit(
    candidate: ExperienceDraft,
    existing: Sequence[object],
    experiences: Sequence[object],
) -> ExperienceDraft:
    """跑三重验证并把结果挂到草稿上（不丢弃草稿）。

    validator 不可用（缺依赖/异常）时**放行**并标记 skipped=True —— 准入闸门
    不应该因为自身故障而阻断主流程（fail-open，但留痕）。
    """
    try:
        from src.skills.validator import validate_skill

        cand = _AdmissionCandidate(
            name=str(candidate.draft.name),
            description=str(candidate.draft.description),
            triggers=tuple(
                str(t).strip()
                for t in str(getattr(candidate.draft, "triggers", "") or "").split(",")
                if str(t).strip()
            ),
            tool_chain=tuple(str(x) for x in candidate.chain),
        )
        result = validate_skill(cand, list(existing), list(experiences))
        report = _build_validation_report(result)
    except Exception as e:  # noqa: BLE001
        # fail-open：准入闸门自身故障不应阻断主流程，但必须留痕。
        log.warning("[distiller] 准入校验失败，本次放行: %s", e)
        return replace(candidate, validation_report={
            "admitted": True, "skipped": True,
            "checks": {}, "scores": {},
            "reasons": [f"validator 不可用: {e}"],
        })
    if not report["admitted"]:
        log.info("[distiller] 草稿 '%s' 未通过准入校验: %s",
                 candidate.draft.name, report["reasons"])
    return replace(candidate, admitted=report["admitted"],
                   validation_report=report)


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
        best = candidates[0]

        # v10.4.0 (P0-1)：准入闸门 —— validator 自 v10.1 定义以来首次进入生产路径。
        # validator 自己的 docstring 就写着“skill 进入棘轮前的准入校验”，
        # 此前零调用方；这里把它接在“草稿产出后、返回给调用方前”。
        if not _admission_enabled():
            return best
        return _admit(best, existing, experiences)


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