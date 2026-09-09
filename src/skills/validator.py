"""三重验证（validator）— skill 进入棘轮前的准入校验。

参考 nuwa/EvoSkill/SkillOpt 思路的规则化版本（无 LLM）：

1. **跨领域**（novelty）：skill 的适用场景与已有 skills 的 intent 分布重叠度低。
   指标：description + triggers 与已有 skill 的关键词交集比例。
   超过 overlap_threshold 即视为与现有 skill 角色重复 → 排他性失败（fail）。
2. **预测力**（predictability）：skill 的工作流工具链步骤能在历史经验中复现。
   指标：实验步骤（自带 tool_chain）在经验样本中出现的覆盖率。
3. **排他性**（conflict）：name 不重复，且与新 skill description 的相似词重叠
   在被认为过于接近的阈值内才仅告警，不算失败？—— 规则化设计：
   直接允许排他性通过仅有冲突 name 时 fail（name 冲突=硬失败），
   description 重叠走"跨领域"那一层。

返回 ValidationResult(pass, reasons, scores)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

try:  # 复用 spectre 的规则化分析；spectre 不可用时降级（保持模块独立）
    from src.skills.spectre import scan_skill
except Exception:  # noqa: BLE001
    scan_skill = None  # type: ignore[assignment]

_NON_WORD_RE = re.compile(r"\W+")


def tokenize(text: str) -> Set[str]:
    """把中文/英文描述切成关键词 token（保留 2 字以上中文词）。"""
    if not text:
        return set()
    toks: Set[str] = set()
    for w in _NON_WORD_RE.split(text):
        if not w:
            continue
        w = w.strip().strip("().,，。、;；:")
        if len(w) >= 2:
            toks.add(w.lower())
    return toks


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tokenize_description(skill: object) -> Set[str]:
    desc = getattr(skill, "description", "") or ""
    return tokenize(str(desc))


@dataclass(frozen=True)
class ValidationResult:
    """校验结论。pass=True 表示三项全部通过。"""

    pass_: bool
    name: str
    check_results: Dict[str, bool]
    reasons: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    # 兼容属性名
    @property
    def passed(self) -> bool:
        return self.pass_

    @property
    def failed(self) -> bool:
        return not self.pass_


class SkillValidator:
    """三重验证器 default thresholds (tunable)."""

    def __init__(
        self,
        overlap_threshold: float = 0.45,
        min_predictability: float = 0.5,
        max_conflict: float = 0.85,
    ) -> None:
        self.overlap_threshold = float(overlap_threshold)
        self.min_predictability = float(min_predictability)
        self.max_conflict = float(max_conflict)

    # -- 1. 跨领域（novelty） -------------------------------------------------

    def cross_domain(
        self, candidate: object, existing: Sequence[object]
    ) -> tuple[bool, float]:
        """候选 vs 已有 skills 的最大关键词重叠比（Jaccard）。"""
        cand_toks = _tokenize_description(candidate)
        trig = getattr(candidate, "triggers", ()) or ()
        for t in trig:
            cand_toks |= tokenize(str(t))
        if not cand_toks:
            return True, 0.0  # 无信息量候选默认通过（宁可放过不可错杀）
        best = 0.0
        for ex in existing:
            ex_toks = _tokenize_description(ex)
            for t in (getattr(ex, "triggers", ()) or ()):
                ex_toks |= tokenize(str(t))
            best = max(best, _jaccard(cand_toks, ex_toks))
        return best <= self.overlap_threshold, best

    # -- 2. 预测力（predictability） -------------------------------------------

    def predictability(
        self, candidate: object, experiences: Sequence[object]
    ) -> tuple[bool, float]:
        """候选 tool_chain 中的步骤在历史经验中的复现率。"""
        steps = self._get_tool_steps(candidate)
        if not steps:
            return False, 0.0  # 无工具链 = 不可预测
        hist_steps = self._collect_step_tokens(experiences)
        if not hist_steps:
            return False, 0.0
        hit = sum(1 for s in steps if s in hist_steps)
        return hit / len(steps) >= self.min_predictability, hit / len(steps)

    def _get_tool_steps(self, candidate: object) -> List[str]:
        obj = candidate
        if hasattr(obj, "chain"):  # ExperienceDraft
            obj = getattr(obj, "draft", obj)
        if hasattr(obj, "tool_chain"):
            return [str(x).strip() for x in obj.tool_chain if str(x).strip()]
        algo = getattr(obj, "algorithm", "") or ""
        if not algo:
            return []
        # algorithm 里每句掐一段来当步骤（保守）
        parts = [p for p in algo.replace(";", "\n").replace("。", "\n").splitlines() if p]
        return [p.strip().rstrip("；") for p in parts if p]

    def _collect_step_tokens(self, experiences: Sequence[object]) -> Set[str]:
        toks: Set[str] = set()
        for e in experiences:
            for t in (getattr(e, "tool_chain", ()) or ()):
                if t:
                    toks.add(str(t))
            chain = getattr(e, "chain", None)
            if chain:
                for t in chain:
                    if t:
                        toks.add(str(t))
        return toks

    # -- 3. 排他性（conflict） --------------------------------------------------

    def exclusivity(
        self, candidate: object, existing: Sequence[object]
    ) -> tuple[bool, float]:
        """候选与现有 skill 是否冲突：name 硬重复 → fail；description 相似度
        超过 max_conflict 也 fail（规则化：高 Jaccard 视作同题 skill）。"""
        cand_name = str(getattr(candidate, "name", "")).strip()
        cand_toks = _tokenize_description(candidate)
        for t in (getattr(candidate, "triggers", ()) or ()):
            cand_toks |= tokenize(str(t))
        if not cand_name:
            return False, 1.0
        best = 0.0
        for ex in existing:
            ex_name = str(getattr(ex, "name", "")).strip()
            if ex_name == cand_name:
                return False, 1.0
            ex_toks = _tokenize_description(ex)
            for t in (getattr(ex, "triggers", ()) or ()):
                ex_toks |= tokenize(str(t))
            best = max(best, _jaccard(cand_toks, ex_toks))
        return best <= self.max_conflict, best

    # -- 全量 ----------------------------------------------------------------

    def validate(
        self,
        candidate: object,
        existing: Sequence[object],
        experiences: Optional[Sequence[object]] = None,
    ) -> ValidationResult:
        """跑完整三重验证。pass 需要 1/2/3 全部 True。"""
        name = str(getattr(candidate, "name", "")).strip()
        domain_ok, domain_score = self.cross_domain(candidate, existing)
        pred_ok, pred_score = self.predictability(candidate, experiences or [])
        excl_ok, excl_score = self.exclusivity(candidate, existing)
        reasons: List[str] = []
        if not domain_ok:
            reasons.append(
                f"跨领域重叠过高（overlap={domain_score:.2f} > "
                f"{self.overlap_threshold}）：与现有 skill 主题重复")
        if not pred_ok:
            reasons.append(
                f"预测力不足（predictability={pred_score:.2f} < "
                f"{self.min_predictability}）：工具链步骤在历史经验中不可复现")
        if not excl_ok:
            reasons.append(
                f"排他性失败（name 冲突或描述相似度过高 = {excl_score:.2f}）："
                "不与现有 skill 冲突")
        return ValidationResult(
            pass_=(domain_ok and pred_ok and excl_ok),
            name=name,
            check_results={
                "cross_domain": domain_ok,
                "predictability": pred_ok,
                "exclusivity": excl_ok,
            },
            reasons=reasons,
            scores={
                "cross_domain_overlap": domain_score,
                "predictability": pred_score,
                "exclusivity_conflict": excl_score,
            },
        )


# 便捷函数（兼容调用方）
def validate_skill(
    candidate: object,
    existing: Sequence[object],
    experiences: Optional[Sequence[object]] = None,
    **kwargs
) -> ValidationResult:
    """一次调用三层验证。"""
    return SkillValidator(**kwargs).validate(candidate, existing, experiences)