"""skill 准入闸门测试（v10.4.0 P0-1）。

背景（真实缺陷）
----------------
``src/skills/validator.py`` 自 v10.1 就实现了三重验证（跨领域/预测力/排他性），
它自己的 docstring 写着"skill 进入棘轮前的准入校验"，但**全仓非测试代码零调用**
（``grep -rn "skills.validator" src/`` 只命中定义文件自身）—— 蒸馏出的草稿
绕过了一切质量闸门。

本文件守护这次接线的三项契约：
  1. 草稿产出后必经 validator，且结果挂到 ``admitted`` / ``validation_report``；
  2. 判定复用 **原始 tool_chain**（而非中文步骤描述），否则复现率恒为 0、误杀全部草稿；
  3. ``VAP_SKILLS_AUTODISTILL=0`` 时完全跳过（零回归）。
"""

from __future__ import annotations

import time

import pytest

from src.core.agent.experience import Experience
from src.skills.distiller import DraftPipeline, _AdmissionCandidate


def _mk_exp(intent: str, chain, sid: str, quality: float = 0.9) -> Experience:
    return Experience(
        intent=intent,
        tool_chain=list(chain),
        success=True,
        quality_score=quality,
        timestamp=time.time(),
        session_id=sid,
    )


def _three(intent: str = "停车场车牌识别", chain=("extract_frames", "transcribe", "analyze_frames")):
    return [_mk_exp(intent, chain, f"s{i}") for i in range(3)]


class _ExistingSkill:
    """模拟已加载的 skill（router 里传入的 existing 元素）。"""

    def __init__(self, name: str, description: str, triggers=()):
        self.name = name
        self.description = description
        self.triggers = list(triggers)


# --------------------------------------------------------------------------
# 1) 接线本体
# --------------------------------------------------------------------------


def test_distiller_calls_validator_on_candidate():
    """蒸馏产物必须带 validation_report（证明 validator 真被调用了）。"""
    draft = DraftPipeline(min_count=3).distill(_three())
    assert draft is not None
    assert draft.validation_report is not None, "validator 未被调用（准入闸门未接线）"
    assert "admitted" in draft.validation_report
    assert "checks" in draft.validation_report
    assert set(draft.validation_report["checks"]) == {
        "cross_domain", "predictability", "exclusivity"
    }


def test_distiller_reports_reasons_when_rejected():
    """被拒时必须有可读原因（不静默丢弃）。"""
    # 与现有 skill 完全同名 → 排他性硬失败
    draft = DraftPipeline(min_count=3).distill(
        _three(),
        existing=[_ExistingSkill("停车场车牌识别", "停车场车牌识别监控")],
    )
    assert draft is not None, "被拒也应返回草稿（不静默丢弃）"
    report = draft.validation_report
    assert report is not None
    # 同名必然与 existing 冲突 → exclusivity 失败
    if not draft.admitted:
        assert report["reasons"], "被拒但没有给出原因"


def test_distiller_accepts_valid_candidate():
    """高质量经验 + 无冲突 existing → 放行。"""
    draft = DraftPipeline(min_count=3).distill(
        _three(),
        existing=[_ExistingSkill("夜间车辆检测", "夜里数车")],
    )
    assert draft is not None
    assert draft.admitted is True, f"合法草稿被误杀: {draft.validation_report}"
    assert draft.validation_report["checks"]["predictability"] is True


def test_admission_uses_raw_tool_chain_not_chinese_steps():
    """回归守卫：准入必须复用**原始工具名**比较复现率。

    若把 SkillDraft（algorithm 存的是中文步骤描述）直接交给 validator，
    predictability 的分子会恒为 0 → 每条草稿都被误杀。这里断言复现率为 1.0。
    """
    chain = ("extract_frames", "transcribe", "analyze_frames")
    draft = DraftPipeline(min_count=3).distill(_three(chain=chain))
    assert draft is not None
    assert draft.validation_report is not None
    assert draft.validation_report["scores"]["predictability"] == 1.0, (
        "复现率不是 1.0 —— 说明准入拿的是中文步骤而非原始 tool_chain"
    )


def test_admission_candidate_exposes_tool_chain_not_chain_attr():
    """_AdmissionCandidate 必须暴露 tool_chain 且**不含** chain 属性。

    WHY: validator._get_tool_steps 对带 ``chain`` 的对象会先解到 ``.draft``，
    从而又回到中文步骤。这个断言锁住那个坑。
    """
    cand = _AdmissionCandidate(
        name="n", description="d", triggers=(), tool_chain=("a", "b")
    )
    assert hasattr(cand, "tool_chain")
    assert not hasattr(cand, "chain")


def test_validator_crash_is_fail_open(monkeypatch):
    """validator 自身抛异常时必须放行并留痕（闸门坏了不能阻断主流程）。"""
    import src.skills.validator as validator_mod

    def _boom(*_a, **_k):
        raise RuntimeError("模拟 validator 崩溃")

    monkeypatch.setattr(validator_mod, "validate_skill", _boom)

    draft = DraftPipeline(min_count=3).distill(_three())
    assert draft is not None, "validator 崩溃不应导致蒸馏返回 None"
    report = draft.validation_report
    assert report is not None
    assert report["admitted"] is True, "fail-open 应放行"
    assert report.get("skipped") is True
    assert any("validator 不可用" in r for r in report["reasons"])


def test_existing_param_is_actually_used():
    """existing 传入必须影响判定（此前该参数从未被使用）。"""
    baseline = DraftPipeline(min_count=3).distill(_three())
    assert baseline is not None and baseline.admitted is True

    triggers = [t for t in str(getattr(baseline.draft, "triggers", "") or "").split(",") if t]
    with_conflict = DraftPipeline(min_count=3).distill(
        _three(),
        existing=[_ExistingSkill(
            str(baseline.draft.name), str(baseline.draft.description), triggers
        )],
    )
    assert with_conflict is not None
    # 同名冲突 → 排他性失败（证明 existing 真的被用上了）
    assert with_conflict.admitted is False
    assert with_conflict.validation_report["checks"]["exclusivity"] is False


# --------------------------------------------------------------------------
# 2) 开关：零回归
# --------------------------------------------------------------------------


def test_autodistill_off_skips_validator(monkeypatch):
    """VAP_SKILLS_AUTODISTILL=0 → 完全跳过准入（行为同 v10.3.1）。"""
    monkeypatch.setenv("VAP_SKILLS_AUTODISTILL", "0")
    draft = DraftPipeline(min_count=3).distill(_three())
    assert draft is not None
    assert draft.admitted is True
    assert draft.validation_report is None, "开关关闭时不应产生准入报告"


def test_autodistill_on_by_default(monkeypatch):
    """未设环境变量 → 默认启用准入。"""
    monkeypatch.delenv("VAP_SKILLS_AUTODISTILL", raising=False)
    draft = DraftPipeline(min_count=3).distill(_three())
    assert draft is not None
    assert draft.validation_report is not None


def test_admission_enabled_helper_reads_env(monkeypatch):
    from src.skills.distiller import _admission_enabled

    monkeypatch.delenv("VAP_SKILLS_AUTODISTILL", raising=False)
    assert _admission_enabled() is True
    monkeypatch.setenv("VAP_SKILLS_AUTODISTILL", "1")
    assert _admission_enabled() is True
    monkeypatch.setenv("VAP_SKILLS_AUTODISTILL", "0")
    assert _admission_enabled() is False
    monkeypatch.setenv("VAP_SKILLS_AUTODISTILL", "false")
    assert _admission_enabled() is True  # 只认 "0"


# --------------------------------------------------------------------------
# 3) 防回退：源码级契约
# --------------------------------------------------------------------------


def test_validator_has_production_caller():
    """反向断言：distiller.py 必须真的调用 validate_skill（防未来回退）。"""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "skills" / "distiller.py"
    text = src.read_text(encoding="utf-8")
    assert "validate_skill" in text, "distiller 又不再调用 validator 了（准入闸门断裂）"
    assert "_admission_enabled" in text


def test_skills_router_passes_existing_skills():
    """反向断言：router 必须把已加载 skills 作为 existing 传入。"""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "web" / "routers" / "skills.py"
    text = src.read_text(encoding="utf-8")
    assert "existing=_load_skills()" in text, "router 又没给 validator 传对照 skills"
    assert '"admitted"' in text, "router 又没把准入结果返回给前端"


def test_draft_is_frozen_still():
    """ExperienceDraft 仍是 frozen dataclass（新增字段不得破坏不可变语义）。"""
    d = DraftPipeline(min_count=3).distill(_three())
    with pytest.raises(Exception):
        d.intent = "changed"  # type: ignore[misc]
