"""validator 三重验证测试。

构造跨领域/重复/不可复现样例，验证三重验证分别返回 pass/fail：
  1. cross_domain：与现有 skill 高度重叠 → fail；新领域 → pass
  2. predictability：工具链在历史经验中可复现 → pass；不可复现 → fail
  3. exclusivity：name 冲突 → fail；高描述相似 → fail；低冲突 → pass
  4. 全量 validate 聚合逻辑

纯规则，无 LLM。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.skills.validator import (
    SkillValidator,
    ValidationResult,
    tokenize,
    validate_skill,
)
from src.skills.schema import Skill


def _skill(name: str, desc: str, triggers=()) -> Skill:
    return Skill(name=name, description=desc,
                 triggers=tuple(triggers), path=Path("."), enabled=True)


def _candidate(name: str, desc: str, triggers=(), chain=()):
    """构造候选：带 tool_chain 属性的普通对象（供 predictability）。"""
    return SimpleNamespace(name=name, description=desc,
                           triggers=tuple(triggers), tool_chain=list(chain))


def _exp(chain, intent="监控找包"):
    return SimpleNamespace(intent=intent, tool_chain=list(chain))


# ---- cross_domain ----

def test_cross_domain_overlap_fails():
    """与现有 skill 高度重叠（同一主题）→ fail。"""
    v = SkillValidator(overlap_threshold=0.45)
    existing = [_skill("surveillance-sparse-corridor", "稀疏走廊/楼梯口监控分析（长时间无人）")]
    cand = _candidate("sparse-v2", "稀疏走廊/楼梯口监控分析（长时间无人）重复版")
    ok, score = v.cross_domain(cand, existing)
    assert ok is False
    assert score > 0.45


def test_cross_domain_new_field_passes():
    """新领域（语音合成）与现有 surveillance 系列几乎不重叠 → pass。"""
    v = SkillValidator()
    existing = [_skill("surveillance-sparse-corridor", "稀疏走廊/楼梯口监控分析")]
    cand = _candidate("tts-voice", "为视频生成配音与旁白，本地 TTS 工作流")
    ok, score = v.cross_domain(cand, existing)
    assert ok is True
    assert score < 0.45


def test_cross_domain_empty_tokens_passes():
    """候选无有效 token（信息量低）→ 默认 pass（宁可放过不可错杀）。"""
    v = SkillValidator()
    existing = [_skill("surv", "稀疏走廊")]
    cand = _candidate("x", "a b c")  # 单字符 token 全被过滤
    ok, score = v.cross_domain(cand, existing)
    assert ok is True


# ---- predictability ----

def test_predictability_steps_reproduced():
    """候选工具链步骤在历史经验中出现 → pass。"""
    v = SkillValidator(min_predictability=0.5)
    cand = _candidate("sparse", "走廊监控", chain=["extract_frames", "detect_motion"])
    exps = [_exp(["extract_frames", "detect_motion", "generate_summary"])]
    ok, score = v.predictability(cand, exps)
    assert ok is True
    assert score == 1.0


def test_predictability_steps_not_reproduced():
    """候选工具链步骤历史经验中未出现 → fail。"""
    v = SkillValidator(min_predictability=0.5)
    cand = _candidate("tts", "配音", chain=["lux_tts", "mix_audio"])
    exps = [_exp(["extract_frames", "analyze_frames"])]
    ok, score = v.predictability(cand, exps)
    assert ok is False
    assert score == 0.0


def test_predictability_no_steps_fails():
    """候选无工具链 → fail（不可预测）。"""
    v = SkillValidator()
    cand = _candidate("x", "描述")
    ok, _ = v.predictability(cand, [])
    assert ok is False


# ---- exclusivity ----

def test_exclusivity_name_conflict_fails():
    """name 与现有重复 → fail（硬冲突）。"""
    v = SkillValidator()
    existing = [_skill("sparse", "走廊监控")]
    cand = _candidate("sparse", "走廊监控新版")
    ok, score = v.exclusivity(cand, existing)
    assert ok is False
    assert score == 1.0


def test_exclusivity_high_similarity_fails():
    """描述相似度过高（max_conflict 内）→ fail。"""
    v = SkillValidator(max_conflict=0.85)
    existing = [_skill("a", "走廊监控分析")]
    cand = _candidate("b", "走廊监控分析")
    ok, score = v.exclusivity(cand, existing)
    assert ok is False
    assert score > 0.85


def test_exclusivity_low_conflict_passes():
    """无冲突 → pass。"""
    v = SkillValidator()
    existing = [_skill("surv", "稀疏走廊监控")]
    cand = _candidate("tts", "视频配音与旁白")
    ok, score = v.exclusivity(cand, existing)
    assert ok is True
    assert score < 0.85


# ---- validate 全量 ----

def test_validate_all_pass():
    """三项全过 → pass。"""
    v = SkillValidator()
    existing = [_skill("surv", "稀疏走廊监控")]
    cand = _candidate("tts", "视频配音与旁白", chain=["lux_tts", "mix_audio"])
    exps = [_exp(["lux_tts", "mix_audio", "render_video"])]
    result = v.validate(cand, existing, exps)
    assert isinstance(result, ValidationResult)
    assert result.pass_ is True
    assert result.passed is True
    assert result.reasons == []
    assert result.check_results["cross_domain"] is True
    assert result.check_results["predictability"] is True
    assert result.check_results["exclusivity"] is True


def test_validate_overlap_fails_with_reason():
    """跨领域重叠过高 → fail 且带 reason。"""
    v = SkillValidator()
    existing = [_skill("surv", "稀疏走廊/楼梯口监控分析（长时间无人）")]
    cand = _candidate("sparse-v2", "稀疏走廊/楼梯口监控分析（长时间无人）重复版",
                      chain=["extract_frames"])
    exps = [_exp(["extract_frames"])]
    result = v.validate(cand, existing, exps)
    assert result.pass_ is False
    assert any("跨领域" in r for r in result.reasons)


def test_validate_predictability_fails():
    """工具链不可复现 → fail。"""
    v = SkillValidator()
    cand = _candidate("newskill", "全新领域描述", chain=["unknown_tool_x"])
    result = v.validate(cand, [], [])  # 无历史经验
    assert result.pass_ is False
    assert any("预测力" in r for r in result.reasons)


def test_validate_skill_standalone():
    """便捷函数 validate_skill 可独立调用。"""
    result = validate_skill(
        _candidate("tts", "视频配音与旁白", chain=["a"]),
        [_skill("surv", "稀疏走廊")],
        [_exp(["a"])],
    )
    assert isinstance(result, ValidationResult)


def test_tokenize_keeps_chinese_words():
    """tokenize 保留 2 字以上中文词（整词保留）。"""
    toks = tokenize("走廊监控分析")
    assert toks == {"走廊监控分析"}  # 无空格词，整词为一个 token