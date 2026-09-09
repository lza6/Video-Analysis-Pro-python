"""distiller 蒸馏测试：从 Experience 聚合出 skill 草稿。

覆盖：
  1. 聚合：同 intent >=3 次 + 稳定 tool_chain → 产出 ExperienceDraft
  2. 复用 skill_generator 场景模板（parking 场景 → 复用模板）
  3. 非模板场景：tool_chain 归纳工作流步骤自建草稿
  4. 未达阈值 / 链不稳定 → None
  5. derive_workflow_steps / make_skill_name / make_skill_description
  6. group_experiences / aggregate_workflow_steps

纯 stdlib + 规则，不调付费 LLM（守红线）。
"""
from __future__ import annotations

from src.core import skill_generator as sg
from src.skills.distiller import (
    ExperienceDraft,
    DraftPipeline,
    aggregate_workflow_steps,
    derive_workflow_steps,
    group_experiences,
    make_skill_name,
)

from src.core.agent.experience import Experience
import time


def _mk_exp(intent: str, chain, sid: str) -> Experience:
    return Experience(
        intent=intent,
        tool_chain=list(chain),
        success=True,
        quality_score=0.9,
        timestamp=time.time(),
        session_id=sid,
    )


def test_distill_meets_threshold_with_scene():
    """同 intent >=3 次 + 稳定链 → 复用场景模板生成草稿。"""
    pipeline = DraftPipeline(min_count=3)
    exps = [_mk_exp("停车场车牌识别", ["extract_frames", "detect_vehicles"], f"s{i}")
            for i in range(3)]
    draft = pipeline.distill(exps)
    assert draft is not None
    assert isinstance(draft, ExperienceDraft)
    assert draft.occurrences == 3
    assert draft.chain == ("extract_frames", "detect_vehicles")
    # 复用 skill_generator 场景模板
    assert draft.draft.name == "surveillance-parking-lpr"
    assert "## 算法" in draft.markdown
    assert draft.markdown.startswith("---\nname: ")


def test_distill_below_threshold_returns_none():
    """2 次（<3）→ None。"""
    pipeline = DraftPipeline(min_count=3)
    exps = [_mk_exp("停车场", ["a"], f"s{i}") for i in range(2)]
    assert pipeline.distill(exps) is None


def test_distill_unstable_chain_returns_none():
    """3 次但链不稳定 → None（避免把多意图捏成一条）。"""
    pipeline = DraftPipeline(min_count=3)
    exps = [
        _mk_exp("停车场", ["a"], "s0"),
        _mk_exp("停车场", ["b"], "s1"),
        _mk_exp("停车场", ["c"], "s2"),
    ]
    assert pipeline.distill(exps) is None


def test_distill_no_scene_builds_rule_draft():
    """非模板场景（无 detect_scene 命中）→ 自建草稿。"""
    pipeline = DraftPipeline(min_count=3)
    exps = [_mk_exp("分析视频找猫", ["search_visual", "extract_frames"], f"s{i}")
            for i in range(3)]
    draft = pipeline.distill(exps)
    assert draft is not None
    assert draft.occurrences == 3
    # 自建 name 含意图 slug
    assert draft.draft.name != "surveillance-parking-lpr"
    assert "搜索/检索 visual" in " ".join(derive_workflow_steps(list(draft.chain)))
    assert "## 工作流" in draft.markdown or "## 算法" in draft.markdown


def test_derive_workflow_steps_frame_chain():
    """典型工具链归纳出可读步骤。"""
    chain = ["extract_frames", "transcribe", "analyze_frames", "generate_summary"]
    steps = derive_workflow_steps(chain)
    assert len(steps) == 4
    assert "抽帧采样" in steps
    assert "提取语音转录" in steps


def test_make_skill_name_slug():
    """make_skill_name 生成合法 slug。"""
    name = make_skill_name("分析视频找猫", ["搜索 visual", "抽帧采样"])
    assert name  # 非空
    assert " " not in name  # 无空格（slug 化）
    assert "一-鿿" not in name  # 中文字符 range 正常


def test_group_experiences_aggregates():
    """group_experiences 形态无关分组（dict 或 dataclass 均可）。"""
    groups = group_experiences([
        {"intent": "监控找包", "tool_chain": ["extract_frames"]},
        {"intent": "监控找包", "tool_chain": ["extract_frames"]},
        _mk_exp("监控找包", ["extract_frames"], "s2"),
    ])
    assert groups["监控找包"]["count"] == 3
    assert len(groups["监控找包"]["chains"]) == 1


def test_aggregate_workflow_steps_stable_only():
    """aggregate_workflow_steps 仅保留 >=3 次 + 稳定链。"""
    exps = [
        _mk_exp("监控找包", ["extract_frames", "analyze_frames"], "s0"),
        _mk_exp("监控找包", ["extract_frames", "analyze_frames"], "s1"),
        _mk_exp("监控找包", ["extract_frames", "analyze_frames"], "s2"),
        _mk_exp("稀疏意图", ["a"], "s3"),
    ]
    out = aggregate_workflow_steps(exps)
    assert "监控找包" in out
    assert "稀疏意图" not in out  # 仅 1 次


def test_reuse_generate_skill_return_draft(tmp_path):
    """generate_skill(return_draft=True) 返回 draft 供 distiller 复用。"""
    result = sg.generate_skill("停车场找车牌", tmp_path, return_draft=True)
    assert result is not None
    assert result["ok"] is True
    draft = result.get("draft")
    assert draft is not None
    assert draft.name == "surveillance-parking-lpr"
    # 默认兼容签名（无 return_draft）不破坏既有行为
    result2 = sg.generate_skill("停车场找车牌", tmp_path, overwrite=True)
    assert result2["ok"] is True
    assert "draft" not in result2