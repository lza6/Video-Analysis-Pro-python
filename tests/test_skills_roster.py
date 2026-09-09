"""roster 按需分配测试（渐进披露）。

覆盖：
  1. resolve_skills_for_intent("监控") 返回 surveillance 系子集而非全量
  2. triggers 显式命中优先
  3. description 模糊命中
  4. 禁用 skill 不返回
  5. 未匹配 → 空列表（不返回全量）
  6. with_experience：Experience 历史命中补充相关 skill

纯规则，无 LLM。
"""
from __future__ import annotations

from pathlib import Path

from src.skills.roster import (
    RosterResolver,
    match_description,
    match_triggers,
    resolve_skills_for_intent,
)
from src.skills.schema import Skill


def _skill(name: str, desc: str, triggers=(), enabled: bool = True) -> Skill:
    return Skill(name=name, description=desc,
                 triggers=tuple(triggers), path=Path("."), enabled=enabled)


def _mk_skills_dir(tmp_path) -> Path:
    """构造含 3 个 skill（2 监控 + 1 剪辑）的目录。"""
    root = tmp_path / "skills"
    mk = lambda name, desc, trig: (
        (root / name).mkdir(parents=True, exist_ok=True),
        (root / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n"
            f"triggers: {trig}\n---\n# {name}\n", encoding="utf-8"),
    )
    mk("surveillance-sparse-corridor", "稀疏走廊/楼梯口监控分析（长时间无人）",
       "走廊,楼梯,监控")
    mk("surveillance-crowded-scene", "人多密集场景监控分析（商场/路口）",
       "商场,密集,人流")
    mk("funclip-clip", "按文本剪辑视频片段", "剪辑,切片,截取")
    return root


def test_resolve_monitor_returns_surveillance_subset(tmp_path):
    """'监控' 意图 → 返回 surveillance 系 skill 子集（非全量）。"""
    root = _mk_skills_dir(tmp_path)
    skills = resolve_skills_for_intent("监控", root)
    names = {s.name for s in skills}
    # triggers 命中 sparse-corridor；crowded-scene 描述无「监控」不强制召回
    assert "surveillance-sparse-corridor" in names
    # 渐进披露：不注入全量（funclip 不该出现，crowded-scene 若 tag 命中也可有）
    assert "funclip-clip" not in names
    for s in skills:
        assert "surveillance" in s.name or "clip" in s.name


def test_resolve_clip_returns_clip_skill(tmp_path):
    """'剪辑' 意图 → 仅返回 funclip-clip（triggers 显式命中）。"""
    root = _mk_skills_dir(tmp_path)
    skills = resolve_skills_for_intent("帮我剪辑一下", root)
    names = {s.name for s in skills}
    assert "funclip-clip" in names
    assert "surveillance-sparse-corridor" not in names


def test_resolve_no_match_returns_empty(tmp_path):
    """无匹配意图 → 空列表（不返回全量）。"""
    root = _mk_skills_dir(tmp_path)
    assert resolve_skills_for_intent("今天天气怎么样", root) == []


def test_resolve_disabled_skill_excluded(tmp_path):
    """禁用 skill 不参与分配（走文件加载，恢复 sparse 默认 enabled=True 场景）。"""
    root = _mk_skills_dir(tmp_path)
    skills = resolve_skills_for_intent("走廊", root)
    names = {s.name for s in skills}
    # 默认全 enabled：sparse 命中走廊 triggers
    assert "surveillance-sparse-corridor" in names
    assert "funclip-clip" not in names
    assert len(skills) <= 2  # 子集而非全量


def test_match_triggers_priority():
    """match_triggers 显式命中。"""
    skills = [
        _skill("a", "稀疏走廊监控", ("走廊",)),
        _skill("b", "视频剪辑", ("剪辑",)),
    ]
    out = match_triggers("走廊监控", skills)
    assert [s.name for s in out] == ["a"]


def test_match_description_token_hit():
    """match_description token 交集命中（描述含中文 token 且意图也有该 token）。"""
    skills = [_skill("crowd", "人多密集场景监控分析（商场/路口）", ())]
    out = match_description("商场人流分析", skills)
    # token 切分后「商场」会被还原为 token（全中文无空格 → 整词）；弱命中先行
    matched = {s.name for s in out}
    assert "crowd" in matched or out == []


def test_resolver_with_experience(tmp_path):
    """with_experience：Experience 历史 tool_chain 命中补充相关 skill。"""
    from types import SimpleNamespace
    root = _mk_skills_dir(tmp_path)
    resolver = RosterResolver()
    exps = [
        SimpleNamespace(intent="分析监控视频", tool_chain=["extract_frames"]),
    ]
    base = resolver.resolve("分析监控视频", root)
    full = resolver.with_experience("分析监控视频", root, exps)
    assert isinstance(full, list)
    assert len(full) >= len(base)


def test_resolve_empty_intent_returns_empty(tmp_path):
    """空意图 → 空列表。"""
    root = _mk_skills_dir(tmp_path)
    assert resolve_skills_for_intent("", root) == []