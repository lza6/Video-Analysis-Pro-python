"""skills loader 单测：内置加载 / 启用切换 / 坏 frontmatter 跳过。

用 tmp_path 隔离，避免污染项目 config/skills。
"""
from __future__ import annotations

from pathlib import Path

from src.skills import load_skills, set_enabled_state
from src.skills.state import _state_path, get_enabled_for, get_enabled_state


def _write_skill(
    root: Path, name: str, description: str, triggers: str = "", enabled_default: bool = True
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    fm = "---\nname: " + name + "\n"
    fm += "description: " + description + "\n"
    if triggers:
        fm += "triggers: " + triggers + "\n"
    fm += "---\n\n# " + name + "\n"
    (skill_dir / "SKILL.md").write_text(fm, encoding="utf-8")
    return skill_dir


def test_load_builtin_skill(tmp_path: Path, monkeypatch) -> None:
    # 隔离 state.json 到 tmp，避免读写项目真实 config
    monkeypatch.setattr("src.skills.state._state_path", lambda: tmp_path / "skills_state.json")
    monkeypatch.setattr("src.skills.loader.get_enabled_state", lambda: {})

    root = tmp_path / "skills"
    _write_skill(
        root,
        "my-skill",
        "测试用 skill",
        "a,b,c",
    )
    skills = load_skills(root)
    assert len(skills) == 1
    s = skills[0]
    assert s.name == "my-skill"
    assert s.description == "测试用 skill"
    assert s.triggers == ("a", "b", "c")
    assert s.enabled is True  # 缺失默认 True
    assert s.path.name == "SKILL.md"


def test_enabled_toggle_persists(tmp_path: Path, monkeypatch) -> None:
    state_file = tmp_path / "skills_state.json"
    monkeypatch.setattr("src.skills.state._state_path", lambda: state_file)
    # 用真实 get_enabled_state（它内部读 state_file，已隔离到 tmp_path）
    monkeypatch.setattr("src.skills.loader.get_enabled_state", get_enabled_state)

    root = tmp_path / "skills"
    _write_skill(root, "toggleable", "可切换")
    skills = load_skills(root)
    assert skills[0].enabled is True

    set_enabled_state("toggleable", False)
    skills_after = load_skills(root)
    assert skills_after[0].enabled is False
    assert get_enabled_for("toggleable") is False
    # 文件确实落盘
    assert state_file.exists()
    # 切回
    set_enabled_state("toggleable", True)
    assert load_skills(root)[0].enabled is True


def test_bad_frontmatter_skipped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.skills.state._state_path", lambda: tmp_path / "skills_state.json")
    monkeypatch.setattr("src.skills.loader.get_enabled_state", lambda: {})

    root = tmp_path / "skills"

    # 1) name 与目录名不一致
    bad_dir = root / "wrong-dir"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "SKILL.md").write_text(
        "---\nname: not-matching\ndescription: 有名字但不匹配\n---\n# x\n",
        encoding="utf-8",
    )
    # 2) 缺 description
    no_desc = root / "no-desc"
    no_desc.mkdir(parents=True, exist_ok=True)
    (no_desc / "SKILL.md").write_text(
        "---\nname: no-desc\n---\n# x\n", encoding="utf-8"
    )
    # 3) 完全无 frontmatter
    no_fm = root / "no-fm"
    no_fm.mkdir(parents=True, exist_ok=True)
    (no_fm / "SKILL.md").write_text("纯正文，无 frontmatter", encoding="utf-8")
    # 4) 合法 skill 作为对照
    _write_skill(root, "good-one", "合法 skill")

    skills = load_skills(root)
    names = [s.name for s in skills]
    assert names == ["good-one"]
