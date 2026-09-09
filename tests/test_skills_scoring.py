"""scoring 棘轮测试。

覆盖：
  1. score_skill 评分卡（结构/触发/可执行/安全）
  2. 高安全风险 skill → 安全分被扣
  3. 棘轮：新 skill 首次落盘；重跑同 skill 只留改进，退步 rejected
  4. 原子写 skills_scores.json（tmp 重定向隔离）

纯规则，无 LLM。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.skills import scoring
from src.skills.scoring import (
    ScoreCard,
    apply_ratchet,
    get_score,
    load_scores,
    score_skill,
)


@pytest.fixture(autouse=True)
def _isolate_scores_path(tmp_path, monkeypatch):
    """把 skills_scores.json 重定向到 tmp（测试隔离，避免污染项目 config）。"""
    scores_file = tmp_path / "skills_scores.json"
    monkeypatch.setattr(scoring, "_scores_path", lambda p=scores_file: p)
    monkeypatch.setattr("src.skills.scoring._scores_path",
                        lambda p=scores_file: p)
    yield scores_file


def _write_full_skill(root: Path, name: str, desc: str, body: str = "") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {desc}\ntriggers: {name},测试\n---\n"
    content += f"\n# {name}\n\n## 适用场景\n\n## 算法\n{body}\n\n"
    content += "## 参数\nsample_fps=1.0\n\n## 何时用\n\n## 何时不该用\n\n## 降级行为\n"
    (d / "SKILL.md").write_text(content, encoding="utf-8")
    return d


def _write_rich_skill(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    content = (
        f"---\nname: {name}\ndescription: 视频分析监控工作流\n"
        "triggers: 监控,视频,分析\n---\n"
        f"# {name}\n\n## 适用场景\n监控视频分析。\n\n"
        "## 算法\n1. extract_frames\n2. analyze_frames\n3. generate_summary\n\n"
        "```bash\n./run.sh\n```\n\n```python\nprint('ok')\n```\n\n"
        "## 参数\nsample_fps=1.0, threshold=0.6\n\n"
        "## 何时用\n用户要监控分析。\n\n## 何时不该用\n\n## 降级行为\n"
    )
    (d / "SKILL.md").write_text(content, encoding="utf-8")
    return d


def test_score_skill_structure_and_trigger(tmp_path):
    """评分卡结构/触发分。"""
    d = _write_rich_skill(tmp_path, "rich-skill")
    card = score_skill(d)
    assert isinstance(card, ScoreCard)
    assert card.name == "rich-skill"
    # 完整 frontmatter + 全部正文区块 → 结构分高
    assert card.structure >= 26
    # 3 个 triggers + 描述含关键词 → 触发分高
    assert card.trigger >= 10
    # 可执行分（代码块 + 列表 + 参数）
    assert card.executable >= 20
    # 安全分：干净 skill 满分
    assert card.security == 20.0
    assert card.total > 70


def test_score_skill_marks_security_deduction(tmp_path):
    """含危险命令的 skill → 安全分被扣。"""
    d = _mk_dangerous(tmp_path, "evil")
    card = score_skill(d)
    assert card.security < 20.0


def _mk_dangerous(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    content = (
        f"---\nname: {name}\ndescription: 危险 skill\n---\n"
        f"# {name}\n\n## 适用场景\n\n## 算法\nos.system('rm -rf /')\n\n"
        "## 参数\n\n## 何时用\n\n## 何时不该用\n\n## 降级行为\n"
    )
    (d / "SKILL.md").write_text(content, encoding="utf-8")
    return d


def test_score_skill_minimal(tmp_path):
    """最小 SKILL.md（仅 frontmatter）→ 结构/触发/可执行分低。"""
    d = tmp_path
    d.mkdir(parents=True, exist_ok=True)
    p = d / "minimal"
    p.mkdir()
    (p / "SKILL.md").write_text(
        "---\nname: minimal\ndescription: minimal\n---\n", encoding="utf-8")
    card = score_skill(p)
    assert card.structure < 20
    # minimal 无正文，可执行分应为 0（"sample_fps"="1" 之类不出现即可）
    assert card.executable < 10
    assert card.trigger < 12


def test_apply_ratchet_new_skill(tmp_path):
    """新 skill 首次落盘 → ratchet=new。"""
    d = _write_full_skill(tmp_path, "brand-new", "全新技能")
    result = apply_ratchet(d)
    assert result["ratchet"] == "new"
    assert result["score"] > 0
    scores = load_scores()
    assert "brand-new" in scores
    assert scores["brand-new"]["status"] == "ok"


def test_apply_ratchet_improved(tmp_path):
    """同 skill 重跑高分（>=旧分）→ 更新落盘。"""
    d = _write_full_skill(tmp_path, "improve-me", "视频分析监控工作流")
    apply_ratchet(d)
    old = get_score("improve-me")
    assert old is not None
    # 重跑同内容（分相同或略高）
    result = apply_ratchet(d)
    assert result["ratchet"] in ("new", "improved")
    new = get_score("improve-me")
    assert new >= old


def test_apply_ratchet_regression_rejected(tmp_path):
    """同 skill 重跑低分（内容变差）→ rejected，保留旧分。"""
    d1 = _write_rich_skill(tmp_path, "regress")
    apply_ratchet(d1)
    good = get_score("regress")
    assert good is not None
    # 用瘦弱版本覆盖（变差）
    d2 = tmp_path / "regress"
    d2.mkdir(parents=True, exist_ok=True)
    (d2 / "SKILL.md").write_text(
        "---\nname: regress\ndescription: x\n---\n# x\n", encoding="utf-8")
    result = apply_ratchet(d2)
    assert result["ratchet"] == "rejected"
    # 旧分保留
    assert get_score("regress") == good
    scores = load_scores()
    assert scores["regress"]["status"] == "rejected"
    assert scores["regress"]["last_good_score"] == good


def test_scores_file_atomic_json(tmp_path):
    """skills_scores.json 是合法 JSON 且原子写入。"""
    d = _write_full_skill(tmp_path, "json-check", "测试")
    apply_ratchet(d)
    raw = (tmp_path / "skills_scores.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert "json-check" in data


def test_get_score_missing_returns_none():
    """无记录 skill → None。"""
    assert get_score("not-exist") is None