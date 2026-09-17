"""SkillAdvisor 建议端点测试（v10.4.0 P0-2）。

背景
----
``src/core/agent/experience.py:356`` 的 ``SkillAdvisor`` 实现了"经验达到阈值 →
建议生成 skill"，但**零生产调用**（只有 ``tests/test_experience.py`` 引用）。
本文件守护 ``GET /api/skills/suggestions`` 这条生产入口：

  1. 无经验 → 空列表（不 500）；
  2. 达到阈值（≥3 次同类 + 稳定链）→ 出现建议且带 draft；
  3. ``VAP_SKILLS_ADVISOR=0`` → 空列表 + disabled 标记（零回归）；
  4. SkillAdvisor 有非测试调用方（防回退）。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.agent.experience import Experience
from src.web.routers import skills as skills_router


def _mk_client() -> TestClient:
    app = FastAPI()
    app.include_router(skills_router.router)
    return TestClient(app)


def _seed_experiences(db_path: Path, intent: str, chain, n: int) -> None:
    from src.core.agent.experience import ExperienceStore

    store = ExperienceStore(db_path)
    for i in range(n):
        store.record(Experience(
            intent=intent,
            tool_chain=list(chain),
            success=True,
            quality_score=0.9,
            timestamp=time.time(),
            session_id=f"s{i}",
        ))


@pytest.fixture()
def fresh_config(tmp_path, monkeypatch):
    """把 CONFIG_DIR 指到临时目录，避免污染真实 config/agent_experiences.db。"""
    monkeypatch.setattr("src.utils.constants.CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_suggestions_empty_when_no_experiences(fresh_config):
    """无经验 → 空列表（不报错）。"""
    client = _mk_client()
    resp = client.get("/api/skills/suggestions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["suggestions"] == []
    assert body["count"] == 0
    assert body["disabled"] is False


def test_suggestions_appear_after_threshold(fresh_config):
    """同类 intent ≥3 次 + 稳定链 → 出现建议且带 draft_skill_md。"""
    _seed_experiences(fresh_config / "agent_experiences.db", "停车场监控",
                      ["extract_frames", "transcribe", "analyze_frames"], 3)
    body = _mk_client().get("/api/skills/suggestions").json()
    assert body["count"] >= 1
    hit = [s for s in body["suggestions"] if "停车场" in s["intent"]]
    assert hit, f"未出现预期建议: {body}"
    assert hit[0]["recommended_tool_chain"] == [
        "extract_frames", "transcribe", "analyze_frames"]
    assert hit[0]["sample_count"] >= 3
    assert hit[0]["has_draft"] is True
    assert hit[0]["draft_skill_md"]


def test_suggestions_below_threshold_are_absent(fresh_config):
    """只有 2 次经验 → 未达阈值，不出现建议。"""
    _seed_experiences(fresh_config / "agent_experiences.db", "临时任务",
                      ["extract_frames"], 2)
    body = _mk_client().get("/api/skills/suggestions").json()
    assert body["suggestions"] == []


def test_suggestions_unstable_chain_is_absent(fresh_config):
    """同类 intent 但 tool_chain 不稳定 → 不出现建议。"""
    db = fresh_config / "agent_experiences.db"
    _seed_experiences(db, "多变任务", ["extract_frames", "a"], 2)
    _seed_experiences(db, "多变任务", ["transcribe", "b"], 2)
    body = _mk_client().get("/api/skills/suggestions").json()
    assert body["suggestions"] == []


def test_suggestions_respects_switch(fresh_config, monkeypatch):
    """VAP_SKILLS_ADVISOR=0 → 空列表 + disabled=True（零回归）。"""
    _seed_experiences(fresh_config / "agent_experiences.db", "停车场监控",
                      ["extract_frames", "transcribe"], 3)
    monkeypatch.setenv("VAP_SKILLS_ADVISOR", "0")
    body = _mk_client().get("/api/skills/suggestions").json()
    assert body["suggestions"] == []
    assert body["disabled"] is True


def test_suggestions_switch_enabled_by_default(fresh_config, monkeypatch):
    """未设环境变量 → 默认启用。"""
    monkeypatch.delenv("VAP_SKILLS_ADVISOR", raising=False)
    body = _mk_client().get("/api/skills/suggestions").json()
    assert body["disabled"] is False


def test_min_count_param_is_honored(fresh_config):
    """min_count=5 时，只有 3 次的经验不应出现。"""
    _seed_experiences(fresh_config / "agent_experiences.db", "停车场监控",
                      ["extract_frames", "transcribe", "analyze_frames"], 3)
    body = _mk_client().get("/api/skills/suggestions?min_count=5").json()
    assert body["suggestions"] == []


# --------------------------------------------------------------------------
# 防回退：源码级契约
# --------------------------------------------------------------------------


def test_skill_advisor_has_production_caller():
    """反向断言：SkillAdvisor 必须在 src/ 里有非测试调用方。"""
    root = Path(__file__).resolve().parents[1] / "src"
    callers = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "SkillAdvisor" in text and "class SkillAdvisor" not in text:
            callers.append(path.name)
    assert callers, "SkillAdvisor 又变成零调用模块了（P0-2 接线断裂）"


def test_suggestions_endpoint_registered():
    """端点必须注册在 router 上（路径固定，前端按此调用）。"""
    paths = {route.path for route in skills_router.router.routes}
    assert "/api/skills/suggestions" in paths
