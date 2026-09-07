"""tests/test_skills_router.py — Skills 管理 router 的 API 层测试。

覆盖 src/web/routers/skills.py：
    GET  /api/skills
    POST /api/skills/{name}/toggle
    POST /api/skills/generate

策略（不真实调付费 LLM，generate 是纯规则模板）：
    - 用 tmp_path 造真实 SKILL.md 目录树，把 src.skills.loader._default_root
      与 src.skills.state._state_path monkeypatch 到 tmp_path，读写真实文件
    - skill_generator 的写目录由 CONFIG_DIR/skills 决定 -> monkeypatch
      src.core.skill_generator 的 skills_dir 侧写入逻辑为 tmp_path
    - toggle 幂等：同一 skill 开关两次
    - 每测试后清理 src.skills 相关模块缓存（可选，state 读文件即可）
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch 必须先于其它 C 扩展加载（Windows DLL 顺序铁律，与项目其它测试一致）
try:
    import torch  # noqa: F401
except OSError:
    torch = None

from fastapi.testclient import TestClient  # noqa: E402

import src.skills.loader as skills_loader
import src.skills.state as skills_state
import src.web.routers.skills as skills_router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    """独立 FastAPI app（仅挂 skills router）+ tmp_path 隔离技能目录。"""
    # 1) 技能目录根 -> tmp_path/skills
    skills_root = tmp_path / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(skills_loader, "_default_root", lambda: skills_root)

    # 2) 启用状态文件 -> tmp_path/skills_state.json（原子写，OS 级替换）
    monkeypatch.setattr(skills_state, "_state_path", lambda: tmp_path / "skills_state.json")

    # 3) skill_generator 写入目录 -> tmp_path/gen_skills（路由动态 import，
    #    monkeypatch 其模块级 _SCENE_TEMPLATES 不起作用，改 patch 写目录来源
    #    CONFIG_DIR/skills 字符串路径，见 generate fixture）
    gen_skills_dir = tmp_path / "gen_skills"
    gen_skills_dir.mkdir(parents=True, exist_ok=True)

    # 4) 路由层缓存：toggle/generate 都是函数内动态 import，天然读最新，
    #    但 src.skills 模块级 state 函数已指向 tmp_path，无需额外清理。

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(skills_router.router)
    with TestClient(app) as c:
        c._gen_skills_dir = gen_skills_dir
        yield c


def _write_skill(root: Path, name: str, *, enabled_marker: str | None = None) -> Path:
    """写一个标准 SKILL.md 目录。enabled_marker 仅为可读性，实际状态走 state 文件。"""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    md = d / "SKILL.md"
    md.write_text(
        f"---\n"
        f"name: {name}\n"
        f"description: 测试技能 {name}\n"
        f"triggers: 关键词1,关键词2\n"
        f"---\n"
        f"\n# {name}\n\n测试正文。\n",
        encoding="utf-8",
    )
    return md


# ---------------------------------------------------------------------------
# GET /api/skills
# ---------------------------------------------------------------------------


def test_list_skills_empty(client, tmp_path):
    """空目录 -> skills 空数组，count 0。"""
    r = client.get("/api/skills")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["skills"] == []
    assert body["count"] == 0


def test_list_skills_returns_loaded(client, tmp_path):
    """有 skill 目录 -> 返回字段完整（name/description/triggers/path/enabled）。"""
    _write_skill(tmp_path / "skills", "demo-skill")
    r = client.get("/api/skills")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1
    s = body["skills"][0]
    assert s["name"] == "demo-skill"
    assert s["description"] == "测试技能 demo-skill"
    assert s["triggers"] == ["关键词1", "关键词2"]
    assert s["path"].endswith("demo-skill" + "SKILL.md") or "demo-skill" in s["path"]
    # 状态文件缺失 -> 默认 enabled True
    assert s["enabled"] is True


def test_list_skills_reflects_disabled_state(client, tmp_path):
    """state 文件标记 disabled -> enabled False。"""
    _write_skill(tmp_path / "skills", "flip-skill")
    (tmp_path / "skills_state.json").write_text(
        '{"flip-skill": false}', encoding="utf-8",
    )
    r = client.get("/api/skills")
    assert r.status_code == 200, r.text
    s = r.json()["skills"][0]
    assert s["name"] == "flip-skill"
    assert s["enabled"] is False


def test_list_skills_skips_invalid_skill(client, tmp_path):
    """缺 frontmatter / name 不一致 -> 被 loader 跳过（不 500）。"""
    bad = tmp_path / "skills" / "bad-one"
    bad.mkdir(parents=True, exist_ok=True)
    # 缺 frontmatter 块
    (bad / "SKILL.md").write_text("# no frontmatter\n", encoding="utf-8")
    r = client.get("/api/skills")
    assert r.status_code == 200, r.text
    assert r.json()["skills"] == []
    assert r.json()["count"] == 0


# ---------------------------------------------------------------------------
# POST /api/skills/{name}/toggle
# ---------------------------------------------------------------------------


def test_toggle_enable(client, tmp_path):
    """toggle 启用 -> ok True + state 文件落盘。"""
    _write_skill(tmp_path / "skills", "parking")
    r = client.post("/api/skills/parking/toggle", json={"enabled": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["name"] == "parking"
    assert body["enabled"] is True
    state = (tmp_path / "skills_state.json").read_text(encoding="utf-8")
    assert '"parking": true' in state


def test_toggle_disable(client, tmp_path):
    """toggle 禁用 -> ok True + state 文件落盘。"""
    _write_skill(tmp_path / "skills", "vehicle")
    r = client.post("/api/skills/vehicle/toggle", json={"enabled": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["enabled"] is False
    state = (tmp_path / "skills_state.json").read_text(encoding="utf-8")
    assert '"vehicle": false' in state


def test_toggle_idempotent(client, tmp_path):
    """同 skill 开关两次 -> 状态跟随最后一次。"""
    _write_skill(tmp_path / "skills", "flip")
    client.post("/api/skills/flip/toggle", json={"enabled": False})
    r = client.post("/api/skills/flip/toggle", json={"enabled": True})
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "name": "flip", "enabled": True}
    state = (tmp_path / "skills_state.json").read_text(encoding="utf-8")
    assert '"flip": true' in state


def test_toggle_empty_name_raises(client):
    """空 name -> set_enabled_state 抛 ValueError -> ok False + error。

    FastAPI 路由 `/{name}/toggle` 中 name 为空串 `//toggle` 不会命中
    （Starlette 按空段匹配，`/api/skills//toggle` 为 404）。空 name 的
    ValueError 分支只能通过直接调用路由函数覆盖。
    """
    req = type("Req", (), {"enabled": True})()
    result = skills_router.toggle_skill("", req)
    assert result["ok"] is False
    assert "不能为空" in result["error"]


def test_toggle_internal_error(client, monkeypatch):
    """set_enabled_state 异常 -> toggle 通用 except 分支。"""
    import types

    def _boom(name, enabled):
        raise RuntimeError("state write boom")

    def _fake_set_enabled(name, enabled):
        return _boom(name, enabled)

    # toggle_skill 在函数内动态 import src.skills.set_enabled_state;
    # construct 一个最小假模块塞进 sys.modules,保留 load_skills 供其它场景
    fake = types.ModuleType("src.skills")
    fake.set_enabled_state = _fake_set_enabled
    try:
        from src.skills import load_skills as _real_load  # noqa: F401
        fake.load_skills = _real_load
    except Exception:
        pass
    monkeypatch.setitem(sys.modules, "src.skills", fake)
    r = client.post("/api/skills/anything/toggle", json={"enabled": True})
    assert r.status_code == 200, r.text
    body = r.json()
    # router 的 except 分支返回 {"ok": False, "error": str(e)}
    assert body["ok"] is False
    assert body["error"] == "state write boom"


# ---------------------------------------------------------------------------
# POST /api/skills/generate
# ---------------------------------------------------------------------------


def test_generate_parking(client, monkeypatch):
    """生成停车场技能 -> ok True + 落盘到 gen 目录。"""
    gen_dir = client._gen_skills_dir
    # router 动态 import src.core.skill_generator.generate_skill + 读 CONFIG_DIR/skills;
    # CONFIG_DIR 是 src.utils.constants 的模块级常量, router 函数内引用, patch 该模块
    monkeypatch.setattr(
        "src.utils.constants.CONFIG_DIR", str(gen_dir.parent),
    )
    r = client.post("/api/skills/generate", json={"text": "停车场车牌识别"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["scene"] == "parking"
    assert body["skill_name"] == "surveillance-parking-lpr"
    assert Path(body["path"]).exists()


def test_generate_unknown_scene(client):
    """无匹配关键词 -> detect_scene None -> ok False + error。"""
    r = client.post("/api/skills/generate", json={"text": "完全无关的文本"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    assert "error" in body


def test_generate_overwrite_conflict(client, monkeypatch):
    """同名 skill 已存在且 overwrite=False -> FileExistsError -> ok False。"""
    gen_dir = client._gen_skills_dir
    # router 计算的 skills_dir = CONFIG_DIR/skills;预置文件须放在该目录(非 gen_dir)
    skills_root = gen_dir.parent / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    target = skills_root / "surveillance-parking-lpr" / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(
        "src.utils.constants.CONFIG_DIR", str(gen_dir.parent),
    )
    r = client.post("/api/skills/generate", json={"text": "停车场车牌识别"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    assert "overwrite" in body["error"] or "已存在" in body["error"]


def test_generate_overwrite_true(client, monkeypatch):
    """overwrite=True -> 覆盖生成成功。"""
    gen_dir = client._gen_skills_dir
    skills_root = gen_dir.parent / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    target = skills_root / "surveillance-parking-lpr" / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(
        "src.utils.constants.CONFIG_DIR", str(gen_dir.parent),
    )
    r = client.post(
        "/api/skills/generate",
        json={"text": "停车场车牌识别", "overwrite": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["scene"] == "parking"


def test_generate_import_failure(client, monkeypatch):
    """src.core.skill_generator import 失败 -> 通用 except 分支。"""
    monkeypatch.setitem(sys.modules, "src.core.skill_generator", None)
    r = client.post("/api/skills/generate", json={"text": "停车场"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    assert body["error"]