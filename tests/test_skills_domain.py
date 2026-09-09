"""B-P2-2 领域 skills（电商/PPT/营销）测试。

覆盖（契约见 batch B-P2-2）：
  ① 4 个 SKILL.md frontmatter 完整 + spectre 扫描 clean
  ② loader 能加载且全部 enabled
  ③ roster 按意图命中（"上架"→ecommerce-merchant / "做PPT"→ppt-deck /
     "小红书文案"→marketing-publish / "比价"→ecommerce-shopper）
  ④ role_template 三个新角色可构建（merchant / shopper / marketer）
  ⑤ Mock 守付费红线：merchant/marketing 的写步骤在无 VAP_ALLOW_WRITE_*
     时明确走审批/不真实执行（断言 SKILL.md 内容含审批标注）

纯规则，无 LLM，无真实外部调用。
"""
from __future__ import annotations

import os
from pathlib import Path

from src.core.subagent.role_template import (
    MARKETER_ROLE,
    MERCHANT_ROLE,
    SHOPPER_ROLE,
    _DOMAIN_WRITE_KEYS,
    build_domain_role,
)
from src.skills.loader import load_skills
from src.skills.roster import resolve_skills_for_intent
from src.skills.spectre import scan_skill
from src.utils.constants import CONFIG_DIR

SKILLS_ROOT = Path(CONFIG_DIR) / "skills"

#: 本批新增的 4 个领域 skill 名
DOMAIN_SKILL_NAMES = (
    "ecommerce-merchant",
    "ecommerce-shopper",
    "ppt-deck",
    "marketing-publish",
)


def _load_domain_skill(name: str):
    """从 config/skills 加载指定 skill，未找到返回 None。"""
    for s in load_skills(SKILLS_ROOT):
        if s.name == name:
            return s
    return None


def _read_skill_md(name: str) -> str:
    """读取指定 skill 的 SKILL.md 全文。"""
    return (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# ① frontmatter 完整 + spectre clean
# ---------------------------------------------------------------------------


def test_all_domain_skills_have_full_frontmatter() -> None:
    """4 个 SKILL.md 均有 name/description/triggers。"""
    from src.skills.loader import FRONTMATTER_RE

    for name in DOMAIN_SKILL_NAMES:
        raw = _read_skill_md(name)
        m = FRONTMATTER_RE.match(raw)
        assert m, f"{name}: 缺少 frontmatter 块"
        fm = m.group(1)
        assert "name: " in fm and "description: " in fm, f"{name}: 缺 name/description"
        assert "triggers:" in fm, f"{name}: 缺 triggers"


def test_all_domain_skills_spectre_clean() -> None:
    """新增 4 个 skill 全部 spectre 扫描 clean（无 security_warning）。"""
    for name in DOMAIN_SKILL_NAMES:
        report = scan_skill(SKILLS_ROOT / name)
        assert report.clean is True, (
            f"{name}: 扫描非 clean {[(f.code, f.severity) for f in report.findings]}"
        )


def test_domain_skills_no_security_warning_via_loader() -> None:
    """经 loader 加载后 security_warning=False 且 enabled=True。"""
    loaded = {s.name: s for s in load_skills(SKILLS_ROOT)}
    for name in DOMAIN_SKILL_NAMES:
        s = loaded.get(name)
        assert s is not None, f"{name} 未被 loader 加载"
        assert s.security_warning is False, f"{name} 出现安全警告"
        assert s.enabled is True, f"{name} 应默认 enabled"


# ---------------------------------------------------------------------------
# ② loader 能加载且全部 enabled
# ---------------------------------------------------------------------------


def test_loader_loads_all_domain_skills() -> None:
    """loader 能加载全部 4 个领域 skill。"""
    names = {s.name for s in load_skills(SKILLS_ROOT)}
    for n in DOMAIN_SKILL_NAMES:
        assert n in names, f"{n} 未加载"


def test_all_loader_skills_enabled() -> None:
    """所有 loader 返回的 skill 均 enabled（含新 skill）。"""
    for s in load_skills(SKILLS_ROOT):
        assert s.enabled is True, f"{s.name} 未 enabled"


# ---------------------------------------------------------------------------
# ③ roster 按意图命中
# ---------------------------------------------------------------------------


def test_roster_hit_ecommerce_merchant() -> None:
    """'上架' 意图 → ecommerce-merchant。"""
    names = {s.name for s in resolve_skills_for_intent("上架", SKILLS_ROOT)}
    assert "ecommerce-merchant" in names


def test_roster_hit_ppt_deck() -> None:
    """'做PPT' 意图 → ppt-deck。"""
    names = {s.name for s in resolve_skills_for_intent("做PPT", SKILLS_ROOT)}
    assert "ppt-deck" in names


def test_roster_hit_marketing_publish() -> None:
    """'小红书文案' 意图 → marketing-publish。"""
    names = {s.name for s in resolve_skills_for_intent("小红书文案", SKILLS_ROOT)}
    assert "marketing-publish" in names


def test_roster_hit_ecommerce_shopper() -> None:
    """'比价' 意图 → ecommerce-shopper。"""
    names = {s.name for s in resolve_skills_for_intent("比价", SKILLS_ROOT)}
    assert "ecommerce-shopper" in names


def test_roster_hit_ppt_deck_lowercase_ascii() -> None:
    """'帮我做个路演ppt'（含小写 ascii ppt）→ ppt-deck。"""
    names = {s.name for s in resolve_skills_for_intent(
        "帮我做个路演ppt", SKILLS_ROOT)}
    assert "ppt-deck" in names


# ---------------------------------------------------------------------------
# ④ role_template 三个新角色可构建
# ---------------------------------------------------------------------------


def test_build_domain_role_all_keys() -> None:
    """merchant/shopper/marketer 均可构建。"""
    for key, expected_display in (
        ("merchant", "merchant"),
        ("shopper", "shopper"),
        ("marketer", "marketer"),
    ):
        role = build_domain_role(key)
        assert role.display_name == expected_display
        assert role.persona  # persona 非空
        assert role.provider == "inprocess"


def test_build_domain_role_unknown_key_raises() -> None:
    """未知角色键 → ValueError。"""
    import pytest

    with pytest.raises(ValueError):
        build_domain_role("nope")


def test_domain_roles_exclude_write_tools_by_default() -> None:
    """写工具不在读白名单内 → 必须经 scope_guard 审批。

    permits_wildcard(写工具名)=False（allow 白名单只含 get_*/search_*/list_*
    等读前缀通配）。写工具（publish/create/update/send）不匹配任何读前缀。
    """
    for key in ("merchant", "shopper", "marketer"):
        role = build_domain_role(key)
        for w in ("publish_post", "create_job", "update_price", "send_message"):
            assert role.tool_filter.permits_wildcard(w) is False, (
                f"{key}: {w} 不应在读白名单内（写操作需审批）"
            )


def test_domain_roles_permit_read_tools() -> None:
    """读工具匹配读前缀通配 → 直接放行。"""
    for key in ("merchant", "shopper", "marketer"):
        role = build_domain_role(key)
        for r in ("get_price", "search_sku", "list_orders", "compare_sku"):
            assert role.tool_filter.permits_wildcard(r) is True, (
                f"{key}: {r} 应在读白名单内"
            )


def test_domain_role_specs_align_with_persona() -> None:
    """MERCHANT_ROLE / SHOPPER_ROLE / MARKETER_ROLE 常量可用。"""
    assert MERCHANT_ROLE["display_name"] == "merchant"
    assert SHOPPER_ROLE["display_name"] == "shopper"
    assert MARKETER_ROLE["display_name"] == "marketer"


# ---------------------------------------------------------------------------
# ⑤ Mock 守付费红线：写步骤在无 VAP_ALLOW_WRITE_* 时走审批
# ---------------------------------------------------------------------------


def test_merchant_skill_requires_approval_for_writes() -> None:
    """merchant SKILL.md 明确标注写操作需审批 + 仅 Mock。"""
    md = _read_skill_md("ecommerce-merchant")
    assert "审批" in md and "scope_guard" in md
    assert "Mock" in md
    # 写操作关键词出现即应伴随审批标注
    for kw in ("上架", "改价", "库存"):
        assert kw in md


def test_merchant_skill_writes_not_executed_without_allow() -> None:
    """merchant 写操作（上架/改价/库存）需审批放行后才执行。

    断言 SKILL.md 工作流在无 VAP_ALLOW_WRITE_* 时止步于「提交审批」。
    """
    md = _read_skill_md("ecommerce-merchant")
    assert "VAP_ALLOW_WRITE" in md or "审批" in md
    assert "提交审批" in md
    assert "未获批准不执行任何写操作" in md


def test_marketing_skill_requires_approval_for_publish() -> None:
    """marketing SKILL.md 明确发布需 VAP_ALLOW_WRITE_PUBLISH + 人工审批，仅 Mock。"""
    md = _read_skill_md("marketing-publish")
    assert "VAP_ALLOW_WRITE_PUBLISH" in md
    assert "人工审批" in md
    assert "Mock" in md
    assert "绝不真实调用" in md


def test_marketing_skill_publish_not_executed_without_allow() -> None:
    """marketing 发布在无放行时止步于「分发清单」。"""
    md = _read_skill_md("marketing-publish")
    assert "分发清单" in md
    assert "未获批时流程止步于" in md


def test_shopper_skill_is_readonly() -> None:
    """shopper SKILL.md 明确纯读 + 仅本地 Mock。"""
    md = _read_skill_md("ecommerce-shopper")
    assert "纯读" in md or "读操作" in md
    assert "Mock" in md
    assert "绝不真实访问" in md


def test_ppt_skill_no_external_publish() -> None:
    """ppt-deck SKILL.md 仅本地渲染，不上传外部平台。"""
    md = _read_skill_md("ppt-deck")
    assert "不上传" in md or "本地渲染" in md
    assert "无 LLM" in md  # 无 LLM 也能出骨架


def test_env_publish_switch_defaults_disabled() -> None:
    """VAP_ALLOW_WRITE_PUBLISH 默认不置 1（付费红线默认关）。"""
    assert os.environ.get("VAP_ALLOW_WRITE_PUBLISH", "0") != "1"


def test_domain_write_keys_exist() -> None:
    """领域写关键词元组非空且含 publish/create。"""
    assert "publish" in _DOMAIN_WRITE_KEYS
    assert "create" in _DOMAIN_WRITE_KEYS
