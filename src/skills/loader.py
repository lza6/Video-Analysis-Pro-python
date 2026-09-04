"""Skill 加载器：递归扫描 root/<name>/SKILL.md，解析 frontmatter。

优先用 pyyaml（项目已有依赖 chromadb 间接带）；不可用则降级手写解析。
frontmatter 带 BOM 剥离（open encoding="utf-8-sig"）。

参考 hermes 校验规则：
- frontmatter 必填 name + description
- name 必须等于所在目录名，否则跳过并日志告警
- triggers 可选（逗号分隔字符串 → tuple）
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.skills.schema import MAX_DESCRIPTION_LEN, Skill
from src.skills.state import get_enabled_state
from src.utils.constants import CONFIG_DIR

logger = logging.getLogger(__name__)

SKILLS_DIR_NAME = "skills"
SKILL_FILENAME = "SKILL.md"
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")

try:
    import yaml  # type: ignore[import-untyped]

    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - 降级路径
    _YAML_AVAILABLE = False


def _default_root() -> Path:
    return Path(CONFIG_DIR) / SKILLS_DIR_NAME


def _parse_frontmatter(fm_text: str) -> dict[str, Any]:
    """解析 frontmatter 文本为 dict。优先 yaml，降级手写 key:value。"""
    if _YAML_AVAILABLE:
        try:
            data = yaml.safe_load(fm_text)
            if isinstance(data, dict):
                return data
            if data is None:
                return {}
            logger.warning("frontmatter 顶层非 dict（%s），降级手写解析", type(data).__name__)
        except yaml.YAMLError as exc:
            logger.warning("yaml 解析失败，降级手写：%s", exc)

    # 手写 key:value 降级解析（不支持嵌套/列表块，满足本场景）
    result: dict[str, Any] = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = KV_RE.match(line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        # 去掉两端引号
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        result[key] = val
    return result


def _normalize_triggers(raw: Any) -> tuple[str, ...]:
    """把 triggers 字段规范为 tuple[str, ...]。

    接受：tuple/list/逗号分隔字符串；其它返回空 tuple。
    """
    if isinstance(raw, (list, tuple)):
        return tuple(str(x).strip() for x in raw if str(x).strip())
    if isinstance(raw, str):
        return tuple(s.strip() for s in raw.split(",") if s.strip())
    return ()


def _build_skill(skill_dir: Path, fm: dict[str, Any], md_path: Path, enabled: bool) -> Skill | None:
    """根据 frontmatter dict 构建 Skill。失败返回 None。"""
    name = str(fm.get("name", "")).strip()
    description = str(fm.get("description", "")).strip()

    if not name:
        logger.warning("跳过 %s：frontmatter 缺 name", md_path)
        return None
    if not description:
        logger.warning("跳过 %s（name=%s）：frontmatter 缺 description", md_path, name)
        return None
    if name != skill_dir.name:
        logger.warning(
            "跳过 %s：frontmatter name=%s 与目录名 %s 不一致",
            md_path, name, skill_dir.name,
        )
        return None
    if len(description) > MAX_DESCRIPTION_LEN:
        logger.warning(
            "跳过 %s（name=%s）：description 长度 %d 超过 %d",
            md_path, name, len(description), MAX_DESCRIPTION_LEN,
        )
        return None

    triggers = _normalize_triggers(fm.get("triggers"))
    try:
        return Skill(
            name=name,
            description=description,
            triggers=triggers,
            path=md_path,
            enabled=enabled,
        )
    except ValueError as exc:
        logger.warning("跳过 %s：Skill 构建失败 %s", md_path, exc)
        return None


def load_skills(root: Path | None = None) -> tuple[Skill, ...]:
    """递归扫描 root 下所有 <name>/SKILL.md，返回 tuple[Skill, ...]。

    enabled 状态从 config/skills_state.json 读，缺失默认 True。
    不可变返回（tuple），结果按 name 排序保证稳定。
    """
    root_path = _default_root() if root is None else Path(root)
    if not root_path.exists():
        return ()

    states = get_enabled_state()
    skills: list[Skill] = []

    for md_path in root_path.rglob(SKILL_FILENAME):
        skill_dir = md_path.parent
        try:
            raw = md_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            logger.warning("读取 %s 失败：%s", md_path, exc)
            continue

        m = FRONTMATTER_RE.match(raw)
        if not m:
            logger.warning("跳过 %s：缺少 frontmatter 块（--- ... ---）", md_path)
            continue

        fm = _parse_frontmatter(m.group(1))
        enabled = states.get(skill_dir.name, True)
        skill = _build_skill(skill_dir, fm, md_path, enabled)
        if skill is not None:
            skills.append(skill)

    return tuple(sorted(skills, key=lambda s: s.name))
