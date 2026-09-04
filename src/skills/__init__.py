"""src.skills — 用户 skills 沉淀机制 v0。

对外导出 Skill dataclass 与 load_skills 加载器。
"""
from src.skills.loader import load_skills
from src.skills.schema import MAX_DESCRIPTION_LEN, Skill
from src.skills.state import (
    STATE_FILENAME,
    get_enabled_for,
    get_enabled_state,
    set_enabled_state,
)

__all__ = [
    "Skill",
    "MAX_DESCRIPTION_LEN",
    "load_skills",
    "get_enabled_state",
    "get_enabled_for",
    "set_enabled_state",
    "STATE_FILENAME",
]
