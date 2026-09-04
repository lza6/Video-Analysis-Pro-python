"""skills_state.json 读写。

原子写：先写 .tmp 再 os.replace，避免崩溃时半截文件。
文件位于 config/skills_state.json，结构 {skill_name: bool}。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from src.utils.constants import CONFIG_DIR

logger = logging.getLogger(__name__)

STATE_FILENAME = "skills_state.json"


def _state_path() -> Path:
    """返回 skills_state.json 路径（相对项目根）。"""
    return Path(CONFIG_DIR) / STATE_FILENAME


def get_enabled_state() -> dict[str, bool]:
    """读取全部 skill 的 enabled 状态。缺失条目默认 True（调用方补齐）。"""
    p = _state_path()
    if not p.exists():
        return {}
    try:
        with p.open(encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): bool(v) for k, v in data.items()}
        logger.warning("skills_state.json 顶层非 dict，忽略：%s", type(data).__name__)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 skills_state.json 失败，按空处理：%s", exc)
    return {}


def get_enabled_for(name: str) -> bool:
    """读取单个 skill 的 enabled 状态，缺失默认 True。"""
    return get_enabled_state().get(name, True)


def set_enabled_state(name: str, enabled: bool) -> None:
    """原子写入单个 skill 的 enabled 状态（合并写整文件）。"""
    if not name:
        raise ValueError("skill name 不能为空")
    states = get_enabled_state()
    states[name] = bool(enabled)
    _atomic_write(states)


def _atomic_write(data: dict[str, Any]) -> None:
    """原子写 JSON：写 .tmp 后 os.replace。"""
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, p)
    except OSError as exc:
        logger.error("写入 skills_state.json 失败：%s", exc)
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
