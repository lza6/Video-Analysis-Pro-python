"""Skill 数据模型。

不可变 dataclass：创建新对象而非就地修改，避免并发竞态。
description 长度校验 ≤200 字符（宽松；60 是 hermes 系统提示索引截断，
非硬限，此处不卡 60）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_DESCRIPTION_LEN = 200


@dataclass(frozen=True)
class Skill:
    """单个 skill 的不可变描述。"""

    name: str
    description: str
    triggers: tuple[str, ...]
    path: Path
    enabled: bool

    def __post_init__(self) -> None:
        # frozen=True 下无法直接赋值，走 object.__setattr__ 完成规范化。
        if not self.name:
            raise ValueError("Skill.name 不能为空")
        desc = self.description or ""
        if len(desc) > MAX_DESCRIPTION_LEN:
            raise ValueError(
                f"Skill.description 长度 {len(desc)} 超过 {MAX_DESCRIPTION_LEN} 字符"
            )
        # 规范化 triggers 为 tuple
        if not isinstance(self.triggers, tuple):
            object.__setattr__(self, "triggers", tuple(self.triggers))
        # path 强制 Path
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))
