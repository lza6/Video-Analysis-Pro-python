"""SkillsManagerTab — 用户 skills 管理面板。

PyQt6 QWidget 子类。顶部 torch 守卫铁律（先于 PyQt6 导入，Windows DLL 顺序）。
列出所有 skill，支持启用/禁用切换、导入 skill 文件夹。
构造签名 history_manager=None 保持与其它 tab 一致（本 tab 不使用，但保留）。
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

# CRITICAL: torch 必须先于 PyQt6 加载（Windows DLL 顺序，见 main_window.py 注释）。
try:
    import torch  # noqa: F401  (DLL load-order fix)
except OSError:
    torch = None  # type: ignore[assignment]

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.skills import Skill, load_skills, set_enabled_state
from src.utils.constants import CONFIG_DIR

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(CONFIG_DIR) / "skills"
DESC_TRUNC = 80  # 列表项 description 截断长度


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


class SkillsManagerTab(QWidget):
    """用户 skills 管理面板：列表 + 详情 + 启用切换 + 导入。"""

    def __init__(self, history_manager: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history_manager = history_manager  # 签名一致，本 tab 不使用
        self._skills: tuple[Skill, ...] = ()
        self._current: Skill | None = None
        self._build_ui()
        self.reload()

    # ---- UI 构建 ----
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("<h3>🧩 Skills 管理</h3>")
        hint = QLabel(
            "管理本地沉淀的 skills。勾选启用/禁用即时写入 "
            "<code>config/skills_state.json</code>；"
            "导入按钮可将外部 skill 文件夹拷贝到 <code>config/skills/</code>。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(title)
        root.addWidget(hint)

        # 顶部操作栏
        bar = QHBoxLayout()
        self.btn_reload = QPushButton("🔄 重新加载")
        self.btn_reload.clicked.connect(self.reload)
        self.btn_import = QPushButton("📁 导入 skill 文件夹")
        self.btn_import.clicked.connect(self._on_import)
        bar.addStretch()
        bar.addWidget(self.btn_reload)
        bar.addWidget(self.btn_import)
        root.addLayout(bar)

        # 主体：左右分栏
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_select)
        splitter.addWidget(self.list_widget)

        detail = QWidget()
        self.form = QFormLayout(detail)
        self.form.setContentsMargins(8, 8, 8, 8)
        self.lbl_name = QLabel("—")
        self.lbl_desc = QLabel("—")
        self.lbl_desc.setWordWrap(True)
        self.lbl_triggers = QLabel("—")
        self.lbl_triggers.setWordWrap(True)
        self.lbl_path = QLabel("—")
        self.lbl_path.setWordWrap(True)
        self.lbl_path.setStyleSheet("color: gray; font-size: 10px;")
        self.chk_enabled = QCheckBox("启用")
        self.chk_enabled.setEnabled(False)
        self.chk_enabled.stateChanged.connect(self._on_toggle_enabled)
        self.form.addRow("名称：", self.lbl_name)
        self.form.addRow("描述：", self.lbl_desc)
        self.form.addRow("触发词：", self.lbl_triggers)
        self.form.addRow("路径：", self.lbl_path)
        self.form.addRow("", self.chk_enabled)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)
        root.addWidget(splitter, stretch=1)

    # ---- 数据加载 ----
    def reload(self) -> tuple[Skill, ...]:
        """重新扫描 skills 目录并刷新列表。返回当前 skills tuple。"""
        self._skills = load_skills()
        self.list_widget.clear()
        for sk in self._skills:
            status = "✅" if sk.enabled else "⛔"
            text = f"{status} {sk.name} — {_truncate(sk.description, DESC_TRUNC)}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, sk.name)
            self.list_widget.addItem(item)
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
        else:
            self._show_empty_detail()
        return self._skills

    def _show_empty_detail(self) -> None:
        self._current = None
        self.lbl_name.setText("—")
        self.lbl_desc.setText("（未选中 skill）")
        self.lbl_triggers.setText("—")
        self.lbl_path.setText("—")
        self.chk_enabled.setEnabled(False)
        self.chk_enabled.setChecked(False)

    # ---- 选中详情 ----
    def _on_select(self, row: int) -> None:
        if row < 0 or row >= len(self._skills):
            self._show_empty_detail()
            return
        sk = self._skills[row]
        self._current = sk
        self.lbl_name.setText(sk.name)
        self.lbl_desc.setText(sk.description)
        self.lbl_triggers.setText(", ".join(sk.triggers) if sk.triggers else "（无）")
        self.lbl_path.setText(str(sk.path))
        # 禁止信号循环：先 block 再 set
        self.chk_enabled.blockSignals(True)
        self.chk_enabled.setEnabled(True)
        self.chk_enabled.setChecked(sk.enabled)
        self.chk_enabled.blockSignals(False)

    def _on_toggle_enabled(self, state: int) -> None:
        if self._current is None:
            return
        enabled = bool(state)
        try:
            set_enabled_state(self._current.name, enabled)
        except OSError as exc:
            logger.error("写入 skills_state.json 失败：%s", exc)
            # 回滚 UI 状态
            self.chk_enabled.blockSignals(True)
            self.chk_enabled.setChecked(not enabled)
            self.chk_enabled.blockSignals(False)
            return
        # 不可变更新：构造新 Skill 替换列表项
        new_skill = Skill(
            name=self._current.name,
            description=self._current.description,
            triggers=self._current.triggers,
            path=self._current.path,
            enabled=enabled,
        )
        self._skills = tuple(
            new_skill if s.name == new_skill.name else s for s in self._skills
        )
        self._current = new_skill
        # 刷新列表项显示
        row = self.list_widget.currentRow()
        if row >= 0:
            status = "✅" if enabled else "⛔"
            text = f"{status} {new_skill.name} — {_truncate(new_skill.description, DESC_TRUNC)}"
            self.list_widget.item(row).setText(text)

    # ---- 导入 skill 文件夹 ----
    def _on_import(self) -> None:
        src = QFileDialog.getExistingDirectory(
            self, "选择 skill 文件夹（应含 SKILL.md）", ""
        )
        if not src:
            return
        src_path = Path(src)
        skill_md = src_path / "SKILL.md"
        if not skill_md.exists():
            logger.warning("导入失败：%s 下无 SKILL.md", src_path)
            return
        dest_dir = SKILLS_DIR / src_path.name
        if dest_dir.exists():
            logger.warning("导入失败：目标已存在 %s", dest_dir)
            return
        try:
            SKILLS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_path, dest_dir)
        except OSError as exc:
            logger.error("拷贝 skill 失败：%s", exc)
            return
        self.reload()
