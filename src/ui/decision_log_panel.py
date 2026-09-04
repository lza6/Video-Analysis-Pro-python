"""Agent 决策日志面板 —— 黑匣子透明化 UI 三层之③ 独立面板。

学 edit-mind 把耗时/帧数当一等公民：QTableWidget 列含状态/风险/耗时，
选中行下方 QFormLayout 展完整 decision/reason/args/output_path/duration。
学 OpenHands Event：cause_id 链路在详情区可见，便于追溯上游步骤。

线程铁律（与 surveillance_tab.SurveillanceWorker 一致）：
  ChatWorker 在后台线程跑工具调用，emit entry_append 信号，Qt 自动把
  信号投递回主线程触发槽——绝不跨线程直接改 QTableWidget（Qt 铁律，
  违反必崩）。主控接线时 connect(worker, entry_append, panel.append_entry)
  即可。
"""
import logging
import os
from pathlib import Path
from typing import Optional

# CRITICAL: torch 必须先于 PyQt6 加载（Windows DLL 顺序，见 main_window 顶部注释）。
try:
    import torch  # noqa: F401
except OSError:
    torch = None  # Headless/CPU-broken 环境也要能展示 UI

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.decision_log import DecisionEntry, DecisionLog

logger = logging.getLogger(__name__)

# 列定义：时间 | 步骤 | 工具 | 决策(摘要) | 状态 | 风险
_COLUMNS = ["时间", "步骤", "工具", "决策(摘要)", "状态", "风险"]
# 决策摘要列截断宽度（字符），完整内容在下方详情区
_DECISION_SUMMARY_LIMIT = 40


class DecisionLogPanel(QWidget):
    """Agent 决策日志面板。

    接线点（供 main_window 统一接入 tab bar）：
        from src.ui.decision_log_panel import DecisionLogPanel
        self.tab_decision_log = DecisionLogPanel(parent=self)
        self.tabs.addTab(self.tab_decision_log, "🧭 决策日志")

    ChatWorker 工具调用点 emit 信号：
        worker.entry_append.connect(self.tab_decision_log.append_entry)
    """

    # 公开信号：供 ChatWorker 在后台线程 emit（Qt 自动跨线程回主线程）
    # 类型 object 而非 DecisionEntry，避免 PyQt6 注册自定义类型的成本
    # 与潜在的 metatype 注册顺序问题；append_entry 内做类型守卫。
    entry_append = pyqtSignal(object)
    log_cleared = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._entries: list[DecisionEntry] = []
        self._setup_ui()
        # 双向连接：信号 → 槽。外部 emit(entry_append) 即可安全投递。
        self.entry_append.connect(self.append_entry)

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ---- 工具栏 ----
        bar = QHBoxLayout()
        self.btn_export = QPushButton("导出 JSON")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self.clear_log)
        bar.addWidget(self.btn_export)
        bar.addWidget(self.btn_clear)
        bar.addStretch(1)
        root.addLayout(bar)

        # ---- 决策表 ----
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        assert header is not None  # Qt 保证返回表头；mypy 消除 union-attr
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        root.addWidget(self.table)

        # ---- 详情区 ----
        self._detail_label = QLabel("选中上方任一行查看完整决策详情")
        self._detail_label.setStyleSheet("color: gray;")
        root.addWidget(self._detail_label)

        self._form = QFormLayout()
        self._form.setContentsMargins(4, 4, 4, 4)
        self._fld_decision = QLineEdit()
        self._fld_decision.setReadOnly(True)
        self._fld_reason = QTextEdit()
        self._fld_reason.setReadOnly(True)
        self._fld_reason.setMaximumHeight(60)
        self._fld_args = QTextEdit()
        self._fld_args.setReadOnly(True)
        self._fld_args.setMaximumHeight(80)
        self._fld_output = QLineEdit()
        self._fld_output.setReadOnly(True)
        self._fld_duration = QLineEdit()
        self._fld_duration.setReadOnly(True)
        self._fld_cause = QLineEdit()
        self._fld_cause.setReadOnly(True)
        self._form.addRow("决策：", self._fld_decision)
        self._form.addRow("原因：", self._fld_reason)
        self._form.addRow("参数：", self._fld_args)
        self._form.addRow("产物路径：", self._fld_output)
        self._form.addRow("耗时(ms)：", self._fld_duration)
        self._form.addRow("触发来源：", self._fld_cause)
        root.addLayout(self._form)

    # ------------------------------------------------------------------ 槽

    def append_entry(self, entry: object) -> None:
        """主线程槽：追加一条决策记录到表格。

        线程安全说明：本方法在主线程执行（由 entry_append 信号经 Qt 事件循环
        投递过来）。ChatWorker 后台线程调用 self.entry_append.emit(entry)
        即可，Qt 保证跨线程信号在接收端队列执行，不直接触碰 UI。
        """
        if not isinstance(entry, DecisionEntry):
            logger.warning("[decision_log] 忽略非 DecisionEntry 投递: %r", entry)
            return
        self._entries.append(entry)
        row = self.table.rowCount()
        self.table.insertRow(row)
        # 取摘要前 N 字，完整 decision 留在详情区
        decision_short = entry.decision
        if len(decision_short) > _DECISION_SUMMARY_LIMIT:
            decision_short = decision_short[:_DECISION_SUMMARY_LIMIT] + "…"
        cells = [
            entry.timestamp,
            entry.step_name,
            entry.action_type,
            decision_short,
            entry.status,
            entry.risk,
        ]
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if col in (4, 5):  # 状态/风险列居中
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col, item)

    def clear_log(self) -> None:
        """清空表格与详情区（不删除已落盘的历史 JSON 文件）。"""
        self._entries.clear()
        self.table.setRowCount(0)
        self._reset_detail()
        self.log_cleared.emit()

    def _on_row_selected(self) -> None:
        items = self.table.selectedItems()
        if not items:
            return
        row = items[0].row()
        if row >= len(self._entries):
            return
        e = self._entries[row]
        self._detail_label.setText(
            f"步骤详情：{e.step_name} · {e.action_type} · {e.id}")
        self._fld_decision.setText(e.decision)
        self._fld_reason.setPlainText(e.reason)
        # 真实工具参数（Critic 轮1 MAJOR-2：黑匣子必须可见工具入参）。
        # args_json 由 ChatWorker 工具调用点传入；非工具步骤为 None。
        self._fld_args.setPlainText(e.args_json or "(无参数)")
        self._fld_output.setText(e.output_path or "(无)")
        self._fld_duration.setText(f"{e.duration_ms:.1f}")
        self._fld_cause.setText(e.cause_id or "(根步骤)")

    def _on_export(self) -> None:
        """导出当前内存日志为 JSON（原子写由 DecisionLog.save 保证）。"""
        path, _ = QFileDialog.getSaveFileName(
            self, "导出决策日志", "decision_log.json", "JSON (*.json)")
        if not path:
            return
        log = DecisionLog(tuple(self._entries))
        try:
            log.save(Path(path))
        except Exception as e:  # noqa: BLE001 — UI 槽不能崩
            logger.exception("[decision_log] 导出失败")
            self._detail_label.setText(f"导出失败：{e}")
            return
        self._detail_label.setText(f"已导出到：{path}")

    def _reset_detail(self) -> None:
        self._detail_label.setText("选中上方任一行查看完整决策详情")
        for w in (self._fld_decision, self._fld_output,
                  self._fld_duration, self._fld_cause):
            w.clear()
        for w in (self._fld_reason, self._fld_args):
            w.clear()

    # ------------------------------------------------------------------ 生命周期

    def closeEvent(self, event) -> None:  # noqa: D401
        """与 main_window.closeEvent 同模式：无后台线程需 stop，直接接受。"""
        super().closeEvent(event)
