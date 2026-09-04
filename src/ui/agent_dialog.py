"""Agent 对话主界面（豆包风格）—— M5 主入口。

软件启动后默认进入此界面（不是原功能 tab 平铺）。结构：

    工具箱侧栏 (QListWidget, 左) | QStackedWidget (右)
      - 对话主页（默认）：消息气泡 + 输入框 + 上传 + 思考链
      - 工具专属 UI 页：外部接入（batch_tab / surveillance / models 等）

对话主页复用 agent_panel.ChatBubble / ThinkingWidget（已有折扇思考块、
气泡布局），不重复造轮子。AgentDialog 只负责"主界面骨架 + 工具箱路由"，
对话逻辑由 AgentOrchestrator（src/core/agent_orchestrator.py）承担，
LLM 流式回复仍由 main_window.ChatWorker 经 on_agent_query 走原链路。

线程铁律：与 surveillance_tab / batch_tab 一致——agent 工具在后台线程
执行时，结果通过 pyqtSignal 回主线程，不跨线程直接改 QListWidget。
"""
import logging
from pathlib import Path
from typing import Optional

# CRITICAL: torch 必须先于 PyQt6 加载（Windows DLL 顺序，见 main_window.py 顶部注释）
try:
    import torch  # noqa: F401
except OSError:
    torch = None  # type: ignore[assignment]

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.ui.agent_panel import ChatBubble, ThinkingWidget

logger = logging.getLogger(__name__)


class _ToolBoxItem:
    """工具箱侧栏一项（icon+name+widget index+tool_id）。不可变语义。"""

    def __init__(self, icon: str, name: str, widget_index: int, tool_id: str = ""):
        self.icon = icon
        self.name = name
        self.widget_index = widget_index
        self.tool_id = tool_id or name


class AgentDialog(QWidget):
    """豆包风格 agent 对话主界面。

    对外信号：
      message_sent(text, attachments)  — 用户发送消息（attachments=路径列表）
      tool_requested(tool_id)           — 用户点击工具箱项
      provider_config_requested        — 用户要求配 key
      model_download_requested(mid)    — 用户要求下模型

    对外 API：
      add_tool_page(icon, name, widget, tool_id) -> index
      append_user_message(text) / append_agent_message(text, model)
      append_thinking(text) / append_tool_call(tool, args, result)
      set_thoughts(text) / clear_messages() / update_last_bubble(chunk)
    """

    message_sent = pyqtSignal(str, list)  # text, attachment_paths
    tool_requested = pyqtSignal(str)
    provider_config_requested = pyqtSignal()
    model_download_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._attachments: list[str] = []
        self._tool_items: list[_ToolBoxItem] = []
        self._last_bubble: Optional[ChatBubble] = None
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_tool_box(), stretch=0)
        self.stacked = QStackedWidget()
        self.stacked.addWidget(self._build_dialog_page())
        root.addWidget(self.stacked, stretch=1)

    def _build_tool_box(self) -> QWidget:
        box = QFrame()
        box.setFixedWidth(200)
        box.setStyleSheet(
            "QFrame { background: #1e1e1e; border-right: 1px solid #333; }"
        )
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 12, 8, 12)
        v.setSpacing(6)

        title = QLabel("🧰 工具箱")
        title.setStyleSheet(
            "color: #fff; font-weight: bold; font-size: 14px; padding: 4px;"
        )
        v.addWidget(title)

        self.list_tools = QListWidget()
        self.list_tools.setStyleSheet(
            """
            QListWidget { background: transparent; border: none; color: #ddd;
                          font-size: 12px; }
            QListWidget::item { padding: 10px 8px; border-radius: 6px; }
            QListWidget::item:hover { background: rgba(255,255,255,0.08); }
            QListWidget::item:selected { background: #2196F3; color: white; }
            """
        )
        # 首页"Agent 对话"固定不可被外部移除（index 0 = 对话页）
        QListWidgetItem("💬 Agent 对话", self.list_tools)
        self.list_tools.setCurrentRow(0)
        self.list_tools.currentRowChanged.connect(self._on_tool_changed)
        v.addWidget(self.list_tools, stretch=1)

        # 快捷操作（愿景5/6：对话式配 key + 帮下模型）
        btn_provider = QPushButton("🔑 配置 Provider")
        btn_provider.setToolTip("Agent 引导你配置 API Provider 与 Key")
        btn_provider.clicked.connect(self.provider_config_requested.emit)
        v.addWidget(btn_provider)
        btn_download = QPushButton("📦 下载模型")
        btn_download.setToolTip("Agent 帮你下载并校验模型（SHA256）")
        btn_download.clicked.connect(
            lambda: self.model_download_requested.emit("yolo_v11n"))
        v.addWidget(btn_download)
        btn_new = QPushButton("🆕 新建会话")
        btn_new.clicked.connect(self.clear_messages)
        v.addWidget(btn_new)
        return box

    def _build_dialog_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        # 消息列表（滚动气泡）
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        self.scroll_content = QWidget()
        self.chat_layout = QVBoxLayout(self.scroll_content)
        self.chat_layout.addStretch()
        self.scroll.setWidget(self.scroll_content)
        v.addWidget(self.scroll, stretch=1)

        # 思考链（折扇，复用 agent_panel.ThinkingWidget）
        self.thinking_widget = ThinkingWidget(self)
        v.addWidget(self.thinking_widget)

        # 附件暂留区
        self.attach_area = QHBoxLayout()
        self.attach_area.setSpacing(4)
        v.addLayout(self.attach_area)

        # 输入区
        input_row = QHBoxLayout()
        self.btn_upload = QPushButton("📎 上传")
        self.btn_upload.setToolTip("上传视频/照片（支持多选）")
        self.btn_upload.setFixedWidth(60)
        self.btn_upload.clicked.connect(self._on_upload)
        self.input_msg = QPlainTextEdit()
        self.input_msg.setPlaceholderText("发送消息给 Agent（可上传视频/照片）…")
        self.input_msg.setMaximumHeight(80)
        self.btn_send = QPushButton("发送 ➤")
        self.btn_send.setFixedWidth(80)
        self.btn_send.setStyleSheet(
            "background: #2196F3; color: white; font-weight: bold;"
        )
        self.btn_send.clicked.connect(self._on_send)
        input_row.addWidget(self.btn_upload)
        input_row.addWidget(self.input_msg, stretch=1)
        input_row.addWidget(self.btn_send)
        v.addLayout(input_row)
        return page

    # ------------------------------------------------------------------ 公开 API

    def add_tool_page(self, icon: str, name: str, widget: QWidget,
                     tool_id: str = "") -> int:
        """添加工具专属 UI 页到 stacked，工具箱侧栏自动加一项。

        返回 stacked index（0 是对话页，工具页从 1 开始）。
        """
        index = self.stacked.addWidget(widget)
        item = QListWidgetItem(f"{icon} {name}", self.list_tools)
        # UserRole 存 stacked index（对话页=0，工具页>=1）
        item.setData(Qt.ItemDataRole.UserRole, index)
        self._tool_items.append(_ToolBoxItem(icon, name, index, tool_id))
        return index

    def append_user_message(self, text: str) -> None:
        bubble = ChatBubble("User", text, is_user=True)
        self._append_bubble(bubble, is_user=True)
        self._last_bubble = None  # 用户气泡不参与 update_last_bubble

    def append_agent_message(self, text: str,
                              model_name: Optional[str] = None) -> None:
        bubble = ChatBubble("Agent", text, model_name=model_name, is_user=False)
        self._append_bubble(bubble, is_user=False)
        self._last_bubble = bubble

    def append_thinking(self, text: str) -> None:
        self.thinking_widget.set_text(text)

    def append_tool_call(self, tool_name: str, args: dict, result: str) -> None:
        """工具调用气泡（愿景：每步可追溯）。"""
        args_str = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())[:120]
        line = (
            f"🛠️ 工具调用: {tool_name}({args_str})\n"
            f"✅ 结果: {str(result)[:200]}"
        )
        self.append_agent_message(line)

    def set_thoughts(self, text: str) -> None:
        self.thinking_widget.set_text(text)

    def update_last_bubble(self, chunk: str) -> None:
        """流式追加到最近一个 agent 气泡（ChatWorker 回调用）。"""
        if self._last_bubble is None:
            self.append_agent_message(chunk)
            return
        prev = getattr(self._last_bubble, "full_text", "") or ""
        self._last_bubble.update_text(prev + chunk)

    def clear_messages(self) -> None:
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._last_bubble = None
        self.thinking_widget.set_text("")
        self._attachments.clear()
        self._clear_attach_area()

    # ------------------------------------------------------------------ 槽

    def _on_tool_changed(self, row: int) -> None:
        if row <= 0:
            self.stacked.setCurrentIndex(0)
            return
        item = self.list_tools.item(row)
        if item is None:
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        idx = int(data) if isinstance(data, int) else 0
        self.stacked.setCurrentIndex(idx)
        tool_id = ""
        # row 0 是对话首页，工具项从 row 1 开始，对应 _tool_items[row-1]
        if 0 <= row - 1 < len(self._tool_items):
            tool_id = self._tool_items[row - 1].tool_id
        if tool_id:
            self.tool_requested.emit(tool_id)

    def _on_upload(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择视频/照片", "",
            "媒体 (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp);;所有 (*.*)"
        )
        for p in paths:
            self.add_attachment(p)

    def _on_send(self) -> None:
        text = self.input_msg.toPlainText().strip()
        if not text and not self._attachments:
            return
        self.append_user_message(text or "(附件)")
        self.message_sent.emit(text, list(self._attachments))
        self.input_msg.clear()
        self._clear_attach_area()
        self._attachments.clear()

    # ------------------------------------------------------------------ 工具

    def add_attachment(self, path: str) -> None:
        """添加附件 chip（愿景2：上传视频/照片）。"""
        if not path or not Path(path).exists():
            return
        self._attachments.append(path)
        chip = QLabel(f"📎 {Path(path).name}")
        chip.setStyleSheet(
            "background: #2196F322; border: 1px solid #2196F3; "
            "border-radius: 8px; padding: 2px 8px; color: #2196F3;"
        )
        self.attach_area.addWidget(chip)

    def get_attachments(self) -> list:
        return list(self._attachments)

    def _clear_attach_area(self) -> None:
        while self.attach_area.count():
            item = self.attach_area.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _append_bubble(self, bubble: ChatBubble, is_user: bool) -> None:
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 4, 0, 4)
        if is_user:
            rl.addStretch()
            rl.addWidget(bubble, stretch=0)
        else:
            rl.addWidget(bubble, stretch=0)
            rl.addStretch()
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row)
        sb = self.scroll.verticalScrollBar()
        was_bottom = sb.value() >= sb.maximum() - 10
        if was_bottom:
            QTimer.singleShot(10, lambda: sb.setValue(sb.maximum()))
