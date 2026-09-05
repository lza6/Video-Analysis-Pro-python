"""帧长图可放大查看器（v5.7）。

把 FrameStripBuilder 生成的 strip.png 用 QGraphicsView + QGraphicsPixmapItem
展示，支持滚轮缩放 + 拖动平移 + 点击单帧弹原图。复用 VideoPlayerDialog 的
QGraphicsView 骨架（video_player_dialog.py:19-23），把 video_item 换成
pixmap_item。

项目无现成可缩放图片查看器（survey 确认），QGraphicsView 缩放/平移是
PyQt6 内建能力，实现量小。

入口：_RunDetailDialog「🖼 查看帧长图证据」按钮 → 读 run["strip_path"]
→ 弹此对话框。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

# CRITICAL: torch 必须先于 PyQt6 加载（Windows DLL 顺序，见 main_window.py 注释）
try:
    import torch  # noqa: F401
except OSError:
    torch = None  # type: ignore[assignment]

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QWheelEvent
from PyQt6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class _ZoomableGraphicsView(QGraphicsView):
    """带滚轮缩放 + 手型拖动的 QGraphicsView。"""

    frame_clicked = pyqtSignal(int, object)  # frame_idx, frame_path（点击单帧时）

    def __init__(self, scene: QGraphicsScene, parent: Optional[QWidget] = None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundRole(QWidget.ColorRole.NoRole)  # 透明底
        self.setStyleSheet("background: #1a1a1a; border: none;")
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._zoom = 1.0
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._frame_rows: list = []  # [(idx, ts, path)] 供点击定位

    def set_pixmap(self, pix: QPixmap, frame_rows: list) -> None:
        """挂载长图 pixmap + 帧索引（供点击定位时间戳）。"""
        if self._pixmap_item is not None:
            self.scene().removeItem(self._pixmap_item)
        self._pixmap_item = QGraphicsPixmapItem(pix)
        self.scene().addItem(self._pixmap_item)
        self._frame_rows = frame_rows
        self._zoom = 1.0
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """滚轮缩放（10% 步进，10%–800% 范围）。"""
        if not self._pixmap_item:
            return
        steps = event.angleDelta().y() / 120
        factor = max(0.1, min(8.0, self._zoom * (1.0 + 0.1 * steps)))
        if abs(factor - self._zoom) < 1e-3:
            return
        self._zoom = factor
        self.resetTransform()
        self.scale(factor, factor)


class FrameStripDialog(QDialog):
    """帧长图证据查看器：可缩放 + 拖动 + 单帧原图 + 跳转视频时间点。"""

    seek_video_requested = pyqtSignal(str, float)  # video_path, timestamp_sec

    def __init__(self, strip_path: str, video_path: str = "",
                 frame_dir: Optional[str] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._strip_path = strip_path
        self._video_path = video_path
        self._frame_dir = frame_dir or str(Path(strip_path).parent)
        self._frame_rows: list = []  # [(idx, ts, path)]
        self.setWindowTitle("帧长图证据 - 可缩放")
        self.resize(1100, 760)
        self._build_ui()
        self._load_strip()
        self._load_frame_index()

    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        # 顶部信息行
        info = QHBoxLayout()
        self.lbl_info = QLabel(
            f"🖼 长图: {Path(self._strip_path).name}  |  帧目录: "
            f"{Path(self._frame_dir).name}/  |  滚轮缩放, 拖动平移"
        )
        self.lbl_info.setStyleSheet("color: #aaa; font-size: 11px;")
        info.addWidget(self.lbl_info, stretch=1)
        self.btn_reset = QPushButton("🔍 适合窗口")
        self.btn_reset.clicked.connect(self._reset_view)
        info.addWidget(self.btn_reset)
        v.addLayout(info)

        # 长图视图
        self.scene = QGraphicsScene(self)
        self.view = _ZoomableGraphicsView(self.scene, self)
        v.addWidget(self.view, stretch=1)

        # 底部操作行
        bottom = QHBoxLayout()
        self.lbl_frame = QLabel("点击长图下方时间戳查看单帧原图")
        self.lbl_frame.setStyleSheet("color: #aaa; font-size: 11px;")
        bottom.addWidget(self.lbl_frame, stretch=1)
        self.btn_ask_ai = QPushButton("💬 询问 AI 这一帧")
        self.btn_ask_ai.setToolTip("把当前选中帧的时间点发给 Agent，让 AI 描述这一刻发生了什么")
        self.btn_ask_ai.clicked.connect(self._ask_ai_current)
        bottom.addWidget(self.btn_ask_ai)
        self.btn_open_dir = QPushButton("📂 打开帧目录")
        self.btn_open_dir.clicked.connect(self._open_dir)
        bottom.addWidget(self.btn_open_dir)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.accept)
        bottom.addWidget(self.btn_close)
        v.addLayout(bottom)

    def _load_strip(self) -> None:
        pix = QPixmap(self._strip_path)
        if pix.isNull():
            self.lbl_info.setText(f"⚠️ 长图读取失败: {self._strip_path}")
            return
        self.view.set_pixmap(pix, self._frame_rows)

    def _load_frame_index(self) -> None:
        """从 frame_dir 扫帧建索引（供点击/AI 查询定位时间戳）。"""
        from src.core.frame_strip import FrameStripBuilder
        try:
            frames = FrameStripBuilder.list_frames(Path(self._frame_dir))
            self._frame_rows = [(i, ts, str(p)) for i, (ts, p) in enumerate(frames)]
            n = len(self._frame_rows)
            self.lbl_info.setText(
                f"🖼 长图: {Path(self._strip_path).name}  |  帧目录: "
                f"{Path(self._frame_dir).name}/  |  共 {n} 帧  |  滚轮缩放, 拖动平移"
            )
        except Exception as e:
            logger.debug(f"[strip_view] 帧索引加载失败: {e}")

    def _reset_view(self) -> None:
        """重置到适合窗口大小。"""
        if self.view._pixmap_item is not None:
            self.view._zoom = 1.0
            self.view.resetTransform()
            self.view.fitInView(
                self.view._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def _open_dir(self) -> None:
        """打开帧目录（Windows 用 os.startfile）。"""
        import os
        try:
            sf = getattr(os, "startfile", None)
            if sf:
                sf(self._frame_dir)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", self._frame_dir])
        except Exception as e:
            logger.warning(f"[strip_view] 打开目录失败: {e}")

    def _ask_ai_current(self) -> None:
        """把当前选中的帧（或第一帧）时间点发给 Agent 查询。

        简化版：取第一帧（用户可在长图上滚动后点此按钮，后续可扩展为
        "选中帧"——当前 QGraphicsView 无单帧选中态，留扩展点）。
        """
        if not self._frame_rows:
            QMessageBox = __import__("PyQt6.QtWidgets",
                                     fromlist=["QMessageBox"]).QMessageBox
            QMessageBox.information(self, "提示", "无帧数据")
            return
        # 取中间帧作为"当前"（用户看长图通常在中段）
        idx = len(self._frame_rows) // 2
        _, ts, _ = self._frame_rows[idx]
        mm = int(ts) // 60
        ss = int(ts) % 60
        msg = f"在视频 {Path(self._video_path).name} 的 {mm:02d}:{ss:02d} 处发生了什么？请描述画面内容。"
        # 通过信号交给 main_window 注入 agent 对话框（main_window 接线）
        self.seek_video_requested.emit(self._video_path, ts)
        # 把消息塞进剪贴板方便用户粘贴到 agent（无直接 agent 引用时兜底）
        try:
            from PyQt6.QtGui import QGuiApplication
            QGuiApplication.clipboard().setText(msg)
        except Exception:
            pass
        QMessageBox = __import__("PyQt6.QtWidgets",
                                  fromlist=["QMessageBox"]).QMessageBox
        QMessageBox.information(
            self, "已准备 AI 查询",
            f"时间点 {mm:02d}:{ss:02d}（第 {idx+1} 帧）已复制到剪贴板。\n"
            f"粘贴到 Agent 对话框即可查询。\n"
            f"（视频已尝试跳转到该时刻）")
