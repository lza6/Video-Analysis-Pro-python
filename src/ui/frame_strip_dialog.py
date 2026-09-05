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

from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QWheelEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class _ZoomableGraphicsView(QGraphicsView):
    """带滚轮缩放 + 手型拖动 + 单帧点击命中的 QGraphicsView。"""

    frame_clicked = pyqtSignal(int, object, float)  # frame_idx, frame_path, ts

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
        self._frame_rows: list = []  # [(idx, ts, path)]
        # 布局参数（与 FrameStripBuilder.build 保持一致，供 hit-test）
        self._cols = 20
        self._thumb_w = 160
        self._thumb_h = 90  # build 时按首帧比例算，set_pixmap 时回填真实值
        self._label_h = 22
        self._gap = 2

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

    def set_layout_params(self, cols: int, thumb_w: int, thumb_h: int,
                           label_h: int, gap: int) -> None:
        """记录长图网格布局参数（与 FrameStripBuilder.build 一致），供 hit-test。"""
        self._cols = cols
        self._thumb_w = thumb_w
        self._thumb_h = thumb_h
        self._label_h = label_h
        self._gap = gap

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

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """点击长图 → hit-test 命中单帧 → 发 frame_clicked 信号。

        左键单击在帧区域内：定位到该帧，显示时间戳 tooltip，发信号给
        FrameStripDialog（更新"当前帧"+可跳转/AI查询）。拖动模式仍由
        ScrollHandDrag 处理（短按拖动不触发此分支，因 setDragMode 已接管）。
        """
        if (event.button() == Qt.MouseButton.LeftButton
                and self._pixmap_item is not None
                and self._frame_rows):
            # 鼠标坐标 → scene 坐标
            sp = self.mapToScene(event.position().toPoint())
            idx = self._hit_test_frame(sp)
            if idx is not None:
                ts = self._frame_rows[idx][1] if idx < len(self._frame_rows) else 0.0
                path = self._frame_rows[idx][2] if idx < len(self._frame_rows) else ""
                QToolTip.showText(
                    event.globalPosition().toPoint(),
                    f"#{idx+1}  {int(ts)//60:02d}:{int(ts)%60:02d}",
                    self,
                )
                self.frame_clicked.emit(idx, path, ts)
                event.accept()
                return
        super().mousePressEvent(event)

    def _hit_test_frame(self, scene_pt: QPointF) -> Optional[int]:
        """scene 坐标 → 命中的帧 idx（不在任何帧矩形内返回 None）。

        用 FrameStripBuilder.cell_rect 的同款公式，保持与 build 一致。
        """
        from src.core.frame_strip import cell_rect, compute_layout
        n = len(self._frame_rows)
        if n == 0:
            return None
        layout = compute_layout(
            n, self._cols, self._thumb_w, self._thumb_h,
            self._label_h, self._gap,
        )
        # 反推：scene_pt 落在哪个 cell 矩形
        gap = self._gap
        cw = self._thumb_w
        # 只算帧区域（不含下方标签条），用于行/列反推
        x = scene_pt.x()
        y = scene_pt.y()
        # 先减去左边距 gap，再除以 (cw+gap) 取列
        col = int((x - gap) // (cw + gap))
        row = int((y - gap) // (self._thumb_h + self._label_h + gap))
        if col < 0 or col >= self._cols or row < 0 or row >= layout["rows"]:
            return None
        idx = row * self._cols + col
        if idx >= n:
            return None
        # 精确矩形内（排除间隙）
        rx, ry, rw, rh = cell_rect(
            idx, n, self._cols, self._thumb_w, self._thumb_h,
            self._label_h, self._gap,
        )
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            return idx
        return None


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
        self._current_idx: Optional[int] = None  # 当前选中帧（v5.7.1 hit-test）
        self._current_ts: float = 0.0
        self._current_path: str = ""
        self._frame_viewer: Optional[QDialog] = None  # 单帧原图弹窗（复用避免重复开）
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
            f"{Path(self._frame_dir).name}/  |  滚轮缩放, 拖动平移, 点击单帧选中"
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
        # 点击单帧 → 选中 + 显示原图 + 更新底部状态
        self.view.frame_clicked.connect(self._on_frame_clicked)
        v.addWidget(self.view, stretch=1)

        # 底部操作行
        bottom = QHBoxLayout()
        self.lbl_frame = QLabel("👆 点击长图任一帧选中（当前：未选中）")
        self.lbl_frame.setStyleSheet("color: #3498db; font-size: 11px; font-weight: bold;")
        bottom.addWidget(self.lbl_frame, stretch=1)
        self.btn_view_frame = QPushButton("🖼 查看原图")
        self.btn_view_frame.setToolTip("放大查看当前选中帧")
        self.btn_view_frame.clicked.connect(self._view_current_frame)
        bottom.addWidget(self.btn_view_frame)
        self.btn_seek = QPushButton("▶ 跳转视频")
        self.btn_seek.setToolTip("打开/复用播放器定位到当前选中帧时刻")
        self.btn_seek.clicked.connect(self._seek_current)
        bottom.addWidget(self.btn_seek)
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

    def _on_frame_clicked(self, idx: int, path: object, ts: float) -> None:
        """点击长图单帧 → 记为当前选中 + 更新状态行 + 自动弹原图。"""
        self._current_idx = idx
        self._current_path = str(path) if path else ""
        self._current_ts = float(ts) if ts else 0.0
        mm = int(self._current_ts) // 60
        ss = int(self._current_ts) % 60
        self.lbl_frame.setText(
            f"👆 当前：第 {idx+1} 帧  {mm:02d}:{ss:02d}  ({self._current_path or '无'})"
        )
        # 自动弹原图（用户点哪帧看哪帧，无需二次点击）
        self._view_current_frame()

    def _view_current_frame(self) -> None:
        """单帧原图查看器（可缩放，复用同一弹窗避免叠开）。"""
        if not self._current_path or not Path(self._current_path).exists():
            return
        # 复用已开弹窗（切换图即可）
        if self._frame_viewer is not None and self._frame_viewer.isVisible():
            pix = QPixmap(self._current_path)
            if not pix.isNull() and hasattr(self._frame_viewer, "_pix_item"):
                self._frame_viewer._pix_item.setPixmap(pix)
                self._frame_viewer.setWindowTitle(
                    f"帧 #{(self._current_idx or 0)+1} 原图 - {Path(self._current_path).name}")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(
            f"帧 #{(self._current_idx or 0)+1} 原图 - {Path(self._current_path).name}")
        dlg.resize(800, 600)
        lv = QVBoxLayout(dlg)
        sc = QGraphicsScene(dlg)
        gv = _ZoomableGraphicsView(sc, dlg)
        pix = QPixmap(self._current_path)
        gv.set_layout_params(1, pix.width(), pix.height(), 0, 0)
        gv.set_pixmap(pix, [])
        gv.frame_clicked = None  # 单帧视图不响应 frame_clicked（复用组件但不启用命中）
        lv.addWidget(gv, stretch=1)
        btn_row = QHBoxLayout()
        mm = int(self._current_ts) // 60
        ss = int(self._current_ts) % 60
        lbl = QLabel(f"⏱ {mm:02d}:{ss:02d}  |  滚轮缩放, 拖动平移")
        lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        btn_row.addWidget(lbl, stretch=1)
        btn_seek = QPushButton("▶ 跳转视频")
        btn_seek.clicked.connect(self._seek_current)
        btn_row.addWidget(btn_seek)
        btn_ask = QPushButton("💬 问 AI")
        btn_ask.clicked.connect(self._ask_ai_current)
        btn_row.addWidget(btn_ask)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        lv.addLayout(btn_row)
        dlg._pix_item = gv._pixmap_item  # type: ignore[attr-defined]
        self._frame_viewer = dlg
        dlg.show()

    def _seek_current(self) -> None:
        """跳转播放器到当前选中帧时刻。"""
        if self._current_idx is None:
            QMessageBox = __import__("PyQt6.QtWidgets",
                                     fromlist=["QMessageBox"]).QMessageBox
            QMessageBox.information(self, "提示", "请先在长图上点击一帧选中")
            return
        self.seek_video_requested.emit(self._video_path, self._current_ts)

    def _load_frame_index(self) -> None:
        """从 frame_dir 扫帧建索引（供点击/AI 查询定位时间戳）+ 回填布局参数。"""
        from src.core.frame_strip import FrameStripBuilder
        try:
            frames = FrameStripBuilder.list_frames(Path(self._frame_dir))
            self._frame_rows = [(i, ts, str(p)) for i, (ts, p) in enumerate(frames)]
            n = len(self._frame_rows)
            self.lbl_info.setText(
                f"🖼 长图: {Path(self._strip_path).name}  |  帧目录: "
                f"{Path(self._frame_dir).name}/  |  共 {n} 帧  |  滚轮缩放, 拖动, 点击选中"
            )
            # 回填真实缩略图高度（build 按首帧比例算的，这里从首帧读）
            if self._frame_rows:
                from PyQt6.QtGui import QImage
                qimg = QImage(self._frame_rows[0][2])
                if not qimg.isNull() and qimg.width() > 0:
                    real_h = int(self.view._thumb_w * qimg.height() / qimg.width())
                    self.view.set_layout_params(
                        cols=self.view._cols,
                        thumb_w=self.view._thumb_w,
                        thumb_h=real_h,
                        label_h=self.view._label_h,
                        gap=self.view._gap,
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
        """把当前选中帧的时间点发给 Agent 查询（v5.7.1 真实命中，不再取中段）。"""
        if self._current_idx is None:
            QMessageBox = __import__("PyQt6.QtWidgets",
                                     fromlist=["QMessageBox"]).QMessageBox
            QMessageBox.information(
                self, "提示", "请先在长图上点击一帧选中，再询问 AI")
            return
        ts = self._current_ts
        mm = int(ts) // 60
        ss = int(ts) % 60
        msg = (f"在视频 {Path(self._video_path).name} 的 {mm:02d}:{ss:02d} 处"
               f"（第 {self._current_idx + 1} 帧）发生了什么？请描述画面内容。")
        # 跳转播放器定位到该时刻（让用户边听 AI 边看画面核对）
        self.seek_video_requested.emit(self._video_path, ts)
        # 把查询消息塞进剪贴板方便用户粘贴到 agent 对话框
        try:
            from PyQt6.QtGui import QGuiApplication
            QGuiApplication.clipboard().setText(msg)
        except Exception:
            pass
        QMessageBox = __import__("PyQt6.QtWidgets",
                                  fromlist=["QMessageBox"]).QMessageBox
        QMessageBox.information(
            self, "已准备 AI 查询",
            f"时间点 {mm:02d}:{ss:02d}（第 {self._current_idx + 1} 帧）"
            f"已复制到剪贴板。\n"
            f"粘贴到 Agent 对话框即可查询。\n"
            f"（视频已尝试跳转到该时刻）")
