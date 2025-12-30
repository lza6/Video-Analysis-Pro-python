from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QFrame, QGraphicsView, QGraphicsScene)
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QRectF, QSize
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget, QGraphicsVideoItem
from src.ui.timeline_widget import TimelineWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush

class VideoPlayerDialog(QDialog):
    def __init__(self, video_path, parent=None, frames=None):
        super().__init__(parent)
        self.setWindowTitle(f"专业播放器 - {video_path.name}")
        self.resize(1100, 800)
        self.frames = frames or []

        layout = QVBoxLayout(self)

        # Graphics View for Overlay
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet("background: black; border: none;")
        layout.addWidget(self.view)

        # Video Item
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        self.video_item.setSize(QSize(960, 540)) # Initial size

        # Player
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_item)
        self.media_player.setSource(QUrl.fromLocalFile(str(video_path)))

        # Overlay Layer (Simplified Rects)
        self.overlay_items = []

        # Timeline Widget (NEW)
        self.timeline = TimelineWidget(parent=self)
        layout.addWidget(self.timeline)
        self.timeline.seek_requested.connect(self.seek_to)

        # Controls
        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self.toggle_play)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)

        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.slider)
        layout.addLayout(ctrl_layout)

        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)

        # Populate markers if frames exist
        for f in self.frames:
            # Color coding (example: reddish for objects, blue for scene)
            color = QColor(255, 82, 82) if f.vision_content else QColor(33, 150, 243)
            self.timeline.add_marker(f.timestamp, "Keyframe", color)

    def update_overlay(self, timestamp_s):
        """根据当前时间更新 AI 叠加层（框和文字）。"""
        # Clear old items
        for item in self.overlay_items:
            self.scene.removeItem(item)
        self.overlay_items = []

        # Find closest frame
        closest_frame = min(self.frames, key=lambda f: abs(f.timestamp - timestamp_s), default=None)
        if closest_frame and abs(closest_frame.timestamp - timestamp_s) < 0.5:
            # Example: Draw a box if YOLO detections exist (simulated here)
            if closest_frame.vision_content:
                rect = self.scene.addRect(QRectF(100, 100, 200, 150), QPen(QColor(255, 0, 0), 3))
                text = self.scene.addText(closest_frame.vision_content, Qt.GlobalColor.red)
                text.setPos(100, 80)
                self.overlay_items.extend([rect, text])

            if closest_frame.ocr_text:
                ocr_text = self.scene.addText(f"OCR: {closest_frame.ocr_text[:50]}...", Qt.GlobalColor.yellow)
                ocr_text.setPos(100, 300)
                self.overlay_items.append(ocr_text)

    def seek_to(self, seconds):
        self.media_player.setPosition(int(seconds * 1000))

    def toggle_play(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def set_position(self, position):
        self.media_player.setPosition(position)

    def position_changed(self, position):
        self.slider.setValue(position)
        self.update_overlay(position / 1000.0)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)
        self.timeline.set_duration(duration / 1000.0)

    def update_duration_label(self, current_ms, total_ms):
        def fmt(ms):
            seconds = (ms // 1000) % 60
            minutes = (ms // 60000)
            return f"{minutes:02}:{seconds:02}"
        self.lbl_time.setText(f"{fmt(current_ms)} / {fmt(total_ms)}")

    def closeEvent(self, event):
        self.player.stop()
        super().closeEvent(event)
