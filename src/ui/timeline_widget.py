from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolTip
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPixmap, QPainterPath

class InteractionMarker:
    def __init__(self, timestamp, type_name, color, description=""):
        self.timestamp = timestamp
        self.type_name = type_name
        self.color = color
        self.description = description

class TimelineWidget(QWidget):
    seek_requested = pyqtSignal(float)
    
    def __init__(self, duration=0, parent=None):
        super().__init__(parent)
        self.duration = duration
        self.markers = []
        self.waveform_data = None # np.array
        self.speaker_segments = [] # [(start, end, label)]
        self.setMinimumHeight(100)
        self.setMouseTracking(True)
        self.hover_ts = -1
        self.thumbnails = {} # ts -> pixmap placeholder
        
    def set_duration(self, duration):
        self.duration = duration
        self.update()
        
    def set_waveform(self, data):
        self.waveform_data = data # Expected to be normalized 0-1
        self.update()

    def set_speaker_segments(self, segments):
        self.speaker_segments = segments
        self.update()

    def add_marker(self, timestamp, type_name, color, description=""):
        self.markers.append(InteractionMarker(timestamp, type_name, color, description))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Track Heights
        wave_h = 40
        track_y = 10
        
        # 1. Draw Waveform (Audio)
        if self.waveform_data is not None and len(self.waveform_data) > 0:
            painter.setPen(QColor(100, 100, 100, 50))
            painter.setBrush(QColor(200, 200, 200, 30))
            painter.drawRect(0, track_y, w, wave_h)
            
            path = QPainterPath()
            path.moveTo(0, track_y + wave_h / 2)
            
            step = len(self.waveform_data) // w if len(self.waveform_data) > w else 1
            for x in range(0, w):
                idx = int((x / w) * len(self.waveform_data))
                val = self.waveform_data[idx] * (wave_h / 2)
                path.lineTo(x, track_y + wave_h / 2 - val)
            for x in range(w-1, -1, -1):
                idx = int((x / w) * len(self.waveform_data))
                val = self.waveform_data[idx] * (wave_h / 2)
                path.lineTo(x, track_y + wave_h / 2 + val)
                
            painter.setFillRule(Qt.FillRule.WindingFill)
            painter.fillPath(path, QColor(33, 150, 243, 150))

        # 2. Draw Speaker Tracks
        speaker_y = track_y + wave_h + 5
        speaker_h = 15
        painter.setBrush(QColor(240, 240, 240))
        painter.drawRect(0, speaker_y, w, speaker_h)
        for start, end, label in self.speaker_segments:
            if self.duration > 0:
                x1 = int((start / self.duration) * w)
                x2 = int((end / self.duration) * w)
                painter.setBrush(QColor(76, 175, 80, 180)) # Greenish for voice
                painter.drawRect(x1, speaker_y, x2 - x1, speaker_h)

        # 3. Draw Markers (Keyframes)
        marker_y = speaker_y + speaker_h + 5
        for m in self.markers:
            if self.duration > 0:
                x = int((m.timestamp / self.duration) * w)
                painter.setPen(QPen(m.color, 2))
                painter.drawLine(x, marker_y, x, h)
                painter.setBrush(m.color)
                painter.drawEllipse(QPoint(x, marker_y + 5), 4, 4)

        # Hover Line & Preview
        if self.hover_ts >= 0 and self.duration > 0:
            hx = int((self.hover_ts / self.duration) * w)
            painter.setPen(QPen(QColor(255, 64, 129), 1, Qt.PenStyle.DashLine))
            painter.drawLine(hx, 0, hx, h)

    def mouseMoveEvent(self, event):
        if self.duration > 0:
            x = event.pos().x()
            self.hover_ts = (x / self.width()) * self.duration
            self.update()
            
            # Show Tooltip with time
            QToolTip.showText(event.globalPosition().toPoint(), f"{self.hover_ts:.1f}s", self)
            
    def leaveEvent(self, event):
        self.hover_ts = -1
        self.update()

    def mousePressEvent(self, event):
        if self.duration > 0:
            x = event.pos().x()
            ts = (x / self.width()) * self.duration
            self.seek_requested.emit(ts)
