from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, 
                             QPushButton, QFrame, QSizePolicy, QGraphicsOpacityEffect)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QPoint

class CarouselWidget(QWidget):
    """
    A horizontal scrolling carousel with floating navigation buttons.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(220) # Reasonable height for thumbnails
        
        # Main Layout (Stack to allow overlay)
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Left Button
        self.btn_left = QPushButton("❮")
        self.btn_left.setFixedSize(30, 220) # Full height
        self.btn_left.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 0.1);
                border: none;
                color: #BDBDBD;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.3);
                color: white;
            }
        """)
        self.btn_left.clicked.connect(self.scroll_left)
        self.main_layout.addWidget(self.btn_left)

        # Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        self.content = QWidget()
        self.content_layout = QHBoxLayout(self.content)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(15)
        
        self.scroll.setWidget(self.content)
        self.main_layout.addWidget(self.scroll)

        # Right Button
        self.btn_right = QPushButton("❯")
        self.btn_right.setFixedSize(30, 220)
        self.btn_right.setStyleSheet(self.btn_left.styleSheet())
        self.btn_right.clicked.connect(self.scroll_right)
        self.main_layout.addWidget(self.btn_right)

    def add_widget(self, widget):
        """Add a card widget to the carousel."""
        self.content_layout.addWidget(widget)

    def clear(self):
        """Clear all widgets."""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def scroll_left(self):
        sb = self.scroll.horizontalScrollBar()
        # Scroll by visible width / 2
        step = self.scroll.viewport().width() // 2
        self.smooth_scroll(sb, sb.value() - step)

    def scroll_right(self):
        sb = self.scroll.horizontalScrollBar()
        step = self.scroll.viewport().width() // 2
        self.smooth_scroll(sb, sb.value() + step)
        
    def smooth_scroll(self, scrollbar, target):
        """Simple animation for scrolling."""
        # Cap target
        target = max(scrollbar.minimum(), min(target, scrollbar.maximum()))
        
        # We can use QPropertyAnimation if we subclass QScrollBar or similar, 
        # but direct setValue for now is responsive enough. 
        # For "smooth", we can just step it? 
        # Keep it simple for stability first.
        scrollbar.setValue(target)
