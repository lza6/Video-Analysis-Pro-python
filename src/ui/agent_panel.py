from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
                             QLineEdit, QPushButton, QFrame, QScrollArea, QComboBox, 
                             QProgressBar, QSizePolicy, QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QCursor

class ThinkingWidget(QFrame):
    """DeepSeek R1 style collapsible thinking block."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("ThinkingWidget")
        self.setStyleSheet("""
            #ThinkingWidget {
                background-color: #F9FAFB;
                border: 1px solid #E5E7EB;
                border-radius: 12px;
                margin: 4px 0px;
            }
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 6, 12, 6)
        self.layout.setSpacing(0)

        # Header (Click to expand)
        self.btn_toggle = QPushButton("💭 深度思考 (展开)")
        self.btn_toggle.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_toggle.setStyleSheet("""
            QPushButton {
                border: none;
                text-align: left;
                color: #6B7280;
                font-size: 11px;
                font-weight: 600;
                background: transparent;
                padding: 4px;
            }
            QPushButton:hover { color: #374151; }
        """)
        self.btn_toggle.clicked.connect(self.toggle_content)
        self.layout.addWidget(self.btn_toggle)

        # Content area (Labels for thoughts)
        self.content_area = QLabel()
        self.content_area.setWordWrap(True)
        self.content_area.setVisible(False) 
        self.content_area.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.content_area.setStyleSheet("""
            QLabel {
                border: none; 
                background: transparent; 
                color: #4B5563; 
                font-family: 'Segoe UI', system-ui; 
                font-size: 10pt; 
                padding: 8px 4px;
                line-height: 1.5;
            }
        """)
        self.layout.addWidget(self.content_area)

    def toggle_content(self):
        is_visible = self.content_area.isVisible()
        self.content_area.setVisible(not is_visible)
        self.btn_toggle.setText("💭 深度思考 (收起)" if not is_visible else "💭 深度思考 (展开)")

    def set_text(self, text):
        if not text.strip():
            self.setVisible(False)
            return
        self.setVisible(True)
        self.content_area.setText(text)

class ChatBubble(QFrame):
    regenerate_requested = pyqtSignal(str, str) # prompt, model

    def __init__(self, sender, text, model_name=None, is_user=True, related_prompt=None):
        super().__init__()
        self.is_user = is_user
        self.related_prompt = related_prompt
        self.model_name = model_name
        self.full_text = text # Store full text for parsing

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(6)
        
        # Shadow Effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 30))
        self.setGraphicsEffect(shadow)

        # Style based on sender
        if is_user:
            bg_color = "#DCF8C6"  # WhatsApp Green
            text_color = "#000000"
            border_radius = "15px 15px 0px 15px"
            margin = "margin-left: 50px;"
        else:
            bg_color = "#FFFFFF"
            text_color = "#333333"
            border_radius = "15px 15px 15px 0px"
            margin = "margin-right: 50px;"
            
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border-radius: 15px; /* fallback */
                border-top-left-radius: {'15px' if is_user else '0px'};
                border-top-right-radius: {'0px' if is_user else '15px'};
                border-bottom-left-radius: 15px;
                border-bottom-right-radius: 15px;
            }}
            QLabel {{
                background-color: transparent;
                color: {text_color};
                font-size: 11pt;
                line-height: 1.4;
            }}
        """)
        
        # 1. Model Name (AI Only)
        if not is_user and model_name:
            lbl_model = QLabel(f"<small><b>{model_name}</b></small>")
            lbl_model.setStyleSheet("color: #1976D2; margin-bottom: 4px;")
            self.main_layout.addWidget(lbl_model)

        # Content Layout
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(4)
        
        # Thinking Widget (Phase 3)
        self.thinking_widget = ThinkingWidget(self)
        self.thinking_widget.setVisible(False)
        content_layout.addWidget(self.thinking_widget)
        
        # Text Label
        self.lbl_text = QLabel(text)
        self.lbl_text.setWordWrap(True)
        self.lbl_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_text.setStyleSheet(f"font-size: 11pt; color: {text_color}; line-height: 1.4;")
        content_layout.addWidget(self.lbl_text)
        
        # If assistant, handle thinking and answer parts
        if not self.is_user:
            self.update_content(text)
            
        self.main_layout.addLayout(content_layout)
            
        # 4. Action Buttons (Always visible)
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 5, 0, 0)
        actions.addStretch()
        
        if not is_user:
            # Refresh/Regenerate Button
            if self.related_prompt:
                btn_refresh = QPushButton("🔄")
                btn_refresh.setFixedSize(28, 28)
                btn_refresh.setToolTip("重新生成 (Regenerate)")
                btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
                btn_refresh.setStyleSheet("""
                    QPushButton { background-color: rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.1); border-radius: 4px; }
                    QPushButton:hover { background-color: rgba(0,0,0,0.15); }
                """)
                # Emit signal with stored prompt and model
                btn_refresh.clicked.connect(lambda: self.regenerate_requested.emit(self.related_prompt, self.model_name or ""))
                actions.insertWidget(0, btn_refresh) # Add to left of actions

            for icon, slot, tooltip in [("📋", self.copy_text, "复制"), ("🗑️", self.delete_me, "删除")]:
                btn = QPushButton(icon)
                btn.setFixedSize(28, 28) # Slightly larger
                btn.setToolTip(tooltip)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setStyleSheet("""
                    QPushButton { 
                        background-color: rgba(0,0,0,0.05); 
                        border: 1px solid rgba(0,0,0,0.1); 
                        border-radius: 4px; 
                    }
                    QPushButton:hover { 
                        background-color: rgba(0,0,0,0.15); 
                        border-color: rgba(0,0,0,0.3);
                    }
                """)
                btn.clicked.connect(slot)
                actions.addWidget(btn)
                
        self.main_layout.addLayout(actions)

    def update_content(self, full_text: str):
        """Robust parsing of <think> tags for UI update."""
        import re
        self.full_text = full_text
        
        # Find thoughts vs answer
        thoughts = re.findall(r'<think>(.*?)</think>', full_text, re.DOTALL)
        thought_content = "\n\n".join([t.strip() for t in thoughts])
        
        # If there's an open <think> but no close yet (streaming)
        partial_thought = ""
        if "<think>" in full_text and "</think>" not in full_text:
             partial_thought = full_text.split("<think>")[-1].strip()
        
        all_thoughts = (thought_content + "\n\n" + partial_thought).strip()
        self.thinking_widget.set_text(all_thoughts)
        
        # Cleaned text for main display
        display_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL)
        display_text = display_text.split("<think>")[0].strip() # Remove partial thinking tag
        
        if not display_text and all_thoughts:
            self.lbl_text.setText("<i>正在深度思考中...</i>")
        else:
            self.lbl_text.setText(display_text)

    def update_text(self, text):
        """Typewriter-compatible update."""
        self.update_content(text)

    def copy_text(self):
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.lbl_text.text())

    def delete_me(self):
        # Remove the wrapper layout from the parent list
        # Standard Qt way to remove wrapper widget:
        parent = self.parentWidget() 
        # parent is likely the widget inside the scroll area or the wrapper widget
        # If we wrapped this bubble in a layout item, we need to remove THAT.
        # But 'setParent(None)' usually works for the widget itself.
        # We need to signal the panel to remove the Row layout.
        self.setParent(None)
        self.deleteLater()
        
    def sizeHint(self):
        # Limit width
        s = super().sizeHint()
        s.setWidth(min(s.width(), 600)) # improved max width
        return s

class ChatInput(QTextEdit):
    image_pasted = pyqtSignal(QPixmap)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        
    def insertFromMimeData(self, source):
        if source.hasImage():
            image = QPixmap.fromImage(source.imageData())
            if not image.isNull():
                self.image_pasted.emit(image)
                return
        super().insertFromMimeData(source)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    pixmap = QPixmap(path)
                    if not pixmap.isNull():
                        self.image_pasted.emit(pixmap)
            event.acceptProposedAction()
        elif event.mimeData().hasImage():
            self.image_pasted.emit(QPixmap.fromImage(event.mimeData().imageData()))
            event.acceptProposedAction()

class AgentPanel(QFrame):
    send_message = pyqtSignal(str, str) # text, model
    regenerate_requested = pyqtSignal(str, str) # prompt, model
    stop_requested = pyqtSignal()
    edit_requested = pyqtSignal(str)   # text to edit
    
    def __init__(self):
        super().__init__()
        self.last_user_prompt = None # Track last prompt for regeneration
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedWidth(400)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. Header with Model Switcher
        header_v = QVBoxLayout()
        header_h = QHBoxLayout()
        lbl_title = QLabel("🤖 AI Agent")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.combo_model = QComboBox()
        self.combo_model.setPlaceholderText("选择对话模型...")
        self.combo_model.setMinimumWidth(150)
        
        self.btn_new = QPushButton("🆕")
        self.btn_new.setFixedSize(30, 30)
        self.btn_new.setToolTip("新建会话 (New Session)")
        self.btn_new.clicked.connect(self.clear)
        
        btn_close = QPushButton("×")
        btn_close.setFixedSize(30, 30)
        btn_close.clicked.connect(self.hide)
        
        header_h.addWidget(lbl_title)
        header_h.addStretch()
        header_h.addWidget(self.combo_model)
        header_h.addWidget(self.btn_new)
        header_h.addWidget(btn_close)
        header_v.addLayout(header_h)
        
        # Context Progress
        self.progress_context = QProgressBar()
        self.progress_context.setMaximum(100)
        self.progress_context.setValue(0)
        self.progress_context.setTextVisible(True)
        self.progress_context.setFormat("上下文占用: %p%")
        self.progress_context.setStyleSheet("height: 12px; font-size: 8pt;")
        header_v.addWidget(self.progress_context)
        
        self.layout.addLayout(header_v)
        
        # 2. Chat Area (Scrollable Widgets)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.chat_layout = QVBoxLayout(self.scroll_content)
        self.chat_layout.addStretch() # Push bubbles to bottom
        self.scroll.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll)
        
        # 3. Thought/Status Monitor
        self.thought_box = QFrame()
        self.thought_box.setFrameShape(QFrame.Shape.StyledPanel)
        # Reduced max height and margin to minimize whitespace
        self.thought_box.setStyleSheet("background: transparent; border: none;") 
        t_layout = QVBoxLayout(self.thought_box)
        t_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_thoughts = QLabel("<i>等待任务...</i>")
        self.lbl_thoughts.setWordWrap(True)
        self.lbl_thoughts.setFont(QFont("Consolas", 8)) # Smaller font
        self.lbl_thoughts.setStyleSheet("color: gray;")
        t_layout.addWidget(self.lbl_thoughts)
        self.layout.addWidget(self.thought_box)
        
        # 4. Input Area
        input_container = QVBoxLayout()
        input_container.setSpacing(4)
        
        # Pending Image Area (NEW)
        self.pending_img_area = QHBoxLayout()
        input_container.addLayout(self.pending_img_area)
        
        input_h = QHBoxLayout()
        self.input_msg = ChatInput()
        self.input_msg.setPlaceholderText("发送消息 (支持粘贴/拖入图片)...")
        self.input_msg.setMaximumHeight(80)
        self.input_msg.image_pasted.connect(self.add_pending_image)
        
        btn_send_layout = QVBoxLayout()
        self.btn_send = QPushButton("发送")
        self.btn_send.setFixedSize(50, 40)
        self.btn_send.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_send.clicked.connect(self.handle_send)
        
        self.btn_stop = QPushButton("🛑")
        self.btn_stop.setFixedSize(50, 30)
        self.btn_stop.setEnabled(False)
        
        btn_send_layout.addWidget(self.btn_send)
        btn_send_layout.addWidget(self.btn_stop)
        
        input_h.addWidget(self.input_msg)
        input_h.addLayout(btn_send_layout)
        input_container.addLayout(input_h)
        
        self.layout.addLayout(input_container)
        
        self.edit_requested.connect(self.on_edit_requested)
        self.btn_stop.clicked.connect(self.stop_requested.emit)

    def on_edit_requested(self, text):
        self.input_msg.setPlainText(text)
        self.input_msg.setFocus()

    def add_pending_image(self, pixmap):
        if not hasattr(self, 'pending_images'): self.pending_images = []
        
        container = QFrame()
        container.setFixedSize(60, 60)
        container.setStyleSheet("border: 1px solid #4CAF50; border-radius: 5px;")
        l = QVBoxLayout(container)
        l.setContentsMargins(2, 2, 2, 2)
        
        img_lbl = QLabel()
        img_lbl.setPixmap(pixmap.scaled(56, 56, Qt.AspectRatioMode.KeepAspectRatio))
        l.addWidget(img_lbl)
        
        self.pending_img_area.addWidget(container)
        self.pending_images.append(pixmap)

    def handle_send(self):
        text = self.input_msg.toPlainText().strip()
        model = self.combo_model.currentText()
        images = getattr(self, 'pending_images', [])
        
        if text or images:
            self.append_message("User", text, is_user=True)
            # Emit text and images (images as pixmaps for now, logic will convert)
            self.send_message.emit(text, model) 
            self.input_msg.clear()
            self.clear_pending_images()
            self.check_context_limit()

    def inject_context(self, context_text):
        """Injects a system/context message into the chat history without sending it to the model yet."""
        # Visual feedback
        self.append_message("System", "已更新视频分析上下文信息。", is_user=False)
        self.lbl_thoughts.setText("<i>上下文已更新，AI 已知悉视频内容。</i>")
        
        # We can store it or emit it? 
        # Since AgentPanel is UI only, main_window needs to handle the logic. 
        # But we can allow main_window to call this for visual effect.
        pass


    def check_context_limit(self):
        count = self.chat_layout.count() - 1 
        # Heuristic for 200k tokens (~150k words/chars). 
        # Assume average message is generous 500 chars. 
        # 200,000 / 500 = 400 messages. 
        # Let's set 200 messages as 100% for safety (system prompt etc).
        total_capacity = 200 
        usage = min(100, int((count / total_capacity) * 100)) 
        self.progress_context.setValue(usage)
        self.progress_context.setFormat(f"上下文占用: {usage}% (估算)")
        
    def compress_history(self):
        # Auto-summarize or just remove oldest pairs
        if self.chat_layout.count() > 150: # Only compress if truly long
            for _ in range(20): # Remove a chunk
                item = self.chat_layout.takeAt(0)
                if item.widget(): item.widget().deleteLater()
            
            # Add a system indicator
            lbl = QLabel("<i>[自动压缩] 为了保持运行流畅，较早的对话上下文已被提炼压缩。</i>")
            lbl.setStyleSheet("color: #FF9800; font-size: 8pt; alignment: center;")
            self.chat_layout.insertWidget(0, lbl)
            self.update_thoughts("上下文已自动压缩以释放内存。")

    def clear_pending_images(self):
        while self.pending_img_area.count():
            item = self.pending_img_area.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.pending_images = []

    def append_message(self, sender, text, is_user=False, model_name=None):
        related = None
        if is_user:
            self.last_user_prompt = text
        else:
            related = getattr(self, 'last_user_prompt', None)
            
        bubble = ChatBubble(sender, text, model_name, is_user, related_prompt=related)
        if not is_user:
            bubble.regenerate_requested.connect(self.regenerate_requested.emit)
        
        # Row Layout for Alignment
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 4, 0, 4)
        
        if is_user:
            row_layout.addStretch()
            row_layout.addWidget(bubble, stretch=0) # Let bubble determine size, max 600
        else:
            row_layout.addWidget(bubble, stretch=0)
            row_layout.addStretch()
            
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row_widget)
        
        # Scroll to bottom logic:
        # Check if we were already at bottom
        sb = self.scroll.verticalScrollBar()
        was_at_bottom = (sb.value() >= sb.maximum() - 10)
        
        self.last_bubble = bubble 
        self.last_full_text = text # Store full raw text for state machine parsing
        
        # Update context mock
        self.check_context_limit()
        
        if was_at_bottom:
             QTimer.singleShot(10, lambda: sb.setValue(sb.maximum()))

    def update_last_bubble(self, chunk):
        if not hasattr(self, 'last_bubble') or not self.last_bubble:
            return

        # Append new chunk to full text
        self.last_full_text += chunk
        full_text = self.last_full_text
        
        # Update Bubble using the robust parser in ChatBubble
        self.last_bubble.update_text(full_text)
        
        # Smart Auto Scroll
        # Only scroll if user hasn't scrolled up
        sb = self.scroll.verticalScrollBar()
        if sb.value() >= sb.maximum() - 100: # Tolerance
             sb.setValue(sb.maximum())

    def update_thoughts(self, text):
        self.lbl_thoughts.setText(f"<i>AI 正在思考: {text}</i>")

    def clear(self):
        # Clear bubbles
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.progress_context.setValue(0)
        self.lbl_thoughts.setText("<i>等待任务...</i>")
