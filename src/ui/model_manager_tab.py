from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, QFrame, QScrollArea, QGridLayout
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path

class ModelCard(QFrame):
    download_requested = pyqtSignal(str) # model_id
    detect_requested = pyqtSignal(str)   # model_id/filename
    
    def __init__(self, name, description, size, model_id):
        super().__init__()
        self.model_id = model_id
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.layout = QVBoxLayout(self)
        
        self.lbl_name = QLabel(f"<b>{name}</b>")
        self.lbl_desc = QLabel(description)
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setStyleSheet("color: gray; font-size: 11px;")
        
        self.lbl_status = QLabel("状态: 未下载")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet("height: 15px; font-size: 10px;")
        
        btn_layout = QHBoxLayout()
        self.btn_action = QPushButton("📥 下载")
        self.btn_action.clicked.connect(lambda: self.download_requested.emit(self.model_id))
        
        self.btn_health = QPushButton("🔍 校验")
        self.btn_health.setFixedWidth(50)
        self.btn_health.setToolTip("检查文件完整性")
        
        btn_layout.addWidget(self.btn_action)
        btn_layout.addWidget(self.btn_health)
        
        self.btn_detect = QPushButton("🔭 检测类型")
        self.btn_detect.clicked.connect(lambda: self.detect_requested.emit(self.model_id))
        self.btn_detect.setVisible(False) # Only for local files
        btn_layout.addWidget(self.btn_detect)
        
        self.layout.addWidget(self.lbl_name)
        self.layout.addWidget(self.lbl_desc)
        self.lbl_size = QLabel(f"大小: {size}")
        self.lbl_size.setToolTip("实际磁盘占用 (models/ 目录下的真实文件大小)")
        self.layout.addWidget(self.lbl_size)
        self.layout.addWidget(self.lbl_status)
        self.layout.addWidget(self.progress)
        self.layout.addLayout(btn_layout)

    def set_actual_size(self, file_path):
        """用模型文件的真实磁盘大小替换'预计大小'。"""
        from pathlib import Path
        p = Path(file_path)
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            self.lbl_size.setText(f"实际占用: {size_mb:.1f} MB")

    def set_downloading(self):
        self.progress.setVisible(True)
        self.btn_action.setEnabled(False)
        self.btn_action.setText("正在下载...")
        self.lbl_status.setText("状态: 下载中")

    def set_ready(self):
        self.progress.setVisible(False)
        self.btn_action.setEnabled(True)
        self.btn_action.setText("🗑️ 删除并重新下载")
        self.btn_action.setStyleSheet("color: #F44336;")
        self.lbl_status.setText("状态: ✅ 已就绪")

class ModelManagerTab(QWidget):
    download_all_requested = pyqtSignal()
    detect_requested = pyqtSignal(str) # filename
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        header_layout = QHBoxLayout()
        header = QLabel("<h3>模型下载与管理</h3>")
        self.btn_download_all = QPushButton("⏬ 全部下载(缺失)")
        self.btn_download_all.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        self.btn_download_all.clicked.connect(self.download_all_requested.emit)
        
        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_download_all)
        self.layout.addLayout(header_layout)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        
        # Add Cards
        self.cards = {}
        
        models = [
            ("YOLOv11n (目标检测)", "用于识别视频每一帧中的物体 (人, 车, 物品等)", "5.4 MB", "yolo_v11n"),
            ("Whisper Base (音频转码)", "高性能音频转文本模型, 支持多语言", "145 MB", "whisper_base"),
            ("Sentence-Transformer (RAG)", "用于文本语义分析与搜索", "23 MB", "st_minilm"),
            ("FFmpeg 核心组件", "视频处理的必要基础库", "依赖系统", "ffmpeg"),
        ]
        
        for i, (name, desc, size, mid) in enumerate(models):
            card = ModelCard(name, desc, size, mid)
            self.grid.addWidget(card, i // 2, i % 2)
            self.cards[mid] = card
            
        self.scroll.setWidget(self.content)
        self.layout.addWidget(self.scroll)
        
        # Help Info
        help_info = QLabel("💡 提示: 模型将保存到 'models' 文件夹。您也可以将自己的 (.gguf / .pt) 模型放入该目录，并在首页'本地模型'模式中调用。")
        help_info.setStyleSheet("color: #2196F3; font-style: italic;")
        help_info.setWordWrap(True)
        self.layout.addWidget(help_info)
        
        self.scroll_local = QScrollArea()
        self.scroll_local.setWidgetResizable(True)
        self.local_content = QWidget()
        self.local_grid = QGridLayout(self.local_content)
        self.scroll_local.setWidget(self.local_content)
        self.scroll_local.setMaximumHeight(300)
        self.layout.addWidget(self.scroll_local)

    def refresh_local_cards(self, model_files):
        # Clear local grid
        while self.local_grid.count():
            item = self.local_grid.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()
            
        known_ids = ["yolo_v11n", "yolo_v8n", "whisper_base", "st_minilm", "ffmpeg"]
        
        row, col = 0, 0
        for f in model_files:
            # Skip if it's already in the "Known/Mandatory" list to avoid duplication
            is_known = any(kid in f.lower() for kid in known_ids)
            if is_known: continue
            
            card = ModelCard(f, "扫描到的本地模型文件", "未知", f)
            card.btn_detect.setVisible(True)
            card.detect_requested.connect(self.detect_requested.emit)
            self.local_grid.addWidget(card, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1

    def update_model_status(self, model_id, exists, actual_path=None):
        if model_id in self.cards:
            if exists:
                self.cards[model_id].set_ready()
            else:
                # Reset if deleted
                pass
        if actual_path is not None:
            self.cards[model_id].set_actual_size(actual_path)
