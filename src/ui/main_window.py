import sys
import os
import logging
import json
import requests
import time

from pathlib import Path

# Fix module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QTabWidget, QPushButton, QTextEdit, QFileDialog, 
                             QProgressBar, QComboBox, QCheckBox, QSlider, QGroupBox, QSplitter,
                             QScrollArea, QToolBox, QSizePolicy, QFrame, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QDateTime
from PyQt6.QtGui import QIcon, QAction, QPixmap
from logging.handlers import RotatingFileHandler
import qdarktheme

# ... (Previous imports remain) ...
from src.core.logic import (VideoProcessor, VideoAnalyzer, AudioProcessor, PromptLoader, 
                        APIGatewayClient, ModelContextManager,
                        logger as core_logger, Frame, AudioTranscript,
                        ADVANCED_FEATURES_AVAILABLE, NVIDIA_GPU_AVAILABLE)
from src.utils.constants import APP_NAME, APP_VERSION, LOG_DIR
from src.utils.config_manager import ConfigurationManager
from src.core.agent_tools import (ToolRegistry, create_get_video_meta_tool, 
                                  create_get_frame_details_tool, create_delete_history_tool,
                                  create_search_web_tool, create_visual_search_tool, create_ocr_tool)
import cv2 # For thumbnails
from src.ui.status_console import StatusConsole
from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal
from src.ui.agent_panel import AgentPanel
from src.ui.model_manager_tab import ModelManagerTab
from src.ui.api_intro_page import APIIntroPage
from src.ui.help_dialog import HelpDialog
from src.ui.carousel_widget import CarouselWidget

class ImageLoaderSignals(QObject):
    loaded = pyqtSignal(object, object) # frame_obj, pixmap

class ImageLoader(QRunnable):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        self.signals = ImageLoaderSignals()

    def run(self):
        try:
            if self.frame.path.exists():
                pix = QPixmap(str(self.frame.path))
                # 预先缩放，减少主线程内存压力
                scaled = pix.scaled(220, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.signals.loaded.emit(self.frame, scaled)
        except: pass

class QtLogHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        try:
            msg = self.format(record)
            self.signal.emit(msg)
        except (AttributeError, RuntimeError, Exception):
            # Fail silently during app shutdown or initialization errors
            pass

class ExtractionWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    
    def __init__(self, video_path, config):
        super().__init__()
        self.video_path = Path(video_path)
        self.config = config
        
    def run(self):
        try:
            self.log.emit(f"开始处理: {self.video_path.name}")
            
            # Setup Output - Use centralized cache folder in app root
            from src.utils.constants import CACHE_DIR
            base_dir = Path(__file__).parent.parent.parent  # App root
            output_dir = base_dir / CACHE_DIR / self.video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Video Processing
            processor = VideoProcessor(self.video_path, output_dir)
            density = self.config.get("extraction_density", 0.2)
            
            # Note: extract_keyframes in logic.py handles sampling
            frames = processor.extract_keyframes(density=density, max_frames=10000)
            self.log.emit(f"提取了 {len(frames)} 个关键帧")
            
            # 2. Audio
            transcript = ""
            if self.config.get("enable_audio"):
                self.log.emit("开始音频转录...")
                audio_proc = AudioProcessor()
                audio_path = audio_proc.extract_audio(self.video_path, output_dir)
                if audio_path:
                    # Check if transcribe returns object or text. logic.py says AudioTranscript object
                    res = audio_proc.transcribe(audio_path)
                    transcript = res.text if res else ""
                    self.log.emit("音频转录完成")
            
            result = {
                "frames": frames,
                "transcript": transcript,
                "output_dir": output_dir
            }
            self.finished.emit(result)
            
        except Exception as e:
            self.log.emit(f"Extraction Error: {e}")
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished.emit({})

class AnalysisWorker(QThread):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, analyzer, frames, transcript, custom_prompt=None):
        super().__init__()
        self.analyzer = analyzer
        self.frames = frames
        self.transcript = transcript
        self.custom_prompt = custom_prompt
        
    def run(self):
        if not self.analyzer: return
        try:
            for chunk in self.analyzer.analyze_video(self.frames, self.transcript, self.custom_prompt):
                self.chunk_received.emit(chunk)
            self.finished.emit()
        except Exception as e:
            self.chunk_received.emit(f"\nError: {e}")
            self.finished.emit()

class ChatWorker(QThread):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, client, model, prompt, image_paths=None, tool_registry=None):
        super().__init__()
        self.client = client
        self.model = model
        self.prompt = prompt
        self.image_paths = image_paths
        self.tool_registry = tool_registry
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        import re
        import json
        
        # XML pattern: <tool name="...">{...}</tool>
        # Supporting multi-turn ReAct
        tool_pattern = re.compile(r'<tool name="(\w+)">(.*?)</tool>', re.DOTALL)
        
        # New pattern for "Thought:", "Action:", "Observation:"
        react_pattern = re.compile(r'Action:\s*(\w+)\((.*?)\)', re.IGNORECASE)
        
        current_prompt = self.prompt
        if "请使用中文总结" not in current_prompt:
             current_prompt += "\n\n请始终使用中文回答我。"
             
        max_turns = 10
        
        try:
            for turn in range(max_turns):
                if not self._is_running: break
                
                full_response = ""
                # Stream content
                for chunk in self.client.chat_stream(self.model, current_prompt, self.image_paths):
                     if not self._is_running: break
                     if chunk:
                         self.chunk_received.emit(chunk)
                         full_response += chunk

                
                if not self._is_running: break

                # Robust tool extraction (XML or Logic)
                match = tool_pattern.search(full_response) or react_pattern.search(full_response)
                
                if match and self.tool_registry:
                    tool_name = match.group(1)
                    args_str = match.group(2).strip()
                    
                    self.chunk_received.emit(f"\n\n🛠️ 正在调用工具: {tool_name}...\n")
                    
                    try:
                        # Handle both JSON and simple comma-sep args
                        if args_str.startswith("{"):
                            args = json.loads(args_str)
                        else:
                            # Positional args logic for Action: tool(val)
                            # This is a bit complex, let's assume JSON for schema tools
                            args = {"query": args_str} if "search" in tool_name else {"seconds": args_str}
                            
                        result = self.tool_registry.execute_tool_call(tool_name, args)
                        
                        tool_output_block = f"\n\nObservation: [工具返回结果]\n{result}\n"
                        self.chunk_received.emit(f"✅ 工具返回: {str(result)[:100]}...\n")
                        self.chunk_received.emit(f"\n🤖 正在根据结果思考下一步...\n")
                        
                        # Add to context
                        current_prompt = f"{current_prompt}\n\nAgent: {full_response}\n{tool_output_block}\nSystem: 请基于以上观察继续回答用户。"
                        continue
                        
                    except Exception as e:
                        self.chunk_received.emit(f"\n❌ 工具执行出错: {e}\n")
                        break
                
                # No more tools, stop
                break
                
            self.finished.emit()
        except Exception as e:
            self.chunk_received.emit(f"Error: {e}")
            self.finished.emit()

class MediaWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(list)
    
    def __init__(self, video_path, frames, output_dir):
        super().__init__()
        self.video_path = video_path
        self.frames = frames
        self.output_dir = output_dir
        
    def run(self):
        try:
            from src.core.logic import create_summary_media_artifacts
            self.log.emit("开始生成摘要媒体 clips...")
            # Default settings matching legacy
            clips, _, video, gif = create_summary_media_artifacts(
                str(self.video_path), 0, self.frames, self.output_dir, self.video_path.stem,
                make_video=True, make_gif=True
            )
            results = []
            if clips: results.extend(clips)
            if video: results.append(video)
            if gif: results.append(gif)
            
            self.log.emit(f"生成完成. 共 {len(results)} 个文件")
            self.finished.emit(results)
        except Exception as e:
             self.log.emit(f"Media Gen Error: {e}")
             self.finished.emit([])

class ModelDownloadWorker(QThread):
    progress = pyqtSignal(str, int) # model_id, percent
    finished = pyqtSignal(str, bool) # model_id, success
    
    def __init__(self, manager, model_id):
        super().__init__()
        self.manager = manager
        self.model_id = model_id
        
    def run(self):
        success = self.manager.download_model(self.model_id, lambda p: self.progress.emit(self.model_id, p))
        self.finished.emit(self.model_id, success)

class OllamaRefreshWorker(QThread):
    models_ready = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def run(self):
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                models = [m['name'] for m in data.get('models', [])]
                self.models_ready.emit(models)
            else:
                self.error.emit(f"HTTP {resp.status_code}")
        except Exception as e:
            self.error.emit(str(e))

class ModelLoadWorker(QThread):
    finished = pyqtSignal(object, str, str) # analyzer, model_name, error_msg

    def __init__(self, client, model_name, prompt_loader):
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.prompt_loader = prompt_loader

    def run(self):
        try:
            from src.core.logic import VideoAnalyzer
            analyzer = VideoAnalyzer(self.client, self.model_name, self.prompt_loader)
            self.finished.emit(analyzer, self.model_name, "")
        except Exception as e:
            self.finished.emit(None, self.model_name, str(e))

class ApiCheckWorker(QThread):
    finished = pyqtSignal(list, str, str) # models, error_msg, chat_endpoint

    def __init__(self, url, key):
        super().__init__()
        self.url = url
        self.key = key

    def run(self):
        try:
            from src.core.logic import APIGatewayClient
            client = APIGatewayClient(self.key, self.url)
            models = client.list_models()
            self.finished.emit(models, "", client.chat_endpoint)
        except Exception as e:
            self.finished.emit([], str(e), "")

class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.bg = QFrame(self)
        self.bg.setObjectName("LoadingBG")
        self.bg.setStyleSheet("#LoadingBG { background-color: rgba(0, 0, 0, 180); border-radius: 10px; }")
        self.bg.setFixedSize(400, 200)
        
        bg_layout = QVBoxLayout(self.bg)
        self.lbl_msg = QLabel("🚀 正在检查必要模型组件...")
        self.lbl_msg.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        self.lbl_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.progress = QProgressBar()
        self.progress.setStyleSheet("QProgressBar { height: 20px; border-radius: 5px; text-align: center; }")
        
        bg_layout.addWidget(self.lbl_msg)
        bg_layout.addWidget(self.progress)
        self.layout.addWidget(self.bg)
        
        self.hide()

    def show_msg(self, msg, progress=None):
        self.lbl_msg.setText(msg)
        if progress is not None:
            self.progress.setValue(progress)
        self.show()
        self.raise_()

    def resizeEvent(self, event):
        self.setGeometry(self.parent().rect())
        super().resizeEvent(event)

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

class DesktopApp(QMainWindow):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1400, 900)
        
        # Setup Logging
        self.log_handler = QtLogHandler(self.log_signal)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Comprehensive Master Log (100MB limit)
        abs_log_dir = os.path.abspath(LOG_DIR)
        os.makedirs(abs_log_dir, exist_ok=True)
        master_log_path = os.path.join(abs_log_dir, "主程序.log")
        
        # Cleanup temporary log name if exists
        temp_log = os.path.join(abs_log_dir, "main_app.log")
        if os.path.exists(temp_log):
            try: os.remove(temp_log)
            except: pass

        self.file_handler = RotatingFileHandler(
            master_log_path, 
            maxBytes=100 * 1024 * 1024, 
            backupCount=3, 
            encoding='utf-8'
        )
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            
        # Terminal Handler
        self.stream_handler = logging.StreamHandler(sys.stdout)
        self.stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(self.log_handler)
        root_logger.addHandler(self.file_handler)
        root_logger.addHandler(self.stream_handler)
        
        core_logger.addHandler(self.log_handler)
        core_logger.addHandler(self.file_handler)
        core_logger.addHandler(self.stream_handler)
        
        self.log_signal.connect(self.append_log)
        
        # Suppress Matplotlib font debugging
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.ticker').setLevel(logging.WARNING)

        self.config_manager = ConfigurationManager()
        self.app_config = self.config_manager.load_main_config()
        self.api_presets = self.config_manager.load_api_presets()
        self.prompt_templates = self.config_manager.load_prompts()
        
        self.init_backend()
        self.setup_ui()
        self.load_settings()
        
        # UI Sync: Force call on_client_changed to align visibility
        self.on_client_changed(self.combo_client.currentIndex())
        
        # Apply startup theme
        self.apply_theme(self.app_config['Application'].get('theme', 'dark'))
        
        logging.info(f"=== {APP_NAME} 启动 (Detailed Logging Enabled) ===")
        
        # Phase 2: Core Components
        self.vram_manager = ModelContextManager()
        
        # Trigger Startup Scan & Self-Healing
        QTimer.singleShot(500, self.run_startup_scan)
        QTimer.singleShot(1000, self.check_ffmpeg)
        
        # Thread Pool for Image Loading
        self.image_pool = QThreadPool()
        self.image_pool.setMaxThreadCount(4)
        
        # VRAM / Memory Monitor Timer
        self.timer_vram = QTimer(self)
        self.timer_vram.timeout.connect(self.update_stats)
        self.timer_vram.start(2000)

    def check_ffmpeg(self):
        """Self-healing FFmpeg check logic."""
        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
            os.environ["IMAGEIO_FFMPEG_EXE"] = path
            logging.info(f"FFmpeg (Self-Healing) active: {path}")
        except ImportError:
            logging.warning("imageio-ffmpeg not found. Recommend: pip install imageio-ffmpeg")
        except Exception as e:
            logging.error(f"FFmpeg check failed: {e}")

    def closeEvent(self, event):
        """Robust resource cleanup on exit."""
        logging.info("Shutting down... releasing VRAM and stopping workers.")
        if hasattr(self, 'timer_vram'):
            self.timer_vram.stop()
            
        # Unload all models
        if self.vram_manager:
            for m in list(self.vram_manager.active_models.keys()):
                self.vram_manager.unload(m)
                
        # Stop potential background works
        # ... logic for worker stops ...
        
        logging.shutdown()
        event.accept()

    def update_stats(self):
        """Update VRAM and CPU stats in status bar or console."""
        try:
            if NVIDIA_GPU_AVAILABLE:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_gb = mem.used / 1024**3
                total_gb = mem.total / 1024**3
                self.status_bar.showMessage(f"VRAM: {used_gb:.1f}G / {total_gb:.1f}G  | CPU: {psutil.cpu_percent()}%")
        except: pass

    def setup_ui(self):
        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        # Root layout: Vertical (Top: Splitter, Bottom: StatusConsole)
        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(5, 5, 5, 5)

        # Container for Sidebar and Results
        middle_widget = QWidget()
        middle_layout = QHBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(0) # No gap between sidebar and tabs
        
        # --- LEFT SIDEBAR (Controls) ---
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header / Theme Toggle
        header_layout = QHBoxLayout()
        header_lbl = QLabel(f"<b>{APP_NAME}</b>")
        header_lbl.setStyleSheet("font-size: 16px;")
        header_layout.addWidget(header_lbl)
        header_layout.addStretch()
        
        self.btn_theme = QPushButton("🌗")
        self.btn_theme.setFixedSize(30, 30)
        self.btn_theme.setToolTip("切换深色/浅色主题 (Toggle Theme)")
        self.btn_theme.clicked.connect(lambda: self._instrumented_call(self.toggle_theme, "切换主题"))
        header_layout.addWidget(self.btn_theme)
        sidebar_layout.addLayout(header_layout)

        # Container for Settings (Scrollable)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_content = QWidget()
        settings_layout = QVBoxLayout(settings_content)
        settings_layout.setContentsMargins(0, 0, 5, 0) # Slight right margin for scrollbar
        
        # 1. Model Config Group
        self.group_model_outer = QGroupBox("1. 模型配置")
        self.group_model_outer.setObjectName("sidebarGroup")
        layout_model = QVBoxLayout(self.group_model_outer)
        layout_model.setSpacing(12)
        
        # Client Type
        self.combo_client = QComboBox()
        self.combo_client.addItems(["Ollama (Local)", "API 网关 (OpenAI/DeepSeek)", "LM Studio (Local/V1)", "本地模型文件 (.gguf/.pt)"])
        self.combo_client.setCurrentIndex(1) # Default to API Gridway
        self.combo_client.currentIndexChanged.connect(self.on_client_changed)
        layout_model.addWidget(QLabel("客户端类型:"))
        layout_model.addWidget(self.combo_client)
        
        # API Config Group (Hidden by default)
        self.grp_api = QGroupBox("API 设置")
        layout_api = QVBoxLayout()
        self.txt_api_url = QTextEdit()
        self.txt_api_url.setPlaceholderText("API Base URL (e.g. http://localhost:1234/v1)")
        self.txt_api_url.setFixedHeight(30)
        self.txt_api_url.setText("http://localhost:1234/v1")
        self.txt_api_url.textChanged.connect(self.on_api_url_changed) # Connect change signal
        layout_api.addWidget(QLabel("API URL:"))
        layout_api.addWidget(self.txt_api_url)

        # Real-time Preview Label
        self.lbl_api_preview = QLabel("")
        self.lbl_api_preview.setStyleSheet("color: gray; font-size: 11px; font-style: italic;")
        self.lbl_api_preview.setWordWrap(True)
        layout_api.addWidget(self.lbl_api_preview)

        self.txt_api_key = QTextEdit()
        self.txt_api_key.setPlaceholderText("API Key (Optional)")
        self.txt_api_key.setFixedHeight(30)
        layout_api.addWidget(QLabel("API Key:"))
        layout_api.addWidget(self.txt_api_key)

        # Model Selection with Auto-Complete
        self.combo_api_model = QComboBox()
        self.combo_api_model.setEditable(True)
        self.combo_api_model.setPlaceholderText("Select or enter Model ID...")
        layout_api.addWidget(QLabel("模型名称:"))
        layout_api.addWidget(self.combo_api_model)

        # Check Connection Button
        self.btn_check_api = QPushButton("🔍 检测连接 & 获取模型")
        self.btn_check_api.clicked.connect(lambda: self._instrumented_call(self.check_api_connection, "检测 API 连接"))
        layout_api.addWidget(self.btn_check_api)

        self.grp_api.setLayout(layout_api)
        self.grp_api.setVisible(False)
        layout_model.addWidget(self.grp_api)

        # Ollama Config Group
        self.grp_ollama = QGroupBox("Ollama 设置")
        layout_ollama = QVBoxLayout()
        self.combo_ollama_model = QComboBox()
        btn_refresh_ollama = QPushButton("🔄 刷新模型")
        btn_refresh_ollama.clicked.connect(lambda: self._instrumented_call(self.refresh_ollama_models, "刷新 Ollama 模型"))
        layout_ollama.addWidget(QLabel("选择模型:"))
        layout_ollama.addWidget(self.combo_ollama_model)
        layout_ollama.addWidget(btn_refresh_ollama)
        self.grp_ollama.setLayout(layout_ollama)
        layout_model.addWidget(self.grp_ollama)

        # API Presets Toolbar
        preset_layout = QHBoxLayout()
        self.combo_presets = QComboBox()
        self.combo_presets.addItem("-- 选择推荐/保存的预设 --")
        self._refresh_preset_combo()
        self.combo_presets.currentIndexChanged.connect(self.on_preset_selected)
        
        btn_add_new = QPushButton("🆕 新增")
        btn_add_new.setToolTip("清空当前输入，准备新增预设")
        btn_add_new.clicked.connect(self.clear_api_fields)
        
        btn_save_current = QPushButton("💾 保存")
        btn_save_current.setToolTip("保存或更新当前配置到预设")
        btn_save_current.clicked.connect(self.save_current_as_preset)
        
        btn_del_preset = QPushButton("🗑️")
        btn_del_preset.setFixedSize(30,30)
        btn_del_preset.setToolTip("删除当前选中的预设")
        btn_del_preset.clicked.connect(self.delete_selected_preset)
        
        preset_layout.addWidget(self.combo_presets)
        preset_layout.addWidget(btn_add_new)
        preset_layout.addWidget(btn_save_current)
        preset_layout.addWidget(btn_del_preset)
        
        layout_model.addLayout(preset_layout)

        # Bottom of Model Config: Unload
        self.btn_unload = QPushButton("🔌 释放/卸载当前模型 (Ollama)")
        self.btn_unload.clicked.connect(self.unload_ollama_model)
        self.btn_unload.setVisible(False) # Only show for Ollama
        layout_model.addWidget(self.btn_unload)

        self.btn_load_model = QPushButton("✅ 加载模型")
        self.btn_load_model.clicked.connect(lambda: self._instrumented_call(self.load_model, "加载模型"))
        layout_model.addWidget(self.btn_load_model)
        
        settings_layout.addWidget(self.group_model_outer)

        # 2. Upload & Settings Group
        group_setup = QGroupBox("1. 上传与分析设置")
        group_setup.setObjectName("sidebarGroup")
        layout_setup = QVBoxLayout(group_setup)
        
        # 1. Video Selection with Drag & Drop
        class DragDropFrame(QFrame):
            file_dropped = pyqtSignal(str)
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setAcceptDrops(True)
                self.setStyleSheet("QFrame { border: 2px dashed #BDBDBD; border-radius: 8px; background: transparent; }")
                
            def dragEnterEvent(self, event):
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    self.setStyleSheet("QFrame { border: 2px dashed #2196F3; border-radius: 8px; background: rgba(33, 150, 243, 0.1); }")
                    
            def dragLeaveEvent(self, event):
                self.setStyleSheet("QFrame { border: 2px dashed #BDBDBD; border-radius: 8px; background: transparent; }")
                
            def dropEvent(self, event):
                self.setStyleSheet("QFrame { border: 2px dashed #BDBDBD; border-radius: 8px; background: transparent; }")
                if event.mimeData().hasUrls():
                    for url in event.mimeData().urls():
                        path = url.toLocalFile()
                        if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            self.file_dropped.emit(path)
                            break # Just take the first valid one
                    event.acceptProposedAction()
                    
        self.drag_area = DragDropFrame()
        drag_layout = QVBoxLayout(self.drag_area)
        drag_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_upload = QPushButton("📂 选择视频文件 (支持拖拽/粘贴)...")
        self.btn_upload.clicked.connect(lambda: self._instrumented_call(self.select_video, "选择视频"))
        self.btn_upload.setStyleSheet("text-align: left; padding: 10px; border: none; background: transparent;")
        self.btn_upload.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.lbl_file = QLabel("未选择文件")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color: gray; padding: 5px;")
        
        drag_layout.addWidget(self.btn_upload)
        drag_layout.addWidget(self.lbl_file)
        
        self.drag_area.file_dropped.connect(self.load_video_from_path)
        
        # Add paste support to the window usually, or handling Ctrl+V global shortcut (complex). 
        # Drag drop is easier. For paste, usually need a focused widget.
        
        layout_setup.addWidget(self.drag_area)
        
        layout_setup.addWidget(QLabel("提示词模板:"))
        self.combo_prompt = QComboBox()
        self._refresh_prompt_combo()
        self.combo_prompt.currentIndexChanged.connect(self.on_prompt_type_changed)
        layout_setup.addLayout(self._create_prompt_toolbar())
        layout_setup.addWidget(self.combo_prompt)
        
        self.txt_custom_prompt = QTextEdit()
        self.txt_custom_prompt.setPlaceholderText("在此输入自定义提示词...")
        self.txt_custom_prompt.setFixedHeight(80)
        self.txt_custom_prompt.setVisible(False)
        layout_setup.addWidget(self.txt_custom_prompt)
        
        self.chk_audio = QCheckBox("启用音频转录 (Audio)")
        self.chk_audio.setChecked(True)
        self.chk_smart = QCheckBox("智能关键帧提取 (Smart Keyframes)")
        self.chk_smart.setChecked(True)
        layout_setup.addWidget(self.chk_audio)
        layout_setup.addWidget(self.chk_smart)
        
        settings_layout.addWidget(group_setup)

        # 3. Advanced Group
        group_advanced = QGroupBox("3. 高级设置")
        group_advanced.setObjectName("sidebarGroup")
        layout_adv = QVBoxLayout(group_advanced)
        layout_adv.setSpacing(12)
        self.slider_frames = QSlider(Qt.Orientation.Horizontal)
        self.slider_frames.setRange(1, 100) # 1% to 100%
        self.slider_frames.setValue(20) # Default 20%
        self.lbl_slider_val = QLabel("提取密度: 20%")
        self.lbl_slider_val.setStyleSheet("color: #2196F3; font-weight: bold;")
        self.slider_frames.valueChanged.connect(self.update_slider_label)
        
        layout_adv.addWidget(QLabel("关键帧提取密度 (1% - 100%):"))
        layout_adv.addWidget(self.slider_frames)
        layout_adv.addWidget(self.lbl_slider_val)
        
        desc_lbl = QLabel("<small>提示：控制提取的精细度。100% 约每秒提取 2 帧，1% 为最低限度 (保证至少 5 帧)。根据视频时长自动调整。</small>")
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: gray;")
        layout_adv.addWidget(desc_lbl)
        
        settings_layout.addWidget(group_advanced)

        settings_layout.addStretch() # Push items to top
        settings_scroll.setWidget(settings_content)
        sidebar_layout.addWidget(settings_scroll)

        # Action Buttons
        self.btn_start = QPushButton("🚀 开始提取数据 (Phase 1)")
        self.btn_start.clicked.connect(lambda: self._instrumented_call(self.start_analysis_phase1, "开始提取 (Phase 1)"))
        self.btn_start.setEnabled(False)
        self.btn_start.setFixedHeight(40)
        
        self.btn_ai = QPushButton("🤖 生成 AI 总结 (Phase 2)")
        self.btn_ai.clicked.connect(lambda: self._instrumented_call(self.start_ai_analysis, "生成 AI 总结 (Phase 2)"))
        self.btn_ai.setEnabled(False) 
        self.btn_ai.setFixedHeight(40)
        
        self.btn_media = QPushButton("🎬 生成摘要媒体 (Phase 3)")
        self.btn_media.clicked.connect(lambda: self._instrumented_call(self.generate_summary_media, "生成摘要媒体 (Phase 3)"))
        self.btn_media.setEnabled(False) # Enabled after Phase 1
        self.btn_media.setFixedHeight(40)

        sidebar_layout.addWidget(self.btn_start)
        sidebar_layout.addWidget(self.btn_ai)
        sidebar_layout.addWidget(self.btn_media)
        
        sidebar_widget.setFixedWidth(460)
        middle_layout.addWidget(sidebar_widget)

        # --- RIGHT PANEL (Results) ---
        self.tabs = QTabWidget()
        
        # Tab 1: AI Report
        from src.core.logic import ModelManager
        self.model_manager = ModelManager()

        self.tab_report = QWidget()
        layout_report = QVBoxLayout(self.tab_report)
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        layout_report.addWidget(self.txt_report)
        self.btn_export = QPushButton("📄 导出 PDF 报告")
        self.btn_export.clicked.connect(lambda: self._instrumented_call(self.export_pdf, "导出 PDF 报告"))
        layout_report.addWidget(self.btn_export)
        self.tabs.addTab(self.tab_report, "📝 AI 摘要报告")
        
        # Tab 2: Gallery and Details (Master-Detail)
        self.tab_gallery = QWidget()
        layout_gallery_main = QVBoxLayout(self.tab_gallery)
        
        # 2.1 Carousel for Gallery
        self.gallery_carousel = CarouselWidget()
        layout_gallery_main.addWidget(self.gallery_carousel, stretch=3)
        
        # 2.2 Frame Details (Collapsible/Fixed Area)
        self.grp_frame_details = QGroupBox("🔍 选定帧详细信息")
        self.grp_frame_details.setFixedHeight(150)
        layout_details = QHBoxLayout(self.grp_frame_details)
        
        self.lbl_frame_img = QLabel("未选择")
        self.lbl_frame_img.setFixedSize(200, 120)
        self.lbl_frame_img.setStyleSheet("border: 1px dashed gray;")
        self.lbl_frame_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_details.addWidget(self.lbl_frame_img)
        
        self.txt_frame_info = QTextEdit()
        self.txt_frame_info.setReadOnly(True)
        layout_details.addWidget(self.txt_frame_info)
        
        layout_gallery_main.addWidget(self.grp_frame_details, stretch=1)
        self.tabs.addTab(self.tab_gallery, "🖼️ 关键帧画廊")

        # Tab 3: Summary Media (Clips/GIFs)
        # Tab 3: Summary Media (Clips/GIFs)
        self.tab_media = QWidget()
        layout_media = QVBoxLayout(self.tab_media)
        self.media_carousel = CarouselWidget()
        layout_media.addWidget(self.media_carousel)
        self.tabs.addTab(self.tab_media, "🎬 摘要媒体 (GIF/Clips)")

        # Tab 4: Metrics (Charts)
        self.tab_metrics = QWidget()
        self.layout_metrics = QVBoxLayout(self.tab_metrics)
        self.tabs.addTab(self.tab_metrics, "📊 元数据与画质")

        # Tab 4: Logs
        self.tab_logs = QWidget()
        layout_logs = QVBoxLayout(self.tab_logs)
        layout_logs.setContentsMargins(0, 0, 0, 0)
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        self.txt_logs.setFrameShape(QFrame.Shape.NoFrame)
        self.txt_logs.setStyleSheet("font-family: Consolas; font-size: 10pt; background: transparent;")
        layout_logs.addWidget(self.txt_logs)
        self.tabs.addTab(self.tab_logs, "📜 系统日志")

        # Tab 5: Model Manager
        self.tab_models = ModelManagerTab()
        self.tab_models.download_all_requested.connect(self.run_startup_scan)
        self.tab_models.detect_requested.connect(self.on_detect_model_type)
        for mid, card in self.tab_models.cards.items():
            card.download_requested.connect(self.start_model_download)
            card.btn_health.clicked.connect(lambda checked, m=mid: logging.info(f"正在校验 {m} 的完整性..."))
        self.tabs.addTab(self.tab_models, "📦 模型管理")
        
        # Periodic check for local models
        self.timer_model_check = QTimer(self)
        self.timer_model_check.timeout.connect(self.check_local_models)
        self.timer_model_check.start(5000)
        
        # Tab 6: API Intro
        self.tab_api_help = APIIntroPage()
        self.tabs.addTab(self.tab_api_help, "💡 获取 API")

        middle_layout.addWidget(self.tabs, stretch=1)
        
        # --- AGENT PANEL (Right Sliding) ---
        self.agent_panel = AgentPanel()
        self.agent_panel.send_message.connect(self.on_agent_query)
        self.agent_panel.regenerate_requested.connect(self.on_agent_query)
        self.agent_panel.stop_requested.connect(self.stop_agent_query)
        self.agent_panel.combo_model.currentIndexChanged.connect(self.on_agent_model_switched)
        middle_layout.addWidget(self.agent_panel)
        
        # Button to toggle Agent
        self.btn_toggle_agent = QPushButton("🤖")
        self.btn_toggle_agent.setFixedSize(30, 30)
        self.btn_toggle_agent.setToolTip("显示/隐藏 Agent 面板")
        self.btn_toggle_agent.clicked.connect(self.toggle_agent_panel)
        header_layout.insertWidget(header_layout.count() - 1, self.btn_toggle_agent)
        
        # History Button
        self.btn_history = QPushButton("🕒 历史记录")
        self.btn_history.clicked.connect(self.show_history_dialog)
        header_layout.insertWidget(header_layout.count() - 1, self.btn_history)
        
        # Help Button
        self.btn_help = QPushButton("📖 使用说明")
        self.btn_help.clicked.connect(self.show_help)
        header_layout.insertWidget(header_layout.count() - 1, self.btn_help)

        root_layout.addWidget(middle_widget)
        
        # --- BOTTOM STATUS CONSOLE ---
        self.status_console = StatusConsole()
        self.status_console.setFrameShape(QFrame.Shape.NoFrame)
        root_layout.addWidget(self.status_console)

        self.status_bar = self.statusBar()
        
        # Overlay for startup/downloads
        self.loading_overlay = LoadingOverlay(self.centralWidget())

    def load_settings(self):
        """Restore previous session settings."""
        try:
            cfg = self.app_config['LastUsed']
            self.combo_client.setCurrentIndex(int(cfg.get('client_type', 1)))
            self.txt_api_url.setPlainText(cfg.get('api_url', ""))
            self.txt_api_key.setPlainText(cfg.get('api_key', ""))
            self.combo_api_model.setEditText(cfg.get('model_name', ""))
            
            show_agent = self.app_config['Application'].getboolean('show_agent_panel', True)
            self.agent_panel.setVisible(show_agent)
        except Exception as e:
            logging.debug(f"Failed to load settings: {e}")

    def save_current_settings(self):
        self.config_manager.update_config("LastUsed", "client_type", self.combo_client.currentIndex())
        self.config_manager.update_config("LastUsed", "api_url", self.txt_api_url.toPlainText())
        self.config_manager.update_config("LastUsed", "api_key", self.txt_api_key.toPlainText())
        self.config_manager.update_config("LastUsed", "model_name", self.combo_api_model.currentText())

    def _refresh_preset_combo(self):
        self.combo_presets.clear()
        self.combo_presets.addItem("-- 选择推荐/保存的预设 --")
        for p in self.api_presets:
            self.combo_presets.addItem(p['name'])

    def _refresh_prompt_combo(self):
        self.combo_prompt.clear()
        for p in self.prompt_templates:
            self.combo_prompt.addItem(p['name'])
        self.combo_prompt.addItem("自定义")

    def _create_prompt_toolbar(self):
        layout = QHBoxLayout()
        btn_new = QPushButton("📄 新建")
        btn_new.setToolTip("新建一个提示词模板")
        btn_new.clicked.connect(self.on_new_prompt)
        
        btn_save = QPushButton("💾 保存")
        btn_save.setToolTip("保存当前提示词修改")
        btn_save.clicked.connect(self.on_save_prompt)
        
        btn_del = QPushButton("🗑️")
        btn_del.setFixedSize(30,30)
        btn_del.setToolTip("删除当前选中的模板")
        btn_del.clicked.connect(self.on_delete_prompt)
        
        layout.addWidget(btn_new)
        layout.addWidget(btn_save)
        layout.addWidget(btn_del)
        layout.addStretch()
        return layout

    def on_prompt_type_changed(self, index):
        is_custom = (self.combo_prompt.currentText() == "自定义")
        self.txt_custom_prompt.setVisible(is_custom)
        # Handle index - it matches prompt_templates until "自定义" (the last one)
        if not is_custom and index >= 0 and index < len(self.prompt_templates):
            self.txt_custom_prompt.setPlainText(self.prompt_templates[index]['content'])
        elif is_custom:
            self.txt_custom_prompt.clear()

    def on_new_prompt(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "新建提示词", "请输入模板名称:")
        if ok and name:
            new_p = {"name": name, "content": ""}
            self.prompt_templates.append(new_p)
            self.config_manager.save_prompts(self.prompt_templates)
            self._refresh_prompt_combo()
            self.combo_prompt.setCurrentText(name)
            self.txt_custom_prompt.setVisible(True)
            self.txt_custom_prompt.clear()

    def on_save_prompt(self):
        idx = self.combo_prompt.currentIndex()
        if idx < 0 or idx >= len(self.prompt_templates):
            # If "Custom" is selected, we should ask for a name to save as new
            self.on_new_prompt()
            return
            
        self.prompt_templates[idx]['content'] = self.txt_custom_prompt.toPlainText()
        self.config_manager.save_prompts(self.prompt_templates)
        logging.info(f"提示词模板 '{self.prompt_templates[idx]['name']}' 已保存")

    def on_delete_prompt(self):
        idx = self.combo_prompt.currentIndex()
        if idx < 0 or idx >= len(self.prompt_templates): return
        name = self.prompt_templates[idx]['name']
        self.prompt_templates.pop(idx)
        self.config_manager.save_prompts(self.prompt_templates)
        self._refresh_prompt_combo()
        logging.info(f"已删除模板: {name}")

    def on_preset_selected(self, index):
        if index <= 0: return
        preset = self.api_presets[index - 1]
        self.txt_api_url.setPlainText(preset['url'])
        self.txt_api_key.setPlainText(preset['key'])
        if 'model' in preset:
            self.combo_api_model.setEditText(preset['model'])
        logging.info(f"已加载预设: {preset['name']}")

    def save_current_as_preset(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "保存预设", "请输入预设名称:")
        if ok and name:
            new_preset = {
                "name": name,
                "url": self.txt_api_url.toPlainText(),
                "key": self.txt_api_key.toPlainText(),
                "model": self.combo_api_model.currentText()
            }
            self.api_presets.append(new_preset)
            self.config_manager.save_api_presets(self.api_presets)
            self._refresh_preset_combo()
            logging.info(f"预设 '{name}' 已保存")

    def clear_api_fields(self):
        self.txt_api_url.setPlainText("https://api.iflow.cn/v1")
        self.txt_api_key.clear()
        self.combo_api_model.setEditText("")
        self.combo_presets.setCurrentIndex(0)
        logging.info("已重置 API 输入框。")

    def delete_selected_preset(self):
        index = self.combo_presets.currentIndex()
        if index <= 0: return
        name = self.combo_presets.currentText()
        self.api_presets.pop(index - 1)
        self.config_manager.save_api_presets(self.api_presets)
        self._refresh_preset_combo()
        logging.info(f"已删除预设: {name}")

    def unload_ollama_model(self):
        if self.combo_client.currentIndex() != 0: return
        model_name = self.combo_ollama_model.currentText()
        if not model_name or model_name == "未找到模型": return
        
        try:
            from src.core.logic import OllamaClient
            client = OllamaClient()
            if client.unload_model(model_name):
                logging.info(f"✅ 已成功卸载模型: {model_name}")
            else:
                logging.info(f"⚠️ 卸载模型 {model_name} 可能失败或该模型未加载")
        except Exception as e:
            logging.info(f"❌ 卸载失败: {e}")

    def toggle_agent_panel(self):
        is_visible = not self.agent_panel.isVisible()
        self.agent_panel.setVisible(is_visible)
        self.config_manager.update_config("Application", "show_agent_panel", is_visible)
        
    def on_agent_query(self, text, model):
        if not self.analyzer:
            self.agent_panel.append_message("Agent", "❌ 模型尚未加载，请先加载模型。")
            return
        
        self.agent_panel.update_thoughts(f"正在使用模型 {model} 思考...")
        self.agent_panel.btn_stop.setEnabled(True)
        self.agent_panel.btn_send.setEnabled(False)
        
        # Create a new AI bubble for streaming
        self.agent_panel.append_message("Agent", "", model_name=model)
        
        # Determine prompt: include video context if available
        context = ""
        if hasattr(self, 'report_full_text') and self.report_full_text:
            context = f"\n\nVideo Context:\n{self.report_full_text}"
        
        full_prompt = f"User Question: {text}"
        if hasattr(self, 'agent_system_context') and self.agent_system_context:
            full_prompt = f"{self.agent_system_context}\n\nUser Question: {text}"
        
        # Inject Tool Definitions if available
        if hasattr(self, 'tool_registry'):
            tool_desc = self.tool_registry.get_tool_descriptions()
            tool_prompt = (f"\n\n[SYSTEM: TOOLS ENABLED]\nYou have access to the following tools:\n{tool_desc}\n"
                           f"To use a tool, output XML format EXACTLY like this: <tool name=\"tool_name\">{{\"arg\": value}}</tool>\n"
                           f"Example: <tool name=\"get_frame_details\">{{\"seconds\": 10.5}}</tool>\n"
                           f"Wait for the <tool_result> block before continuing.\n")
            full_prompt = f"{tool_prompt}\n{full_prompt}"

        # Check if model supports vision. If not, don't send images.
        model_is_vision = False
        vision_keywords = ["vl", "vision", "llava", "qwen-vl", "moondream", "internvl", "minicpm-v", "qwen3"]
        if any(kw in model.lower() for kw in vision_keywords):
            model_is_vision = True
            
        pending_pixmaps = getattr(self.agent_panel, 'pending_images', [])
        image_paths = []
        if model_is_vision and pending_pixmaps:
            import tempfile
            from pathlib import Path
            temp_dir = Path(tempfile.gettempdir()) / "video_analysis_agent_images"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            for i, pix in enumerate(pending_pixmaps):
                tmp_path = temp_dir / f"agent_input_{int(time.time())}_{i}.jpg"
                pix.save(str(tmp_path), "JPG")
                image_paths.append(str(tmp_path))
            
            # Clear pending images after consuming
            self.agent_panel.clear_pending_images()
        elif not model_is_vision and pending_pixmaps:
            logging.info(f"Chat model {model} is text-only. Skipping attached images.")

        # Enhanced Logging: User query and images
        img_info = f" (附带 {len(image_paths)} 张图片/提帧)" if image_paths else ""
        logging.info(f"💬 Agent 用户提问: {text}{img_info}")
        if image_paths:
            for i, p in enumerate(image_paths):
                logging.info(f"   🖼️ 素材 {i+1}: {p}")

        # Use existing client but specific model
        self.chat_worker = ChatWorker(self.analyzer.client, model, full_prompt, image_paths=image_paths, tool_registry=getattr(self, 'tool_registry', None))
        self.chat_worker.chunk_received.connect(self.on_chat_chunk)
        self.chat_worker.finished.connect(self.on_chat_finished)
        self.chat_worker.start()

    def on_chat_chunk(self, chunk):
        if chunk is None: return
        # Update the last bubble
        self.agent_panel.update_last_bubble(str(chunk))


    def on_chat_finished(self):
        self.agent_panel.update_thoughts("回答完成。")
        self.agent_panel.btn_stop.setEnabled(False)
        self.agent_panel.btn_send.setEnabled(True)
        
        # Safely clean up worker
        if hasattr(self, 'chat_worker') and self.chat_worker:
            self.chat_worker.deleteLater()
            # Do NOT set to None immediately if we want to be 100% safe from 'Destroyed while thread is running'
            # But in PyQt6, deleteLater is usually sufficient as it waits for thread finish.


    def stop_agent_query(self):
        if hasattr(self, 'chat_worker') and self.chat_worker:
            self.chat_worker.stop()
            logging.info("🛑 用户中止了 Agent 对话回复")
            self.on_chat_finished()

    def update_slider_label(self, val):
        self.lbl_slider_val.setText(f"当前值: {val} 帧/分钟")

    def start_model_download(self, model_id):
        logging.info(f"📥 准备下载模型组件: {model_id} ...")
        self.tab_models.cards[model_id].set_downloading()
        worker = ModelDownloadWorker(self.model_manager, model_id)
        worker.progress.connect(self.on_download_progress)
        worker.finished.connect(self.on_download_finished)
        worker.start()
        # Keep reference to prevent GC
        if not hasattr(self, '_download_workers'): self._download_workers = []
        self._download_workers.append(worker)

    def on_download_progress(self, model_id, percent):
        self.tab_models.cards[model_id].progress.setValue(percent)
        # Log every 20% to avoid spamming but show activity
        if percent > 0 and percent % 20 == 0:
            logging.info(f"⏳ {model_id} 下载进度: {percent}%")
            
        if hasattr(self, 'loading_overlay') and self.loading_overlay.isVisible():
            self.loading_overlay.show_msg(f"正在下载必要组件: {model_id}...", percent)

    def on_download_finished(self, model_id, success):
        self.check_local_models()
        if success:
            logging.info(f"✅ 模型 {model_id} 下载完成")
        else:
            logging.info(f"❌ 模型 {model_id} 下载失败")
        
        if hasattr(self, 'loading_overlay') and self.loading_overlay.isVisible():
            self.loading_overlay.hide()
            from PyQt6.QtWidgets import QMessageBox
            if success:
                QMessageBox.information(self, "组件更新", f"必要组件 {model_id} 已下载并就绪。")
            else:
                QMessageBox.warning(self, "组件更新", f"必要组件 {model_id} 下载失败，可能会影响分析。")

    def check_local_models(self):
        for mid, card in self.tab_models.cards.items():
            path = self.model_manager.get_model_path(mid)
            exists = path is not None and path.exists()
            self.tab_models.update_model_status(mid, exists)
        
        # Also refresh found files
        found_files = self.model_manager.list_local_models()
        self.tab_models.refresh_local_cards(found_files)

    def run_startup_scan(self):
        """Checks for mandatory models and offers to download missing ones."""
        mandatory = ["yolo_v11n"] 
        missing = []
        for m in mandatory:
            if not self.model_manager.get_model_path(m):
                missing.append(m)
        
        if missing:
            self.loading_overlay.show_msg(f"发现缺失必要组件: {', '.join(missing)}，准备自动下载...", 0)
            # Use a slightly longer delay so the user sees the overlay
            QTimer.singleShot(2000, lambda: self._sequential_download(missing))
        else:
            logging.info("✅ 所有必要模型组件已就绪。")

    def _sequential_download(self, missing_list):
        if not missing_list:
            if self.loading_overlay.isVisible(): self.loading_overlay.hide()
            return
        
        current = missing_list[0]
        remaining = missing_list[1:]
        
        self.start_model_download(current)

    def on_detect_model_type(self, model_filename):
        try:
            m_type = self.model_manager.detect_model_type(model_filename)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "模型能力探测", f"模型文件: {model_filename}\n基本类型: {m_type}\n\n提示: 如果检测为 VL，软件将尝试使用视觉能力分析视频帧。")
            logging.info(f"探测模型 {model_filename} 类型为: {m_type}")
        except Exception as e:
            logging.info(f"探测模型失败: {e}")
        # Note: on_download_finished will need to be updated to trigger the next one if remaining
        self._remaining_downloads = remaining

    def on_download_finished(self, model_id, success):
        self.check_local_models()
        if success:
            logging.info(f"✅ 模型 {model_id} 下载完成")
        else:
            logging.info(f"❌ 模型 {model_id} 下载失败")
        
        # Check if we were in a sequential download
        if hasattr(self, '_remaining_downloads') and self._remaining_downloads:
            next_model = self._remaining_downloads.pop(0)
            self.start_model_download(next_model)
        else:
            if hasattr(self, 'loading_overlay') and self.loading_overlay.isVisible():
                self.loading_overlay.hide()
                from PyQt6.QtWidgets import QMessageBox
                if success:
                    QMessageBox.information(self, "组件更新", "所有必要组件已下载并就绪。")
                else:
                    QMessageBox.warning(self, "组件更新", "部分必要组件下载失败，可能会影响分析。")

    def toggle_theme(self):
        """Randomly switch between various themed styles."""
        logging.info("UI操作: 随机切换主题")
        themes = ["dark", "light", "dark_blue", "dark_purple", "dark_green", "light_blue"]
        import random
        new_theme = random.choice([t for t in themes if t != self.app_config['Application'].get('theme', 'dark')])
        
        self.apply_theme(new_theme)
        self.config_manager.update_config("Application", "theme", new_theme)

    def apply_theme(self, theme_name):
        try:
            # Base themes from qdarktheme
            base = "dark" if "dark" in theme_name else "light"
            qdarktheme.setup_theme(base)
            
            # Custom accent overrides (QSS)
            accents = {
                "dark_blue": ("#2196F3", "#1976D2"),
                "dark_purple": ("#9C27B0", "#7B1FA2"),
                "dark_green": ("#4CAF50", "#388E3C"),
                "light_blue": ("#03A9F4", "#0288D1")
            }
            
            if theme_name in accents:
                primary, secondary = accents[theme_name]
                qss = f"""
                QMainWindow {{ background-color: {base == "dark" and "#1e1e1e" or "#f5f5f5"}; }}
                QGroupBox#sidebarGroup {{ 
                    border: none; 
                    border-top: 2px solid {primary}; 
                    margin-top: 20px; 
                    padding-top: 10px;
                    font-weight: bold;
                }}
                QGroupBox#sidebarGroup::title {{ 
                    color: {primary}; 
                    subcontrol-origin: margin; 
                    left: 10px; 
                    top: -5px;
                }}
                QScrollBar:vertical {{
                    background: transparent;
                    width: 10px;
                    margin: 0px;
                }}
                QScrollBar::handle:vertical {{
                    background: {primary}33;
                    min-height: 30px;
                    border-radius: 5px;
                }}
                QScrollBar::handle:vertical:hover {{
                    background: {primary}66;
                }}
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                    background: none;
                    height: 0px;
                }}
                QScrollBar:horizontal {{
                    background: transparent;
                    height: 10px;
                }}
                QScrollBar::handle:horizontal {{
                    background: {primary}33;
                    min-width: 30px;
                    border-radius: 5px;
                }}
                QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
                QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                    background: none;
                    width: 0px;
                }}
                QPushButton {{ 
                    border: 1px solid {primary}33; 
                    border-radius: 6px; 
                    padding: 8px; 
                    background-color: {base == "dark" and "rgba(255,255,255,0.03)" or "rgba(0,0,0,0.03)"};
                }}
                QPushButton:hover {{ background-color: {primary}22; border: 1px solid {primary}; }}
                QPushButton#btn_primary {{ background-color: {primary}; color: white; }}
                QTabWidget::pane {{ border: 1px solid {primary}22; border-radius: 5px; }}
                """
                self.setStyleSheet(qss)
            else:
                # Default Sleek Style for standard dark/light
                accent = "#2196F3"
                qss = f"""
                QGroupBox#sidebarGroup {{ border: none; border-top: 1px solid {accent}44; margin-top: 15px; padding-top: 5px; }}
                QGroupBox#sidebarGroup::title {{ color: {accent}; font-weight: bold; }}
                QScrollBar:vertical {{ background: transparent; width: 8px; }}
                QScrollBar::handle:vertical {{ background: {accent}44; border-radius: 4px; }}
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
                QScrollBar:horizontal {{ background: transparent; height: 8px; }}
                QScrollBar::handle:horizontal {{ background: {accent}44; border-radius: 4px; }}
                QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}
                """
                self.setStyleSheet(qss)
                
            logging.info(f"主题已切换为: {theme_name}")
        except Exception as e:
            logging.error(f"主题应用失败: {e}")
            qdarktheme.setup_theme("dark")

    def on_client_changed(self, index):
        """Handle UI changes based on inference client selection."""
        # 0: Ollama, 1: API, 2: LM Studio, 3: Local File
        is_ollama = (index == 0)
        is_api = (index == 1)
        is_lmstudio = (index == 2)
        is_local = (index == 3)
        
        self.grp_api.setVisible(is_api or is_lmstudio)
        self.grp_ollama.setVisible(is_ollama or is_local)
        self.btn_unload.setVisible(is_ollama)
        
        if is_lmstudio:
            self.txt_api_url.setPlainText("http://localhost:1234/v1")
            self.txt_api_key.setPlainText("lm-studio")
        elif is_api:
            # Maybe restore last used API config
            pass
            
        if is_local:
            self.grp_ollama.setTitle("本地模型文件 (.gguf/.pt)")
            self.refresh_local_model_list()
        else:
            self.grp_ollama.setTitle("Ollama 设置")
            if is_ollama:
                self.refresh_ollama_models()
        
        self.sync_agent_models()

    def sync_agent_models(self):
        """Syncs the current client's model list to the Agent panel."""
        self.agent_panel.combo_model.clear()
        client_type = self.combo_client.currentIndex()
        if client_type == 0 or client_type == 3: # Ollama or Local
            for i in range(self.combo_ollama_model.count()):
                self.agent_panel.combo_model.addItem(self.combo_ollama_model.itemText(i))
        else: # API or LM Studio
            for i in range(self.combo_api_model.count()):
                self.agent_panel.combo_model.addItem(self.combo_api_model.itemText(i))

    def refresh_local_model_list(self):
        self.combo_ollama_model.clear()
        models = self.model_manager.list_local_models()
        if not models:
            self.combo_ollama_model.addItem("未发现模型 (请放入 models/ 目录)")
        else:
            self.combo_ollama_model.addItems(models)
            
        self.sync_agent_models()

    def refresh_ollama_models(self):
        """Fetch models from Ollama API asynchronously."""
        self.combo_ollama_model.clear()
        self.combo_ollama_model.addItem("正在刷新...")
        
        self.ollama_worker = OllamaRefreshWorker()
        self.ollama_worker.models_ready.connect(self._on_ollama_models_ready)
        self.ollama_worker.error.connect(self._on_ollama_refresh_error)
        self.ollama_worker.start()

    def _on_ollama_models_ready(self, models):
        self.combo_ollama_model.clear()
        if models:
            self.combo_ollama_model.addItems(models)
            self.status_bar.showMessage(f"已刷新: 找到 {len(models)} 个 Ollama 模型")
        else:
            self.combo_ollama_model.addItem("未找到模型")
        
        # Start monitor if not started
        if self.timer_monitor is None:
            self.timer_monitor = QTimer(self)
            self.timer_monitor.timeout.connect(self.update_system_stats)
            self.timer_monitor.start(3000)

    def _on_ollama_refresh_error(self, err_msg):
        self.combo_ollama_model.clear()
        self.combo_ollama_model.addItem("连接失败 (Ollama 未启动)")
        self.status_bar.showMessage(f"Ollama 刷新失败: {err_msg}")

    def update_system_stats(self):
        if self.combo_client.currentIndex() == 0: 
             from src.core.logic import OllamaClient, NVIDIA_GPU_AVAILABLE
             if NVIDIA_GPU_AVAILABLE:
                 try:
                     client = OllamaClient()
                     status = client.get_status()
                     vram_used = status.get('vram_used', 0)
                     vram_total = status.get('vram_total', 1)
                     gpu_util = status.get('gpu_util', 0)
                     self.grp_ollama.setTitle(f"Ollama (Local) - GPU: {gpu_util}% | VRAM: {vram_used:.1f}/{vram_total:.1f} GB")
                     self.status_console.resource_monitor.update_vram(vram_used, vram_total)
                 except: pass

    def on_api_url_changed(self):
        from src.core.logic import APIGatewayClient
        url = self.txt_api_url.toPlainText().strip()
        if not url:
            self.lbl_api_preview.setText("")
            return
            
        base, chat, models = APIGatewayClient.parse_endpoint(url)
        self.lbl_api_preview.setText(f"预览: {chat}")
        if url.endswith("#"):
            self.lbl_api_preview.setStyleSheet("color: orange; font-style: italic;")
            self.lbl_api_preview.setText(f"Raw Mode: {chat}")
        else:
            self.lbl_api_preview.setStyleSheet("color: #4CAF50; font-style: italic;")

    def check_api_connection(self):
        from src.core.logic import APIGatewayClient
        url = self.txt_api_url.toPlainText().strip()
        key = self.txt_api_key.toPlainText().strip()
        
        if not url:
            logging.info("❌ API URL 不能为空")
            return
            
        self.btn_check_api.setEnabled(False)
        self.btn_check_api.setText("正在检测...")
        logging.info(f"正在尝试连接 API: {url} (后台线程)...")
        
        # Use Thread for non-blocking UI
        self.api_check_worker = ApiCheckWorker(url, key)
        self.api_check_worker.finished.connect(self.on_api_check_finished)
        self.api_check_worker.start()

    def on_api_check_finished(self, models, error_msg, chat_endpoint):
        self.btn_check_api.setEnabled(True)
        if error_msg:
            logging.info(f"❌ 连接失败: {error_msg}")
            self.btn_check_api.setText("❌ 连接失败")
        else:
            if models:
                self.combo_api_model.clear()
                self.combo_api_model.addItems(models)
                self.combo_api_model.setCurrentIndex(0)
                logging.info(f"✅ 连接成功! 发现 {len(models)} 个模型。")
                self.btn_check_api.setText("✅ 连接成功")
                self.lbl_api_preview.setText(f"Chat Endpoint: {chat_endpoint}")
            else:
                logging.info("⚠️ 连接成功但未找到模型 (可能是格式不兼容或无权限列表)")
                self.btn_check_api.setText("⚠️ 无模型列表")
                
            self.sync_agent_models()
        
        # Reset button text after 2s
        QTimer.singleShot(2000, lambda: self.btn_check_api.setText("🔍 检测连接 & 获取模型"))

    def plot_metrics(self, frames):
        try:
            logging.info("正在生成高级画质分析图表...")
            QTimer.singleShot(100, lambda: self._generate_plot_internal())
        except Exception as e:
            logging.info(f"绘图准备失败: {e}")

    def _generate_plot_internal(self):
        try:
            from src.core.logic import get_advanced_video_metrics
            # Correct import for Matplotlib Qt backend
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            
            if not self.video_path: return
            avg_metrics, fig = get_advanced_video_metrics(str(self.video_path))
            if fig:
                for i in reversed(range(self.layout_metrics.count())): 
                    item = self.layout_metrics.itemAt(i)
                    if item.widget(): item.widget().setParent(None)
                
                canvas = FigureCanvasQTAgg(fig)
                # Set size policy to expand
                canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self.layout_metrics.addWidget(canvas)
                logging.info("📊 画质分析图表已生成。")
        except Exception as e:
            logging.info(f"生成图表出错: {e}")

    def _instrumented_call(self, func, action_name, *args, **kwargs):
        """Helper to log UI actions and catch errors."""
        logging.info(f"UI操作: 点击按钮 '{action_name}' -> 调用 {func.__name__}")
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"UI指令 '{action_name}' 执行错误: {e}", exc_info=True)
            logging.info(f"⚠️ 执行失败: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "执行错误", f"在执行 '{action_name}' 时发生错误:\n{e}")

    def show_help(self):
        """Show the help/readme dialog."""
        dialog = HelpDialog(self)
        dialog.exec()

    def on_agent_model_switched(self, index):
        """Global model switch from Agent panel."""
        if index < 0: return
        model_name = self.agent_panel.combo_model.currentText()
        client_type = self.combo_client.currentIndex()
        
        # Sync back to primary combos
        if client_type == 0 or client_type == 3:
            self.combo_ollama_model.setCurrentText(model_name)
        else:
            self.combo_api_model.setCurrentText(model_name)
            
        logging.info(f"🔄 全局模型切换: {model_name}")
        # Optionally reload model immediately if one was already loaded
        if self.analyzer:
            self.load_model()

    def on_sidebar_model_changed(self, text):
        """Sync sidebar model selection to Agent panel."""
        if not text: return
        if not hasattr(self, 'agent_panel'): return
        
        # Avoid loop: check if different
        current_agent = self.agent_panel.combo_model.currentText()
        if text != current_agent:
            self.agent_panel.combo_model.blockSignals(True)
            self.agent_panel.combo_model.setCurrentText(text)
            self.agent_panel.combo_model.blockSignals(False)

    def init_backend(self):
        self.analyzer = None
        self.video_path = None
        self.timer_monitor = None
        self.prompt_loader = PromptLoader()
        
        from src.core.history_manager import HistoryManager
        from src.utils.constants import CONFIG_DIR
        self.history_manager = HistoryManager(CONFIG_DIR)
        
        # Run auto-cleanup in background to avoid startup delay
        QTimer.singleShot(5000, lambda: self.history_manager.cleanup_old_sessions(7))

        # Init Agent Tools
        self.tool_registry = ToolRegistry()
        # Context provider returns self (DesktopApp instance)
        context_provider = lambda: self
        self.tool_registry.set_context_provider(context_provider)
        
        self.tool_registry.register_tool(
            "get_video_meta", 
            "Get metadata about the current video (path, duration, etc.)", 
            create_get_video_meta_tool(context_provider)
        )
        self.tool_registry.register_tool(
            "get_frame_details", 
            "Get details (caption, OCR) for a specific second in the video. Args: {'seconds': 10.5}", 
            create_get_frame_details_tool(context_provider),
            {"seconds": "float"}
        )
        self.tool_registry.register_tool(
            "delete_this_history", 
            "Delete the current analysis session/history. Use with caution.", 
            create_delete_history_tool(context_provider)
        )

    def append_log(self, msg):
        # Safeguard: prevent crash if logging happens before UI is ready
        if not hasattr(self, 'txt_logs') or self.txt_logs is None:
            return
            
        # Msg already has timestamp from Formatter if coming via QtLogHandler
        self.txt_logs.append(msg)
        
        # Limit log lines (blocks) to 500 to save memory
        doc = self.txt_logs.document()
        if doc.blockCount() > 500:
            cursor = self.txt_logs.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(cursor.MoveOperation.Down, cursor.MoveMode.KeepAnchor, doc.blockCount() - 500)
            cursor.removeSelectedText()
            
        self.txt_logs.verticalScrollBar().setValue(self.txt_logs.verticalScrollBar().maximum())
        
        if hasattr(self, 'status_bar') and self.status_bar:
            # Strip timestamp for status bar if present (simple heuristic)
            clean_msg = msg.split(' - ')[-1] if ' - ' in msg else msg
            self.status_bar.showMessage(clean_msg, 3000)
    def on_worker_log(self, msg):
        """Redirects worker logs to the central logging system."""
        logging.info(msg)

    def load_video_from_path(self, path):
        """Handle video file loading from drag & drop or selection"""
        from pathlib import Path
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        
        path_obj = Path(path)
        if path_obj.exists() and path_obj.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            self.video_path = path_obj
            self.lbl_file.setText(f"已选择: {path_obj.name}")
            self.lbl_file.setStyleSheet("color: #4CAF50; font-weight: bold; border: 1px solid #4CAF50; padding: 5px; border-radius: 4px;")
            self.btn_start.setEnabled(True)
            logging.info(f"已加载视频文件: {path}")
        else:
            logging.warning(f"尝试加载无效文件: {path}")

    def select_video(self):
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if file_path:
            self.load_video_from_path(file_path)

    def show_history_dialog(self):
        from PyQt6.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QHeaderView
        
        dlg = QDialog(self)
        dlg.setWindowTitle("历史记录管理")
        dlg.resize(700, 450)
        layout = QVBoxLayout(dlg)
        
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["时间", "视频名称", "输出目录", "操作"])
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(3, 80)
        
        history = self.history_manager.get_history()
        table.setRowCount(len(history))
        
        for i, session in enumerate(history):
            ts = session.get('timestamp', '')[:19].replace('T', ' ')
            table.setItem(i, 0, QTableWidgetItem(ts))
            table.setItem(i, 1, QTableWidgetItem(session.get('video_name', 'Unknown')))
            table.setItem(i, 2, QTableWidgetItem(session.get('output_dir', '')))
            
            btn_del = QPushButton("删除")
            # Connect Delete with closure over s_id
            btn_del.clicked.connect(lambda checked, s_id=session['id']: self._delete_session_proxy(s_id, dlg))
            table.setCellWidget(i, 3, btn_del)
            
        layout.addWidget(table)
        
        btn_clear = QPushButton("🗑️ 清空所有历史 (!!!)")
        btn_clear.setStyleSheet("background-color: #F44336; color: white; padding: 10px; font-weight: bold;")
        btn_clear.clicked.connect(lambda: [self.history_manager.clear_all_history(), dlg.accept(), self.show_history_dialog()])
        layout.addWidget(btn_clear)
        
        dlg.exec()

    def _delete_session_proxy(self, s_id, dlg):
         if self.history_manager.delete_session(s_id):
             dlg.accept()
             self.show_history_dialog() # Refresh

    def load_model(self):
        client_idx = self.combo_client.currentIndex()
        try:
            if client_idx == 0 or client_idx == 3: # Ollama or Local
                model_name = self.combo_ollama_model.currentText()
            else: # API or LM
                model_name = self.combo_api_model.currentText().strip()
                
            if not model_name or "请选择" in model_name or "未找到" in model_name:
                raise ValueError("请选择一个有效的模型进行加载")

            self.btn_load_model.setText("⏳ 正在载入核心组件...")
            self.btn_load_model.setEnabled(False)
            logging.info(f"正在后台初始化分析器: {model_name}...")
            logging.info(f"注意: 系统将同时加载本地辅助模型 'all-MiniLM' 用于语义分析，这与您的 API 模型无关。")

            if client_idx == 0:
                from src.core.logic import OllamaClient
                client = OllamaClient()
            elif client_idx == 1 or client_idx == 2:
                api_url = self.txt_api_url.toPlainText().strip()
                api_key = self.txt_api_key.toPlainText().strip()
                from src.core.logic import APIGatewayClient
                client = APIGatewayClient(api_key, api_url)
            else: # Local file
                from src.core.logic import LocalModelClient
                client = LocalModelClient(model_name)

            self.load_worker = ModelLoadWorker(client, model_name, self.prompt_loader)
            self.load_worker.finished.connect(self.on_model_loaded)
            self.load_worker.start()

        except Exception as e:
            logging.info(f"❌ 加载指令发送失败: {e}")
            self.btn_load_model.setText("❌ 加载失败 (重试)")
            self.btn_load_model.setEnabled(True)

    def on_model_loaded(self, analyzer, model_name, error_msg):
        self.btn_load_model.setEnabled(True)
        if analyzer:
            self.analyzer = analyzer
            logging.info(f"✅ 模型分析器加载成功: {model_name}")
            self.btn_load_model.setText(f"✅ 已就绪: {model_name}")
            self.btn_load_model.setStyleSheet("background-color: #4CAF50; color: white;")
            if self.video_path: self.btn_start.setEnabled(True)
        else:
            logging.info(f"❌ 模型加载失败: {error_msg}")
            self.btn_load_model.setText("❌ 加载失败 (重试)")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "加载失败", f"初始化后台分析引擎时出错:\n{error_msg}")

    def start_analysis_phase1(self):
        if not self.video_path: return
        self.tabs.setCurrentWidget(self.tab_logs)
        logging.info(">>> 启动 Phase 1: 数据提取...")
        self.status_console.add_task("Phase 1: 数据提取")
        self.agent_panel.update_thoughts("正在从视频中提取关键帧与音频数据...")
        self.save_current_settings()
        
        config = {
            "enable_audio": self.chk_audio.isChecked(),
            "smart_extraction": self.chk_smart.isChecked(),
            "extraction_density": self.slider_frames.value() / 100.0 # 0.01 to 1.0
        }
        
        self.worker = ExtractionWorker(self.video_path, config)
        self.worker.log.connect(self.on_worker_log)
        self.worker.finished.connect(self.on_phase1_finished)
        self.worker.start()
        self.btn_start.setEnabled(False)

    def on_phase1_finished(self, result):
        if not result:
            logging.info("❌ Phase 1 提取失败。")
            self.status_console.finish_task("Phase 1: 数据提取", False)
            self.btn_start.setEnabled(True)
            return
            
        logging.info("✅ Phase 1 完成。正在填充画廊...")
        self.status_console.finish_task("Phase 1: 数据提取", True)
        self.agent_panel.update_thoughts("Phase 1 完成。等待 AI 总结 (Phase 2)...")
        self.frames = result['frames']
        self.transcript = result['transcript']
        self.output_dir = result['output_dir']
        self.video_duration = result.get('duration', 0.0)
        
        # Save to History
        self.history_manager.add_session(self.video_path, self.output_dir)
        
        # Populate Gallery
        self.populate_gallery(self.frames)
        
        # Generate Metrics Plot
        self.plot_metrics(self.frames)
        
        # Update UI state
        self.btn_start.setText("🔄 重新提取 (Phase 1)")
        self.btn_start.setEnabled(True)
        self.btn_ai.setEnabled(True)
        self.btn_media.setEnabled(True)
        self.btn_ai.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.tabs.setCurrentWidget(self.tab_gallery)

    def generate_summary_media(self):
        if not self.video_path or not self.output_dir: return
        logging.info(">>> 启动 Phase 3: 生成摘要媒体 (Clips/GIF)...")
        self.status_console.add_task("Phase 3: 生成摘要媒体")
        self.agent_panel.update_thoughts("正在合成摘要视频与 GIF...")
        self.tabs.setCurrentWidget(self.tab_logs)
        self.btn_media.setEnabled(False)
        
        # We need frames (self.frames).
        self.media_worker = MediaWorker(self.video_path, self.frames, self.output_dir)
        self.media_worker.log.connect(self.on_worker_log)
        self.media_worker.finished.connect(self.on_media_finished)
        self.media_worker.start()

    def on_media_finished(self, results):
        self.btn_media.setEnabled(True)
        self.btn_media.setText("✅ 媒体生成完成")
        if not results:
            logging.info("⚠️ 未生成任何媒体文件。")
            self.status_console.finish_task("Phase 3: 生成摘要媒体", False)
            return
            
        logging.info(f"✅ 摘要媒体生成成功。")
        self.status_console.finish_task("Phase 3: 生成摘要媒体", True)
        self.agent_panel.update_thoughts("所有任务已完成。")
        self.tabs.setCurrentWidget(self.tab_media)
        
        # Determine strict correlation: create_summary_media_artifacts returns (clips, selected_frames, video, gif)
        # But our Worker returns a flat list 'results' which combines them?
        # Wait, MediaWorker.run says:
        # results = []
        # if clips: results.extend(clips)
        # if video: results.append(video)
        # if gif: results.append(gif)
        # This loses structure. We need to match clips to frames.
        # Let's fix MediaWorker to return a dict or tuple. 
        # But I can't easily change it now without rewriting the class I just restored.
        # However, I know 'clips' are first N items, and they correspond to 'selected_frames' (which I also need).
        # Actually logic.py returns selected_frames. Worker ignores it!
        # I should assume I can't easily map back without updating Worker.
        # For now, just display the files.
        # Or... I can update MediaWorker right now? No I just wrote it.
        # I'll rely on file naming: *_clip_00.mp4, etc.
        # Better: MediaWorker returns 'results' list.
        # I can just populate gallery with these files.
        self.populate_media_gallery(results)

    def populate_media_gallery(self, file_paths):
        self.media_carousel.clear()
        
        from PyQt6.QtGui import QDesktopServices, QPixmap
        from PyQt6.QtCore import QUrl
        
        for path_str in file_paths:
            path = Path(path_str)
            if not path.exists(): continue
            
            # Create container (Card)
            item_widget = QFrame()
            item_widget.setFixedSize(200, 200) # Fixed size for carousel items
            item_widget.setStyleSheet("QFrame { background: rgba(0,0,0,0.1); border-radius: 8px; }")
            layout = QVBoxLayout(item_widget)
            layout.setContentsMargins(5,5,5,5)
            
            # Thumbnail
            pix = None
            if path.suffix.lower() == '.gif':
                pix = QPixmap(str(path))
            elif path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                pix = self._get_video_thumbnail(path)
                
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if pix:
                img_lbl.setPixmap(pix.scaled(180, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                img_lbl.setText("🎬 No Preview")
                img_lbl.setStyleSheet("border: 1px dashed gray; color: gray;")
            layout.addWidget(img_lbl)
            
            # Label
            name_lbl = QLabel(path.name)
            name_lbl.setWordWrap(True)
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_lbl.setStyleSheet("font-size: 10px; color: #555;")
            layout.addWidget(name_lbl)
            
            # Button
            btn = QPushButton("▶️ Play")
            btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; border-radius: 4px; padding: 2px; }")
            btn.clicked.connect(lambda checked, p=path: self.open_in_app_player(p))
            layout.addWidget(btn)
            
            self.media_carousel.add_widget(item_widget)

    def open_in_app_player(self, path):
        """Opens video in in-app player or default viewer."""
        if path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            try:
                from src.ui.video_player_dialog import VideoPlayerDialog
                dlg = VideoPlayerDialog(path, self)
                dlg.exec()
            except ImportError:
                # Fallback if dialog issue
                from PyQt6.QtGui import QDesktopServices
                from PyQt6.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _get_video_thumbnail(self, video_path):
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                from PyQt6.QtGui import QImage, QPixmap
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                return QPixmap.fromImage(qimg)
        except Exception as e:
            logging.error(f"Thumbnail error: {e}")
        return None

    def populate_gallery(self, frames):
        self.gallery_carousel.clear()
        
        for i, frame in enumerate(frames):
            # Container
            item_widget = QFrame()
            item_widget.setFixedSize(220, 180) # Fixed size
            item_widget.setStyleSheet("QFrame { background: rgba(0,0,0,0.05); border-radius: 8px; }")
            item_layout = QVBoxLayout(item_widget)
            item_layout.setContentsMargins(5,5,5,5)
            
            # Clickable Image
            if frame.path.exists():
                from PyQt6.QtGui import QPixmap
                pix = QPixmap(str(frame.path)).scaled(200, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                btn_img = ClickableLabel()
                btn_img.setPixmap(pix)
                btn_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
                btn_img.setCursor(Qt.CursorShape.PointingHandCursor)
                btn_img.clicked.connect(lambda f=frame: self.show_frame_details(f))
                item_layout.addWidget(btn_img)
            
            # Label
            lbl = QLabel(f"{frame.timestamp:.2f}s")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; color: #333;")
            item_layout.addWidget(lbl)
            
            self.gallery_carousel.add_widget(item_widget)
            
            # Add to Chat Button (NEW)
            btn_chat = QPushButton("💬 对话")
            btn_chat.setToolTip("将此帧添加到 AI 对话以便讨论")
            btn_chat.setStyleSheet("background-color: #2196F3; color: white;")
            btn_chat.clicked.connect(lambda checked, f=frame: self.on_add_frame_to_chat(f))
            item_layout.addWidget(btn_chat)



    def on_add_frame_to_chat(self, frame):
        """Send a frame's pixmap to the Agent panel."""
        if not self.agent_panel.isVisible():
            self.toggle_agent_panel()
        
        pixmap = QPixmap(str(frame.path))
        if not pixmap.isNull():
            self.agent_panel.add_pending_image(pixmap)
            logging.info(f"已将帧 {frame.timestamp:.2f}s 添加到对话暂留区")
            self.tabs.setCurrentWidget(self.tab_logs) # Optional: focus on logs or just stay

    def show_frame_details(self, frame):
        self.grp_frame_details.setTitle(f"🔍 选定帧详细信息: {frame.timestamp:.2f}s")
        if frame.path.exists():
             pix = QPixmap(str(frame.path)).scaled(self.lbl_frame_img.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
             self.lbl_frame_img.setPixmap(pix)
             
        # Format info
        info = f"<b>时间戳:</b> {frame.timestamp:.2f}秒<br>"
        info += f"<b>文件:</b> {frame.path.name}<br>"
        if frame.metrics:
            info += "<br><b>画质指标:</b><br>"
            metric_map = {
                "brightness": "亮度",
                "contrast": "对比度",
                "saturation": "饱和度",
                "sharpness": "清晰度"
            }
            for k, v in frame.metrics.items():
                label = metric_map.get(k, k)
                info += f"{label}: {v:.2f}<br>"
            
            # Tips
            if frame.metrics.get("sharpness", 0) < 50:
                info += "<span style='color:orange'>⚠️ 画面较模糊</span>"
        self.txt_frame_info.setHtml(info)

    def plot_metrics(self, frames):
        try:
            # Lazy imports
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            # We need to construct a Plot
            # timestamps, brightness, sharpness
            timestamps = [f.timestamp for f in frames]
            brightness = [f.metrics.get('brightness', 0) for f in frames] # Assuming metrics keys
            sharpness = [f.metrics.get('sharpness', 0) for f in frames]

            # Clear previous plot
            for i in reversed(range(self.layout_metrics.count())): 
                item = self.layout_metrics.itemAt(i)
                if item.widget(): item.widget().setParent(None)

            # --- Fix Chinese Font Rendering ---
            font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['font.sans-serif'] = font_list
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(timestamps, brightness, label="亮度", color='orange')
            ax.plot(timestamps, sharpness, label="清晰度", color='green')
            ax.set_xlabel("时间 (秒)")
            ax.set_ylabel("数值")
            ax.legend(prop={'family': font_list}) # Support Chinese font with multiple fallbacks
            ax.grid(True)
            
            canvas = FigureCanvasQTAgg(fig)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.layout_metrics.addWidget(canvas)
            logging.info("📊 基础画质趋势图已生成 (Phase 1)。")
        except Exception as e:
            logging.info(f"绘图失败: {e}")

    def update_report(self, text: str):
        """Callback for AI streaming chunks."""
        from PyQt6.QtGui import QTextCursor
        # Append to variable
        if not hasattr(self, 'report_full_text'): self.report_full_text = ""
        self.report_full_text += text
        
        cursor = self.txt_report.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.txt_report.setTextCursor(cursor)
        self.txt_report.ensureCursorVisible()

    def start_ai_analysis(self):
        if not self.analyzer:
            logging.info("错误: 模型未加载。")
            return
            
        self.tabs.setCurrentWidget(self.tab_report)
        logging.info(">>> 启动 Phase 2: AI 分析...")
        self.status_console.add_task("Phase 2: AI 分析")
        self.agent_panel.update_thoughts("正在调用大模型对提取的数据进行深度分析...")
        
        # Reset report text
        self.txt_report.clear()
        self.report_full_text = ""
        
        custom_p = self.txt_custom_prompt.toPlainText() if self.txt_custom_prompt.isVisible() else None
        self.ai_worker = AnalysisWorker(self.analyzer, self.frames, self.transcript, custom_p)
        self.ai_worker.chunk_received.connect(self.update_report)
        self.ai_worker.finished.connect(self.on_ai_finished)
        self.ai_worker.start()
        self.btn_ai.setEnabled(False)

    def on_ai_finished(self):
        logging.info("✅ AI 分析任务已全部完成。")
        self.status_console.finish_task("Phase 2: AI 分析", True)
        self.agent_panel.update_thoughts("Phase 2 完成。您可以查看报告或生成摘要媒体。")
        self.btn_ai.setEnabled(True)
        self.btn_ai.setText("重新生成总结")
        
        # Inject Context into Agent
        self.inject_agent_system_context()

    def inject_agent_system_context(self):
        """Builds a context prompt from Phase 1 & 2 results and injects it into the Agent."""
        if not self.frames: return
        
        # Build Context String
        duration = getattr(self, 'video_duration', 0)
        video_name = self.video_path.name if self.video_path else "Unknown Video"
        
        context_str = f"--- System Context ---\n"
        context_str += f"Current Video: {video_name}\n"
        context_str += f"Duration: {duration:.2f} seconds\n"
        context_str += f"Recognized Transcript:\n{self.transcript[:2000]}...\n" if hasattr(self, 'transcript') and self.transcript else "No transcript available.\n"
        context_str += f"\nAI Analysis Report:\n{getattr(self, 'report_full_text', '')}\n"
        context_str += "--------------------\n"
        
        logging.info("Injecting video context into Agent Panel...")
        self.agent_system_context = context_str
        self.agent_panel.inject_context(context_str)

    def export_pdf(self):
        report_md = self.txt_report.toMarkdown() # or toPlainText since we display MD? 
        # Actually txt_report contains rendered HTML if we used setHtml? No, we used setMarkdown?
        # Let's check update_report (not visible here, but assuming it sets text).
        # We need the raw markdown for export function.
        # self.txt_report.toPlainText() is safest if we set raw text.
        
        if not self.output_dir:
            logging.info("无法导出: 无输出目录")
            return
            
        logging.info("正在导出 PDF...")
        try:
            from src.core.logic import export_report_as_pdf
            path = export_report_as_pdf(self.txt_report.toPlainText(), self.output_dir)
            if path:
                logging.info(f"PDF 导出成功: {path}")
                # Open folder?
            else:
                logging.info("PDF 导出失败 (请检查日志)")
        except Exception as e:
            logging.info(f"导出出错: {e}")

def run_main():
    """Main entry point for the desktop application."""
    try:
        import qdarktheme
        from PyQt6.QtWidgets import QApplication
        
        qdarktheme.enable_hi_dpi()
        app = QApplication(sys.argv)
        qdarktheme.setup_theme("dark")
        
        # Check for local ffmpeg override
        models_dir = Path("models").resolve()
        ffmpeg_path = models_dir / "ffmpeg.exe"
        if ffmpeg_path.exists():
            os.environ["IMAGEIO_FFMPEG_EXE"] = str(ffmpeg_path)
            logging.info(f"Using local FFmpeg override: {ffmpeg_path}")

        window = DesktopApp()
        window.show()
        
        return app.exec()
    except Exception as e:
        import traceback
        error_msg = f"Startup Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        
        # Write to crash log file
        try:
            with open("crash_log.txt", "w", encoding="utf-8") as f:
                f.write(error_msg)
        except: pass
        
        try:
             from PyQt6.QtWidgets import QMessageBox, QApplication
             if not QApplication.instance():
                 app = QApplication(sys.argv)
             else:
                 app = QApplication.instance()
             QMessageBox.critical(None, "Application Startup Failed", error_msg)
        except:
             pass
        return 1

if __name__ == "__main__":
    sys.exit(run_main())
    def init_backend(self):
        """Register all tools including search and OCR."""
        from src.core.agent_tools import (create_highlight_cut_tool, create_visual_grounding_tool)
        
        self.tool_registry = ToolRegistry()
        self.tool_registry.set_context_provider(lambda: self)
        
        self.tool_registry.register_tool("get_video_meta", "获取视频基本信息", create_get_video_meta_tool(lambda: self))
        self.tool_registry.register_tool("get_frame_details", "获取特定时间的画面细节", create_get_frame_details_tool(lambda: self), {"seconds": "时间(秒)"})
        self.tool_registry.register_tool("search_web", "搜索互联网内容", create_search_web_tool(), {"query": "搜索词"})
        self.tool_registry.register_tool("search_visual", "按语义搜索视频画面", create_visual_search_tool(lambda: self), {"query": "画面描述"})
        self.tool_registry.register_tool("run_ocr", "对特定画面执行文字识别", create_ocr_tool(lambda: self), {"seconds": "时间(秒)"})
        
        # Phase 2 Tools
        self.tool_registry.register_tool("create_highlights", "自动剪辑生成精彩集锦", create_highlight_cut_tool(lambda: self), {"description": "集锦描述"})
        self.tool_registry.register_tool("point_and_jump", "在画面中定位物体并跳转", create_visual_grounding_tool(lambda: self), {"query": "目标描述"})

    def seek_video(self, ts):
        """Unified jump method for Agent tools."""
        logging.info(f"Agent requested jump to {ts}s")
        # 1. If video player dialog is open, seek it
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QDialog) and "播放器" in widget.windowTitle():
                 if hasattr(widget, 'media_player'):
                     widget.media_player.setPosition(int(ts * 1000))
                     return
        
        # 2. Otherwise log it (could expand to seek main preview if exists)
        self.append_log(f"Agent 已定位到关键时刻: {ts:.1f}s")
