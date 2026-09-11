
import sys

# Application Info
APP_NAME = "TingFeng Hermes"
APP_VERSION = "10.3.1"

# Environment
VENV_SUBDIR_NAME = "venv"
USER_VALIDATION_FILE_NAME = "env_validated.marker"
INITIAL_SYS_EXECUTABLE = sys.executable

# Defaults
DEFAULT_THEME_NAME = "Dark"

# Logic Constants
MAX_CONSECUTIVE_TASK_FAILURES_BEFORE_RESET = 3
MAX_ITERATION_RETRIES = 3
MAX_RESPONSE_LENGTH = 4096
RESOURCE_MONITOR_INTERVAL = 1.0
FULL_TEXT_DUPLICATION_CHECK_INTERVAL = 5.0

# Themes
THEMES = {
    "Light": {
        "bg_color": "#ffffff",
        "fg_color": "#000000",
        "accent_color": "#007bff",
        "button_bg": "#e0e0e0",
        "button_fg": "#000000"
    },
    "Dark": {
        "bg_color": "#2b2b2b",
        "fg_color": "#ffffff",
        "accent_color": "#3daee9",
        "button_bg": "#3c3f41",
        "button_fg": "#ffffff"
    }
}

# Dependencies checked by the launcher import-verification script.
# Mirrors requirements.txt + requirements-web.txt (keep in sync).
# v9.0.0: PyQt6/pyqtdarktheme 全清(桌面壳走 Electron,后端纯 FastAPI,无 Qt 依赖)。
REQUIRED_PACKAGES = [
    "opencv-python-headless>=4.8.0",
    "numpy",
    "scenedetect>=0.6.2",
    "ultralytics>=8.3.0",
    "torch",
    "faster-whisper>=1.0.0",
    "sentence-transformers>=2.3.0",
    "chromadb>=1.5.9",
    "markdown2",
    "requests",
    "psutil",
    "moviepy>=2.0",
    "imageio-ffmpeg",
    # Web 后端(v8 Web UI 重构新增,launcher 需验证)
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "python-multipart>=0.0.12",
    "sse-starlette>=2.1.3",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
]

# Paths
CONFIG_DIR = "config"
LOG_DIR = "logs"
CACHE_DIR = "cache"  # Central folder for all generated resources（ASCII，规避 cv2 中文路径问题）
MAIN_CONFIG_FILENAME = "app_config.ini"
PROVIDERS_FILENAME = "providers.json"
TASK_STATES_FILENAME = "task_states.json"
PRESETS_FILENAME = "provider_presets.json"
ACTIVE_PRESET_FILENAME = "active_preset.json"
