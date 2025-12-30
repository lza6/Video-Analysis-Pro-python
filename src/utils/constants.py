
import sys
import os

# Application Info
APP_NAME = "Video Analysis Pro"
APP_VERSION = "4.0.0"

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

# Dependencies to check/install
REQUIRED_PACKAGES = [
    "PyQt6>=6.6.0",
    "pyqtdarktheme>=2.1.0",
    "opencv-python-headless>=4.8.0",
    "scenedetect>=0.6.2",
    "ultralytics>=8.0.0",
    "sentence-transformers>=2.2.2",

    "markdown2",
    "numpy",
    "requests",
    "gradio", # Kept for backward compatibility if needed, or remove if fully migrating
    "moviepy"
]

# Paths
CONFIG_DIR = "config"
LOG_DIR = "logs"
CACHE_DIR = "软产生的缓存"  # Central folder for all generated resources
MAIN_CONFIG_FILENAME = "app_config.ini"
PROVIDERS_FILENAME = "providers.json"
TASK_STATES_FILENAME = "task_states.json"
PRESETS_FILENAME = "provider_presets.json"
ACTIVE_PRESET_FILENAME = "active_preset.json"
