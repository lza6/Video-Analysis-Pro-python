import configparser
import os
import json
import logging
import logging as _log_mod
from src.utils.constants import CONFIG_DIR, MAIN_CONFIG_FILENAME, DEFAULT_THEME_NAME

# 凭据安全存储：优先 OS 密钥环（Windows DPAPI / macOS Keychain / Linux SecretService）。
# 不可用时降级到 ini 明文（旧行为），并在日志中明确警示。
try:
    import keyring
    _KEYRING_AVAILABLE = True
except Exception:
    _KEYRING_AVAILABLE = False

_KEYRING_SERVICE = "VideoAnalysisPro"


def _secure_set(key: str, value: str):
    """把 API Key 等敏感凭据存入 OS 密钥环；不可用时退回 ini。"""
    if _KEYRING_AVAILABLE and value:
        try:
            keyring.set_password(_KEYRING_SERVICE, key, value)
            return  # 成功 → 不落 ini
        except Exception as e:
            _log_mod.warning(f"keyring 不可用，凭据将明文存于 ini: {e}")
    # 回退：仅存一个标记位（"__keyring__"）而非真实 Key
    return None


def _secure_get(key: str, fallback: str = "") -> str:
    if _KEYRING_AVAILABLE:
        try:
            val = keyring.get_password(_KEYRING_SERVICE, key)
            if val:
                return val
        except Exception:
            pass
    return fallback


def is_keyring_available() -> bool:
    """v6.0 6.1：检测密钥环是否可用（启动时状态栏红点警告用）。

    返回 _KEYRING_AVAILABLE + 一次真实 get_password 探测（有些系统 import
    成功但无后端服务，get_password 会抛）。探测用伪 key 不影响真实数据。
    """
    if not _KEYRING_AVAILABLE:
        return False
    try:
        # 探测：get 一个不存在的 key，应返回 None 不抛
        keyring.get_password(_KEYRING_SERVICE, "__keyring_probe__")
        return True
    except Exception:
        return False


def audit_ini_key_cleared(ini_path: str = None) -> bool:
    """v6.0 6.1：启动时二次校验 app_config.ini 的 api_key 标记位已清空。

    save_current_settings 成功入密钥环后清空 ini 的 api_key 字段，但历史
    残留可能未清。本函数启动时检查 LastUsed.api_key 是否为空或"__keyring__"
    标记位。返回 True 表示已清空（安全），False 表示有明文残留（不安全）。
    """
    from src.utils.constants import CONFIG_DIR, MAIN_CONFIG_FILENAME
    p = ini_path or os.path.join(CONFIG_DIR, MAIN_CONFIG_FILENAME)
    if not os.path.exists(p):
        return True  # 无 ini 文件，无残留
    try:
        cp = configparser.ConfigParser()
        cp.read(p, encoding="utf-8")
        if "LastUsed" not in cp:
            return True
        api_key = cp.get("LastUsed", "api_key", fallback="")
        # 空 / "__keyring__" 标记位 = 已清空；其他 = 明文残留
        return api_key in ("", "__keyring__")
    except Exception:
        return True  # 读失败不阻断，保守返回 True


class ConfigurationManager:
    def __init__(self):
        self.config_dir = CONFIG_DIR
        self.config_path = os.path.join(self.config_dir, MAIN_CONFIG_FILENAME)
        self.presets_path = os.path.join(self.config_dir, "api_presets.json")
        self.prompts_path = os.path.join(self.config_dir, "prompts.json")
        self.config = configparser.ConfigParser()
        
    def load_main_config(self):
        """Loads the main configuration file, creating defaults if missing."""
        if not os.path.exists(self.config_path):
            self._create_default_config()
        
        try:
            self.config.read(self.config_path, encoding='utf-8')
        except Exception as e:
            logging.error(f"Error reading config: {e}")
            self._create_default_config()
            
        return self.config

    def _create_default_config(self):
        """Creates a default configuration file."""
        self.config["Application"] = {
            "theme": DEFAULT_THEME_NAME,
            "version": "3.1.0",
            "show_agent_panel": "True"
        }
        self.config["Environment"] = {
            "venv_path": ""
        }
        self.config["LastUsed"] = {
            "client_type": "1",
            "api_url": "",   # 中立默认：用户自行填入（OpenAI / DeepSeek / 自建网关等）
            "api_key": "",
            "model_name": ""
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        self._save_config()

    def update_config(self, section, key, value):
        """Updates a specific config value and saves."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = str(value)
        self._save_config()
        
    def _save_config(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                self.config.write(f)
        except Exception as e:
            logging.error(f"Failed to save config: {e}")

    # API Presets Management
    def load_api_presets(self):
        if not os.path.exists(self.presets_path):
            return []
        try:
            with open(self.presets_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def save_api_presets(self, presets):
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            with open(self.presets_path, 'w', encoding='utf-8') as f:
                json.dump(presets, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save presets: {e}")

    # Prompt Templates Management
    def load_prompts(self):
        if not os.path.exists(self.prompts_path):
            # Return some defaults if none exist
            return [
                {"name": "内容总结与评估", "content": "请对该视频内容进行全面总结，评估其视频质量、剪辑技巧及核心价值。"},
                {"name": "技术质量分析", "content": "请从分辨率、帧率、色彩平衡和对焦等方面分析该视频的技术质量。"},
                {"name": "情感与风格识别", "content": "请识别视频所传达的主要情感、环境氛围以及视觉风格。"}
            ]
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def save_prompts(self, prompts):
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            with open(self.prompts_path, 'w', encoding='utf-8') as f:
                json.dump(prompts, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save prompts: {e}")
