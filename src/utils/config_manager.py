import configparser
import os
import json
import logging
from src.utils.constants import CONFIG_DIR, MAIN_CONFIG_FILENAME, DEFAULT_THEME_NAME

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
            "api_url": "https://api.iflow.cn/v1",
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
        except:
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
        except:
            return []

    def save_prompts(self, prompts):
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            with open(self.prompts_path, 'w', encoding='utf-8') as f:
                json.dump(prompts, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save prompts: {e}")
