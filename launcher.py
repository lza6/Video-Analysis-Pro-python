import sys
import os
import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext
import logging
import subprocess
import shutil
import time
from datetime import datetime, timezone, timedelta
import webbrowser
import threading
import json
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
import io
import traceback
from logging.handlers import RotatingFileHandler

# =====================================================================================
# 核心常量和配置 (Core Constants and Configuration)
# =====================================================================================
from src.utils.constants import (
    APP_NAME, VENV_SUBDIR_NAME, USER_VALIDATION_FILE_NAME, INITIAL_SYS_EXECUTABLE,
    DEFAULT_THEME_NAME, MAX_CONSECUTIVE_TASK_FAILURES_BEFORE_RESET, MAX_ITERATION_RETRIES,
    MAX_RESPONSE_LENGTH, RESOURCE_MONITOR_INTERVAL, FULL_TEXT_DUPLICATION_CHECK_INTERVAL,
    THEMES, REQUIRED_PACKAGES, CONFIG_DIR, LOG_DIR, MAIN_CONFIG_FILENAME
)

# --- 应用版本 (Application Version) ---
APP_VERSION = "v4.0.0"

# =====================================================================================
# 辅助函数和工具 (Helper Functions & Utilities)
# =====================================================================================

def get_venv_python_executable(venv_dir_path: str) -> Optional[str]:
    """根据虚拟环境目录获取Python解释器路径。"""
    if not venv_dir_path:
        return None
    if sys.platform == "win32":
        return os.path.join(venv_dir_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_dir_path, "bin", "python")

def is_running_in_target_venv(target_venv_python_exe: str) -> bool:
    """检查当前是否在目标虚拟环境中运行。"""
    if not target_venv_python_exe:
        return False
    current_python_exe = os.path.normpath(os.path.abspath(sys.executable))
    target_python_exe = os.path.normpath(os.path.abspath(target_venv_python_exe))
    return current_python_exe == target_python_exe

def get_clean_env_for_subprocess() -> Dict[str, str]:
    """获取一个干净的、用于子进程的环境变量字典。"""
    clean_env = os.environ.copy()
    for key in ["PYTHONPATH", "PYTHONHOME"]:
        if key in clean_env:
            del clean_env[key]
    return clean_env

def run_import_verification_script(python_executable: str) -> Tuple[bool, str]:
    """运行一个子进程来验证Python环境中是否安装了所有必要的库。"""
    # 从 REQUIRED_PACKAGES 提取包名（去除版本号）
    package_names = []
    for pkg in REQUIRED_PACKAGES:
        # 提取包名部分（去除版本号和比较符）
        pkg_name = pkg.split('>=')[0].split('==')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
        # 处理特殊包名 mapping
        if pkg_name == 'py-cpuinfo':
            pkg_name = 'cpuinfo'
        elif pkg_name == 'google-generativeai':
            pkg_name = 'google.generativeai'
        
        # 对于 opencv-python-headless 验证 import cv2
        if 'opencv' in pkg_name:
            pkg_name = 'cv2'
        elif pkg_name == 'sentence-transformers':
            pkg_name = 'sentence_transformers'
            
        package_names.append(pkg_name)

    verification_script_content = f"""
import sys
import importlib

required_packages = {package_names}
missing_packages = []
import_errors = []

for package in required_packages:
    try:
        # 特殊处理
        if package == 'gradio':
            import gradio 
        elif package == 'qdarktheme':
            import qdarktheme
        else:
            importlib.import_module(package)
    except ImportError as e:
        missing_packages.append(package)
        import_errors.append(f"{{package}}: {{e}}")

if not missing_packages:
    print("SUCCESS")
    sys.exit(0)
else:
    print(f"ERROR: Missing packages: {{', '.join(missing_packages)}}")
    print("Details:")
    for error in import_errors:
        print(error)
    sys.exit(1)
"""
    try:
        process = subprocess.run(
            [python_executable, "-c", verification_script_content],
            capture_output=True, text=True, check=True,
            env=get_clean_env_for_subprocess(),
            encoding='utf-8'
        )
        output = process.stdout.strip()
        if "SUCCESS" in output:
            return True, "所有核心库均已安装。"
        else:
            return False, output
    except subprocess.CalledProcessError as e:
        error_message = f"验证脚本执行失败。返回码: {e.returncode}\\n"
        error_message += f"--- STDOUT ---\\n{e.stdout.strip()}\\n"
        error_message += f"--- STDERR ---\\n{e.stderr.strip()}\\n"
        return False, error_message
    except FileNotFoundError:
        return False, f"Python可执行文件未找到: {python_executable}"
    except Exception as e:
        return False, f"运行验证脚本时发生未知错误: {e}"

# =====================================================================================
# 主启动逻辑 (Main Startup Logic)
# =====================================================================================

def handle_setup_completion_and_relaunch(success: bool, venv_python_exe_to_run: Optional[str] = None):
    """处理环境设置完成后的重新启动逻辑。"""
    global _setup_master_root_instance
    if _setup_master_root_instance and _setup_master_root_instance.winfo_exists():
        _setup_master_root_instance.destroy()
        _setup_master_root_instance = None

    if success and venv_python_exe_to_run:
        try:
            # On Windows, try to use pythonw.exe for GUI apps to avoid console window
            if sys.platform == "win32" and "python.exe" in venv_python_exe_to_run:
                pythonw = venv_python_exe_to_run.replace("python.exe", "pythonw.exe")
                if os.path.exists(pythonw):
                    venv_python_exe_to_run = pythonw
            
            logging.info(f"环境设置成功。使用虚拟环境Python重新启动: {venv_python_exe_to_run}")
            current_script_path = os.path.abspath(sys.argv[0])
            clean_env_relaunch = get_clean_env_for_subprocess()
            
            if sys.platform == "win32":
                # Use DETACHED_PROCESS (0x00000008) to separate console
                # or CREATE_NO_WINDOW (0x08000000)
                creation_flags = 0x00000008 
                subprocess.Popen([venv_python_exe_to_run, current_script_path] + sys.argv[1:], 
                                 env=clean_env_relaunch, 
                                 creationflags=creation_flags, 
                                 close_fds=True)
            else:
                subprocess.Popen([venv_python_exe_to_run, current_script_path] + sys.argv[1:], env=clean_env_relaunch)
            sys.exit(0)
        except Exception as e_relaunch:
            logging.critical(f"使用虚拟环境Python重新启动失败: {e_relaunch}", exc_info=True)
            root_err_relaunch = tk.Tk(); root_err_relaunch.withdraw()
            messagebox.showerror("重新启动失败", f"尝试使用隔离环境重新启动脚本失败: {e_relaunch}\\n请尝试手动从命令行激活隔离环境并运行此脚本。", master=root_err_relaunch)
            if root_err_relaunch.winfo_exists(): root_err_relaunch.destroy()
            sys.exit(1)
    else:
        logging.error("环境设置失败或未提供虚拟环境Python路径用于重新启动。")
        sys.exit(1)

def cleanup_environment_and_config(configured_venv_path: str, reason: str = ""):
    """清理虚拟环境和配置文件。"""
    logging.warning(f"清理环境和配置。原因: {reason}")
    config_file = os.path.join(CONFIG_DIR, MAIN_CONFIG_FILENAME)
    if os.path.exists(config_file):
        try:
            os.remove(config_file)
            logging.info(f"已移除配置文件: {config_file}")
        except Exception as e:
            logging.error(f"移除配置文件 {config_file} 失败: {e}", exc_info=True)

    if configured_venv_path and os.path.isdir(configured_venv_path):
        try:
            shutil.rmtree(configured_venv_path)
            logging.info(f"已移除虚拟环境目录: {configured_venv_path}")
        except Exception as e:
            logging.error(f"移除虚拟环境目录 {configured_venv_path} 失败: {e}", exc_info=True)

_setup_master_root_instance = None

if __name__ == "__main__":
    # 确保日志和配置目录存在
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 配置日志
    log_file_path = os.path.join(LOG_DIR, "launcher.log")
    file_handler = RotatingFileHandler(
        log_file_path, 
        maxBytes=100 * 1024 * 1024, 
        backupCount=3, 
        encoding='utf-8'
    )
    stream_handler = logging.StreamHandler()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        handlers=[file_handler, stream_handler]
    )
    
    logging.info(f"--- {APP_NAME} 启动程序 ---")
    logging.info(f"初始Python: {INITIAL_SYS_EXECUTABLE}")
    logging.info(f"当前 sys.executable: {sys.executable}")
    logging.info(f"命令行参数: {sys.argv}")

    # 导入config_manager并加载配置
    try:
        from src.utils.config_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        app_config_main = config_manager.load_main_config()
    except Exception as e:
        logging.critical(f"无法加载配置管理器: {e}", exc_info=True)
        root_err = tk.Tk(); root_err.withdraw()
        messagebox.showerror("严重错误", f"无法加载配置管理器: {e}")
        sys.exit(1)

    configured_venv_path = app_config_main.get("Environment", "venv_path", fallback=None)
    logging.info(f"从配置读取的虚拟环境路径: {configured_venv_path}")

    needs_full_setup = True

    if configured_venv_path and os.path.isdir(configured_venv_path):
        expected_venv_python = get_venv_python_executable(configured_venv_path)
        user_validation_file_full_path = os.path.join(configured_venv_path, USER_VALIDATION_FILE_NAME)
        logging.info(f"预期的虚拟环境Python: {expected_venv_python}")
        logging.info(f"预期的验证文件: {user_validation_file_full_path}")

        if expected_venv_python and os.path.exists(expected_venv_python):
            if os.path.exists(user_validation_file_full_path):
                logging.info("虚拟环境Python和验证文件均存在。" )
                if is_running_in_target_venv(expected_venv_python):
                    logging.info("当前正在目标虚拟环境中运行。")
                    try:
                        desktop_app_path = "src/ui/main_window.py"
                        if os.path.exists(desktop_app_path):
                             logging.info(f"正在直接启动桌面应用模块: {desktop_app_path}")
                             
                             # Fix module search path before import
                             sys.path.append(os.getcwd())
                             
                             try:
                                 from src.ui.main_window import run_main
                                 exit_code = run_main()
                                 if exit_code != 0:
                                     logging.error(f"桌面应用异常退出. Exit Code: {exit_code}")
                                     sys.exit(exit_code)
                                 else:
                                     logging.info("桌面应用正常退出。")
                                     sys.exit(0)
                             except Exception as e_import:
                                 logging.error(f"直接导入启动失败: {e_import}。尝试回退到子进程启动方式。")
                                 # Fallback to subprocess if direct import/run fails
                                 p = subprocess.Popen(
                                     [sys.executable, "-m", "src.ui.main_window"],
                                     stdout=None, 
                                     stderr=None
                                 )
                                 p.wait()
                                 sys.exit(p.returncode)

                        else:
                             logging.warning("未找到桌面应用文件。回退到 Gradio 应用。")
                             p = subprocess.Popen([sys.executable, "app.py"])
                             p.wait()
                             sys.exit(p.returncode)
                        
                        needs_full_setup = False
                        sys.exit(0)
                        
                    except ImportError as e:
                         # This block is somewhat theoretical since we are using subprocess above
                         logging.critical(f"Import failed: {e}", exc_info=True)
                else: # 不在虚拟环境中，但虚拟环境看起来是好的
                    logging.info("未在目标虚拟环境中运行。尝试重新启动到虚拟环境。" )
                    needs_full_setup = False
                    handle_setup_completion_and_relaunch(True, expected_venv_python)
                    sys.exit(1) # 确保当前进程退出
            else: # 验证文件丢失，需要重新验证
                logging.warning("验证文件缺失。重新验证虚拟环境。" )
                verification_ok, verification_details_msg = run_import_verification_script(expected_venv_python)
                if verification_ok:
                    logging.info("虚拟环境重新验证成功。创建验证文件。" )
                    try:
                        with open(user_validation_file_full_path, "w", encoding='utf-8') as f_val:
                            f_val.write(f"环境由 {APP_NAME} 于 {time.strftime('%Y-%m-%d %H:%M:%S')} 自动验证并标记。\\n")
                        needs_full_setup = False
                        logging.info("验证文件已创建。重新启动到虚拟环境。" )
                        handle_setup_completion_and_relaunch(True, expected_venv_python)
                        sys.exit(0)
                    except Exception as e_create_val:
                        logging.error(f"重新验证后创建验证文件失败: {e_create_val}", exc_info=True)
                        cleanup_environment_and_config(configured_venv_path, f"快速验证后创建验证文件失败: {e_create_val}")
                else: # 重新验证失败，环境损坏
                    logging.error(f"虚拟环境重新验证失败: {verification_details_msg}")
                    cleanup_environment_and_config(configured_venv_path, f"快速验证失败: {verification_details_msg}")
        else: # python.exe丢失或路径无效
            logging.warning("预期的虚拟环境Python可执行文件缺失或虚拟环境路径不是目录。正在清理。" )
            cleanup_environment_and_config(configured_venv_path, "Python解释器缺失或路径无效")

    if needs_full_setup:
        logging.info("需要完整的环境设置。启动设置向导。" )
        try:
            from src.utils.ui_components import EnvironmentSetupWindow, InitialThemeSelectorDialog
            from src.utils.constants import THEMES, DEFAULT_THEME_NAME
        except ImportError as e:
            logging.critical(f"无法导入UI组件进行设置: {e}", exc_info=True)
            root_err = tk.Tk(); root_err.withdraw()
            messagebox.showerror("严重错误", f"无法加载UI组件: {e}\\n程序无法继续。" )
            sys.exit(1)

        _setup_master_root_instance = tk.Tk()
        _setup_master_root_instance.withdraw()

        initial_theme_name_for_setup = app_config_main.get("Application", "theme", fallback=DEFAULT_THEME_NAME)
        if initial_theme_name_for_setup not in THEMES:
            initial_theme_name_for_setup = DEFAULT_THEME_NAME
        
        if not os.path.exists(os.path.join(CONFIG_DIR, MAIN_CONFIG_FILENAME)) or not app_config_main.has_option("Application", "theme"):
             theme_dialog = InitialThemeSelectorDialog(_setup_master_root_instance, "选择向导主题", THEMES, DEFAULT_THEME_NAME)
             selected_theme_for_setup = theme_dialog.result_theme_name
             if selected_theme_for_setup and selected_theme_for_setup in THEMES:
                 initial_theme_name_for_setup = selected_theme_for_setup

        initial_theme_settings_for_setup = THEMES[initial_theme_name_for_setup]

        setup_window = EnvironmentSetupWindow(
            master_tk_instance=_setup_master_root_instance,
            on_setup_complete_callback=handle_setup_completion_and_relaunch,
            theme_settings=initial_theme_settings_for_setup
        )
        _setup_master_root_instance.mainloop()

    sys.exit(0)
