
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import subprocess
import sys
import os
import venv
from src.utils.constants import REQUIRED_PACKAGES, APP_NAME, VENV_SUBDIR_NAME, INITIAL_SYS_EXECUTABLE
from src.utils.config_manager import ConfigurationManager
import logging

class InitialThemeSelectorDialog(tk.Toplevel):
    def __init__(self, parent, title, themes, default_theme):
        super().__init__(parent)
        self.title("选择主题") # Localized
        self.themes = themes
        self.result_theme_name = default_theme
        
        tk.Label(self, text="请选择应用主题:").pack(pady=10) # Localized
        
        self.theme_var = tk.StringVar(value=default_theme)
        theme_combo = ttk.Combobox(self, textvariable=self.theme_var, values=list(themes.keys()), state="readonly")
        theme_combo.pack(pady=5, padx=20)
        
        tk.Button(self, text="确认", command=self.on_confirm).pack(pady=15) # Localized
        
        self.geometry("300x150")
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def on_confirm(self):
        self.result_theme_name = self.theme_var.get()
        self.destroy()

class EnvironmentSetupWindow:
    def __init__(self, master_tk_instance, on_setup_complete_callback, theme_settings):
        self.master = master_tk_instance
        self.callback = on_setup_complete_callback
        self.theme = theme_settings
        self.config_manager = ConfigurationManager()
        
        self.master.title(f"{APP_NAME} - 环境初始化") # Localized
        self.master.geometry("700x500") # Expanded for console
        self.master.configure(bg=self.theme["bg_color"])
        
        self.setup_ui()
        self.start_setup_thread()
        self.master.deiconify() 

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        bg = self.theme["bg_color"]
        fg = self.theme["fg_color"]
        
        main_frame = tk.Frame(self.master, bg=bg)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(main_frame, text="正在初始化应用环境...", 
                 font=("Microsoft YaHei", 16, "bold"), bg=bg, fg=fg).pack(pady=10)
        
        self.status_label = tk.Label(main_frame, text="正在检查依赖...", 
                                     font=("Microsoft YaHei", 10), bg=bg, fg=fg)
        self.status_label.pack(pady=5)
        
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(pady=15)
        
        # Real-time console look
        self.log_text = tk.Text(main_frame, height=15, width=80, bg="#0c0c0c", fg="#cccccc", 
                                font=("Consolas", 9), state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(pady=5, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text['yscrollcommand'] = scrollbar.set

    def log(self, message):
        self.master.after(0, self._append_log, message)

    def _append_log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        logging.info(message)

    def run_command_realtime(self, command_args):
        try:
            process = subprocess.Popen(
                command_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                encoding='utf-8', 
                errors='replace'
            )
            for line in process.stdout:
                self.log(line.strip())
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command_args)
        except Exception as e:
            raise e

    def start_setup_thread(self):
        threading.Thread(target=self.run_setup, daemon=True).start()

    def run_setup(self):
        try:
            # 1. Create Venv
            venv_path = os.path.join(os.getcwd(), VENV_SUBDIR_NAME)
            self.status_label.config(text="正在创建虚拟环境...")
            self.log(f"目标虚拟环境路径: {venv_path}")
            
            if not os.path.exists(venv_path):
                builder = venv.EnvBuilder(with_pip=True)
                builder.create(venv_path)
                self.log("虚拟环境创建成功。")
            else:
                self.log("检测到已存在的虚拟环境。")
            
            self.progress["value"] = 20
            
            # 2. Upgrade pip and install deps
            pip_exe = os.path.join(venv_path, "Scripts", "pip.exe") if sys.platform == "win32" else os.path.join(venv_path, "bin", "pip")
            python_exe = os.path.join(venv_path, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(venv_path, "bin", "python")
            
            self.status_label.config(text="正在更新依赖组件 (可能需要几分钟)...")
            self.log("正在升级 pip 工具...")
            self.run_command_realtime([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
            
            req_file_path = "requirements.txt"
            if os.path.exists(req_file_path):
                self.log(f"从 {req_file_path} 安装依赖...")
                self.run_command_realtime([pip_exe, "install", "-r", req_file_path])
            else:
                self.log("requirements.txt 未找到，使用内置列表安装...")
                self.run_command_realtime([pip_exe, "install", *REQUIRED_PACKAGES])
            
            self.log("所有依赖安装完成。")
            
            # 3. Create Validation Marker
            marker_path = os.path.join(venv_path, "env_validated.marker")
            with open(marker_path, "w") as f:
                f.write("Validated")
            
            # 4. Save Config
            self.config_manager.update_config("Environment", "venv_path", venv_path)
            
            self.status_label.config(text="设置完成。正在启动应用...")
            self.progress["value"] = 100
            
            # Delay slightly to show completion
            self.master.after(1500, lambda: self.callback(True, python_exe))
            
        except Exception as e:
            self.log(f"错误: {e}")
            self.status_label.config(text="错误: 设置失败")
            messagebox.showerror("设置失败", f"环境设置过程中出错: {str(e)}")
            self.callback(False, None)
