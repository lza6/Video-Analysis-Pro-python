from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QFrame, QScrollArea, QSizePolicy
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QIcon
import psutil
import time

class TaskItem(QFrame):
    def __init__(self, task_name):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedHeight(40)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 2, 10, 2)
        
        self.lbl_status = QLabel("⏳") # Icon
        self.lbl_name = QLabel(task_name)
        self.lbl_timer = QLabel("0s")
        self.start_time = time.time()
        self.end_time = None
        
        self.layout.addWidget(self.lbl_status)
        self.layout.addWidget(self.lbl_name)
        self.layout.addStretch()
        self.layout.addWidget(self.lbl_timer)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)

    def update_timer(self):
        if self.end_time is None:
            elapsed = int(time.time() - self.start_time)
            self.lbl_timer.setText(f"{elapsed}s")

    def set_status(self, success=True):
        self.end_time = time.time()
        self.timer.stop()
        elapsed = int(self.end_time - self.start_time)
        self.lbl_timer.setText(f"{elapsed}s")
        if success:
            self.lbl_status.setText("✅")
            self.setStyleSheet("background-color: rgba(76, 175, 80, 0.1);")
        else:
            self.lbl_status.setText("❌")
            self.setStyleSheet("background-color: rgba(244, 67, 54, 0.1);")

class ResourceMonitor(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 5, 10, 5)
        
        # CPU
        self.cpu_lbl = QLabel("CPU: 0%")
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setFixedHeight(10)
        self.cpu_bar.setTextVisible(False)
        
        # RAM
        self.ram_lbl = QLabel("RAM: 0%")
        self.ram_bar = QProgressBar()
        self.ram_bar.setRange(0, 100)
        self.ram_bar.setFixedHeight(10)
        self.ram_bar.setTextVisible(False)
        
        # VRAM (Mock/Place-holder for now, will be updated by main window via Ollama)
        self.vram_lbl = QLabel("VRAM: N/A")
        self.vram_bar = QProgressBar()
        self.vram_bar.setRange(0, 100)
        self.vram_bar.setFixedHeight(10)
        self.vram_bar.setTextVisible(False)

        self.layout.addWidget(self.cpu_lbl)
        self.layout.addWidget(self.cpu_bar)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.ram_lbl)
        self.layout.addWidget(self.ram_bar)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.vram_lbl)
        self.layout.addWidget(self.vram_bar)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_stats)
        self.timer.start(2000)

    def refresh_stats(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        self.cpu_lbl.setText(f"CPU: {cpu}%")
        self.cpu_bar.setValue(int(cpu))
        self.ram_lbl.setText(f"RAM: {ram}%")
        self.ram_bar.setValue(int(ram))

    def update_vram(self, used, total):
        percent = int(used / total * 100) if total > 0 else 0
        self.vram_lbl.setText(f"VRAM: {used:.1f}/{total:.1f} GB")
        self.vram_bar.setValue(percent)

class StatusConsole(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedHeight(120)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # Left side: Task Checklist
        self.task_area = QScrollArea()
        self.task_area.setWidgetResizable(True)
        self.task_content = QWidget()
        self.task_layout = QVBoxLayout(self.task_content)
        self.task_layout.addStretch()
        self.task_area.setWidget(self.task_content)
        self.layout.addWidget(self.task_area, stretch=3)
        
        # Right side: Resources
        self.resource_monitor = ResourceMonitor()
        self.layout.addWidget(self.resource_monitor, stretch=2)
        
        self.active_tasks = {}

    def add_task(self, name):
        task = TaskItem(name)
        # Add to top (before the stretch)
        self.task_layout.insertWidget(self.task_layout.count() - 1, task)
        self.active_tasks[name] = task
        return task

    def finish_task(self, name, success=True):
        if name in self.active_tasks:
            self.active_tasks[name].set_status(success)
            # We don't remove it immediately so the user can see the timer
