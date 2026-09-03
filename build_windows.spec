# -*- mode: python ; coding: utf-8 -*-
# ============================================================================
# Video Analysis Pro — Windows 打包配置 (PyInstaller onedir)
# 构建:  py -3.10 -m PyInstaller build_windows.spec --noconfirm
# 产物:  dist/VideoAnalysisPro/VideoAnalysisPro.exe  (onedir, 启动快、误报少)
#
# 设计要点:
#  - onedir 而非 onefile: 启动快 ~10x、杀软误报率显著更低
#  - config/ 提示词模板打进包
#  - FFmpeg 由 imageio-ffmpeg 自带二进制提供（无外部依赖）
#  - 模型 (yolo11n.pt 等) 首次运行由应用自动下载到 models/，不塞进安装包
# ============================================================================
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

datas = [
    ("config", "config"),
]
binaries = []
hiddenimports = []

# ultralytics / torch 家族需要完整收集
for pkg in ("ultralytics",):
    r = collect_all(pkg)
    datas += r[0]; binaries += r[1]; hiddenimports += r[2]

# chromadb 动态导入链
for pkg in ("chromadb",):
    try:
        r = collect_all(pkg)
        datas += r[0]; binaries += r[1]; hiddenimports += r[2]
    except Exception:
        pass

hiddenimports += [
    "torch", "torchvision", "torchaudio",
    "cv2", "ultralytics.nn.tasks",
    "sentence_transformers",
    "faster_whisper",
    "scenedetect",
    "moviepy",
    "PyQt6.QtMultimedia", "PyQt6.QtMultimediaWidgets",
    "matplotlib.backends.backend_qtagg",
]

a = Analysis(
    ["launcher.py"],
    pathex=[os.path.abspath(".")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 大体积可选依赖不进默认包（P2 决策: OCR 可选下载）
        "paddle", "paddleocr", "paddlepaddle",
        "pyannote", "decord",
        "gradio",
        # 开发工具
        "pytest", "pyinstaller",
        # 多 Qt 绑定冲突: PyInstaller 不支持同时打包多个 Qt 绑定。
        # 构建环境可能残留 PyQt5/PySide2，必须显式排除。
        "PyQt5", "PyQt5.QtCore", "PyQt5.QtWidgets", "PyQt5.QtGui",
        "PySide2", "PySide6",
        "IPython", "jedi", "parso",  # REPL 依赖，桌面应用不需要
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VideoAnalysisPro",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI 应用不出黑窗口
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="VideoAnalysisPro",
)
