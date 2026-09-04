import sys
from pathlib import Path

import pytest

# 让 tests 能 import src 包
sys.path.insert(0, str(Path(__file__).parent.parent))

# torch 必须先于 PyQt6 加载（Windows DLL 顺序），与 main_window 同样处理
try:
    import torch  # noqa: F401
except OSError:
    pass


@pytest.fixture(scope="session")
def qapp():
    """全局 QApplication fixture（offscreen 平台，session 级单例）。

    此前 test_e2e_smoke / test_e2e_full_pipeline 依赖的 qapp 由同包其它测试
    文件（test_ui_components 等）的 module 级 fixture 提供——单跑该文件时
    'fixture qapp not found'（CI 打包 job 曾因它红）。收进根 conftest 统一。
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
