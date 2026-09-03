import sys
from pathlib import Path

# 让 tests 能 import src 包
sys.path.insert(0, str(Path(__file__).parent.parent))

# torch 必须先于 PyQt6 加载（Windows DLL 顺序），与 main_window 同样处理
try:
    import torch  # noqa: F401
except OSError:
    pass
