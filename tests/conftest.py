import sys
from pathlib import Path

import pytest

# 让 tests 能 import src 包
sys.path.insert(0, str(Path(__file__).parent.parent))

# torch 必须先于其它 C 扩展加载(Windows DLL 顺序坑,与 Qt 无关,本项目已去 PyQt6)。
# conftest 顶部 import torch 守卫:后续测试若 import cv2/torch 相关模块,
# DLL 解析顺序已定,避免 vcruntime 旧副本冲突。
try:
    import torch  # noqa: F401
except OSError:
    pass


@pytest.fixture(scope="session")
def qapp():
    """占位 fixture(已无 PyQt6 依赖)。

    v9.0.0 起桌面壳走 Electron,后端纯 FastAPI,测试无 Qt 事件循环需求。
    保留此 fixture 名以兼容历史测试签名(返回 None,无人再读它的属性)。
    """
    yield None
