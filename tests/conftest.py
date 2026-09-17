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


@pytest.fixture(autouse=True)
def _reset_ip_rate_limiter():
    """每个测试前后重置进程级 IP 限流器（v10.4.0 修测试隔离缺陷）。

    WHY: `src/web/security.py` 的 `_ip_limiter` 是**进程级全局**，60s 滑动窗口。
    一个 pytest 进程里跑几十个文件时，同一 IP（TestClient 固定为 'testclient'）
    的请求会累积，默认上限 10 req/min 被击穿 → 后续用例收到 429，表现为
    “随机失败”（单独跑该文件却全绿）。

    这是**测试隔离缺陷**，不是被测代码的 bug：限流本身是预期行为。重置后
    用例结果与执行顺序无关；专门测限流的用例会自己调 `init_ip_limiter`，
    不受影响。
    """
    try:
        from src.web import security as _security
    except Exception:  # noqa: BLE001 — 导入失败不该拖垮整个测试会话
        yield
        return
    _security._ip_limiter = None
    yield
    _security._ip_limiter = None


@pytest.fixture(scope="session")
def qapp():
    """占位 fixture(已无 PyQt6 依赖)。

    v9.0.0 起桌面壳走 Electron,后端纯 FastAPI,测试无 Qt 事件循环需求。
    保留此 fixture 名以兼容历史测试签名(返回 None,无人再读它的属性)。
    """
    yield None
