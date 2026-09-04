"""SurveillanceTab 冒烟测试（offscreen）。

不发起真实 RTSP / 付费 API 请求——只验证：
  1. Tab 构造不崩（PyQt6 控件树正常建立）
  2. 空 URL 点"开始"给出错误提示而非崩溃（边界校验）
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from src.ui.surveillance_tab import SurveillanceTab


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class TestSurveillanceTabSmoke:
    def test_construct_does_not_crash(self, qapp):
        tab = SurveillanceTab(config_manager=None)
        assert tab.__class__.__name__ == "SurveillanceTab"
        # 关键控件存在
        assert tab.txt_rtsp_url is not None
        assert tab.txt_key_image is not None
        assert tab.combo_backend is not None
        assert tab.list_hits is not None

    def test_start_with_empty_url_shows_error_not_crash(self, qapp):
        tab = SurveillanceTab(config_manager=None)
        # 空 URL 直接点开始
        tab._on_start()
        status = tab.lbl_status.text()
        assert "错误" in status or "RTSP" in status
        assert tab._worker is None  # 不应启动 worker


class TestSurveillanceWorkerImportable:
    def test_worker_class_importable(self, qapp):
        from src.ui.surveillance_tab import SurveillanceWorker
        assert SurveillanceWorker.__name__ == "SurveillanceWorker"
