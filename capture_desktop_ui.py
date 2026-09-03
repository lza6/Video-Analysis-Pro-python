"""捕获 Video Analysis Pro 桌面端 UI 截图（offscreen Qt 渲染）。

策略：
- QT_QPA_PLATFORM=offscreen 无头渲染 + QT_SCALE_FACTOR=2 (retina)
- 通过 src.utils.theme_compat 桥接 qdarktheme 0.1.7 → 2.x API（生产代码同款，非 monkeypatch shim）
- 每个 tab 截图：AI 报告 / 关键帧画廊 / 摘要媒体 / 元数据 / 系统日志 / 模型管理 / 获取 API
- 保存到 website/docs/screenshots/desktop/raw/，再由 sharp 压缩
"""
import sys
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_SCALE_FACTOR", "2")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

app = QApplication(sys.argv)

import qdarktheme
from src.utils import theme_compat  # noqa: F401 桥接 qdarktheme 0.1.7→2.x setup_theme
qdarktheme.setup_theme("dark")
qdarktheme.enable_hi_dpi()

from src.ui.main_window import DesktopApp  # noqa: E402

OUT_DIR = "website/docs/screenshots/desktop/raw"
os.makedirs(OUT_DIR, exist_ok=True)

win = DesktopApp()
win.resize(1440, 900)
win.show()
app.processEvents()

TABS = [
    ("report", "tab_report"),
    ("gallery", "tab_gallery"),
    ("media", "tab_media"),
    ("metrics", "tab_metrics"),
    ("logs", "tab_logs"),
    ("models", "tab_models"),
    ("api-help", "tab_api_help"),
]

results = []


def grab(filename):
    for _ in range(3):
        win.repaint()
        app.processEvents()
    pix = win.grab()
    out = os.path.join(OUT_DIR, filename)
    pix.save(out)
    return os.path.getsize(out)


def capture_all():
    # 默认 tab 整体
    size = grab("desktop-overview.png")
    results.append(("overview", size))

    for label, attr in TABS:
        try:
            tab = getattr(win, attr)
            win.tabs.setCurrentWidget(tab)
            for _ in range(5):
                win.repaint()
                app.processEvents()
            size = grab(f"desktop-tab-{label}.png")
            results.append((label, size))
        except Exception as e:
            results.append((label, f"ERR {e}"))

    for label, size in results:
        if isinstance(size, int):
            print(f"  ✓ tab-{label} -> {size // 1024}KB")
        else:
            print(f"  ✗ tab-{label}: {size}")
    print(f"\n✓ captured {sum(1 for _, s in results if isinstance(s, int))}")
    app.quit()


QTimer.singleShot(400, capture_all)
sys.exit(app.exec())
