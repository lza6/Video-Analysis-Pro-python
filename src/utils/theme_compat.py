"""主题兼容层 —— 桥接 pyqtdarktheme 0.1.7 与 2.x API。

原因：requirements.txt 声明 `pyqtdarktheme>=2.1.0`（带 `setup_theme`），
但实际安装的 0.1.7 只有 `load_palette`/`load_stylesheet`，无 `setup_theme`/`enable_hi_dpi`。
本模块在 import qdarktheme 后立即 monkey-patch，使生产代码（main_window.py）的
`qdarktheme.setup_theme("dark")` 调用在任何已发布版本上都能工作。

真实修复：未来统一 requirements.txt 与实际可装版本；本兼容层是过渡兜底。
"""
from __future__ import annotations

import qdarktheme

if not hasattr(qdarktheme, "setup_theme"):
    _orig_load = getattr(qdarktheme, "load_stylesheet", None)

    def _setup_theme(theme: str = "dark", **_kwargs) -> None:
        """0.1.7 → 2.x setup_theme 兼容：apply stylesheet to current QApplication."""
        try:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is None:
                return
            css = _orig_load(theme) if _orig_load else ""
            if css:
                app.setStyleSheet(css)
        except Exception:
            # 兜底：无 QApplication 时静默跳过（与 2.x 行为一致）
            pass

    qdarktheme.setup_theme = _setup_theme  # type: ignore[attr-defined]

if not hasattr(qdarktheme, "enable_hi_dpi"):
    # PyQt6 默认已启用 HiDPI（自 5.6 起），2.x 也已移除该函数，空操作即可
    qdarktheme.enable_hi_dpi = lambda *a, **k: None  # type: ignore[attr-defined]
