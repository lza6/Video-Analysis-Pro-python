"""监控分析 Tab —— 把孤岛后端模块接进主 UI。

接线两个已存在但未暴露的核心模块：
  - src.core.rtsp_stream.RtspMonitor      实时 RTSP 拉流 + 运动检测 + VLM 命中
  - src.core.surveillance_agent.SurveillanceAgent  离线视频目录批量搜索
  - src.core.llm_gateway.build_backend      构造 LLM 后端（监控分析的判图引擎）

线程铁律（与 main_window.seek_video 一致）：
  RtspMonitor 内部已用 daemon 线程拉流 + VLM 工作线程，命中回调来自非主线程。
  本 Tab 用 SurveillanceWorker(QThread) 持有 RtspMonitor，命中通过 pyqtSignal
  回主线程，绝不跨线程直接改 QListWidget（Qt 线程铁律）。

停止/退出与 main_window.closeEvent 同模式：worker.stop() + wait(3000)。
"""
import logging
import os
from pathlib import Path
from typing import Optional

# CRITICAL: torch 必须先于 PyQt6 加载（Windows DLL 顺序，见 main_window 顶部注释）。
try:
    import torch  # noqa: F401
except OSError:
    torch = None  # Headless/CPU-broken 环境也要能展示 UI

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# 后端 import 延迟到运行时（构造时不强制 cv2/numpy/torch 可用，headless 测试仍可 import Tab）
def _build_llm_backend(api_key: str, base_url: str, model: str,
                       protocol_hint: str = "anthropic"):
    """通过 llm_gateway.build_backend 构造后端（监控 VLM 判图用）。

    无 key/无 url 时返回 None，由调用方决定降级行为。
    """
    if not api_key or not base_url or not model:
        return None
    from src.core.llm_gateway import build_backend, detect_protocol
    protocol = detect_protocol(base_url, model) if protocol_hint == "auto" else protocol_hint
    try:
        return build_backend(protocol, api_key, base_url, model)
    except Exception as e:
        logger.warning(f"[surveillance] LLM 后端构造失败: {e}")
        return None


class SurveillanceWorker(QThread):
    """RTSP 监控后台线程。持有 RtspMonitor，命中信号回主线程。

    QThread 本身不跑业务循环——RtspMonitor.start() 启动的是它自己的 daemon
    拉流线程；本 QThread 仅用于"持有 monitor 并安全 stop"的生命周期管理，
    避免主 UI 退出时 Qt 报 "QThread: Destroyed while still running"。
    """

    hit_detected = pyqtSignal(float, str, float)   # timestamp, detail, confidence
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, rtsp_url: str, backend, key_item_image: str,
                 item_description: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._rtsp_url = rtsp_url
        self._backend = backend
        self._key_item_image = key_item_image
        self._item_description = item_description
        self._monitor = None  # type: Optional[object]

    def run(self):  # noqa: D401 (QThread 钩子)
        """启动监控并保持线程存活直到 stop()。

        RtspMonitor.start 内部已 spawn daemon 拉流线程，本方法只需让它存活
        并轮询命中事件转发到信号（不阻塞、不跨线程改 UI）。
        """
        if not self._rtsp_url:
            self.error_occurred.emit("RTSP URL 为空")
            return
        try:
            from src.core.rtsp_stream import RtspMonitor
            self._monitor = RtspMonitor(
                rtsp_url=self._rtsp_url,
                backend=self._backend,
                key_item_image=self._key_item_image,
                item_description=self._item_description or "关键物品",
            )
            self._monitor.start(fps=1.0)
            self.status_changed.emit("监控运行中")
            # 轮询新命中事件（每 1s 扫一次 events 列表尾部的 hit）
            last_seen = 0
            while not self.isInterruptionRequested() and self._monitor is not None:
                self.msleep(1000)
                evs = getattr(self._monitor, "events", [])
                if len(evs) > last_seen:
                    for ev in evs[last_seen:]:
                        if getattr(ev, "kind", "") == "hit":
                            self.hit_detected.emit(
                                float(getattr(ev, "timestamp", 0.0)),
                                str(getattr(ev, "detail", "")),
                                float(getattr(ev, "confidence", 0.0)),
                            )
                    last_seen = len(evs)
        except Exception as e:
            logger.exception("[surveillance] worker 异常")
            self.error_occurred.emit(f"监控异常: {e}")

    def stop(self):
        """主线程调用：先停 RtspMonitor 拉流线程，再请求 QThread 退出。"""
        if self._monitor is not None:
            try:
                self._monitor.stop()
            except Exception as e:
                logger.warning(f"[surveillance] monitor.stop 异常: {e}")
        self.requestInterruption()
        self.quit()


class SurveillanceTab(QWidget):
    """监控分析 Tab（RTSP 实时监控 + 离线搜索入口）。

    接线点（供 main_window 统一接入 tab bar）：
        from src.ui.surveillance_tab import SurveillanceTab
        self.tab_surveillance = SurveillanceTab(self.config_manager)
        self.tabs.addTab(self.tab_surveillance, "🛰️ 监控分析")
    """

    def __init__(self, config_manager=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config_manager = config_manager
        self._worker: Optional[SurveillanceWorker] = None
        self._setup_ui()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ---- 配置区 ----
        cfg_box = QGroupBox("监控配置")
        cfg_layout = QVBoxLayout(cfg_box)

        # RTSP URL
        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("RTSP 流:"))
        self.txt_rtsp_url = QLineEdit()
        self.txt_rtsp_url.setPlaceholderText("rtsp://user:pass@ip/stream")
        url_row.addWidget(self.txt_rtsp_url, stretch=1)
        cfg_layout.addLayout(url_row)

        # 关键物品图片
        item_row = QHBoxLayout()
        item_row.addWidget(QLabel("关键物品图:"))
        self.txt_key_image = QLineEdit()
        self.txt_key_image.setPlaceholderText("选择一张关键物品参考图（jpg/png）")
        item_row.addWidget(self.txt_key_image, stretch=1)
        self.btn_browse = QPushButton("浏览…")
        self.btn_browse.clicked.connect(self._on_browse_image)
        item_row.addWidget(self.btn_browse)
        cfg_layout.addLayout(item_row)

        # LLM 后端选择
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("判图后端:"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["Anthropic", "OpenAI Chat", "Ollama"])
        backend_row.addWidget(self.combo_backend, stretch=1)
        cfg_layout.addLayout(backend_row)

        # 开始 / 停止
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始监控")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("■ 停止监控")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        cfg_layout.addLayout(btn_row)

        root.addWidget(cfg_box)

        # ---- 命中列表 ----
        hit_box = QGroupBox("命中事件")
        hit_layout = QVBoxLayout(hit_box)
        self.list_hits = QListWidget()
        self.list_hits.setAlternatingRowColors(True)
        hit_layout.addWidget(self.list_hits)
        root.addWidget(hit_box, stretch=1)

        # ---- 状态 ----
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color: gray;")
        root.addWidget(self.lbl_status)

    # ------------------------------------------------------------------ 槽

    def _on_browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择关键物品参考图", "", "图片 (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.txt_key_image.setText(path)

    def _read_api_config(self) -> tuple:
        """从 config_manager 读 api_url/api_key/model_name。

        返回 (api_url, api_key, model)。config_manager 缺失时返回空串三元组，
        调用方据此给出"未配置 LLM"错误提示而非崩溃。
        """
        if self._config_manager is None:
            return ("", "", "")
        try:
            cfg = self._config_manager.load_main_config()
            last = cfg["LastUsed"] if "LastUsed" in cfg else {}
            return (
                last.get("api_url", ""),
                last.get("api_key", ""),
                last.get("model_name", ""),
            )
        except Exception as e:
            logger.warning(f"[surveillance] 读 api 配置失败: {e}")
            return ("", "", "")

    def _resolve_protocol(self) -> str:
        """按 combo 选择映射 llm_gateway 协议名。Ollama 走 openai_chat（兼容）。"""
        idx = self.combo_backend.currentIndex()
        return ["anthropic", "openai_chat", "openai_chat"][idx] if idx >= 0 else "anthropic"

    def _on_start(self):
        url = self.txt_rtsp_url.text().strip()
        key_img = self.txt_key_image.text().strip()
        if not url:
            self._set_status("错误：RTSP URL 为空", error=True)
            return
        if not key_img or not Path(key_img).exists():
            self._set_status("错误：请选择有效的关键物品参考图", error=True)
            return

        api_url, api_key, model = self._read_api_config()
        protocol = self._resolve_protocol()
        backend = _build_llm_backend(api_key, api_url, model, protocol_hint=protocol)
        if backend is None:
            self._set_status(
                "错误：LLM 后端未配置（请在主面板填 api_url/api_key/model 后重试）",
                error=True)
            return

        self.list_hits.clear()
        self._worker = SurveillanceWorker(
            rtsp_url=url,
            backend=backend,
            key_item_image=key_img,
            item_description="关键物品",
            parent=self,
        )
        self._worker.hit_detected.connect(self._on_hit)
        self._worker.status_changed.connect(self._set_status)
        self._worker.error_occurred.connect(
            lambda msg: self._set_status(msg, error=True))
        self._worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_status("监控启动中…")

    def _on_stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        self._worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_status("已停止")

    def _on_hit(self, ts: float, detail: str, confidence: float):
        """主线程槽（由 hit_detected 信号触发，Qt 自动跨线程回主线程）。"""
        from datetime import datetime
        when = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts > 0 else "--:--:--"
        conf = f"{confidence:.2f}" if confidence else "0.00"
        text = f"[{when}] 命中 (conf={conf})  {detail[:80]}"
        item = QListWidgetItem(text)
        self.list_hits.addItem(item)

    def _set_status(self, msg: str, error: bool = False):
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet("color: #c0392b;" if error else "color: gray;")

    # ------------------------------------------------------------------ 生命周期

    def closeEvent(self, event):
        """与 main_window.closeEvent 同模式：stop()+wait(3000) 后台线程。"""
        self._on_stop()
        super().closeEvent(event)
