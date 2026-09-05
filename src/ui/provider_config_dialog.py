"""对话式 Provider 配置弹窗（v5.8 断点 B3）。

之前 main_window.on_agent_dialog_config_provider 只切 tab + 贴引导文案，
没真调 orchestrator.configure_provider_dialog（测活性 + 入密钥环）。
本弹窗做两阶段对话式配置：
  1. 用户选 provider 类型 + 填 url/key/model
  2. 点「测活性」→ 调 orchestrator.configure_provider_dialog 真测 list_models
  成功 → 调 config_manager._secure_set 入密钥环 + 发 saved 信号给 main_window

不真实调付费 API（红线）：list_models 是 GET /v1/models，只测连接不发推理请求。
密钥环失败降级 ini 明文 + 日志告警（config_manager 已有逻辑）。
"""
from __future__ import annotations

import logging
from typing import Optional

# CRITICAL: torch 必须先于 PyQt6 加载（Windows DLL 顺序，见 main_window.py 顶部注释）
try:
    import torch  # noqa: F401
except OSError:
    torch = None  # type: ignore[assignment]

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# provider 下拉项（与 main_window 客户端类型对齐，但这里只列 API 类）
_PROVIDER_OPTIONS = [
    ("NVIDIA Integrate (内置多 Key)", "https://integrate.api.nvidia.com/v1"),
    ("OpenAI 兼容网关 (DeepSeek/ yjs.im 等)", "https://api.yjs.im/v1"),
    ("自定义 OpenAI 兼容端点", ""),
]


class ProviderConfigDialog(QDialog):
    """对话式 Provider 配置弹窗。

    成功入库后发 saved(dict) 信号，dict 含：
      ok, api_url, api_key, model, guide
    main_window 接 saved → 回 agent 消息 + 同步 UI。
    """

    saved = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None,
                 orchestrator=None, config_manager=None):
        super().__init__(parent)
        self._orch = orchestrator
        self._cfg_mgr = config_manager
        self.setWindowTitle("配置 Provider（Agent 测活性 + 入库）")
        self.resize(520, 320)
        self._build_ui()

    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setSpacing(10)

        v.addWidget(QLabel("Provider 类型:"))
        self.combo_provider = QComboBox()
        for label, url in _PROVIDER_OPTIONS:
            self.combo_provider.addItem(label, url)
        self.combo_provider.currentIndexChanged.connect(self._on_provider_changed)
        v.addWidget(self.combo_provider)

        v.addWidget(QLabel("API URL:"))
        self.txt_url = QLineEdit()
        self.txt_url.setPlaceholderText("https://integrate.api.nvidia.com/v1")
        v.addWidget(self.txt_url)

        v.addWidget(QLabel("API Key（存系统密钥环，不入库）:"))
        self.txt_key = QLineEdit()
        self.txt_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_key.setPlaceholderText("nvapi-xxx 或 sk-xxx")
        v.addWidget(self.txt_key)

        v.addWidget(QLabel("模型名（可选，测活后自动填）:"))
        self.txt_model = QLineEdit()
        self.txt_model.setPlaceholderText("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
        v.addWidget(self.txt_model)

        btn_row = QHBoxLayout()
        self.btn_test = QPushButton("🔍 测活性 + 入库")
        self.btn_test.setDefault(True)
        self.btn_test.clicked.connect(self._on_test)
        btn_row.addWidget(self.btn_test)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.reject)
        btn_row.addStretch(1)
        v.addLayout(btn_row)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        v.addWidget(self.lbl_status)

        # 默认填第一个 provider 的 URL
        self._on_provider_changed(0)

    def _on_provider_changed(self, idx: int) -> None:
        url = self.combo_provider.itemData(idx) or ""
        if url:
            self.txt_url.setText(url)
        # NVIDIA Integrate 默认填首选视频模型
        if idx == 0:
            self.txt_model.setText(
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")

    def _on_test(self) -> None:
        """点「测活性 + 入库」→ 调 orchestrator.configure_provider_dialog。"""
        if self._orch is None:
            self._set_status("⚠️ Agent 未初始化", error=True)
            return
        provider_label = self.combo_provider.currentText()
        api_url = self.txt_url.text().strip()
        api_key = self.txt_key.text().strip()
        model = self.txt_model.text().strip()
        if not api_url or not api_key:
            self._set_status("⚠️ 请填 API URL 和 API Key", error=True)
            return
        self.btn_test.setEnabled(False)
        self.btn_test.setText("正在测活性…")
        self.repaint()
        try:
            result = self._orch.configure_provider_dialog(
                provider=provider_label, api_url=api_url,
                api_key=api_key, model=model)
        except Exception as e:
            result = {"ok": False, "models": [], "error": str(e),
                       "guide": f"测活性异常：{e}"}
        finally:
            self.btn_test.setEnabled(True)
            self.btn_test.setText("🔍 测活性 + 入库")

        if not result.get("ok"):
            self._set_status(
                f"❌ 连接失败：{result.get('error', '未知')}\n"
                f"{result.get('guide', '')}", error=True)
            return

        # 成功 → 入密钥环
        models = result.get("models", [])
        self._set_status(
            f"✅ 连接成功，发现 {len(models)} 个模型。\n"
            f"建议选：{models[0] if models else model}\n"
            f"正在入库（密钥环优先）…", error=False)
        # 入库：密钥环优先，失败降级 ini + 日志告警
        try:
            from src.utils.config_manager import _secure_set
            _secure_set("api_key", api_key)
            # 同步写入 LastUsed（url/model 明文，key 标记位清空）
            if self._cfg_mgr is not None:
                self._cfg_mgr.update_config("LastUsed", "api_url", api_url)
                self._cfg_mgr.update_config("LastUsed", "api_key", "")
                self._cfg_mgr.update_config(
                    "LastUsed", "model_name",
                    models[0] if models else model)
                self._cfg_mgr.update_config(
                    "LastUsed", "client_type", 4 if "NVIDIA" in provider_label else 1)
        except Exception as e:
            logger.warning(f"[provider_config] 密钥环入库失败，降级 ini: {e}")
            if self._cfg_mgr is not None:
                self._cfg_mgr.update_config("LastUsed", "api_url", api_url)
                self._cfg_mgr.update_config("LastUsed", "api_key", api_key)
                self._cfg_mgr.update_config(
                    "LastUsed", "model_name",
                    models[0] if models else model)

        # 选中首个模型回填
        chosen = models[0] if models else model
        self.txt_model.setText(chosen)
        self.saved.emit({
            "ok": True,
            "api_url": api_url,
            "api_key": api_key,
            "model": chosen,
            "provider": provider_label,
            "models": models,
            "guide": result.get("guide", ""),
        })
        self._set_status(
            f"✅ 已入库（密钥环）。模型 {chosen} 已就绪，可关闭此窗。",
            error=False)
        self.accept()

    def _set_status(self, text: str, error: bool = False) -> None:
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(
            "color: #c0392b; font-size: 11px;" if error
            else "color: #27ae60; font-size: 11px;")
