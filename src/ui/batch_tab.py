"""批量监控 UI Tab —— 把 BatchRunner + RunStore 接进主 UI。

接线两个核心模块：
  - src.core.run_store.RunStore        运行记录数据库（list/get/delete/clear）
  - src.core.batch_runner.BatchRunner  批量监控视频分析 QObject（T2 产物）

线程铁律（与 surveillance_tab.SurveillanceWorker / decision_log_panel 一致）：
  BatchRunner 在后台线程跑视频分析，emit pyqtSignal 信号，Qt 自动把
  信号投递回主线程触发槽——绝不跨线程直接改 QTableWidget / QPlainTextEdit
  （Qt 铁律，违反必崩）。主控接线时 connect(runner.XXX, tab.on_xxx) 即可。

T2 兜底：batch_runner.py 可能还没落地，本模块 try/except 兜底 import，
运行时缺失则「开始批量」「继续未完成」「取消」三按钮禁用，状态栏提示
"批量引擎未安装"。历史记录区独立可用（只依赖 RunStore，不依赖 BatchRunner）。

接线点（供 main_window 统一接入 tab bar）：
    from src.ui.batch_tab import BatchTab
    self.tab_batch = BatchTab(config_manager=self.config_manager,
                               run_store=RunStore())
    self.tabs.addTab(self.tab_batch, "🎞 批量监控")
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

# CRITICAL: torch 必须先于 PyQt6 加载（Windows DLL 顺序，见 main_window.py 注释）。
try:
    import torch  # noqa: F401  (DLL load-order fix)
except OSError:
    torch = None  # type: ignore[assignment]

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.run_store import RunStore

logger = logging.getLogger(__name__)

# 历史表列定义：时间 | 视频名 | 状态 | 分片(成功/总) | 命中数 | 耗时 | 首字均(ms) | 操作
_HISTORY_COLUMNS = ["时间", "视频名", "状态", "分片(成功/总)", "命中数", "耗时", "首字均(ms)", "操作"]

# 历史树列定义（顶级视频行）。子级分片行复用前 8 列，但语义不同：
#   顶级：时间 | 视频名 | 状态 | 分片(成功/总) | 命中数 | 耗时 | 首字均(ms) | 操作
#   子级：idx | 时间戳 | 状态 | match | confidence | attempts | first_token_ms | （空）
# 子级用空字符串占位保持列对齐，第 0 列 idx 与顶级时间列共享。
_TREE_COLUMNS = _HISTORY_COLUMNS

# 画面变化阈值选项（贴心设计：每档带具体案例说明，帮用户直观选档位）。
#   - 数据值是百分比（0-100），对应 batch_runner 帧差判定阈值
#   - 案例描述贴近真实监控场景，避免用户面对裸数字
# 顺序：从最敏感到最宽松；默认选"20% - 标准"（监控分析常见档）。
_FRAME_CHANGE_OPTIONS: list[tuple[int, str, str]] = [
    (5,  "5% - 极敏感",   "案例：光线微变 / 远处人影掠过"),
    (10, "10% - 敏感",   "案例：有人经过门廊"),
    (20, "20% - 标准",   "案例：人走近镜头"),
    (30, "30% - 宽松",   "案例：大件物品出现"),
    (50, "50% - 极宽松", "案例：大幅画面切换 / 镜头转场"),
]
_DEFAULT_FRAME_CHANGE_PCT = 20

# BatchRunner 可选 import（T2 batch_runner.py 可能未落地）
try:
    from src.core.batch_runner import BatchRunner as _BatchRunner  # type: ignore
    from src.core.batch_runner import BatchConfig as _BatchConfig  # type: ignore
    _BATCH_RUNNER_AVAILABLE = True
except ImportError:
    _BatchRunner = None  # type: ignore[assignment]
    _BatchConfig = None  # type: ignore[assignment]
    _BATCH_RUNNER_AVAILABLE = False


class BatchTab(QWidget):
    """批量监控分析 Tab：配置区 + 进度区 + 历史记录区。

    依赖：RunStore（必需，历史记录持久化）+ BatchRunner（可选，T2 未落地则禁用
    启动按钮，历史记录区仍可用）。
    """

    # 公开信号：供主控接线（保留出口，便于将来主控 emit 触发批量启动）。
    batch_requested = pyqtSignal(dict)

    def __init__(self, config_manager=None, run_store: Optional[RunStore] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config_manager = config_manager
        self._run_store = run_store or RunStore()
        self._runner = None  # type: Optional[object]
        self._setup_ui()
        self._wire_runner_availability()
        self.refresh_history()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(self._build_config_group())
        root.addWidget(self._build_progress_group())
        root.addWidget(self._build_history_group(), stretch=1)

    def _build_config_group(self) -> QGroupBox:
        box = QGroupBox("批量任务配置")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        # 视频目录
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("视频目录:"))
        self.txt_video_dir = QLineEdit("D:/监控/")
        self.btn_browse_dir = QPushButton("浏览…")
        self.btn_browse_dir.clicked.connect(self._on_browse_dir)
        row1.addWidget(self.txt_video_dir, stretch=1)
        row1.addWidget(self.btn_browse_dir)
        v.addLayout(row1)

        # 关键物品图
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("关键物品图:"))
        self.txt_key_image = QLineEdit("D:/监控/关键物品.jpg")
        self.btn_browse_img = QPushButton("浏览…")
        self.btn_browse_img.clicked.connect(self._on_browse_image)
        row2.addWidget(self.txt_key_image, stretch=1)
        row2.addWidget(self.btn_browse_img)
        v.addLayout(row2)

        # 物品描述
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("物品描述:"))
        self.txt_item_desc = QLineEdit("黑色旅行袋 白色提手 商标图案")
        row3.addWidget(self.txt_item_desc, stretch=1)
        v.addLayout(row3)

        # 参数行：分片时长 / max_tokens / reasoning_budget / 清理分片
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("分片时长(s):"))
        self.spin_segment_sec = QSpinBox()
        self.spin_segment_sec.setRange(10, 3600)
        self.spin_segment_sec.setValue(120)
        row4.addWidget(self.spin_segment_sec)
        row4.addSpacing(12)
        row4.addWidget(QLabel("max_tokens:"))
        self.spin_max_tokens = QSpinBox()
        self.spin_max_tokens.setRange(1024, 1_000_000)
        self.spin_max_tokens.setValue(65536)
        row4.addWidget(self.spin_max_tokens)
        row4.addSpacing(12)
        row4.addWidget(QLabel("reasoning_budget:"))
        self.spin_reasoning_budget = QSpinBox()
        self.spin_reasoning_budget.setRange(0, 200_000)
        self.spin_reasoning_budget.setValue(8192)
        row4.addWidget(self.spin_reasoning_budget)
        row4.addSpacing(12)
        self.chk_clean_segments = QCheckBox("分析完清理分片")
        self.chk_clean_segments.setChecked(False)
        row4.addWidget(self.chk_clean_segments)
        row4.addSpacing(12)
        # v5.7.1：帧证据留存策略（auto=只留长图删jpg省盘 / always=全留便于AI重查 / never=全删）
        row4.addWidget(QLabel("帧证据:"))
        self.combo_keep_frames = QComboBox()
        self.combo_keep_frames.addItem("智能（留长图删帧，省盘）", "auto")
        self.combo_keep_frames.addItem("全留（留长图+帧，可AI重查）", "always")
        self.combo_keep_frames.addItem("全删（最省盘，无长图）", "never")
        self.combo_keep_frames.setCurrentIndex(0)
        row4.addWidget(self.combo_keep_frames)
        row4.addStretch(1)
        v.addLayout(row4)

        # 画面变化阈值行（贴心设计：下拉每档带案例说明，用户不必猜数字含义）
        # 阈值含义：相邻抽帧像素差异超过 X% 才送 VLM 判断，避免静止画面浪费 API。
        row4b = QHBoxLayout()
        row4b.addWidget(QLabel("画面变化阈值:"))
        self.combo_frame_change = QComboBox()
        for pct, label, _case in _FRAME_CHANGE_OPTIONS:
            # 显示形如 "20% - 标准（案例：人走近镜头）"
            # 用户选档即看到案例，无需翻文档；默认 20%。
            display = f"{label}（{_case}）"
            self.combo_frame_change.addItem(display, pct)
        # 默认选 20% - 标准
        default_idx = next(
            (i for i, (p, _l, _c) in enumerate(_FRAME_CHANGE_OPTIONS)
             if p == _DEFAULT_FRAME_CHANGE_PCT),
            0,
        )
        self.combo_frame_change.setCurrentIndex(default_idx)
        row4b.addWidget(self.combo_frame_change)
        row4b.addStretch(1)
        v.addLayout(row4b)

        # 操作按钮
        row5 = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始批量")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_resume = QPushButton("⏵ 继续未完成")
        self.btn_resume.clicked.connect(self._on_resume)
        self.btn_cancel = QPushButton("■ 取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        row5.addWidget(self.btn_start)
        row5.addWidget(self.btn_resume)
        row5.addWidget(self.btn_cancel)
        row5.addStretch(1)
        v.addLayout(row5)
        return box

    def _build_progress_group(self) -> QGroupBox:
        box = QGroupBox("批量进度")
        v = QVBoxLayout(box)
        v.setSpacing(6)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        # v6.0 5.4 深色主题打磨：进度条渐变色 + 命中时金色高亮
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #444; border-radius: 4px; "
            "background: #1e1e1e; text-align: center; color: #ddd; }"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #2196F3, stop:1 #21cbcb); "
            "border-radius: 3px; }")
        v.addWidget(self.progress_bar)
        # 预计完成时间（实时刷新：基于已跑片平均耗时 × 剩余片数 / 并发数）
        # 没有任何已完成片时显示"—"，有 1+ 片完成后才给出 ETA，避免误报。
        self.lbl_eta = QLabel("预计完成时间：—（等待首批分片完成）")
        self.lbl_eta.setStyleSheet("color: #2980b9;")
        self.lbl_eta.setWordWrap(True)
        v.addWidget(self.lbl_eta)
        self.lbl_current = QLabel("就绪")
        self.lbl_current.setStyleSheet("color: gray;")
        self.lbl_current.setWordWrap(True)
        v.addWidget(self.lbl_current)
        self.log_segments = QPlainTextEdit()
        self.log_segments.setReadOnly(True)
        self.log_segments.setMaximumHeight(140)
        v.addWidget(self.log_segments)
        return box

    def _build_history_group(self) -> QGroupBox:
        box = QGroupBox("历史记录")
        v = QVBoxLayout(box)
        v.setSpacing(6)
        bar = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 刷新")
        self.btn_refresh.clicked.connect(self.refresh_history)
        self.btn_view_detail = QPushButton("📋 查看详情")
        self.btn_view_detail.clicked.connect(self._on_view_detail)
        self.btn_delete = QPushButton("🗑 删除")
        self.btn_delete.clicked.connect(self._on_delete_selected)
        self.btn_clear_all = QPushButton("🧹 一键清理全部")
        self.btn_clear_all.clicked.connect(self._on_clear_all)
        bar.addWidget(self.btn_refresh)
        bar.addWidget(self.btn_view_detail)
        bar.addWidget(self.btn_delete)
        bar.addWidget(self.btn_clear_all)
        bar.addStretch(1)
        v.addLayout(bar)

        # 历史树：顶级=视频一行，子级=分片（可展开看每片详情）。
        # 列：时间 | 视频名 | 状态 | 分片(成功/总) | 命中数 | 耗时 | 首字均(ms) | 操作
        # 子级分片行：idx | 时间戳 | 状态 | match | confidence | attempts | first_token_ms
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(_TREE_COLUMNS))
        self.tree.setHeaderLabels(_TREE_COLUMNS)
        self.tree.setSelectionBehavior(QTreeWidget.SelectionBehavior.SelectRows)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QTreeWidget.EditTrigger.NoEditTriggers)
        self.tree.setAlternatingRowColors(True)
        self.tree.setIndentation(20)
        self.tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        self.tree.itemExpanded.connect(self._on_tree_item_expanded)
        header = self.tree.header()
        assert header is not None  # Qt 保证返回表头；mypy 消除 union-attr
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # 兼容旧测试引用 self.table（统一指向 tree 的 API 表面）：
        # 现有测试通过 self.table.rowCount()/item()/selectRow() 验证渲染，故保留
        # 一个薄壳对象把 rowCount/item/selectRow/_run_id_of_row 转发到 tree。
        self.table = _TreeShim(self.tree)
        v.addWidget(self.tree, stretch=1)
        return box

    # ------------------------------------------------------------------ BatchRunner 接缝

    def _wire_runner_availability(self) -> None:
        """T2 未落地时禁用启动/取消按钮，状态栏提示。"""
        if not _BATCH_RUNNER_AVAILABLE:
            self.btn_start.setEnabled(False)
            self.btn_resume.setEnabled(False)
            self.btn_cancel.setEnabled(False)
            self.lbl_current.setText("⚠ 批量引擎未安装（src/core/batch_runner.py 缺失）")
            self.lbl_current.setStyleSheet("color: #c0392b;")
            return
        self.btn_start.setEnabled(True)
        self.btn_resume.setEnabled(True)

    def _build_runner(self):
        """构造 BatchRunner 实例并连接信号。

        v5.8 断点 B7 修复（P0 生产阻断）：_collect_config 返回 dict，
        但 BatchRunner.__init__(config: BatchConfig) 用属性访问——之前直接传
        dict 立即抛 AttributeError: 'dict' object has no attribute 'key_item_image'，
        被 _on_start 的 try/except 吞成"错误：dict object..."状态文案，**生产环境
        点"开始批量"必崩**，37 个单测只断言 dict 字段没断言能构造 BatchRunner
        （测试绿但功能坏）。现做 BatchConfig(**dict) 转换。
        v5.8 断点 B2：传 on_segment_judged 回调给 BatchRunner（agent 每轮介入）。
        """
        if not _BATCH_RUNNER_AVAILABLE:
            return None
        cfg_dict = self._collect_config()
        try:
            cfg = _BatchConfig(**cfg_dict)
        except TypeError as e:
            # dict 含 BatchConfig 不识别的字段（如未来扩展）→ pop 掉未知字段兜底
            from dataclasses import fields as _dc_fields
            valid = {f.name for f in _dc_fields(_BatchConfig)}
            cfg_dict_clean = {k: v for k, v in cfg_dict.items() if k in valid}
            logger.warning(
                f"[batch] _collect_config 含未知字段被过滤: {e}；"
                f"cleaned={set(cfg_dict) - valid}")
            cfg = _BatchConfig(**cfg_dict_clean)
        router = self._build_router()
        # v5.8 断点 B2：agent 每轮介入回调（stop/deep_dive/continue）
        on_seg = getattr(self, '_agent_decide_segment', None)
        runner = _BatchRunner(config=cfg, run_store=self._run_store,
                              router=router, on_segment_judged=on_seg)
        # 接缝：BatchRunner 信号 → 本 Tab 槽（Qt 自动跨线程回主线程）
        # 用 *args 槽兼容 T2 多种签名，详见各槽注释。
        runner.run_started.connect(self._on_run_started)
        runner.video_started.connect(self._on_video_started)
        runner.segment_done.connect(self._on_segment_done)
        # v5.9 I5.9-ui-2：分片进度同步投到 agent 对话框
        runner.segment_done.connect(self._on_segment_done_to_agent)
        runner.video_done.connect(self._on_video_done)
        runner.batch_progress.connect(self._on_batch_progress)
        runner.batch_finished.connect(self._on_batch_finished)
        runner.error.connect(self._on_runner_error)
        return runner

    def _build_router(self):
        """v5.8：从 .env 加载 NVIDIA 11 key 构造 ProviderRouter。

        复用 provider_router.load_from_env（读 VAP_NV_API_KEYS 逗号分隔）+
        load_router_config_from_env（读退避/重试/并发参数，断点 router-1/2）。
        失败则返回 None（batch_runner 会兜底提示无 nvidia key）。
        """
        try:
            from src.core.provider_router import (
                load_from_env, load_router_config_from_env, ProviderRouter)
            from pathlib import Path
            env_path = Path(__file__).resolve().parents[2] / ".env"
            env_str = str(env_path) if env_path.exists() else None
            keys = load_from_env(env_str)
            if not keys:
                logger.warning("[batch] .env 未配 VAP_NV_API_KEYS，"
                               "批量监控无 NVIDIA key 可用")
                return None
            cfg = load_router_config_from_env(env_str)
            return ProviderRouter(
                keys,
                rate_limit_per_min=40,
                backoff_sec=cfg.get("backoff_sec", 1.5),
                same_key_retries=cfg.get("same_key_retries", 2),
                max_concurrent_per_key=cfg.get("max_concurrent_per_key", 2),
            )
        except Exception as e:
            logger.warning(f"[batch] router 构造失败: {e}")
            return None

    def _collect_config(self) -> dict:
        """从 UI 收集批量配置（传给 BatchRunner）。"""
        return {
            "video_dir": self.txt_video_dir.text().strip(),
            "key_item_image": self.txt_key_image.text().strip(),
            "item_description": self.txt_item_desc.text().strip(),
            "segment_sec": self.spin_segment_sec.value(),
            "max_tokens": self.spin_max_tokens.value(),
            "reasoning_budget": self.spin_reasoning_budget.value(),
            "clean_segments": self.chk_clean_segments.isChecked(),
            "keep_frames": self.combo_keep_frames.currentData() or "auto",
            # 画面变化阈值（百分比）。BatchRunner 用它做帧差过滤，避免静止画面送 VLM。
            "frame_change_pct": self.combo_frame_change.currentData() or _DEFAULT_FRAME_CHANGE_PCT,
        }

    # ------------------------------------------------------------------ 槽：UI 浏览

    def _on_browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择视频目录", "D:/")
        if path:
            self.txt_video_dir.setText(path)

    def _on_browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择关键物品参考图", "", "图片 (*.jpg *.jpeg *.png *.bmp)"
        )
        if path:
            self.txt_key_image.setText(path)

    # ------------------------------------------------------------------ 槽：批量控制

    def _on_start(self) -> None:
        if not _BATCH_RUNNER_AVAILABLE:
            QMessageBox.warning(
                self, "批量引擎未安装",
                "src/core/batch_runner.py 缺失或 import 失败。\n"
                "请检查依赖是否完整安装（pip install -r requirements.txt）。")
            return
        # v6.0 5.3 小白易用弹窗：无前置条件时弹窗 + 明确下一步（不只状态栏文案）
        video_dir = self.txt_video_dir.text().strip()
        if not video_dir:
            QMessageBox.information(
                self, "请先填视频目录",
                "请在「视频目录」栏填写监控视频所在路径（如 D:/监控/），\n"
                "或点「浏览」选择目录。")
            return
        if not Path(video_dir).is_dir():
            QMessageBox.warning(
                self, "视频目录无效",
                f"目录不存在或非目录：{video_dir}\n"
                f"请检查路径拼写或点「浏览」重新选择。")
            return
        key_img = self.txt_key_image.text().strip()
        if not key_img:
            QMessageBox.information(
                self, "请先填关键物品图",
                "请在「关键物品参考图」栏填写要查找的物品图片路径（如 D:/监控/关键物品.jpg），\n"
                "或点「浏览」选择图片。")
            return
        if not Path(key_img).exists():
            QMessageBox.warning(
                self, "关键物品图不存在",
                f"图片文件不存在：{key_img}\n请检查路径或重新选择。")
            return
        videos = self._scan_videos(video_dir)
        if not videos:
            QMessageBox.information(
                self, "目录下无视频",
                f"目录 {video_dir} 下无支持的视频文件（.mp4/.avi/.mov/.mkv）。\n"
                f"请确认视频放在该目录或换一个目录。")
            return
        self.log_segments.clear()
        self.progress_bar.setValue(0)
        self._runner = self._build_runner()
        if self._runner is None:
            QMessageBox.critical(
                self, "批量引擎初始化失败",
                "BatchRunner 构造失败，可能是配置字段不匹配或 router 不可用。\n"
                "请检查 .env 的 VAP_NV_API_KEYS 是否配置，或查看日志。")
            return
        try:
            self._runner.run_batch(videos)
        except Exception as e:
            logger.exception("[batch_tab] run_batch 异常")
            QMessageBox.critical(
                self, "批量启动失败",
                f"启动批量分析时出错：{e}\n请查看日志排查。")
            return
        self.btn_start.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._set_status(f"批量启动：{len(videos)} 个视频")

    def _on_resume(self) -> None:
        if not _BATCH_RUNNER_AVAILABLE:
            self._set_status("批量引擎未安装", error=True)
            return
        self._runner = self._build_runner()
        if self._runner is None:
            self._set_status("错误：批量引擎初始化失败", error=True)
            return
        try:
            self._runner.resume_batch()
        except Exception as e:
            logger.exception("[batch_tab] resume_batch 异常")
            self._set_status(f"错误：{e}", error=True)
            return
        self.btn_start.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._set_status("继续未完成任务…")

    def _on_cancel(self) -> None:
        if self._runner is not None:
            try:
                self._runner.cancel()
            except Exception as e:
                logger.warning(f"[batch_tab] cancel 异常: {e}")
        self._runner = None
        self.btn_start.setEnabled(_BATCH_RUNNER_AVAILABLE)
        self.btn_resume.setEnabled(_BATCH_RUNNER_AVAILABLE)
        self.btn_cancel.setEnabled(False)
        self._set_status("已取消")

    def _scan_videos(self, video_dir: str) -> list:
        """扫描目录下的视频文件（mp4/avi/mov/mkv）。"""
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        try:
            return [str(p) for p in sorted(Path(video_dir).iterdir())
                    if p.suffix.lower() in exts and p.is_file()]
        except Exception as e:
            logger.warning(f"[batch_tab] 扫描视频目录失败: {e}")
            return []

    # ------------------------------------------------------------------ 槽：BatchRunner 信号
    # 这些槽在主线程执行（Qt 跨线程信号自动队列投递回接收端所在线程）。
    # 用 *args 兼容 T2 的多种信号签名（详见接缝说明）。

    def _on_run_started(self, *args) -> None:
        """run_started 信号槽。兼容 (int total) 或 () 或 (dict,)。"""
        total = 0
        if args:
            if isinstance(args[0], dict):
                total = int(args[0].get("total", 0))
            else:
                total = int(args[0])
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(0)
        self._set_status(f"批量开始，共 {total} 个视频")

    def _on_video_started(self, *args) -> None:
        """video_started 信号槽。兼容 (name, idx, total) 或 (dict,)。"""
        if not args:
            return
        if isinstance(args[0], dict):
            d = args[0]
            vname = str(d.get("video_name", ""))
            idx = d.get("idx", 0)
            total = d.get("total", 0)
        else:
            vname = str(args[0])
            idx = args[1] if len(args) > 1 else 0
            total = args[2] if len(args) > 2 else 0
        self.lbl_current.setText(f"正在分析 {vname}（{idx}/{total}）")
        self.lbl_current.setStyleSheet("color: #2ecc71;")

    def _on_segment_done(self, *args) -> None:
        """segment_done 信号槽。兼容 (idx, total, hits) / (idx, total, hits, status) / (dict,)。"""
        if not args:
            return
        if isinstance(args[0], dict):
            s = args[0]
            seg_idx = s.get("seg_idx", "?")
            seg_total = s.get("seg_total", s.get("segments_total", "?"))
            hits = s.get("hits", s.get("hits_count", 0))
            status = s.get("status", "")
        else:
            seg_idx = args[0] if len(args) > 0 else "?"
            seg_total = args[1] if len(args) > 1 else "?"
            hits = args[2] if len(args) > 2 else 0
            status = args[3] if len(args) > 3 else ""
        self.lbl_current.setText(
            f"  第 {seg_idx}/{seg_total} 片完成，命中 {hits}（{status}）"
        )
        self.log_segments.appendPlainText(
            f"[分片 {seg_idx}/{seg_total}] status={status} hits={hits}"
        )

    def _on_video_done(self, *args) -> None:
        """video_done 信号槽。兼容 (name, hits) 或 (dict,)。"""
        if not args:
            self.refresh_history()
            return
        if isinstance(args[0], dict):
            d = args[0]
            vname = str(d.get("video_name", ""))
            hits = d.get("hits_count", 0)
        else:
            vname = str(args[0])
            hits = args[1] if len(args) > 1 else 0
        self.log_segments.appendPlainText(f"[视频完成] {vname} 命中 {hits}")
        self.refresh_history()

    def _on_batch_progress(self, *args) -> None:
        """batch_progress 信号槽。兼容 (done, total) 或 (dict,)。"""
        if not args:
            return
        if isinstance(args[0], dict):
            d = args[0]
            done = int(d.get("done", 0))
            total = int(d.get("total", 0))
        else:
            done = int(args[0]) if len(args) > 0 else 0
            total = int(args[1]) if len(args) > 1 else 0
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)
        # 每次进度刷新同步重算 ETA（基于已完成分片平均耗时 × 剩余分片）
        self._refresh_eta(done, total)

    # ------------------------------------------------------------------ ETA

    def _refresh_eta(self, done: int = 0, total: int = 0) -> None:
        """实时刷新预计完成时间（基于已跑片平均耗时 × 剩余 / 并发）。

        公式：平均单分片耗时 = 已完成分片总耗时 / 已完成数
              剩余分片 = max(total - done, 0)
              剩余秒 = 剩余分片 × 平均单分片耗时
        并发数：BatchRunner 当前是单视频串行处理（一个线程），并发=1。
                 若未来并行多视频，从 _runner.concurrency 读取（M3 暂 1）。
        无已完成分片时显示"—"提示，避免误报。

        数据源：从 RunStore 拉 list_runs，汇总已完成分片的 elapsed_sec 求平均。
        """
        try:
            runs = self._run_store.list_runs(limit=500)
        except Exception as e:
            logger.warning(f"[batch_tab] ETA 查询失败: {e}")
            return
        total_seg_done = 0
        total_seg_sec = 0.0
        for r in runs:
            # segments_ok + segments_failed 是"已完成"分片（含失败，失败也耗了 API）
            ok = int(r.get("segments_ok") or 0)
            failed = int(r.get("segments_failed") or 0)
            seg_done = ok + failed
            elapsed = r.get("vlm_elapsed_sec")
            if seg_done > 0 and elapsed:
                total_seg_done += seg_done
                total_seg_sec += float(elapsed)
        if total_seg_done == 0 or total <= 0:
            self.lbl_eta.setText("预计完成时间：—（等待首批分片完成）")
            return
        avg_per_seg = total_seg_sec / max(total_seg_done, 1)
        remaining = max(total - max(done, 0), 0)
        concurrency = 1  # M3 串行处理；未来从 _runner 读取
        eta_sec = (remaining * avg_per_seg) / max(concurrency, 1)
        # 转人读：小时/分钟/秒
        if eta_sec >= 3600:
            h = int(eta_sec // 3600)
            m = int((eta_sec % 3600) // 60)
            eta_txt = f"约 {h}h{m}m"
        elif eta_sec >= 60:
            m = int(eta_sec // 60)
            s = int(eta_sec % 60)
            eta_txt = f"约 {m}m{s}s"
        else:
            eta_txt = f"约 {int(eta_sec)}s"
        from datetime import datetime as _dt
        finish_at = _dt.fromtimestamp(_dt.now().timestamp() + eta_sec).strftime("%H:%M")
        self.lbl_eta.setText(
            f"预计完成时间：{finish_at}（{eta_txt}，平均 {avg_per_seg:.1f}s/片，剩余 {remaining} 片）"
        )

    def _on_batch_finished(self, *args) -> None:
        """batch_finished 信号槽。兼容 (total, success, failed) 或 (dict,)。"""
        if not args:
            self._set_status("批量完成")
        elif isinstance(args[0], dict):
            d = args[0]
            self._set_status(
                f"批量完成：共 {d.get('total', 0)}，"
                f"成功 {d.get('success', 0)}，失败 {d.get('failed', 0)}"
            )
        else:
            total = args[0] if len(args) > 0 else 0
            success = args[1] if len(args) > 1 else 0
            failed = args[2] if len(args) > 2 else 0
            self._set_status(f"批量完成：共 {total}，成功 {success}，失败 {failed}")

    # v5.7：帧长图查看器 → main_window 跳转视频时间点（解决伪证据）
    strip_seek_requested = pyqtSignal(str, float)  # video_path, timestamp_sec

    # v5.9 I5.9-ui-2：批量进度投递到 agent 对话框（agent 触发后能看到进度）
    batch_progress_to_agent = pyqtSignal(str, int, int, bool, float)
    # (video_name, seg_idx, hits_so_far, match, confidence)

    def _on_strip_seek(self, video_path: str, ts: float) -> None:
        """转发长图查看器的跳转请求给 main_window（打开播放器定位到 ts）。"""
        self.strip_seek_requested.emit(video_path, ts)

    def _on_segment_done_to_agent(self, run_id: str, seg_idx: int,
                                   match: bool, conf: float) -> None:
        """v5.9 I5.9-ui-2：分片判断完 → 投进度到 agent 对话框。

        batch_runner.segment_done 信号接这里，每片结果转成文本投到
        batch_progress_to_agent 信号，main_window 接后 append_tool_call 到
        agent_dialog（用户在对话框看进度，不必切到批量 tab）。
        节流：每 5 个分片投一次 + 命中时立即投（避免高频刷爆 UI）。
        """
        # 节流：非命中片每 5 片投一次，命中片立即投
        self._seg_since_last_report = getattr(
            self, "_seg_since_last_report", 0) + 1
        if not match and self._seg_since_last_report < 5:
            return
        self._seg_since_last_report = 0
        try:
            run = self._run_store.get_run(run_id) if run_id else None
            video_name = (run.get("video_name", "?") if run else "?")
            hits = int(run.get("hits_count", 0)) if run else 0
            total = int(run.get("segments_total", 0)) if run else 0
            ok = int(run.get("segments_ok", 0)) if run else 0
        except Exception:
            video_name, hits, total, ok = "?", 0, 0, 0
        match_txt = "✅ 命中" if match else "—"
        # 进度文本（供日志/调试；实际投递只发结构化信号给 main_window 拼）
        _progress = (
            f"📊 {video_name} | 分片 {seg_idx+1}/{total}（已跑 {ok}）"
            f" | {match_txt} conf={conf:.2f} | 累计命中 {hits}"
        )
        logger.debug(f"[batch] agent 进度: {_progress}")
        self.batch_progress_to_agent.emit(video_name, seg_idx, hits, match, conf)

    def _agent_decide_segment(self, payload: dict) -> str:
        """v5.8 断点 B2：batch_runner 每轮介入回调 → agent 决策。

        规则版（无 LLM 也跑）：连续命中数≥2 且最近一片 confidence>0.8 → stop
        （已找到目标，停后续省调用）；某片 confidence 0.6-0.8 灰色地带 →
        deep_dive（二次验证深挖）；其余 continue。可通过 strip_seek_requested
        同款机制把决策投到 agent_dialog（此处先简化只返回决策字符串）。
        """
        try:
            hits_so_far = int(payload.get("hits_so_far", 0))
            conf = float(payload.get("confidence", 0.0))
            match = bool(payload.get("match"))
            seg_idx = payload.get("seg_idx", -1)
            if match and hits_so_far >= 2 and conf > 0.8:
                logger.info(
                    f"[batch] agent 决策 stop（命中 {hits_so_far} 片，"
                    f"seg {seg_idx} conf={conf}）")
                return "stop"
            if not match and 0.6 <= conf < 0.8:
                logger.info(
                    f"[batch] agent 决策 deep_dive（seg {seg_idx} "
                    f"灰色地带 conf={conf}）")
                return "deep_dive"
            return "continue"
        except Exception:
            return "continue"
        self.btn_start.setEnabled(_BATCH_RUNNER_AVAILABLE)
        self.btn_resume.setEnabled(_BATCH_RUNNER_AVAILABLE)
        self.btn_cancel.setEnabled(False)
        self.refresh_history()

    def _on_runner_error(self, *args) -> None:
        """error 信号槽。兼容 (str msg) 或 (dict,)。"""
        if not args:
            return
        if isinstance(args[0], dict):
            msg = str(args[0].get("message", args[0].get("error", "")))
        else:
            msg = str(args[0])
        self._set_status(f"错误：{msg}", error=True)
        self.log_segments.appendPlainText(f"[ERROR] {msg}")

    # ------------------------------------------------------------------ 槽：历史记录

    def refresh_history(self) -> None:
        """从 RunStore 拉取 runs 列表渲染到树（顶级=视频，子级=分片可展开）。

        run_id 存在顶级行第 0 列 UserRole（_run_id_of_tree_item 反查用）。
        子级分片行展开后显示：idx/时间戳/状态/match/confidence/attempts/first_token_ms。
        """
        self.tree.clear()
        try:
            runs = self._run_store.list_runs(limit=200)
        except Exception as e:
            logger.warning(f"[batch_tab] 加载历史失败: {e}")
            return
        for r in runs:
            elapsed = r.get("total_elapsed_sec")
            elapsed_txt = f"{float(elapsed):.1f}s" if elapsed else "—"
            # 顶级行 8 列：时间|视频名|状态|分片(成功/总)|命中|耗时|首字均|操作
            # 子级分片行：复用 8 列：idx|时间戳|状态|match|conf|attempts|first_token_ms|（空）
            top = QTreeWidgetItem([
                str(r.get("started_at") or ""),
                str(r.get("video_name") or ""),
                str(r.get("status") or ""),
                f"{int(r.get('segments_ok') or 0)}/{int(r.get('segments_total') or 0)}",
                str(int(r.get("hits_count") or 0)),
                elapsed_txt,
                self._fmt_avg_first_token(r),
                "双击查看",
            ])
            # 居中列（状态/分片/命中/耗时/首字均）
            for col in (2, 3, 4, 5, 6):
                top.setTextAlignment(col, Qt.AlignmentFlag.AlignCenter)
            # run_id 存第 0 列 UserRole（_run_id_of_tree_item 反查用）
            top.setData(0, Qt.ItemDataRole.UserRole, str(r.get("run_id") or ""))
            self._fill_segment_children(top, r)
            self.tree.addTopLevelItem(top)

    @staticmethod
    def _fmt_avg_first_token(r: dict) -> str:
        """格式化首字耗时均值（ms）。segments 里 first_token_ms 取均值。

        无 first_token_ms 字段或无分片时返回 "—"（老库兼容）。
        """
        segs = r.get("segments") if isinstance(r.get("segments"), list) else None
        if not segs:
            # list_runs 不带 segments，只返回顶层字段 → 这里返回占位
            return "—"
        vals = [int(s.get("first_token_ms") or 0) for s in segs
                if s.get("first_token_ms") is not None]
        if not vals:
            return "—"
        return f"{sum(vals) / len(vals):.0f}"

    def _fill_segment_children(self, top: QTreeWidgetItem, r: dict) -> None:
        """为顶级视频行挂分片子节点（可展开看每片详情）。

        分片子行 8 列对齐：
          idx | 时间戳 | 状态 | match | confidence | attempts | first_token_ms | （空）
        RunStore.list_runs 不返回 segments（性能），所以子节点懒加载：
        只有当用户展开时才 get_run 拉详情。这里先加一个占位子节点 "（展开加载…）"。
        """
        # 懒加载：list_runs 不带 segments，先插一个占位子项；展开时替换为真实分片。
        placeholder = QTreeWidgetItem(["（展开加载分片…）", "", "", "", "", "", "", ""])
        placeholder.setData(0, Qt.ItemDataRole.UserRole, "__placeholder__")
        top.addChild(placeholder)

    def _load_segment_children(self, top: QTreeWidgetItem, run_id: str) -> None:
        """展开顶级行时懒加载分片子节点（从 RunStore.get_run 拉 segments）。

        替换占位子项为真实分片行。若 get_run 返回 None（run 已被删），清空子节点。
        """
        # 移除占位子项
        for i in range(top.childCount()):
            child = top.child(0)
            if child is not None and child.data(0, Qt.ItemDataRole.UserRole) == "__placeholder__":
                top.removeChild(child)
            else:
                break
        run = self._run_store.get_run(run_id)
        if run is None:
            return
        segs = run.get("segments") or []
        if not segs:
            empty = QTreeWidgetItem(["（无分片记录）", "", "", "", "", "", "", ""])
            top.addChild(empty)
            return
        for s in segs:
            ftms = s.get("first_token_ms")
            ftms_txt = f"{int(ftms)}ms" if ftms is not None else "—"
            child = QTreeWidgetItem([
                f"#{s.get('seg_idx', '')}",
                str(s.get("abs_timestamp") or f"{float(s.get('start_sec') or 0):.0f}s"),
                str(s.get("status") or ""),
                "命中" if s.get("match") else "—",
                f"{float(s.get('confidence') or 0):.2f}" if s.get("confidence") is not None else "—",
                str(s.get("attempts") or 0),
                ftms_txt,
                "",
            ])
            for col in (2, 3, 4, 5, 6):
                child.setTextAlignment(col, Qt.AlignmentFlag.AlignCenter)
            top.addChild(child)

    def _on_tree_item_expanded(self, item: QTreeWidgetItem) -> None:
        """顶级行展开时懒加载分片子节点。"""
        run_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(run_id, str) or not run_id:
            return
        # 仅当还有占位子项时才加载（避免重复加载）
        if item.childCount() == 0:
            return
        first = item.child(0)
        if (first is not None
                and first.data(0, Qt.ItemDataRole.UserRole) == "__placeholder__"):
            self._load_segment_children(item, run_id)

    def _on_tree_item_double_clicked(self, item: QTreeWidgetItem, *_args) -> None:
        """双击顶级行 → 弹详情；双击子行 → 无操作（详情在弹窗内看）。"""
        run_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(run_id, str) or not run_id or run_id == "__placeholder__":
            return
        # 子行（非顶级）run_id 为 None/空，跳过
        if self.tree.indexOfTopLevelItem(item) < 0:
            return
        self._show_detail_for_run_id(run_id)

    # 兼容旧测试：旧 self.table.* 接口转 发到 tree。
    def _run_id_of_row(self, row: int) -> Optional[str]:
        """从表格行的 UserRole 取 run_id（refresh_history 时写入）。"""
        top = self.tree.topLevelItem(row)
        if top is None:
            return None
        v = top.data(0, Qt.ItemDataRole.UserRole)
        return str(v) if v else None

    def _on_view_detail(self) -> None:
        row = self.tree.currentIndex().row()
        if row < 0:
            QMessageBox.information(self, "提示", "请先选中一行")
            return
        run_id = self._run_id_of_row(row)
        if not run_id:
            QMessageBox.information(self, "提示", "请先选中一行")
            return
        self._show_detail_for_run_id(run_id)

    def _on_double_click_row(self, *_args) -> None:
        row = self.tree.currentIndex().row()
        if row < 0:
            return
        run_id = self._run_id_of_row(row)
        if not run_id:
            return
        self._show_detail_for_run_id(run_id)

    def _show_detail_for_run_id(self, run_id: str) -> None:
        run = self._run_store.get_run(run_id)
        if run is None:
            QMessageBox.warning(self, "提示", "运行记录已删除")
            return
        dlg = _RunDetailDialog(run, self)
        dlg.exec()

    def _on_delete_selected(self) -> None:
        row = self.tree.currentIndex().row()
        if row < 0:
            QMessageBox.information(self, "提示", "请先选中一行")
            return
        run_id = self._run_id_of_row(row)
        if not run_id:
            return
        top = self.tree.topLevelItem(row)
        vname = top.text(1) if top is not None else ""
        if QMessageBox.question(
            self, "确认", f"删除运行记录 {vname}？\n（磁盘 clip 文件保留）"
        ) != QMessageBox.StandardButton.Yes:
            return
        self._run_store.delete_run(run_id, purge_files=False)
        self.refresh_history()
        self._set_status(f"已删除 {vname}")

    def _on_clear_all(self) -> None:
        if QMessageBox.question(
            self, "确认", "清空全部运行记录？\n（同时删除磁盘 clip 文件）"
        ) != QMessageBox.StandardButton.Yes:
            return
        self._run_store.clear_all(purge_files=True)
        self.refresh_history()
        self._set_status("已清空全部运行记录")

    # ------------------------------------------------------------------ 工具

    def _set_status(self, msg: str, error: bool = False) -> None:
        self.lbl_current.setText(msg)
        self.lbl_current.setStyleSheet("color: #c0392b;" if error else "color: gray;")

    def closeEvent(self, event):
        """与 main_window.closeEvent 同模式：取消运行中的批量任务。"""
        self._on_cancel()
        super().closeEvent(event)


class _TreeShim:
    """薄壳：把旧 QTableWidget 风格 API 转发到 QTreeWidget。

    保留它是因为 M3 之前的历史表是 QTableWidget，测试用 self.table.rowCount() /
    item() / selectRow() 验证渲染。M3 改 QTreeWidget 后这些方法语义变化，
    shim 让旧测试无改动通过，同时不引入额外耦合。

    行号语义：顶级视频行的索引（0..N-1），与 QTreeWidget.topLevelItem 对齐。
    item(row, col) 返回的 _ShimItem 持有对 QTreeWidgetItem 的引用 + 列号，
    支持 .text() / .setData() / .data() / .textAlignment()（仅子集，够测试用）。
    """

    def __init__(self, tree: QTreeWidget):
        self._tree = tree

    def rowCount(self) -> int:
        return self._tree.topLevelItemCount()

    def setRowCount(self, n: int) -> None:
        # QTreeWidget 无 setRowCount；clear + 留空等价于"置 0 行"
        if n == 0:
            self._tree.clear()

    def insertRow(self, row: int) -> None:
        # 测试不直接调（refresh_history 内部走 tree.addTopLevelItem）；
        # 留空实现避免老调用路径抛 AttributeError。
        return None

    def selectRow(self, row: int) -> None:
        item = self._tree.topLevelItem(row)
        if item is not None:
            self._tree.setCurrentItem(item)

    def currentRow(self) -> int:
        cur = self._tree.currentItem()
        if cur is None:
            return -1
        return self._tree.indexOfTopLevelItem(cur)

    def item(self, row: int, col: int):
        top = self._tree.topLevelItem(row)
        if top is None:
            return None
        return _ShimItem(top, col)


class _ShimItem:
    """适配器：让旧测试用 QTableWidgetItem 风格 API 访问 QTreeWidgetItem 单元格。"""

    def __init__(self, tree_item: QTreeWidgetItem, col: int):
        self._it = tree_item
        self._col = col

    def text(self) -> str:
        return self._it.text(self._col)

    def setData(self, role, value) -> None:
        # 旧测试往第 0 列写 run_id（UserRole）；转发到 tree item
        self._it.setData(self._col, role, value)

    def data(self, role):
        return self._it.data(self._col, role)

    def setTextAlignment(self, alignment) -> None:
        self._it.setTextAlignment(self._col, alignment)


class _RunDetailDialog(QDialog):
    """运行详情对话框：segments + clips 路径。

    clips 路径双击打开所在文件夹（Windows 用 os.startfile，Linux 用 xdg-open）。
    """

    def __init__(self, run: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._run = run
        self.setWindowTitle(f"运行详情 - {run.get('video_name', '')}")
        self.resize(640, 480)
        self._build_ui()

    def _build_strip_bar(self) -> QHBoxLayout:
        """v5.7：帧长图证据操作行。

        无变化视频本来 0 命中零证据，长图让用户能核对算法是否漏判；
        有变化视频也能核对 AI 判断真伪。strip_path 写在 runs 表（v5.7 新列）。
        """
        bar = QHBoxLayout()
        bar.setSpacing(6)
        strip_path = self._run.get("strip_path") or ""
        if strip_path and Path(strip_path).exists():
            lbl = QLabel("🖼 帧长图证据已生成")
            lbl.setStyleSheet("color: #27ae60; font-weight: bold;")
            bar.addWidget(lbl)
            btn = QPushButton("🖼 查看帧长图（可缩放）")
            btn.setStyleSheet(
                "background: #2196F3; color: white; font-weight: bold; "
                "padding: 4px 12px;"
            )
            btn.clicked.connect(self._on_view_strip)
            bar.addWidget(btn)
        else:
            bar.addWidget(QLabel("🖼 帧长图证据：未生成（此 run 早于 v5.7 或生成失败）"))
        bar.addStretch(1)
        return bar

    def _on_view_strip(self) -> None:
        """打开帧长图查看器。"""
        strip_path = self._run.get("strip_path") or ""
        if not strip_path or not Path(strip_path).exists():
            QMessageBox.information(self, "提示", "长图文件不存在")
            return
        try:
            from src.ui.frame_strip_dialog import FrameStripDialog
            video_path = self._run.get("video_path") or ""
            frame_dir = str(Path(strip_path).parent)
            dlg = FrameStripDialog(
                strip_path=strip_path,
                video_path=video_path,
                frame_dir=frame_dir,
                parent=self,
            )
            # 接线：长图查看器的 seek 信号 → 由 BatchTab 转发给 main_window
            # （BatchTab 有 _on_strip_seek 槽转发给上层）
            top = self.parent()
            while top is not None and not isinstance(top, BatchTab):
                top = top.parent()
            if isinstance(top, BatchTab):
                dlg.seek_video_requested.connect(top._on_strip_seek)
            dlg.exec()
        except Exception as e:
            logger.warning(f"[batch_tab] 打开长图失败: {e}")
            QMessageBox.warning(self, "打开失败", f"长图查看器错误: {e}")

    def _build_summary_box(self) -> QGroupBox:
        """单视频汇总区：总耗时 / API 调用次数 / 平均首字耗时 / 命中数 / 命中率 / 覆盖率。

        v5.9 I5.9-ui-1：补齐 6 指标卡片（原 5 个 + 命中率）。
        - 命中率 = hits_count / segments_total × 100%（命中分片占总分片比例）
        - 覆盖率 = segments_ok / segments_total × 100%（已完成分片覆盖比例）
        - API 调用次数 = segments.attempts 之和（每片重试也算一次调用）
        - 平均首字耗时 = segments.first_token_ms 的均值（无则 "—"）
        """
        box = QGroupBox("单视频汇总")
        h = QHBoxLayout(box)
        h.setSpacing(12)
        segs = self._run.get("segments") or []
        total_elapsed = self._run.get("total_elapsed_sec")
        total_elapsed_txt = (
            f"{float(total_elapsed):.1f}s" if total_elapsed is not None else "—"
        )
        # API 调用次数 = sum(attempts)，无 attempts 字段则用 len(segs) 兜底
        api_calls = sum(int(s.get("attempts") or 0) for s in segs) or len(segs)
        # 平均首字耗时
        ftms_vals = [int(s.get("first_token_ms") or 0) for s in segs
                     if s.get("first_token_ms") is not None]
        avg_ft = f"{sum(ftms_vals) / len(ftms_vals):.0f}ms" if ftms_vals else "—"
        hits = int(self._run.get("hits_count") or 0)
        seg_total = int(self._run.get("segments_total") or 0) or len(segs)
        seg_ok = int(self._run.get("segments_ok") or 0) or len(segs)
        hit_rate = (hits / seg_total * 100) if seg_total > 0 else 0.0
        coverage = (seg_ok / seg_total * 100) if seg_total > 0 else 0.0
        for label, val in (
            ("总耗时", total_elapsed_txt),
            ("API 调用次数", str(api_calls)),
            ("平均首字耗时", avg_ft),
            ("命中数", str(hits)),
            ("命中率", f"{hit_rate:.1f}%"),
            ("覆盖率", f"{coverage:.1f}%"),
        ):
            col = QVBoxLayout()
            col.addWidget(QLabel(label))
            lbl_val = QLabel(val)
            # 命中率>0 用金色高亮（命中视频视觉强调）
            if label == "命中率" and hit_rate > 0:
                lbl_val.setStyleSheet("font-weight: bold; color: #f39c12;")
            else:
                lbl_val.setStyleSheet("font-weight: bold; color: #2c3e50;")
            col.addWidget(lbl_val)
            h.addLayout(col)
        h.addStretch(1)
        return box

    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setSpacing(6)

        # 顶部汇总区（单视频汇总：耗时 / API 调用次数 / 平均首字耗时 / 命中数 / 覆盖率）
        # 覆盖率 = 命中数 / 分片总数 × 100%（命中分片占总分片比例）
        summary = self._build_summary_box()
        v.addWidget(summary)

        # v5.7：帧长图证据按钮（无变化视频零证据的核心修复）
        strip_bar = self._build_strip_bar()
        v.addLayout(strip_bar)

        # 顶部元信息
        meta = QHBoxLayout()
        meta.addWidget(QLabel(f"状态: {self._run.get('status', '')}"))
        meta.addWidget(QLabel(f"命中: {self._run.get('hits_count', 0)}"))
        meta.addWidget(QLabel(
            f"分片: {self._run.get('segments_ok', 0)}/{self._run.get('segments_total', 0)}"
        ))
        v.addLayout(meta)

        # 分片表
        v.addWidget(QLabel("分片详情:"))
        segs = self._run.get("segments") or []
        # 分片表加 first_token_ms / attempts / confidence / 耗时 列（M3 增强）
        seg_table = QTableWidget(0, 8)
        seg_table.setHorizontalHeaderLabels(
            ["idx", "start(s)", "状态", "match", "confidence",
             "attempts", "first_token_ms", "耗时(s)"]
        )
        seg_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for s in segs:
            row = seg_table.rowCount()
            seg_table.insertRow(row)
            ftms = s.get("first_token_ms")
            elapsed = s.get("elapsed_sec")
            cells = [
                QTableWidgetItem(str(s.get("seg_idx", ""))),
                QTableWidgetItem(f"{float(s.get('start_sec') or 0):.1f}"),
                QTableWidgetItem(str(s.get("status", ""))),
                QTableWidgetItem("是" if s.get("match") else "否"),
                QTableWidgetItem(
                    f"{float(s.get('confidence')):.2f}"
                    if s.get("confidence") is not None else "—"),
                QTableWidgetItem(str(s.get("attempts") or 0)),
                QTableWidgetItem(f"{int(ftms)}" if ftms is not None else "—"),
                QTableWidgetItem(
                    f"{float(elapsed):.2f}" if elapsed is not None else "—"),
            ]
            for col, it in enumerate(cells):
                if col in (2, 3, 4, 5, 6, 7):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                seg_table.setItem(row, col, it)
        seg_header = seg_table.horizontalHeader()
        assert seg_header is not None
        seg_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        v.addWidget(seg_table, stretch=1)

        # 命中片段
        v.addWidget(QLabel("命中片段 (双击打开所在文件夹):"))
        clips = self._run.get("clips") or []
        clip_table = QTableWidget(0, 3)
        clip_table.setHorizontalHeaderLabels(["hit_idx", "时间戳", "clip 路径"])
        clip_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for c in clips:
            row = clip_table.rowCount()
            clip_table.insertRow(row)
            clip_table.setItem(row, 0, QTableWidgetItem(str(c.get("hit_idx", ""))))
            clip_table.setItem(row, 1, QTableWidgetItem(str(c.get("abs_timestamp", ""))))
            clip_table.setItem(row, 2, QTableWidgetItem(str(c.get("clip_path", ""))))
        clip_header = clip_table.horizontalHeader()
        assert clip_header is not None
        clip_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        clip_table.doubleClicked.connect(self._on_open_clip_folder)
        v.addWidget(clip_table, stretch=1)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        v.addWidget(btn_close)

    def _on_open_clip_folder(self, *_args) -> None:
        """双击 clip 行 → 打开所在文件夹。"""
        table = self.sender()
        if not isinstance(table, QTableWidget):
            return
        row = table.currentRow()
        if row < 0:
            return
        item = table.item(row, 2)
        if item is None:
            return
        clip_path = item.text().strip()
        if not clip_path:
            return
        try:
            folder = str(Path(clip_path).parent)
            startfile = getattr(os, "startfile", None)
            if startfile is not None:
                startfile(folder)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            logger.warning(f"[batch_tab] 打开文件夹失败: {e}")
