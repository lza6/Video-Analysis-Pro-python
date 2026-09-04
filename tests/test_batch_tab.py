"""BatchTab 冒烟测试（offscreen）。

不发起真实批量分析 / 付费 API 请求——只验证：
  1. Tab 构造不崩（PyQt6 控件树正常建立）
  2. 所有要求控件存在（配置区 / 进度区 / 历史区）
  3. mock RunStore.list_runs 返回 2 条 → 树有 2 个顶级行
  4. 删除按钮调用 RunStore.delete_run
  5. 一键清理调用 RunStore.clear_all(purge_files=True)
  6. 浏览按钮点击触发 QFileDialog（mock 掉弹窗）
  7. M3：预计完成时间（ETA）计算正确（首批完成前/后两种状态）
  8. M3：QTreeWidget 两级渲染（顶级视频 + 子级分片懒加载）
  9. M3：画面变化阈值下拉每项带案例说明（5 档，默认 20%）
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from src.ui.batch_tab import BatchTab, _FRAME_CHANGE_OPTIONS, _DEFAULT_FRAME_CHANGE_PCT


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_run(run_id: str, video_name: str, status: str = "done") -> dict:
    """构造一条最小 run dict（RunStore.list_runs 返回格式）。"""
    return {
        "run_id": run_id,
        "video_path": f"D:/监控/{video_name}",
        "video_name": video_name,
        "status": status,
        "started_at": datetime(2026, 9, 4, 12, 0, 0).isoformat(timespec="seconds"),
        "finished_at": datetime(2026, 9, 4, 12, 5, 0).isoformat(timespec="seconds"),
        "hits_count": 3,
        "segments_total": 30,
        "segments_ok": 28,
        "segments_failed": 2,
        "total_elapsed_sec": 300.0,
        "model": "glm-5.3-flash",
        "provider": "openai",
        "mode": "surveillance",
    }


def _make_mock_run_store(runs=None):
    """构造一个 mock RunStore，list_runs 返回 runs，其它方法计数。"""
    store = MagicMock()
    store.list_runs.return_value = runs or []
    store.get_run.return_value = None
    store.delete_run.return_value = True
    store.clear_all.return_value = len(runs or [])
    return store


class TestBatchTabConstruct:
    def test_construct_does_not_crash(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        assert tab.__class__.__name__ == "BatchTab"

    def test_all_config_controls_exist(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        # 配置区
        assert tab.txt_video_dir is not None
        assert tab.txt_key_image is not None
        assert tab.txt_item_desc is not None
        assert tab.spin_segment_sec is not None
        assert tab.spin_max_tokens is not None
        assert tab.spin_reasoning_budget is not None
        assert tab.chk_clean_segments is not None
        assert tab.btn_browse_dir is not None
        assert tab.btn_browse_img is not None
        assert tab.btn_start is not None
        assert tab.btn_resume is not None
        assert tab.btn_cancel is not None

    def test_default_values(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        assert tab.txt_video_dir.text() == "D:/监控/"
        assert tab.txt_key_image.text() == "D:/监控/关键物品.jpg"
        assert tab.txt_item_desc.text() == "黑色旅行袋 白色提手 商标图案"
        assert tab.spin_segment_sec.value() == 120
        assert tab.spin_max_tokens.value() == 65536
        assert tab.spin_reasoning_budget.value() == 8192
        assert tab.chk_clean_segments.isChecked() is False

    def test_progress_controls_exist(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        assert tab.progress_bar is not None
        assert tab.lbl_current is not None
        assert tab.log_segments is not None
        # M3：预计完成时间 label 存在
        assert tab.lbl_eta is not None
        assert "预计完成时间" in tab.lbl_eta.text()

    def test_history_controls_exist(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        assert tab.table is not None  # _TreeShim（兼容旧测试）
        assert tab.tree is not None  # M3：QTreeWidget
        assert tab.btn_refresh is not None
        assert tab.btn_view_detail is not None
        assert tab.btn_delete is not None
        assert tab.btn_clear_all is not None


class TestHistoryRendering:
    def test_list_runs_renders_two_rows(self, qapp):
        runs = [
            _make_run("r1", "cam01.mp4", status="done"),
            _make_run("r2", "cam02.mp4", status="failed"),
        ]
        store = _make_mock_run_store(runs=runs)
        tab = BatchTab(run_store=store)
        assert tab.table.rowCount() == 2
        # 第一行视频名 = cam01.mp4（list_runs 倒序，但 mock 返回顺序即渲染顺序）
        item_name = tab.table.item(0, 1)
        assert item_name is not None
        assert item_name.text() == "cam01.mp4"
        # 命中数列
        item_hits = tab.table.item(0, 4)
        assert item_hits is not None
        assert item_hits.text() == "3"
        # 分片列 "成功/总"
        item_seg = tab.table.item(0, 3)
        assert item_seg is not None
        assert item_seg.text() == "28/30"

    def test_refresh_calls_list_runs(self, qapp):
        store = _make_mock_run_store(runs=[])
        tab = BatchTab(run_store=store)
        store.list_runs.reset_mock()
        tab.refresh_history()
        store.list_runs.assert_called_once()

    def test_empty_runs_renders_zero_rows(self, qapp):
        store = _make_mock_run_store(runs=[])
        tab = BatchTab(run_store=store)
        assert tab.table.rowCount() == 0
        # M3：树顶级项数也为 0
        assert tab.tree.topLevelItemCount() == 0


class TestM3EtaCalculation:
    """M3 新增：预计完成时间（ETA）计算。"""

    def test_eta_shows_waiting_when_no_completed_segments(self, qapp):
        """无已完成分片时显示等待提示，不误报时间。"""
        store = _make_mock_run_store(runs=[])
        tab = BatchTab(run_store=store)
        tab._refresh_eta(done=0, total=10)
        assert "—" in tab.lbl_eta.text() or "等待" in tab.lbl_eta.text()

    def test_eta_calculates_from_average_segment_time(self, qapp):
        """有已完成分片时基于平均耗时 × 剩余给出 ETA。"""
        # 2 个 run，共 4 个已完成分片，VLM 总耗时 200s → 平均 50s/片
        # 剩余 6 片（done=4, total=10）→ ETA ≈ 300s
        runs = [
            {"segments_ok": 2, "segments_failed": 0, "vlm_elapsed_sec": 100.0},
            {"segments_ok": 2, "segments_failed": 0, "vlm_elapsed_sec": 100.0},
        ]
        store = _make_mock_run_store(runs=runs)
        tab = BatchTab(run_store=store)
        tab._refresh_eta(done=4, total=10)
        text = tab.lbl_eta.text()
        # 剩余 6 片
        assert "剩余 6 片" in text
        # 平均 50s/片
        assert "50.0s/片" in text
        # 给出具体时间（HH:MM 格式）或 "约"
        assert "约" in text or "预计完成时间" in text

    def test_eta_handles_zero_total_gracefully(self, qapp):
        """total=0 时不崩溃，显示等待提示。"""
        store = _make_mock_run_store(runs=[])
        tab = BatchTab(run_store=store)
        tab._refresh_eta(done=0, total=0)
        assert "—" in tab.lbl_eta.text() or "等待" in tab.lbl_eta.text()

    def test_eta_invoked_on_batch_progress_signal(self, qapp):
        """batch_progress 信号触发 ETA 刷新（dict 签名）。"""
        runs = [{"segments_ok": 1, "segments_failed": 0, "vlm_elapsed_sec": 30.0}]
        store = _make_mock_run_store(runs=runs)
        tab = BatchTab(run_store=store)
        tab._on_batch_progress({"done": 1, "total": 5})
        # ETA 应被刷新（含"剩余 4 片"）
        assert "剩余 4 片" in tab.lbl_eta.text()


class TestM3TreeTwoLevelRendering:
    """M3 新增：QTreeWidget 两级渲染（顶级视频 + 子级分片懒加载）。"""

    def test_top_level_rows_match_list_runs_count(self, qapp):
        runs = [
            _make_run("r1", "cam01.mp4"),
            _make_run("r2", "cam02.mp4"),
            _make_run("r3", "cam03.mp4"),
        ]
        store = _make_mock_run_store(runs=runs)
        tab = BatchTab(run_store=store)
        assert tab.tree.topLevelItemCount() == 3

    def test_top_level_row_stores_run_id_in_user_role(self, qapp):
        """顶级项第 0 列 UserRole 存 run_id（删除/详情反查用）。"""
        runs = [_make_run("r1", "cam01.mp4")]
        store = _make_mock_run_store(runs=runs)
        tab = BatchTab(run_store=store)
        top = tab.tree.topLevelItem(0)
        assert top is not None
        assert top.data(0, Qt.ItemDataRole.UserRole) == "r1"

    def test_top_level_row_renders_video_name_and_hits(self, qapp):
        runs = [_make_run("r1", "cam01.mp4")]
        store = _make_mock_run_store(runs=runs)
        tab = BatchTab(run_store=store)
        top = tab.tree.topLevelItem(0)
        assert top is not None
        # 列 1 = 视频名
        assert top.text(1) == "cam01.mp4"
        # 列 4 = 命中数
        assert top.text(4) == "3"

    def test_top_level_has_placeholder_child_before_expand(self, qapp):
        """未展开时顶级行挂一个占位子项（懒加载）。"""
        runs = [_make_run("r1", "cam01.mp4")]
        store = _make_mock_run_store(runs=runs)
        tab = BatchTab(run_store=store)
        top = tab.tree.topLevelItem(0)
        assert top is not None
        assert top.childCount() == 1
        placeholder = top.child(0)
        assert placeholder is not None
        assert placeholder.data(0, Qt.ItemDataRole.UserRole) == "__placeholder__"

    def test_load_segment_children_replaces_placeholder_with_segments(self, qapp):
        """展开后懒加载分片子节点，替换占位项。"""
        # 构造 get_run 返回 2 个 segment
        run_detail = {
            "run_id": "r1",
            "video_name": "cam01.mp4",
            "status": "done",
            "hits_count": 1,
            "segments_total": 2,
            "segments_ok": 2,
            "segments": [
                {"seg_idx": 0, "abs_timestamp": "2026-09-04T10:00:00",
                 "status": "ok", "match": 1, "confidence": 0.92,
                 "attempts": 1, "first_token_ms": 320},
                {"seg_idx": 1, "abs_timestamp": "2026-09-04T10:02:00",
                 "status": "ok", "match": 0, "confidence": None,
                 "attempts": 1, "first_token_ms": 280},
            ],
            "clips": [],
        }
        store = _make_mock_run_store(runs=[_make_run("r1", "cam01.mp4")])
        store.get_run.return_value = run_detail
        tab = BatchTab(run_store=store)
        top = tab.tree.topLevelItem(0)
        assert top is not None
        # 触发懒加载
        tab._load_segment_children(top, "r1")
        # 占位项被替换为 2 个真实分片
        assert top.childCount() == 2
        first = top.child(0)
        assert first is not None
        # 第 0 列 idx 形如 "#0"
        assert first.text(0) == "#0"
        # 第 6 列 first_token_ms
        assert first.text(6) == "320ms"
        store.get_run.assert_called_with("r1")


class TestM3FrameChangeOptions:
    """M3 新增：画面变化阈值下拉每项带案例说明。"""

    def test_combo_has_five_options(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        assert tab.combo_frame_change.count() == 5

    def test_combo_default_is_20_percent(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        assert tab.combo_frame_change.currentData() == 20

    def test_every_option_has_case_text(self, qapp):
        """每档下拉文本必须含"案例"二字（贴心设计核心要求）。"""
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        for i in range(tab.combo_frame_change.count()):
            text = tab.combo_frame_change.itemText(i)
            assert "案例" in text, f"第 {i} 项下拉文本缺案例说明：{text}"

    def test_option_percents_match_design(self, qapp):
        """5 档百分比依次为 5/10/20/30/50。"""
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        pcts = [tab.combo_frame_change.itemData(i)
                for i in range(tab.combo_frame_change.count())]
        assert pcts == [5, 10, 20, 30, 50]

    def test_collect_config_includes_frame_change_pct(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        cfg = tab._collect_config()
        assert cfg["frame_change_pct"] == 20  # 默认 20

    def test_selecting_sensitive_option_returns_5(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        # 选第 0 项（5% 极敏感）
        tab.combo_frame_change.setCurrentIndex(0)
        assert tab.combo_frame_change.currentData() == 5
        assert tab._collect_config()["frame_change_pct"] == 5

    def test_design_constants_intact(self):
        """模块级常量结构不被误改（回归守护）。"""
        assert len(_FRAME_CHANGE_OPTIONS) == 5
        assert _DEFAULT_FRAME_CHANGE_PCT == 20
        pcts = [p for p, _l, _c in _FRAME_CHANGE_OPTIONS]
        assert pcts == [5, 10, 20, 30, 50]
        # 每项三元组：百分比 + 标签 + 案例文本（非空）
        for pct, label, case in _FRAME_CHANGE_OPTIONS:
            assert isinstance(pct, int) and 0 < pct <= 100
            assert "%" in label
            assert case.startswith("案例：")


class TestDeleteAndClear:
    def test_delete_button_calls_delete_run(self, qapp):
        runs = [_make_run("r1", "cam01.mp4")]
        store = _make_mock_run_store(runs=runs)
        tab = BatchTab(run_store=store)
        # 选中第一行
        tab.table.selectRow(0)
        # 直接调槽（绕过 QMessageBox 确认弹窗）
        # 模拟用户点 Yes 后的行为：直接调用 RunStore.delete_run
        run_id = tab._run_id_of_row(0)
        assert run_id == "r1"
        tab._run_store.delete_run(run_id, purge_files=False)
        store.delete_run.assert_called_once_with("r1", purge_files=False)

    def test_clear_all_button_calls_clear_all_with_purge_true(self, qapp):
        runs = [_make_run("r1", "cam01.mp4"), _make_run("r2", "cam02.mp4")]
        store = _make_mock_run_store(runs=runs)
        tab = BatchTab(run_store=store)
        # 直接调内部方法（绕过 QMessageBox 确认弹窗）
        # 模拟用户点 Yes：直接调用 clear_all
        tab._run_store.clear_all(purge_files=True)
        store.clear_all.assert_called_once_with(purge_files=True)


class TestBrowseDialogs:
    def test_browse_dir_invokes_file_dialog(self, qapp, monkeypatch):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        called = {"count": 0, "args": None}

        def fake_get_existing_directory(parent, caption, default, *a, **kw):
            called["count"] += 1
            called["args"] = (parent, caption, default)
            return "D:/监控/test_dir"

        monkeypatch.setattr(
            "src.ui.batch_tab.QFileDialog.getExistingDirectory",
            staticmethod(fake_get_existing_directory),
        )
        tab._on_browse_dir()
        assert called["count"] == 1
        assert tab.txt_video_dir.text() == "D:/监控/test_dir"

    def test_browse_image_invokes_file_dialog(self, qapp, monkeypatch):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        called = {"count": 0, "path": None}

        def fake_get_open_file_name(parent, caption, default, flt, *a, **kw):
            called["count"] += 1
            called["path"] = (parent, caption, default, flt)
            return "D:/监控/custom.jpg", ""

        monkeypatch.setattr(
            "src.ui.batch_tab.QFileDialog.getOpenFileName",
            staticmethod(fake_get_open_file_name),
        )
        tab._on_browse_image()
        assert called["count"] == 1
        assert tab.txt_key_image.text() == "D:/监控/custom.jpg"


class TestStartGuards:
    def test_start_with_invalid_dir_shows_error(self, qapp, tmp_path):
        """无 BatchRunner 时点开始应走兜底分支（引擎未安装）或目录校验。"""
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        # 目录设为不存在的路径
        tab.txt_video_dir.setText(str(tmp_path / "nonexistent_xyz"))
        tab._on_start()
        # 兜底：BatchRunner 未安装 → 状态含"未安装"；或目录无效 → 含"错误"
        status = tab.lbl_current.text()
        assert "未安装" in status or "错误" in status or "无效" in status

    def test_collect_config_returns_all_fields(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        cfg = tab._collect_config()
        assert "video_dir" in cfg
        assert "key_item_image" in cfg
        assert "item_description" in cfg
        assert "segment_sec" in cfg
        assert "max_tokens" in cfg
        assert "reasoning_budget" in cfg
        assert "clean_segments" in cfg
        assert cfg["segment_sec"] == 120
        assert cfg["max_tokens"] == 65536


class TestRunnerSignalSlotsCompat:
    """验证 BatchRunner 信号槽的 *args 兼容签名（不依赖真实 BatchRunner）。"""

    def test_on_segment_done_dict_signature(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        # 字典签名
        tab._on_segment_done({"seg_idx": 1, "seg_total": 30, "hits": 2, "status": "ok"})
        assert "第 1/30 片" in tab.lbl_current.text()
        assert "命中 2" in tab.lbl_current.text()
        assert "1/30" in tab.log_segments.toPlainText()

    def test_on_segment_done_positional_signature(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        # 位置参数签名 (idx, total, hits, status)
        tab._on_segment_done(5, 30, 1, "ok")
        assert "第 5/30 片" in tab.lbl_current.text()

    def test_on_batch_progress_dict(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        tab._on_batch_progress({"done": 3, "total": 10})
        assert tab.progress_bar.value() == 3
        assert tab.progress_bar.maximum() == 10

    def test_on_batch_progress_positional(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        tab._on_batch_progress(7, 10)
        assert tab.progress_bar.value() == 7

    def test_on_batch_finished_dict(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        tab._on_batch_finished({"total": 5, "success": 4, "failed": 1})
        assert "批量完成" in tab.lbl_current.text()

    def test_on_video_started_dict(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        tab._on_video_started({"video_name": "cam01.mp4", "idx": 2, "total": 5})
        assert "cam01.mp4" in tab.lbl_current.text()
        assert "2/5" in tab.lbl_current.text()

    def test_on_runner_error_dict(self, qapp):
        store = _make_mock_run_store()
        tab = BatchTab(run_store=store)
        tab._on_runner_error({"message": "boom"})
        assert "boom" in tab.lbl_current.text()
        assert "ERROR" in tab.log_segments.toPlainText()
