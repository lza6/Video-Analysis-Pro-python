"""Qt 组件冒烟测试（offscreen）：ChatBubble think 解析 / VideoPlayerDialog 属性 / KB 工具注册。"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class TestChatBubbleThinkParsing:
    def _bubble(self, qapp, text):
        from src.ui.agent_panel import ChatBubble
        return ChatBubble("Agent", text, is_user=False)

    def test_complete_think_tags_extracted(self, qapp):
        bubble = self._bubble(qapp, "<think>先想一下</think>答案在此")
        assert "先想一下" in bubble.thinking_widget.content_area.text()
        assert bubble.lbl_text.text() == "答案在此"

    def test_streaming_partial_think(self, qapp):
        bubble = self._bubble(qapp, "部分回答 <think>思考中")
        assert "思考中" in bubble.thinking_widget.content_area.text()

    def test_no_think_tags(self, qapp):
        bubble = self._bubble(qapp, "纯文本回答")
        assert bubble.lbl_text.text() == "纯文本回答"


class TestVideoPlayerDialogGuards:
    """回归：closeEvent 曾引用不存在的 self.player。

    QMediaPlayer 在没有 QApplication 的进程里实例化会触发 Windows fatal
    exception 0xc0000139（DLL 入口点缺失）——进程级崩溃无法被 pytest 捕获，
    因此实例化路径放到独立子进程中运行，主进程只做签名断言。
    子进程内先建 QApplication（与 run_main 的真实启动顺序一致）。
    """

    def test_close_event_uses_media_player(self, qapp, tmp_path):
        import cv2
        import numpy as np
        import subprocess
        import sys

        # 生成一个 1 秒的最小测试视频
        video_path = tmp_path / "tiny.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 10, (32, 32))
        for _ in range(10):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        writer.release()

        child = subprocess.run(
            [sys.executable, "-c", (
                "import sys; sys.path.insert(0, '.');"
                "from pathlib import Path;"
                "from PyQt6.QtWidgets import QApplication;"
                "app = QApplication([]);"
                "from src.ui.video_player_dialog import VideoPlayerDialog;"
                f"dlg = VideoPlayerDialog(Path(r'''{video_path}'''), None);"
                "assert hasattr(dlg, 'media_player');"
                "dlg.close(); print('PLAYER_CLOSE_OK')"
            )],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        )
        assert child.returncode == 0, f"player crashed: {child.stderr[-500:]}"
        assert "PLAYER_CLOSE_OK" in child.stdout

    def test_accepts_frames_kwarg(self, qapp):
        """回归：主窗口从未传 frames，时间轴标记恒空。签名需支持 frames。"""
        import inspect
        from src.ui.video_player_dialog import VideoPlayerDialog
        sig = inspect.signature(VideoPlayerDialog.__init__)
        assert "frames" in sig.parameters


class TestAdvancedFeaturesFlag:
    def test_flag_reflects_reality(self):
        """回归：旧实现 try:pass 使标志恒 True。"""
        import importlib
        import src.core.logic as logic

        expected = True
        for mod in ("moviepy", "matplotlib.pyplot", "seaborn"):
            try:
                importlib.import_module(mod)
            except Exception:
                expected = False
                break
        assert logic.ADVANCED_FEATURES_AVAILABLE == expected


class TestPromptLoader:
    def test_video_summary_loads_curated_template(self):
        from src.core.logic import PromptLoader
        p = PromptLoader().get_prompt("Video Summary")
        # config/prompts/frame_analysis/video_summary.txt 的中文精选模板
        assert len(p) > 100
        assert "{user_prompt}" in p

    def test_format_placeholders_compatible(self):
        from src.core.logic import PromptLoader
        p = PromptLoader().get_prompt("Video Summary")
        out = p.format(user_prompt="x", audio_transcript="y", frame_info="z")
        assert "x" in out


class TestVersionSingleSource:
    def test_launcher_uses_constants_version(self):
        import src.utils.constants as c
        assert c.APP_VERSION == "4.5.0"
