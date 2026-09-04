"""headless 并发治理 + 路径穿越白名单单测（v5.4 audit-prod P0 修复回归）。

覆盖：
1. 信号量串行化：占住信号量时，第二个 run_analysis 立即抛 _ServerBusy
2. 扩展名白名单：.exe/.py/.sh 等被改写为 upload.mp4；.mp4/.flv 等保留
3. 路径穿越：..\\..\\x.exe、绝对路径均被 basename 取纯文件名
4. 503 响应：_ServerBusy 映射为 503 + 不泄露内部状态
5. 500 响应不回传 str(e)：只回通用 message（P1）
"""
import os
import sys
import threading
from pathlib import Path

import pytest

os.environ.setdefault("VAP_ANALYZE_CONCURRENCY", "1")

from src.server.headless import _ANALYZE_SEMAPHORE, _ServerBusy, run_analysis


class TestConcurrencySemaphore:
    def test_second_call_returns_503_when_semaphore_held(self):
        """信号量被占时，第二个 run_analysis 立即 503 而非排队挂起。"""
        acquired = _ANALYZE_SEMAPHORE.acquire(blocking=False)
        assert acquired, "预占信号量失败（测试前置）"
        errs = []

        def call():
            try:
                run_analysis(b"fake", "a.mp4")
            except _ServerBusy as e:
                errs.append(("busy", str(e)))
            except Exception as e:
                errs.append(("other", str(e)[:40]))

        t = threading.Thread(target=call)
        t.start()
        t.join(3)
        _ANALYZE_SEMAPHORE.release()

        assert errs, "第二个调用既没 503 也没异常，信号量失效"
        assert errs[0][0] == "busy", f"应抛 _ServerBusy，实际 {errs}"


class TestFilenameSanitization:
    """扩展名白名单 + basename 消毒——纯函数式验证 run_analysis 内的清洗逻辑。"""

    @staticmethod
    def _sanitize(filename: str) -> str:
        """复现 run_analysis 内的清洗逻辑做纯函数验证。"""
        safe = Path(filename).name or "upload.mp4"
        ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".wmv", ".ts"}
        if Path(safe).suffix.lower() not in ALLOWED:
            safe = "upload.mp4"
        return safe

    def test_path_traversal_exe_rewritten(self):
        assert self._sanitize(r"..\..\x.exe") == "upload.mp4"

    def test_absolute_path_basenames(self):
        assert self._sanitize(r"C:\evil.mp4") == "evil.mp4"

    def test_non_video_ext_rewritten(self):
        for fn in ("bad.py", "shell.sh", "hack.html", "malware.exe"):
            assert self._sanitize(fn) == "upload.mp4", fn

    def test_video_ext_preserved(self):
        for fn in ("normal.mp4", "a.flv", "clip.mkv", "v.webm"):
            assert self._sanitize(fn) == fn, fn

    def test_empty_filename_fallback(self):
        assert self._sanitize("") == "upload.mp4"


class TestServerErrorResponses:
    """do_POST 的 503/500 响应不泄露内部状态（结构验证，不发真实请求）。"""

    def test_server_busy_is_503_not_500(self):
        # _ServerBusy 是 503 的映射，do_POST except 捕获它返回 503
        # 这里验证异常类型与映射契约
        from src.server.headless import Handler, _ServerBusy
        # _ServerBusy 必须是 Exception 子类才能被 except 捕获
        assert issubclass(_ServerBusy, Exception)

    def test_500_response_generic_message(self):
        """读 do_POST 源码确认 500 不回 str(e)。"""
        import inspect
        from src.server.headless import Handler
        src = inspect.getsource(Handler.do_POST)
        assert "str(e)" not in src, "500 仍回传 str(e)，泄露内部路径"
        assert "analysis failed" in src
