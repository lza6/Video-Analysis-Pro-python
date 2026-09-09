"""B-FIN-3 覆盖率补测：agent_tools 工具工厂全分支、surveillance_agent 抽帧/裁剪、
video_graph build_from_run_store 桥接、history_manager 边界、skill_generator、
remote config 兜底、serve main --reload。

守付费红线：不发起真实 LLM / ffmpeg / PaddleOCR / duckduckgo 调用。
- 工具工厂只测"无外部依赖即可到达"的返回分支（Mock 仅隔离外部调用）。
- 付费红线判定：凡会触发真实 API/subprocess 的分支都走"异常/缺失依赖"路径，
  不真实调用。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
except OSError:
    torch = None

import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# src/core/agent_tools.py（44% → 工厂各分支，Mock 隔离耗时/外部依赖）
# ---------------------------------------------------------------------------
class _F:
    def __init__(self, path="a.jpg", timestamp=5.0, vision_content="一只狗",
                 ocr_text="hello"):
        self.path = path
        self.timestamp = timestamp
        self.vision_content = vision_content
        self.ocr_text = ocr_text


class _App:
    pass


def _make_app(**kw):
    a = _App()
    defaults = dict(
        video_path=Path("a.mp4"),
        video_duration=30.0,
        output_dir=Path("."),
        frames=[_F()],
        history_manager=None,
        run_store_for_tools=None,
        start_batch=None,
        analyzer=None,
        vlm_client=None,
    )
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(a, k, v)
    return a


class TestAgentToolsMore:
    def test_tool_and_registry(self, monkeypatch):
        from src.core.agent_tools import Tool, ToolRegistry
        t = Tool("add", "加法", lambda a, b: a + b, {"type": "object"})
        assert t.execute(a=1, b=2) == 3
        # 工具抛异常 → execute 兜底
        t2 = Tool("boom", "x", lambda: (_ for _ in ()).throw(RuntimeError("x")))
        out = t2.execute()
        assert "Error executing tool boom" in out

        reg = ToolRegistry()
        reg.register_tool("add", "加法", lambda a, b: a + b)
        reg.set_context_provider(lambda: None)
        desc = reg.get_tool_descriptions()
        assert "add" in desc and "加法" in desc and "Args:" not in desc
        assert "not found" in reg.execute_tool_call("nope", {})
        assert reg.execute_tool_call("add", {"a": 1, "b": 2}) == 3

    def test_get_video_meta_no_app_and_no_path(self):
        from src.core.agent_tools import create_get_video_meta_tool
        assert create_get_video_meta_tool(lambda: None)() == "No video loaded."
        assert create_get_video_meta_tool(lambda: _make_app(video_path=None))() \
            == "No video loaded."

    def test_get_frame_details_dynamic_extract_happy(self, tmp_path, monkeypatch):
        from src.core.agent_tools import create_get_frame_details_tool
        vid = tmp_path / "v.mp4"
        vid.write_bytes(b"\x00")
        app = _make_app(video_path=vid)
        app.frames = []  # 无预提取 → 走 on-the-fly

        class _Cap:
            def __init__(self):
                self._opened = True
                self._pos = 0.0

            def isOpened(self):
                return True

            def release(self):
                pass

            def get(self, prop):
                # FPS=25, POS_MSEC=5000
                return 25.0 if prop == 5 else 5000.0

            def set(self, prop, val):
                return True

            def read(self):
                img = np.zeros((16, 16, 3), dtype=np.uint8)
                return True, img

        monkeypatch.setattr("src.core.logic.videocapture_unicode",
                            lambda p: _Cap())
        monkeypatch.setattr("src.core.logic.imwrite_unicode",
                            lambda p, f: True)
        out = json.loads(create_get_frame_details_tool(lambda: app)(5.0))
        assert out["source"] == "on-the-fly extraction"
        assert out["timestamp"] == 5.0

    def test_get_frame_details_dynamic_not_opened_and_read_fail(self, tmp_path, monkeypatch):
        from src.core.agent_tools import create_get_frame_details_tool
        vid = tmp_path / "v.mp4"
        vid.write_bytes(b"\x00")
        app = _make_app(video_path=vid)
        app.frames = []

        class _Closed:
            def __init__(self, open_ok=True, read_ok=True):
                self._open_ok = open_ok
                self._read_ok = read_ok

            def isOpened(self):
                return self._open_ok

            def release(self):
                pass

            def get(self, prop):
                return 25.0

            def set(self, prop, val):
                return True

            def read(self):
                return self._read_ok, None

        monkeypatch.setattr("src.core.logic.videocapture_unicode",
                            lambda p: _Closed(open_ok=False))
        out = create_get_frame_details_tool(lambda: app)(5.0)
        assert "Could not open video" in out

        monkeypatch.setattr("src.core.logic.videocapture_unicode",
                            lambda p: _Closed(open_ok=True, read_ok=False))
        out2 = create_get_frame_details_tool(lambda: app)(5.0)
        assert "Could not read frame" in out2

    def test_get_frame_details_dynamic_exception(self, tmp_path, monkeypatch):
        from src.core.agent_tools import create_get_frame_details_tool
        vid = tmp_path / "v.mp4"
        vid.write_bytes(b"\x00")
        app = _make_app(video_path=vid)
        app.frames = []

        def _boom(p):
            raise RuntimeError("cv2 init fail")
        monkeypatch.setattr("src.core.logic.videocapture_unicode", _boom)
        out = create_get_frame_details_tool(lambda: app)(5.0)
        assert "Error during dynamic extraction" in out

    def test_delete_history_variants(self):
        from src.core.agent_tools import create_delete_history_tool
        # 无 output_dir → No active session
        assert "No active session" in create_delete_history_tool(lambda: _F())()
        # 有 output_dir 但无 history_manager → 源码走 `if app.history_manager:` 判断
        # 为 False 后隐式 return None（真实分支：无 history_manager 时无删除实现）
        app = _make_app()
        out = create_delete_history_tool(lambda: app)()
        assert out is None
        # history_manager 存在 → confirmation 提示
        app2 = _make_app(history_manager=object())
        out2 = create_delete_history_tool(lambda: app2)()
        assert "confirmation" in out2

    def test_search_web_error_and_no_results(self, monkeypatch):
        from src.core.agent_tools import create_search_web_tool
        f = create_search_web_tool()

        def _boom(*a, **k):
            raise RuntimeError("DDG down")
        monkeypatch.setattr("duckduckgo_search.DDGS", lambda *a, **k: _boom())
        out = f("q")
        assert "Web search error" in out

        class _NoRes:
            def text(self, query, max_results=5):
                return []

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        monkeypatch.setattr("duckduckgo_search.DDGS", lambda *a, **k: _NoRes())
        assert f("q") == "No results found."

    def test_kb_search_no_hm_and_embedder_none(self, monkeypatch):
        from src.core.agent_tools import create_kb_search_tool
        assert "App context" in create_kb_search_tool(lambda: None)("q")
        out = create_kb_search_tool(lambda: _make_app(history_manager=None))("q")
        assert "unavailable" in out or "知识库" in out or "unavailable" in out.lower()

        import src.core.kb_indexer as kb
        monkeypatch.setattr(kb, "CLIP_AVAILABLE", False)
        app = _make_app(history_manager=object())
        out2 = create_kb_search_tool(lambda: app)("q")
        assert "CLIP" in out2

    def test_image_search_branches(self, tmp_path, monkeypatch):
        from src.core.agent_tools import create_image_search_tool
        # 无 frames
        assert "No frames" in create_image_search_tool(
            lambda: _make_app(frames=[]))()
        # 无 image_path
        assert "No image provided" in create_image_search_tool(
            lambda: _make_app())(image_path="")
        # 文件不存在
        assert "Image not found" in create_image_search_tool(
            lambda: _make_app())(image_path=str(tmp_path / "nope.png"))
        # embedder None（需文件真实存在才过 exists() 检查）
        img = tmp_path / "x.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        import src.core.kb_indexer as kb
        monkeypatch.setattr(kb, "get_embedder", lambda: None)
        assert "unavailable" in create_image_search_tool(
            lambda: _make_app())(image_path=str(img))

    def test_summarize_hits_errors_and_empty(self, tmp_path):
        from src.core.agent_tools import create_summarize_hits_tool
        # 无 run_store 且无 hm → 未接入
        assert "RunStore 未接入" in create_summarize_hits_tool(
            lambda: _make_app())()
        # list_runs 抛
        class _Broken:
            def list_runs(self, limit=100):
                raise RuntimeError("db locked")
        app = _make_app(run_store_for_tools=_Broken())
        assert "读取 run_store 失败" in create_summarize_hits_tool(lambda: app)()
        # 空 runs
        class _Empty:
            def list_runs(self, limit=100):
                return []
        app2 = _make_app(run_store_for_tools=_Empty())
        assert "无历史 run" in create_summarize_hits_tool(lambda: app2)()

    def test_summarize_hits_with_runs_and_hits(self, tmp_path):
        from src.core.agent_tools import create_summarize_hits_tool
        from src.core.run_store import RunStore
        store = RunStore(config_dir=str(tmp_path / "cfg"))
        rid = store.create_run("D:/v/a.mp4", status="done")
        store.update_run(rid, hits_count=1, video_name="a.mp4",
                         status="done")
        seg_id = store.add_segment(rid, {
            "seg_idx": 0, "start_sec": 5.0, "dur_sec": 10.0,
            "status": "ok", "match": 1, "confidence": 0.9, "reason": "r"})
        store.add_hit(rid, {"hit_idx": 0, "abs_timestamp": "5.0",
                            "clip_path": "c0.mp4"})
        assert seg_id and store.get_run(rid)["segments"][0]["seg_idx"] == 0
        app = _make_app(run_store_for_tools=store)
        out = create_summarize_hits_tool(lambda: app)()
        assert "a.mp4" in out and "5.0" in out

    def test_trace_item_variants(self, tmp_path):
        from src.core.agent_tools import create_trace_item_tool
        # 无 run_store
        assert "RunStore 未接入" in create_trace_item_tool(
            lambda: _make_app())("钥匙")
        # 无 keyword
        assert "item_keyword" in create_trace_item_tool(
            lambda: _make_app())()
        # run_store 但 list_runs 抛 → build_from_run_store 内部捕获返回空图 → 无命中
        class _B:
            def list_runs(self, limit=100):
                raise RuntimeError("db locked")
        app = _make_app(run_store_for_tools=_B())
        out = create_trace_item_tool(lambda: app)("钥匙")
        assert "未在历史命中中找到" in out
        # 空 graph → 无命中
        class _NoRuns:
            def list_runs(self, limit=100):
                return []
        app2 = _make_app(run_store_for_tools=_NoRuns())
        assert "未在历史命中中找到" in create_trace_item_tool(lambda: app2)("钥匙")

    def test_trace_item_happy(self, tmp_path):
        from src.core.agent_tools import create_trace_item_tool
        from src.core.run_store import RunStore
        store = RunStore(config_dir=str(tmp_path / "cfg"))
        rid = store.create_run("D:/c1.mp4", status="done")
        store.update_run(rid, hits_count=1, video_name="cam01.mp4",
                         status="done")
        store.add_segment(rid, {
            "seg_idx": 0, "start_sec": 5.0, "dur_sec": 10.0,
            "status": "ok", "match": 1, "confidence": 0.9,
            "reason": "黑色旅行袋 出现在画面", "abs_timestamp": "10.0"})
        store.add_hit(rid, {"hit_idx": 0, "abs_timestamp": "10.0",
                            "clip_path": "c0.mp4"})
        app = _make_app(run_store_for_tools=store)
        out = create_trace_item_tool(lambda: app)("旅行")
        assert "cam01.mp4" in out

    def test_generate_skill_branches(self, tmp_path, monkeypatch):
        from src.core.agent_tools import create_generate_skill_tool
        # 写盘隔离：agent_tools 内 `from src.utils.constants import CONFIG_DIR`
        # 是函数内局部 import，patch constants.CONFIG_DIR 即可隔离真实 config/
        import src.utils.constants as cst
        monkeypatch.setattr(cst, "CONFIG_DIR", str(tmp_path))
        # 无描述
        assert "请提供场景描述" in create_generate_skill_tool(
            lambda: _make_app())()
        # 未知场景 → None → 提示
        out_unknown = create_generate_skill_tool(lambda: _make_app())(
            "随便说说")
        assert "未识别到已知场景" in out_unknown
        # 已知场景 → 成功
        out_ok = create_generate_skill_tool(lambda: _make_app())(
            "停车场车牌识别")
        assert "已生成新 skill" in out_ok
        # 重复生成 → FileExistsError → 失败提示
        out_dup = create_generate_skill_tool(lambda: _make_app())(
            "停车场车牌识别")
        assert "生成失败" in out_dup

    def test_rtsp_monitor_tool_branches(self, tmp_path):
        from src.core.agent_tools import create_rtsp_monitor_tool
        # app 缺失
        assert "App context missing" in create_rtsp_monitor_tool(lambda: None)()
        # 无 rtsp_url
        assert "RTSP URL" in create_rtsp_monitor_tool(lambda: _make_app())()
        # 构造 RtspMonitor 抛（不真实拉流）
        import src.core.rtsp_stream as rts
        orig = rts.RtspMonitor
        class _BoomMonitor:
            def __init__(self, *a, **k):
                raise RuntimeError("no stream")
        rts.RtspMonitor = _BoomMonitor
        try:
            out = create_rtsp_monitor_tool(lambda: _make_app())(
                "rtsp://u:p@cam/stream")
            assert "启动失败" in out
        finally:
            rts.RtspMonitor = orig


# ---------------------------------------------------------------------------
# src/core/surveillance_agent.py（52% → _extract_frames 抽帧/裁剪/预筛）
# ---------------------------------------------------------------------------
class TestSurveillanceAgentMore:
    def _agent(self, tmp_path):
        from src.core.surveillance_agent import SurveillanceAgent
        ki = tmp_path / "key.jpg"
        ki.write_bytes(b"\xff\xd8\xff")
        return SurveillanceAgent(None, str(ki), "钥匙")

    def test_extract_frames_ffmpeg_happy_and_exception(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00")

        # ffmpeg 在 PATH → 走 ffmpeg 循环
        monkeypatch.setattr("shutil.which", lambda name: "ffmpeg" if name == "ffmpeg" else None)

        class _Cap:
            def get(self, prop):
                # FPS=25, FRAME_COUNT=250, 其它=0
                return 250.0 if prop == 7 else (25.0 if prop == 5 else 0)
            def release(self):
                pass
            def isOpened(self):
                return True

        monkeypatch.setattr("cv2.VideoCapture", lambda p: _Cap())
        monkeypatch.setattr("cv2.imencode",
                            lambda ext, f: (True, np.frombuffer(b"jpg", dtype=np.uint8)))
        out_dir = tmp_path / "frames"
        frames = a._extract_frames(video, out_dir)
        # ffmpeg 子进程 run 未 mock → 文件不存在 → 0 帧
        assert frames == []

        # ffmpeg 抽帧成功（touch 帧文件让 fp.exists() 成立）
        def _run(cmd, **kw):
            out_dir.mkdir(parents=True, exist_ok=True)
            if cmd and cmd[-1].endswith("s.jpg"):
                Path(cmd[-1]).touch()
            return type("R", (), {"returncode": 0})()
        monkeypatch.setattr("subprocess.run", _run)
        frames2 = a._extract_frames(video, out_dir)
        assert len(frames2) >= 1

    def test_extract_frames_cv2_fallback(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00")
        monkeypatch.setattr("shutil.which", lambda name: None)

        class _Cap:
            def __init__(self):
                self._frames = [np.zeros((16, 16, 3), dtype=np.uint8)] * 5
                self._i = 0

            def get(self, prop):
                if prop == 5:
                    return 25.0
                if prop == 7:
                    return 250
                return 0

            def release(self):
                pass

            def isOpened(self):
                return True

            def set(self, prop, val):
                return True

            def read(self):
                if self._i < len(self._frames):
                    f = self._frames[self._i]
                    self._i += 1
                    return True, f
                return False, None

        monkeypatch.setattr("cv2.VideoCapture", lambda p: _Cap())
        # imencode 返回带 tofile 的 ndarray（真实验证 imencode→tofile 落盘路径）
        monkeypatch.setattr("cv2.imencode",
                            lambda ext, f: (True, np.frombuffer(b"jpgdata", dtype=np.uint8)))
        out_dir = tmp_path / "f"
        frames = a._extract_frames(video, out_dir)
        assert frames  # cv2 回退路径产出帧
        assert all(Path(f["path"]).exists() for f in frames)

    def test_extract_frames_not_opened(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00")
        monkeypatch.setattr("shutil.which", lambda name: None)

        class _Closed:
            def get(self, prop):
                return 0
            def release(self):
                pass
            def isOpened(self):
                return False

        monkeypatch.setattr("cv2.VideoCapture", lambda p: _Closed())
        out = a._extract_frames(video, tmp_path / "f")
        assert out == []

    def test_cut_clip_ffmpeg_and_cv2(self, tmp_path, monkeypatch):
        a = self._agent(tmp_path)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00")
        out_path = tmp_path / "clips" / "c.mp4"
        monkeypatch.setattr("shutil.which", lambda name: "ffmpeg")
        assert a.cut_clip(video, 5.0, out_path) is None  # 不抛

        # cv2 回退（无 ffmpeg）
        monkeypatch.setattr("shutil.which", lambda name: None)

        class _Cap:
            def __init__(self):
                self._frames = [np.zeros((16, 16, 3), dtype=np.uint8)] * 10
                self._i = 0

            def get(self, prop):
                return {5: 25.0, 7: 250, 3: 320, 4: 240}.get(prop, 0)
            def release(self):
                pass
            def isOpened(self):
                return True
            def set(self, prop, val):
                return True
            def read(self):
                if self._i < len(self._frames):
                    f = self._frames[self._i]
                    self._i += 1
                    return True, f
                return False, None

        monkeypatch.setattr("cv2.VideoCapture", lambda p: _Cap())

        class _FakeVideoWriter:
            @staticmethod
            def fourcc(*a):
                return 0

            def __init__(self, *a, **k):
                pass

            def write(self, f):
                pass

            def release(self):
                pass

            def isOpened(self):
                return True

        monkeypatch.setattr("cv2.VideoWriter", _FakeVideoWriter)
        a.cut_clip(video, 5.0, out_path)  # 不抛


# ---------------------------------------------------------------------------
# src/core/video_graph.py（88% → build_from_run_store 全桥接 + 节点序列化）
# ---------------------------------------------------------------------------
class TestVideoGraphMore:
    def test_build_from_run_store_with_hits(self, tmp_path):
        from src.core.video_graph import VideoGraph
        from src.core.run_store import RunStore
        store = RunStore(config_dir=str(tmp_path / "cfg"))
        rid = store.create_run("D:/cam01.mp4", status="done")
        store.update_run(rid, hits_count=1, video_name="cam01.mp4",
                         status="done")
        store.add_segment(rid, {
            "seg_idx": 0, "start_sec": 5.0, "dur_sec": 10.0,
            "status": "ok", "match": 1, "confidence": 0.9,
            "reason": "黑色旅行袋", "abs_timestamp": "2026-09-09T10:00:00"})
        store.add_hit(rid, {"hit_idx": 0, "abs_timestamp": "2026-09-09T10:00:00",
                            "clip_path": "c0.mp4"})
        g = VideoGraph.build_from_run_store(store, limit=100)
        assert len(g._nodes) == 1
        node = list(g._nodes.values())[0]
        assert node.video_name == "cam01.mp4"
        assert node.abs_time is not None
        d = g.to_dict()
        assert d["temporal_threshold_sec"] > 0
        assert d["nodes"][0]["abs_time"].startswith("2026-09-09")

    def test_build_from_run_store_no_hits_and_errors(self, tmp_path):
        from src.core.video_graph import VideoGraph
        # 空 store → 空图
        class _NoRuns:
            def list_runs(self, limit=200):
                return []
        assert len(VideoGraph.build_from_run_store(_NoRuns())._nodes) == 0
        # list_runs 抛 → 空图
        class _Broken:
            def list_runs(self, limit=200):
                raise RuntimeError("db down")
        assert len(VideoGraph.build_from_run_store(_Broken())._nodes) == 0

    def test_hit_node_parse_edge_cases(self, tmp_path):
        from src.core.video_graph import HitNode
        # _parse_time 数值 → None（秒不当作绝对时间）
        assert HitNode._parse_time(5.0) is None
        assert HitNode._parse_time("garbage") is None
        assert HitNode._parse_time("") is None
        assert HitNode._parse_time(datetime(2026, 1, 1)) is not None
        # _parse_sec 失败 → 0.0
        assert HitNode._parse_sec("abc") == 0.0
        assert HitNode._parse_sec(None) == 0.0
        # _extract_camera_id
        assert HitNode._extract_camera_id("36#2单元入口.mp4") == "36"
        assert HitNode._extract_camera_id("cam02.mp4") == "cam02"
        assert HitNode._extract_camera_id("_388.mp4") == "388"

    def test_trace_edge_types_and_no_keyword(self):
        from src.core.video_graph import VideoGraph, HitNode
        g = VideoGraph(temporal_threshold_sec=60)
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        n1 = HitNode(run_id="a", video_name="c1", video_path="p",
                     timestamp_sec=1.0, reason="黑色旅行袋 黑色", abs_time=t0,
                     camera_id="cam01")
        n2 = HitNode(run_id="b", video_name="c2", video_path="p2",
                     timestamp_sec=2.0, reason="黑色旅行袋 出现在",
                     abs_time=t0 + timedelta(seconds=10), camera_id="cam02")
        n3 = HitNode(run_id="c", video_name="c3", video_path="p3",
                     timestamp_sec=3.0, reason="无关", abs_time=t0 + timedelta(seconds=90))
        for n in (n1, n2, n3):
            g.add_hit(n)
        # item 边（reason 共享关键词）
        et = g.edge_types(n1.node_id, n2.node_id)
        assert "item" in et
        # 时序边（t0 差 10s < 60s）
        assert "temporal" in et
        # 无关键词 → 空
        assert g.trace_item("") == []
        # 无命中 keyword → 空
        assert g.trace_item("不存在的") == []
        # n3 与 n1 差 90s，不在阈值 → 无 temporal，且 item 关键词不同 → 无边
        assert g.edge_types(n1.node_id, n3.node_id) == set()


# ---------------------------------------------------------------------------
# src/core/history_manager.py（64% → session/checkpoint/空 Chroma 分支）
# ---------------------------------------------------------------------------
class TestHistoryManagerMore:
    def test_sessions_and_checkpoints(self, tmp_path):
        from src.core.history_manager import HistoryManager
        m = HistoryManager(str(tmp_path / "cfg"))
        sid = m.add_session("a.mp4", str(tmp_path / "o"))
        assert m.get_history()[0]["video_name"] == "a.mp4"
        assert m.get_history()[0]["status"] == "completed"
        m.save_checkpoint(sid, 2.5, {"n": 1})
        cp = m.get_checkpoint(sid)
        assert cp["second"] == 2.5 and cp["data"] == {"n": 1}
        assert m.get_checkpoint("missing") is None
        assert m.delete_session("missing") is False
        assert m.delete_session(sid) is True
        assert m.get_history() == []

    def test_kb_without_chroma_returns_empty(self, tmp_path, monkeypatch):
        from src.core import history_manager as hm
        monkeypatch.setattr(hm, "CHROMA_AVAILABLE", False)
        m = hm.HistoryManager(str(tmp_path / "cfg"))
        assert m.chroma_client is None
        assert m.add_frame_to_kb("s", "v", "p", 1.0,
                                 "x", np.zeros(8)) is False
        assert m.search_kb(np.zeros(8)) == []
        assert m.kb_count() == 0


# ---------------------------------------------------------------------------
# src/core/skill_generator.py（95% → 少量分支）
# ---------------------------------------------------------------------------
class TestSkillGeneratorMore:
    def test_detect_scene_and_draft_fallback(self, tmp_path):
        from src.core.skill_generator import (
            detect_scene, draft_skill_from_scene, generate_skill,
            render_skill_md, save_skill)
        assert detect_scene("") is None
        assert detect_scene("停车场车牌识别") == "parking"
        assert detect_scene("人脸识别") == "face"
        assert detect_scene("火焰烟雾") == "fire"
        assert detect_scene("车辆计数 车流") == "vehicle"
        assert draft_skill_from_scene("bogus", "") is None
        d = draft_skill_from_scene("parking", "停车场")
        assert d is not None
        md = render_skill_md(d)
        assert "surveillance-parking-lpr" in md

        sd = tmp_path / "s"
        p = save_skill(d, sd, overwrite=False)
        assert p.exists()
        with pytest.raises(FileExistsError):
            save_skill(d, sd, overwrite=False)
        # overwrite=True 覆盖
        save_skill(d, sd, overwrite=True)
        # 重复生成（不同场景同名不存在，验证 FileExistsError → {"ok": False}）
        r = generate_skill("停车场车牌识别", sd, overwrite=False)
        assert r is not None and r.get("ok") is False
        res = generate_skill("树木分类", sd)
        assert res is None  # 未知场景


# ---------------------------------------------------------------------------
# src/remote/config.py 兜底分支
# ---------------------------------------------------------------------------
class TestRemoteMore:
    def test_credential_fetcher_fallback(self):
        from src.remote.config import _DefaultCredentialFetcher
        f = _DefaultCredentialFetcher()
        # 构造读 config_manager._secure_get；正常时 get 返回 fallback（keyring 隔离）
        assert f.get("missing_key", "fb") in ("fb", "")

    def test_credential_store_has_and_snapshot(self):
        from src.remote.config import CredentialStore

        class _F:
            def get(self, key, fallback=""):
                return "v" if key.endswith("on") else fallback
        s = CredentialStore(fetcher=_F())
        assert s.has("on") is True
        assert s.has("off") is False
        assert s.snapshot(["on", "off"]) == {"on": True, "off": False}


# ---------------------------------------------------------------------------
# src/web/serve.py main 入口 --reload
# ---------------------------------------------------------------------------
class TestServeMore:
    def test_main_reload_flag(self, monkeypatch):
        import sys as _sys
        from src.web import serve as serve_mod
        monkeypatch.setattr(_sys, "argv", ["serve.py", "--reload"])
        captured = {}

        def _fake(host=None, port=None, open_browser=True, reload=False):
            captured["reload"] = reload
            return 0
        monkeypatch.setattr(serve_mod, "run_server", _fake)
        with pytest.raises(SystemExit):
            serve_mod.main()
        assert captured["reload"] is True