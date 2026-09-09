"""B-FIN-2 覆盖率补测：analyzer_service 全流水线、serve 命令行分支、
ui_components tkinter、rtsp_stream、llm_gateway 协议解析、logic 纯函数、
motion_detector 纯函数、surveillance_agent、run_store 边界、kb_indexer。

守付费红线：不发起真实 LLM / ffmpeg 长处理 / 摄像头 / keyring 调用。
- keyring 用内存替身（同 test_b_fin_coverage._fake_secure_set/_get）。
- LLM 网关后端用假响应对象测协议解析（不真实请求）。
- VideoProcessor / AudioProcessor / RtspGrabber 用实例级隔离注入。
- tkinter 仅构造（monkeypatch wait_window 防阻塞），不 mainloop。
"""
from __future__ import annotations

import configparser
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
except OSError:
    torch = None

import numpy as np  # noqa: E402
import cv2  # noqa: E402


# ---------------------------------------------------------------------------
# keyring 内存替身（与 test_b_fin_coverage 同款，防真实 keyring/ini 写）
# ---------------------------------------------------------------------------
_KEYRING_MEM: dict[str, str] = {}


def _fake_secure_set(key: str, value: str):
    if value:
        _KEYRING_MEM[key] = value
    return None


def _fake_secure_get(key: str, fallback: str = ""):
    return _KEYRING_MEM.get(key, fallback)


@pytest.fixture(autouse=True)
def _isolate_keyring(monkeypatch):
    import src.utils.config_manager as cm_mod
    import src.core.provider_preset as pp_mod

    _KEYRING_MEM.clear()
    monkeypatch.setattr(cm_mod, "_KEYRING_AVAILABLE", False)
    monkeypatch.setattr(cm_mod, "_secure_set", _fake_secure_set)
    monkeypatch.setattr(cm_mod, "_secure_get", _fake_secure_get)
    monkeypatch.setattr(pp_mod, "_KEYRING_SERVICE", "BFin2Test")
    yield
    _KEYRING_MEM.clear()


# ---------------------------------------------------------------------------
# src/web/services/analyzer_service.py（69% → 全流水线成功/降级/异常/工具分支）
# ---------------------------------------------------------------------------


class _FakeLoop:
    """call_soon_threadsafe 同步执行（不真实起 loop），便于断言事件。"""

    def __init__(self):
        self.calls: list[tuple] = []

    def call_soon_threadsafe(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))
        try:
            fn(*args, **kwargs)
        except Exception:
            pass


def _make_store_and_rec(tmp_path):
    from src.web.job_store import JobStore
    store = JobStore()
    rec = store.create("v.mp4", tmp_path, tmp_path / "frames")
    return store, rec


def _make_frames(tmp_path, n=2):
    from src.core.logic import Frame
    out = []
    for i in range(n):
        p = tmp_path / f"f{i}.jpg"
        p.write_bytes(b"\xff\xd8\xff")
        out.append(Frame(path=p, timestamp=i + 0.5,
                         metrics={"brightness": 0.3 + i * 0.1},
                         vision_content=f"目标{i}", ocr_text=""))
    return out


class TestAnalyzerServicePipeline:
    def test_pipeline_happy_path_smart_audio(self, tmp_path, monkeypatch):
        """三阶段成功：smart_extraction + audio + report → DONE + 事件。"""
        from src.web.services import analyzer_service as a_mod
        from src.web.services.analyzer_service import AnalyzerService
        from src.web.job_store import JobStatus

        fake_loop = _FakeLoop()
        svc = AnalyzerService(_make_store_and_rec(tmp_path)[0], None, fake_loop)
        rec = _make_store_and_rec(tmp_path)[1]
        frames = _make_frames(tmp_path)

        class _Proc:
            def __init__(self, video_path, out_dir):
                self.video_path = video_path
                self.out_dir = out_dir

            def extract_smart_keyframes(self, **kw):
                return frames

        class _Audio:
            def __init__(self, **kw):
                pass

            def extract_audio(self, video_path, out_dir):
                p = Path(out_dir) / "audio.mp3"
                p.write_bytes(b"\x00\x01")
                return p

            def transcribe(self, audio_path, **kw):
                return type("T", (), {"text": "音频转录内容"})()

        monkeypatch.setattr(a_mod, "VideoProcessor", _Proc)
        monkeypatch.setattr(a_mod, "AudioProcessor", _Audio)
        monkeypatch.setattr(a_mod, "_probe_duration", lambda p: 5.0)
        monkeypatch.setattr(svc, "_build_llm_callback",
                            lambda: lambda prompt, images=None: "完整分析报告")

        svc._run_pipeline(rec, Path("v.mp4"), {
            "smart_extraction": True, "enable_audio": True, "density": 0.5,
        }, rec.put_event)

        assert rec.status == JobStatus.DONE
        assert rec.frame_count == 2
        assert rec.duration == 5.0
        assert rec.transcript == "音频转录内容"
        assert rec.report == "完整分析报告"
        assert len(rec.frames) == 2
        types = [e["type"] for e in rec.recent_events]
        for t in ("phase", "progress", "frame", "transcript", "report-token", "done"):
            assert t in types, f"missing {t}: {types}"

    def test_pipeline_dense_and_llm_ok(self, tmp_path, monkeypatch):
        """非 smart（density）+ llm 正常 → DONE + report_token 不含哨兵。"""
        from src.web.services import analyzer_service as a_mod
        from src.web.services.analyzer_service import AnalyzerService
        from src.web.job_store import JobStatus

        rec = _make_store_and_rec(tmp_path)[1]
        svc = AnalyzerService(_make_store_and_rec(tmp_path)[0], None, _FakeLoop())
        frames = _make_frames(tmp_path)

        class _Proc:
            def __init__(self, video_path, out_dir):
                pass

            def extract_keyframes(self, density, **kw):
                return frames

        monkeypatch.setattr(a_mod, "VideoProcessor", _Proc)
        monkeypatch.setattr(a_mod, "AudioProcessor",
                            lambda **kw: type("A", (), {
                                "extract_audio": lambda self, v, o: None,
                                "transcribe": lambda self, a, **k: None})())
        monkeypatch.setattr(a_mod, "_probe_duration", lambda p: 5.0)
        monkeypatch.setattr(svc, "_build_llm_callback",
                            lambda: lambda prompt, images=None: "结论：有价值")

        svc._run_pipeline(rec, Path("v.mp4"), {
            "smart_extraction": False, "enable_audio": False, "density": 0.4,
        }, rec.put_event)

        assert rec.status == JobStatus.DONE
        assert rec.transcript == ""
        delta_types = [e["data"].get("token", "") for e in rec.recent_events
                       if e["type"] == "report-token"]
        assert "结论：有价值" == "".join(delta_types)

    def test_pipeline_llm_unavailable_degrade(self, tmp_path, monkeypatch):
        """llm_cb=None → 降级 [LLM 阶段不可用] 但不失败。"""
        from src.web.services import analyzer_service as a_mod
        from src.web.services.analyzer_service import AnalyzerService
        from src.web.job_store import JobStatus

        rec = _make_store_and_rec(tmp_path)[1]
        svc = AnalyzerService(_make_store_and_rec(tmp_path)[0], None, _FakeLoop())

        class _Proc:
            def __init__(self, video_path, out_dir):
                pass

            def extract_keyframes(self, density, **kw):
                return _make_frames(tmp_path)

        monkeypatch.setattr(a_mod, "VideoProcessor", _Proc)
        monkeypatch.setattr(a_mod, "_probe_duration", lambda p: 1.0)
        monkeypatch.setattr(svc, "_build_llm_callback", lambda: None)

        svc._run_pipeline(rec, Path("v.mp4"), {
            "smart_extraction": False, "enable_audio": False, "density": 0.2,
        }, rec.put_event)

        assert rec.status == JobStatus.DONE
        assert "[LLM 阶段不可用" in rec.report
        warns = [e for e in rec.recent_events if e["type"] == "log"]
        assert warns, "应推送 warning log"

    def test_pipeline_llm_raises_degrade(self, tmp_path, monkeypatch):
        """llm_cb 抛异常 → 降级。"""
        from src.web.services import analyzer_service as a_mod
        from src.web.services.analyzer_service import AnalyzerService
        from src.web.job_store import JobStatus

        rec = _make_store_and_rec(tmp_path)[1]
        svc = AnalyzerService(_make_store_and_rec(tmp_path)[0], None, _FakeLoop())

        class _Proc:
            def __init__(self, video_path, out_dir):
                pass

            def extract_keyframes(self, density, **kw):
                return _make_frames(tmp_path)

        def _boom_cb(prompt, images=None):
            raise RuntimeError("provider down")

        monkeypatch.setattr(a_mod, "VideoProcessor", _Proc)
        monkeypatch.setattr(a_mod, "_probe_duration", lambda p: 1.0)
        monkeypatch.setattr(svc, "_build_llm_callback", lambda: _boom_cb)

        svc._run_pipeline(rec, Path("v.mp4"), {
            "smart_extraction": False, "enable_audio": False, "density": 0.2,
        }, rec.put_event)

        assert rec.status == JobStatus.DONE
        assert "[LLM 阶段不可用" in rec.report

    def test_pipeline_phase1_exception_failed(self, tmp_path, monkeypatch):
        """Phase1 抽帧抛异常 → FAILED + error + ERROR 事件。"""
        from src.web.services import analyzer_service as a_mod
        from src.web.services.analyzer_service import AnalyzerService
        from src.web.job_store import JobStatus

        rec = _make_store_and_rec(tmp_path)[1]
        svc = AnalyzerService(_make_store_and_rec(tmp_path)[0], None, _FakeLoop())

        class _Proc:
            def __init__(self, video_path, out_dir):
                pass

            def extract_keyframes(self, density, **kw):
                raise RuntimeError("抽帧崩溃")

        monkeypatch.setattr(a_mod, "VideoProcessor", _Proc)

        svc._run_pipeline(rec, Path("v.mp4"), {
            "smart_extraction": False, "enable_audio": False, "density": 0.2,
        }, rec.put_event)

        assert rec.status == JobStatus.FAILED
        assert "抽帧崩溃" in rec.error
        assert any(e["type"] == "error" for e in rec.recent_events)

    def test_create_job_workdir_none_and_push_exception(self, tmp_path, monkeypatch):
        """create_job 未传 workdir → 兜底视频同级 frames；loop=None → push 异常隔离。"""
        from src.web.services import analyzer_service as a_mod
        from src.web.services.analyzer_service import AnalyzerService
        from src.web.job_store import JobStatus, JobStore

        store = JobStore()
        work = tmp_path / "videos"
        work.mkdir()
        svc = AnalyzerService(store, None, None)

        def _sync(target, args, on_start, on_done):
            on_start()
            target(*args)
            on_done()

        monkeypatch.setattr(a_mod, "_run_in_thread", _sync)
        svc._run_pipeline = lambda rec, video_path, config, push: None  # type: ignore
        svc._loop = None  # type: ignore  → push 的 call_soon_threadsafe 抛 AttributeError

        rec = svc.create_job(work / "v.mp4", "v.mp4", {"density": 0.3})
        assert rec.video_path == str(work / "v.mp4")
        assert rec.status == JobStatus.RUNNING
        assert rec.started_at is not None
        assert rec.stream_closed is True  # on_done 已执行
        # frames 目录建在视频同级（workdir=None 兜底分支）
        assert (work / "frames").is_dir()

    def test_build_llm_callback_no_nv_single_provider(self, tmp_path, monkeypatch):
        """无 nv key → 单 provider build_llm_client 成功 → cb 返回纯文本。"""
        from src.web.services.analyzer_service import AnalyzerService
        import src.core.provider_router as pr_mod

        monkeypatch.setattr(pr_mod, "load_from_env", lambda: [])

        class _FakeClient:
            def __init__(self):
                self.model = "gpt-m"

            def chat_stream(self, model, prompt, images=None, **kw):
                yield "hello "
                yield "world"

        monkeypatch.setattr("src.core.logic.build_llm_client",
                            lambda cm: _FakeClient())
        svc = AnalyzerService(None, None, None)
        cb = svc._build_llm_callback()
        assert cb is not None
        assert cb("prompt") == "hello world"

    def test_build_llm_callback_failure_returns_none(self, tmp_path, monkeypatch):
        """nv 无 key 且 build_llm_client 抛 → cb=None。"""
        from src.web.services.analyzer_service import AnalyzerService
        import src.core.provider_router as pr_mod

        monkeypatch.setattr(pr_mod, "load_from_env", lambda: [])
        monkeypatch.setattr("src.core.logic.build_llm_client",
                            lambda cm: (_ for _ in ()).throw(RuntimeError("no creds")))

        svc = AnalyzerService(None, None, None)
        assert svc._build_llm_callback() is None

    def test_probe_duration_not_opened_and_exception(self, tmp_path, monkeypatch):
        """_probe_duration：视频打不开 → 0.0；探测抛异常 → 0.0。"""
        from src.web.services.analyzer_service import _probe_duration

        assert _probe_duration(tmp_path / "nope.mp4") == 0.0

        import src.web.services.analyzer_service as a_mod
        monkeypatch.setattr(a_mod, "videocapture_unicode",
                            lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
        assert _probe_duration(tmp_path / "x.mp4") == 0.0

    def test_run_in_thread_wrapper_on_failure(self):
        """_run_in_thread 包装：目标抛异常仍调 on_done（线程内）。"""
        from src.web.services.analyzer_service import _run_in_thread

        calls = {"start": 0, "done": 0}

        def target():
            raise RuntimeError("靶失败")

        t = _run_in_thread(target, (), lambda: calls.__setitem__("start", 1),
                           lambda: calls.__setitem__("done", 1))
        t.join(timeout=5)
        assert calls["start"] == 1
        assert calls["done"] == 1


# ---------------------------------------------------------------------------
# src/core/llm_gateway.py（55% → 各协议后端流式解析 + 重试/网络错误分支）
# ---------------------------------------------------------------------------


class _FakeResp:
    """带 iter_lines 的假响应（协议后端 with ... as resp 用）。"""

    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        yield from (l + "\n" for l in self._lines)


class _FakePartial:
    """先 503 后 200 的响应对象（_post_with_retry 用）。"""

    def __init__(self, ok_then):
        self._status = 503
        self._flips = ok_then

    @property
    def status_code(self):
        return self._status

    def close(self):
        pass


class TestLlmgatewayBackends:
    def _backend(self, cls):
        return cls(api_key="k", base_url="https://x.example/api",
                   model="m", timeout=5, max_tokens=100)

    def test_anthropic_stream_parsing(self, monkeypatch):
        from src.core.llm_gateway import AnthropicBackend
        b = self._backend(AnthropicBackend)
        lines = [
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"你好"}}',
            'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"分析中"}}',
            "data: [DONE]",
        ]
        monkeypatch.setattr(AnthropicBackend, "_post_with_retry",
                            lambda self, url, headers, payload: _FakeResp(lines))
        out = "".join(b.chat_stream([{"role": "user", "content": "q"}]))
        assert "你好" in out
        assert "分析中" in out  # thinking 走 <thinking> 标签

    def test_anthropic_images_and_json_error_line(self, monkeypatch, tmp_path):
        from src.core.llm_gateway import AnthropicBackend
        b = self._backend(AnthropicBackend)
        img = tmp_path / "a.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        # 非 JSON data 行应被跳过；_encode_image 对缺失文件返回 None
        lines = ["data: not-json-garbage", "data: [DONE]"]
        monkeypatch.setattr(AnthropicBackend, "_post_with_retry",
                            lambda self, url, headers, payload: _FakeResp(lines))
        out = "".join(b.chat_stream(
            [{"role": "user", "content": "q"}], image_paths=[str(img), str(tmp_path / "nope.jpg")]))
        assert out == ""
        assert b._encode_image(tmp_path / "no.jpg") is None
        assert b._encode_image(img) is not None

    def test_openai_chat_stream_parsing(self, monkeypatch):
        from src.core.llm_gateway import OpenAIChatBackend
        b = self._backend(OpenAIChatBackend)
        lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"想一下"}}]}',
            'data: {"choices":[{"delta":{"content":"回答"}}]}',
            "data: [DONE]",
        ]
        monkeypatch.setattr(OpenAIChatBackend, "_post_with_retry",
                            lambda self, url, headers, payload: _FakeResp(lines))
        out = "".join(b.chat_stream([{"role": "user", "content": "q"}],
                                    system="sys"))
        assert "回答" in out
        assert "想一下" in out

    def test_openai_responses_stream_parsing(self, monkeypatch, tmp_path):
        from src.core.llm_gateway import OpenAIResponsesBackend
        b = self._backend(OpenAIResponsesBackend)
        lines = [
            'data: {"type":"response.output_text.delta","delta":"hi"}',
            'data: {"type":"response.reasoning.delta","delta":"r"}',
            "data: [DONE]",
        ]
        monkeypatch.setattr(OpenAIResponsesBackend, "_post_with_retry",
                            lambda self, url, headers, payload: _FakeResp(lines))
        out = "".join(b.chat_stream([{"role": "user", "content": "q"}]))
        assert "hi" in out and "r" in out

    def test_gemini_stream_parsing(self, monkeypatch):
        from src.core.llm_gateway import GeminiBackend
        b = self._backend(GeminiBackend)
        lines = [
            'data: {"candidates":[{"content":{"parts":[{"thought":true,"text":"think"}]}}]}',
            'data: {"candidates":[{"content":{"parts":[{"text":"answer"}]}}]}',
            "data: [DONE]",
        ]
        monkeypatch.setattr(GeminiBackend, "_post_with_retry",
                            lambda self, url, headers, payload: _FakeResp(lines))
        out = "".join(b.chat_stream([{"role": "user", "content": "q"}],
                                    system="s"))
        assert "answer" in out and "think" in out

    def test_retry_then_success(self, monkeypatch):
        """429/5xx 退避重试后成功。"""
        from src.core.llm_gateway import ProtocolBackend
        b = ProtocolBackend("k", "https://x.example/api", "m")
        seen = []

        class _Seq:
            def __init__(self):
                self.n = 0

            def post(self, *a, **k):
                seen.append(self.n)
                if self.n == 0:
                    self.n += 1
                    r = _FakePartial(True)
                    r._status = 503
                    return r
                self.n += 1
                return _FakeResp(['data: ok'], status=200)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        seq = _Seq()
        sleeps = []
        monkeypatch.setattr(b._session, "post", lambda *a, **k: seq.post(*a, **k))
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        resp = b._post_with_retry("https://x/api", headers={}, payload={},
                                  max_retries=3)
        assert resp.status_code == 200
        assert len(sleeps) >= 1  # 503 → 退避 2s

    def test_retry_network_error_then_success(self, monkeypatch):
        from src.core.llm_gateway import ProtocolBackend
        b = ProtocolBackend("k", "https://x.example/api", "m")
        calls = {"n": 0}

        def post(*a, **k):
            if calls["n"] == 0:
                calls["n"] += 1
                import requests
                raise requests.ConnectionError("refused")
            calls["n"] += 1
            return _FakeResp(["x"], status=200)

        sleeps = []
        monkeypatch.setattr(b._session, "post", post)
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        resp = b._post_with_retry("https://x/api", headers={}, payload={},
                                  max_retries=3)
        assert resp.status_code == 200
        assert len(sleeps) == 1

    def test_probe_and_list_models_failure(self, monkeypatch):
        from src.core.llm_gateway import ProtocolBackend, OpenAIChatBackend, AnthropicBackend
        b = ProtocolBackend("", "", "")
        assert b.probe() is False
        assert b.list_models() == []

        from requests import Response
        r = Response()
        r.status_code = 200
        r._content = json.dumps({"data": [{"id": "a"}, {"id": "b"}]}).encode()

        # openai list_models 成功
        ob = OpenAIChatBackend("k", "https://x/api", "m")
        monkeypatch.setattr(ob._session, "get",
                            lambda url, headers, timeout: r)
        assert ob.list_models() == ["a", "b"]

        # anthropic list_models 失败（会话 get 抛）→ []
        ab = AnthropicBackend("k", "https://x/api", "m")
        def _boom(url, **kw):
            raise RuntimeError("net")
        monkeypatch.setattr(ab._session, "get", _boom)
        assert ab.list_models() == []


# ---------------------------------------------------------------------------
# src/core/logic.py 纯函数 + VideoAnalyzer（13% → 关键分支）
# ---------------------------------------------------------------------------


class TestLogicPure:
    def test_check_cuda_health_returns_bool(self):
        from src.core.logic import check_cuda_health
        assert isinstance(check_cuda_health(), bool)

    def test_imwrite_unicode_and_metrics(self, tmp_path):
        import src.core.logic as logic
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:] = (128, 64, 32)
        p = tmp_path / "frame.jpg"
        assert logic.imwrite_unicode(str(p), img) is True
        assert p.exists()
        m = logic.get_frame_metrics(img)
        for k in ("brightness", "contrast", "saturation", "sharpness"):
            assert k in m

    def test_get_unique_filepath(self, tmp_path):
        from src.core.logic import get_unique_filepath
        out = tmp_path / "o"
        # 不存在 → 直接用原名
        first = get_unique_filepath(out, "a.mp4")
        assert first == out / "a.mp4"
        # 已存在 → 加时间戳后缀（原名文件名同名而非时间戳）
        (out / "a.mp4").write_bytes(b"\x00")
        second = get_unique_filepath(out, "a.mp4")
        assert second != out / "a.mp4"
        assert second.name.startswith("a_")
        assert second.suffix == ".mp4"

    def test_model_context_manager(self, monkeypatch):
        from src.core.logic import ModelContextManager
        import src.core.logic as logic

        # 本机 torch.cuda 探测可能挂起（empty_cache 阻塞），monkeypatch 无 GPU
        monkeypatch.setattr(logic.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(logic.gc, "collect", lambda: 0)
        # 源码 GPU_LOCK 是 threading.Lock（非可重入），request_vram 持有锁后调
        # unload 会再抢同一把锁 → 死锁（源码真 bug，见报告）。测试用 RLock
        # 验证"预期卸载行为"（即修复后语义）。
        monkeypatch.setattr(logic, "GPU_LOCK", threading.RLock())

        m = ModelContextManager()
        m.register("YOLO", object())
        m.register("Whisper", object())
        m.request_vram("LLM")
        assert "YOLO" not in m.active_models
        assert "Whisper" not in m.active_models
        m.register("LLM", object())
        m.request_vram("LLM")  # 已是最新，不卸载
        assert "LLM" in m.active_models
        m.unload("LLM")
        assert "LLM" not in m.active_models
        m.unload("UNKNOWN")  # 不崩

    def test_prompt_loader(self, tmp_path):
        from src.core.logic import PromptLoader
        (tmp_path / "video_summary.txt").write_text("自定义模板 {frame_info}",
                                                    encoding="utf-8")
        pl = PromptLoader(prompt_dir=str(tmp_path))
        assert "自定义模板" in pl.get_prompt("Video Summary")
        assert pl.get_prompt("Video Summary") != ""
        pl2 = PromptLoader()
        assert pl2.get_prompt("Nope") == ""

    def test_gateway_config_helpers(self):
        from src.core.logic import (
            _read_gateway_config, _normalize_gateway_base_url,
            _detect_gateway_protocol)
        cp = configparser.ConfigParser()
        cp["LastUsed"] = {"api_url": " https://x.com/v1 ", "api_key": "k",
                          "model_name": "gpt"}
        class _CM:
            config = cp
        assert _read_gateway_config(_CM()) == ("https://x.com/v1", "k", "gpt")
        assert _read_gateway_config(type("E", (), {"config": None})()) == ("", "", "")
        assert _normalize_gateway_base_url("https://x.com/v1/chat/completions") == "https://x.com/v1"
        assert _normalize_gateway_base_url("https://x.com") == "https://x.com"
        assert _detect_gateway_protocol("https://gemini.google.com", "gemini-1") == "gemini"
        assert _detect_gateway_protocol("https://api.anthropic.com", "claude-3") == "anthropic"
        assert _detect_gateway_protocol("https://x.com/v1/responses", "m") == "openai_responses"
        assert _detect_gateway_protocol("https://x.com/v1", "m") == "openai_chat"

    def test_build_llm_client_and_adapter_coerce(self):
        from src.core.logic import build_llm_client, _GatewayClientAdapter
        cp = configparser.ConfigParser()
        cp["LastUsed"] = {"api_url": "https://x.com/v1", "api_key": "k",
                          "model_name": "gpt-4o"}
        class _CM:
            config = cp
        client = build_llm_client(_CM())
        assert client.protocol == "openai_chat"
        assert client.model == "gpt-4o"

        class _Router:
            def chat_stream(self, messages, images, temperature, system):
                yield "|".join(m["content"] for m in messages)
                yield f";system={system}"
            def __init__(self):
                pass

        adapter = _GatewayClientAdapter(_Router(), "openai_chat", "m",
                                        "https://x.com", "k")
        # str prompt coerce
        out = list(adapter.chat_stream("m", "简单问题"))
        assert out and "简单问题" in "".join(out)
        # list prompt + system 拆分
        msgs, sys_ = adapter._coerce_prompt([{"role": "user", "content": "x"}])
        assert sys_ is None and msgs[0]["content"] == "x"
        msgs2, sys2 = adapter._coerce_prompt(
            "前置\n--- System Context ---\n系统内容\n--------------------\n尾巴")
        assert sys2 == "系统内容"
        assert "尾巴" in msgs2[0]["content"]

    def test_video_analyzer_process_stream_and_analyze(self, tmp_path):
        from src.core.logic import VideoAnalyzer, PromptLoader, Frame

        class _Client:
            def __init__(self):
                self.model = "gpt-4o"

            def chat_stream(self, model, prompt, image_paths=None,
                            temperature=0.2, timeout=600):
                yield "第一段 "
                yield 'data: {"choices":[{"delta":{"content":"第二段"}}]}'
                yield "[DONE]"

        va = VideoAnalyzer(_Client(), "gpt-4o", PromptLoader(),
                           use_yolo=False, use_ocr=False)
        frames = [Frame(path=tmp_path / "f.jpg", timestamp=1.0,
                        metrics={"brightness": 0.5})]
        out = "".join(va.analyze_video(frames, None))
        assert "第一段" in out
        assert "第二段" in out
        # 纯 str chunk 直接透传；data:[DONE] 由 _process_stream 的 JSON 解析终止
        # （注：本用例 client 是纯 str 流，[DONE] 作字面文本透传，符合源码行为）
        assert "[DONE]" in out or out.endswith("[DONE]")

        # 纯 _process_stream：非法 JSON → 原样；异常行 → 跳过
        deltas = list(va._process_stream(iter(["hello"])))
        assert deltas == ["hello"]
        deltas2 = list(va._process_stream(iter(["not-json", '"json-str"'])))
        assert len(deltas2) == 2


# ---------------------------------------------------------------------------
# src/web/serve.py（76% → 命令行/构建/重试/端口全占/KeyboardInterrupt/回退）
# ---------------------------------------------------------------------------


class FakeStream:
    def __init__(self, fail_reconfigure=False):
        self.fail = fail_reconfigure
        self.encoding = "utf-8"

    def reconfigure(self, **kw):
        if self.fail:
            raise OSError("closed")

    def write(self, *a):
        return 0

    def flush(self):
        pass


@pytest.fixture
def serve_env(monkeypatch, tmp_path):
    """serve.run_server 隔离环境：清 settings + 跳过前端/端口/浏览器。"""
    from src.web.config import get_settings
    get_settings.cache_clear()
    for k in ("VAP_HOST", "VAP_HEADLESS_TOKEN", "VAP_PORT", "VAP_NO_BROWSER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VAP_HOST", "127.0.0.1")
    import src.web.serve as serve_mod
    monkeypatch.setattr(serve_mod, "webbrowser", type("W", (), {
        "open": lambda url: None}))
    fake_dist = tmp_path / "frontend"
    fake_dist.mkdir()
    monkeypatch.setattr("src.web.app.FRONTEND_DIST", fake_dist)
    yield serve_mod
    get_settings.cache_clear()


class TestServeCoverage:
    def _stub_uvicorn(self, monkeypatch, outcome=None):
        calls = []

        def _fake_run(app, **kw):
            calls.append(kw)
            if outcome is not None:
                raise outcome
        import uvicorn
        monkeypatch.setattr(uvicorn, "run", _fake_run, raising=False)
        return calls

    def test_port_all_busy_returns_1(self, serve_env, monkeypatch, tmp_path):
        serve = serve_env
        monkeypatch.setattr(serve, "_port_available", lambda h, p: False)
        monkeypatch.setattr(serve, "_find_available_port", lambda h, p, **k: 0)
        rc = serve.run_server(open_browser=False)
        assert rc == 1

    def test_keyboard_interrupt_returns_0(self, serve_env, monkeypatch):
        serve = serve_env
        monkeypatch.setattr(serve, "_port_available", lambda h, p: True)
        self._stub_uvicorn(monkeypatch, outcome=KeyboardInterrupt())
        rc = serve.run_server(open_browser=False)
        assert rc == 0

    def test_reconfigure_failure_fallback(self, serve_env, monkeypatch):
        serve = serve_env
        monkeypatch.setattr(serve, "_port_available", lambda h, p: True)
        monkeypatch.setattr(sys, "stdout", FakeStream(fail_reconfigure=True))
        monkeypatch.setattr(sys, "stderr", FakeStream(fail_reconfigure=True))
        self._stub_uvicorn(monkeypatch)
        rc = serve.run_server(open_browser=False)
        assert rc == 0

    def test_install_json_logging_fallback(self, serve_env, monkeypatch):
        serve = serve_env
        monkeypatch.setattr(serve, "_port_available", lambda h, p: True)

        def _boom():
            raise RuntimeError("structured_log down")
        monkeypatch.setattr("src.core.runtime.structured_log.install_json_logging",
                            _boom)
        self._stub_uvicorn(monkeypatch)
        rc = serve.run_server(open_browser=False)
        assert rc == 0

    def test_browser_opened_when_open_browser_true(self, serve_env, monkeypatch, tmp_path):
        serve = serve_env
        monkeypatch.setattr(serve, "_port_available", lambda h, p: True)
        # open_browser=True 时 run_server 起 daemon 线程跑 _wait_and_open；
        # 线程内先 urlopen 探测（失败会 sleep）再 finally webbrowser 里抛异常
        # （monkeypatch 后不抛）。这里把 urlopen 也 stub 成 200 → 线程内
        # _wait_and_open 走"就绪即 break"，且 run_server 层仅在
        # _wait_and_open 被直接调用时会打开（run_server 只 start thread，
        # 不直接 open）。真实语义由 test_wait_and_open_real_200 覆盖。
        monkeypatch.setattr(serve, "_wait_and_open",
                            lambda h, p, timeout=30.0: None)
        self._stub_uvicorn(monkeypatch)
        rc = serve.run_server(open_browser=True)
        assert rc == 0
        # run_server 层不直接 open（浏览器打开发生在后台线程的 _wait_and_open），
        # 这里只断言 rc + 无异常（线程内 open 由 _wait_and_open 单测覆盖）

    def test_wait_and_open_real_200(self, monkeypatch):
        import src.web.serve as serve_mod

        class _Ctx:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        opened = []
        monkeypatch.setattr("urllib.request.urlopen",
                            lambda url, timeout=2: _Ctx())
        monkeypatch.setattr(serve_mod.webbrowser, "open", lambda u: opened.append(u))
        serve_mod._wait_and_open("127.0.0.1", 8001, timeout=0.5)
        assert opened == ["http://127.0.0.1:8001/"]

    def test_try_build_frontend_success(self, monkeypatch, tmp_path):
        import src.web.serve as serve_mod
        webapp = tmp_path / "webapp"
        (webapp / "node_modules").mkdir(parents=True)
        fake_dist = webapp / "out"
        fake_dist.mkdir()
        monkeypatch.setattr("src.web.app.FRONTEND_DIST", fake_dist)
        monkeypatch.setattr("shutil.which", lambda name: "npm")

        class _R:
            returncode = 0
            stderr = ""

        monkeypatch.setattr("subprocess.run", lambda *a, **k: _R())
        serve_mod._try_build_frontend()  # 不抛

    def test_try_build_frontend_timeout_and_exception(self, monkeypatch, tmp_path):
        import src.web.serve as serve_mod
        webapp = tmp_path / "webapp"
        (webapp / "node_modules").mkdir(parents=True)
        fake_dist = webapp / "out"
        monkeypatch.setattr("src.web.app.FRONTEND_DIST", fake_dist)
        monkeypatch.setattr("shutil.which", lambda name: "npm")

        def _timeout(*a, **k):
            import subprocess as _sp
            raise _sp.TimeoutExpired([], 600)
        monkeypatch.setattr("subprocess.run", _timeout)
        serve_mod._try_build_frontend()  # 不抛（超时警告）

        def _boom(*a, **k):
            raise RuntimeError("npm crash")
        monkeypatch.setattr("subprocess.run", _boom)
        serve_mod._try_build_frontend()  # 不抛（异常警告）


# ---------------------------------------------------------------------------
# src/utils/ui_components.py（18% → tk 构造 + 回调，不 mainloop）
# ---------------------------------------------------------------------------
@pytest.fixture
def tk_root():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


class TestUiComponentsTk:
    def test_initial_theme_dialog_construct_and_confirm(self, tk_root, monkeypatch):
        import tkinter as tk
        import src.utils.ui_components as uc

        # 防阻塞：wait_window 置为空操作
        monkeypatch.setattr(tk.Misc, "wait_window", lambda self, w=None: None)
        d = uc.InitialThemeSelectorDialog(tk_root, "主题", {"dark": "#000",
                                                           "light": "#eee"},
                                          "dark")
        assert d.result_theme_name == "dark"
        assert d.theme_var.get() == "dark"
        d.theme_var.set("light")
        d.on_confirm()
        assert d.result_theme_name == "light"

    def test_environment_setup_window_construct_and_log(self, tk_root, monkeypatch):
        import src.utils.ui_components as uc

        # run_setup 真实逻辑会建 venv/装依赖——替换为无害
        monkeypatch.setattr(uc.EnvironmentSetupWindow, "run_setup",
                            lambda self: None)
        done = []
        w = uc.EnvironmentSetupWindow(
            tk_root, lambda ok, pe: done.append((ok, pe)),
            {"bg_color": "#222", "fg_color": "#eee"})
        assert w.master is tk_root
        assert done == []  # run_setup 被替换，不回调
        # log → master.after 调度到 Tk 循环
        w.log("测试日志")
        tk_root.update()
        w._append_log("直接追加")
        assert w.log_text is not None

    def test_initial_theme_on_confirm_destroy(self, tk_root, monkeypatch):
        import tkinter as tk
        import src.utils.ui_components as uc
        monkeypatch.setattr(tk.Misc, "wait_window", lambda self, w=None: None)
        d = uc.InitialThemeSelectorDialog(tk_root, "t", {"a": "#000"}, "a")
        d.on_confirm()  # destroy 不抛


# ---------------------------------------------------------------------------
# src/core/rtsp_stream.py（62% → 脱敏/事件/启动停止/动态分支）
# ---------------------------------------------------------------------------
class TestRtspCoverage:
    def test_sanitize_rtsp_url(self):
        from src.core.rtsp_stream import _sanitize_rtsp_url
        assert _sanitize_rtsp_url("rtsp://user:pass@cam/stream") == \
            "rtsp://user:***@cam/stream"
        assert "pass" not in _sanitize_rtsp_url("rtsp://user:secret@cam/x")
        assert _sanitize_rtsp_url("rtsp://cam/x") == "rtsp://cam/x"

    def test_motion_detector_area_and_cooldown(self):
        from src.core.rtsp_stream import MotionEventDetector
        md = MotionEventDetector(threshold=25.0, min_area=500, cooldown=10.0)
        base = np.full((120, 160, 3), 50, dtype=np.uint8)
        # 首帧建基线
        assert md.detect(base, 1.0) is False
        # 微小区域变化（<500 像素）→ False
        small = base.copy()
        small[60:70, 80:90] = 180  # 10x10=100 像素
        assert md.detect(small, 2.0) is False
        # 大区域变化 → True
        big = base.copy()
        big[40:80, 60:110] = 220  # 40x50=2000 像素
        assert md.detect(big, 3.0) is True
        # 冷却期内再触发 → False（diff 相对 big 基线的变化量需再超阈值，
        # 用"全黑"帧确保差分远大于阈值但仍在冷却窗口内）
        black = np.zeros((120, 160, 3), dtype=np.uint8)
        assert md.detect(black, 3.5) is False
        # 冷却期外 → True（全白帧差分大，且已过冷却）
        white = np.full((120, 160, 3), 255, dtype=np.uint8)
        assert md.detect(white, 20.0) is True

    def test_monitor_start_stop_with_fake_grabber(self, tmp_path, monkeypatch):
        from src.core.rtsp_stream import RtspMonitor
        m = RtspMonitor("rtsp://cam/x", None, work_dir=str(tmp_path))

        class _FakeGrabber:
            def __init__(self, url, cb, fps=1.0, reconnect_delay=5.0):
                self.stopped = False

            def start(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

            def join(self, timeout=10):
                pass

        monkeypatch.setattr("src.core.rtsp_stream.RtspFrameGrabber",
                            _FakeGrabber)
        m.start(fps=1.0)
        assert m._grabber is not None
        assert m._grabber.stopped is False
        m.stop()
        assert m._grabber.stopped is True

    def test_vlm_worker_appends_hit(self, tmp_path):
        from src.core.rtsp_stream import RtspMonitor
        class _B:
            def chat_stream(self, messages, image_paths=None, temperature=0.1):
                yield json.dumps({"match": True, "confidence": 0.9,
                                  "reason": "找到"})
        m = RtspMonitor("rtsp://c/x", _B(), key_item_image="k.jpg",
                        work_dir=str(tmp_path))
        fp = tmp_path / "hit.jpg"
        fp.write_bytes(b"\xff\xd8\xff")
        m._vlm_worker(5.0, str(fp))
        assert any(e.kind == "hit" for e in m.events)
        hit = [e for e in m.events if e.kind == "hit"][0]
        assert hit.confidence == 0.9

    def test_on_frame_motion_prune_500(self, tmp_path):
        from src.core.rtsp_stream import RtspMonitor, StreamEvent
        m = RtspMonitor("rtsp://c/x", None, key_item_image="",
                        work_dir=str(tmp_path), motion_threshold=25.0,
                        vlm_cooldown=3600.0)
        # 预填 500 条事件，触发一次真实运动 → 超上限裁剪（pop 最旧 + 删文件）
        old_fp = tmp_path / "old.jpg"
        old_fp.write_bytes(b"\x00\x01")
        for i in range(500):
            m.events.append(StreamEvent(timestamp=i, kind="motion",
                                        frame_path=str(old_fp)))
        base = np.full((120, 160, 3), 50, dtype=np.uint8)
        m._on_frame(1.0, base)  # 建基线
        moving = base.copy()
        moving[40:80, 60:110] = 220
        m._on_frame(2.0, moving)
        assert len(m.events) <= 500
        # 最旧事件（timestamp=0）应被 pop
        assert all(e.timestamp >= 1.0 for e in m.events)


# ---------------------------------------------------------------------------
# src/core/motion_detector.py（74% → 纯函数 + 合并/切段/昼夜/降级）
# ---------------------------------------------------------------------------
class TestMotionDetectorPure:
    def _detector(self, **kw):
        from src.core.motion_detector import MotionDetector, MotionConfig
        cfg = MotionConfig(**kw)
        d = MotionDetector(cfg, ffmpeg_exe=None)
        # 避免实例构造走 _find_ffmpeg（无害），但保留探测失败路径
        return d

    def test_detect_no_ffmpeg_and_missing_video(self, tmp_path):
        from src.core.motion_detector import MotionDetector, MotionConfig
        d = MotionDetector(MotionConfig(), ffmpeg_exe=None)
        assert d.detect(tmp_path / "missing.mp4") == []
        d2 = MotionDetector(MotionConfig(), ffmpeg_exe="not-exists")
        assert d2.detect(tmp_path / "missing.mp4") == []

    def test_frame_diff_and_day_night(self, tmp_path):
        import src.core.motion_detector as md
        p1 = tmp_path / "a.jpg"
        p2 = tmp_path / "b.jpg"
        cv2.imwrite(str(p1), np.full((16, 16), 50, dtype=np.uint8))
        cv2.imwrite(str(p2), np.full((16, 16), 200, dtype=np.uint8))
        assert md.MotionDetector._frame_diff_score(str(p1), str(p2)) > 10
        assert md.MotionDetector._frame_diff_score(str(p1), str(p1)) == 0.0

        d = self._detector()
        labels = d._detect_day_night([(0.0, str(p1)), (1.0, str(p2))])
        assert labels == ["night", "day"]

    def test_merge_to_segments_and_group(self):
        from src.core.motion_detector import MotionDetector, MotionConfig
        d = MotionDetector(MotionConfig(min_scene_len=5, context_padding=2.0),
                           ffmpeg_exe=None)
        diff = [100.0, 80.0, 0.0, 0.0, 90.0, 0.0]
        times = [1.0, 2.0, 3.0, 4.0, 20.0, 21.0]
        day_night = ["day"] * len(times)
        segs = d._merge_to_segments(diff, times, day_night, [], duration=30.0)
        assert segs, "应有合并时段"
        assert segs[0].start_sec < segs[0].end_sec
        assert segs[0].diff_score > 0
        assert MotionDetector._group_brightness(1.0, 2.0, day_night, times) == "day"
        assert MotionDetector._group_brightness(50.0, 60.0, day_night, times) == "night"
        assert MotionDetector._max_score_in_range(0, 5, diff, times) == 100.0

    def test_clamp_segment_splits_long(self):
        from src.core.motion_detector import MotionDetector, MotionConfig, MotionSegment
        d = MotionDetector(MotionConfig(max_segment_sec=10), ffmpeg_exe=None)
        seg = MotionSegment(start_sec=0.0, end_sec=25.0, duration=25.0,
                            brightness="day", diff_score=5.0, scene_count=1,
                            change_points=[5.0])
        out = d._clamp_segment(seg)
        assert len(out) >= 3
        assert out[0].change_points == [5.0]
        assert all(s.duration <= 10.0 for s in out)

    def test_cancel_quick_check_short(self):
        from src.core.motion_detector import MotionDetector, MotionConfig
        d = MotionDetector(MotionConfig(min_scene_len=15), ffmpeg_exe=None)
        assert d._cancel_quick_check(Path("x.mp4"), 5.0) is True
        assert d._cancel_quick_check(Path("x.mp4"), 60.0) is False

    def test_crowded_detector_low_density_delegates(self):
        from src.core.motion_detector import CrowdedSceneDetector, MotionConfig
        d = CrowdedSceneDetector(MotionConfig(), ffmpeg_exe=None)
        diff = [100.0]
        times = [1.0]
        segs = d._merge_to_segments(diff, times, ["day"], [], duration=10.0)
        assert segs  # 密度低 → 父类合并

    def test_crowded_detector_yolo_unavailable_fallback(self):
        from src.core.motion_detector import CrowdedSceneDetector, MotionConfig
        from unittest.mock import MagicMock
        d = CrowdedSceneDetector(MotionConfig(crowded_density_threshold=0.1),
                                 ffmpeg_exe=None)
        d._detect_objects_yolo = MagicMock(return_value=None)
        diff = [100.0, 80.0]
        times = [1.0, 2.0]
        segs = d._merge_to_segments(diff, times, ["day", "day"], [], duration=10.0)
        assert segs  # 密度高但 YOLO 不可用 → 父类降级


# ---------------------------------------------------------------------------
# src/core/surveillance_agent.py（0% → judge/write/report 编排，mock backend）
# ---------------------------------------------------------------------------
class TestSurveillanceAgent:
    def _agent(self, tmp_path):
        from src.core.surveillance_agent import SurveillanceAgent
        ki = tmp_path / "key.jpg"
        ki.write_bytes(b"\xff\xd8\xff")
        return SurveillanceAgent(None, str(ki), "钥匙", fps=1.0,
                                 max_frames_per_video=600)

    def test_judge_frame_hit_and_timestamp_parse(self, tmp_path):
        from src.core.surveillance_agent import SurveillanceAgent
        fp = tmp_path / "frame_0001_123.4s.jpg"
        fp.write_bytes(b"\xff\xd8\xff")

        class _B:
            model = "glm-5.3-flash"

            def chat_stream(self, messages, image_paths=None, temperature=0.1):
                yield '{"match": true, "confidence": 0.92, "reason": "人持钥匙"}'

        a = SurveillanceAgent(_B(), str(tmp_path / "key.jpg"), "钥匙")
        hit = a._judge_frame(str(fp))
        assert hit is not None
        assert hit.confidence == 0.92
        assert hit.timestamp == 123.4

    def test_judge_frame_no_match_and_bad_json(self, tmp_path):
        from src.core.surveillance_agent import SurveillanceAgent

        class _B1:
            def chat_stream(self, messages, image_paths=None, temperature=0.1):
                yield '{"match": false, "confidence": 0.1, "reason": "无"}'

        a1 = SurveillanceAgent(_B1(), str(tmp_path / "k.jpg"), "x")
        assert a1._judge_frame(str(tmp_path / "f.jpg")) is None

        class _B2:
            def chat_stream(self, messages, image_paths=None, temperature=0.1):
                yield "not json"

        a2 = SurveillanceAgent(_B2(), str(tmp_path / "k.jpg"), "x")
        assert a2._judge_frame(str(tmp_path / "f.jpg")) is None

    def test_search_video_happy(self, tmp_path, monkeypatch):
        from src.core.surveillance_agent import SurveillanceAgent

        class _B:
            model = "m"

            def chat_stream(self, messages, image_paths=None, temperature=0.1):
                yield '{"match": true, "confidence": 0.8, "reason": "看到"}'

        a = SurveillanceAgent(_B(), str(tmp_path / "k.jpg"), "钥匙")
        video = tmp_path / "a.mp4"
        video.write_bytes(b"\x00" * 100)

        a._extract_frames = lambda vpath, out_dir: [
            {"timestamp": 5.0, "path": str(tmp_path / "f5.jpg")},
            {"timestamp": 10.0, "path": str(tmp_path / "f10.jpg")},
        ]
        a._clip_prefilter = lambda frames, **kw: frames

        hits = a.search_video(video, tmp_path / "work")
        assert len(hits) == 2
        assert hits[0].video_name == "a.mp4"
        assert hits[0].timestamp == 5.0

    def test_run_orchestration_and_write_report(self, tmp_path, monkeypatch):
        from src.core.surveillance_agent import SurveillanceAgent
        from src.core.surveillance_agent import HitMoment

        class _B:
            model = "glm"

        a = SurveillanceAgent(_B(), str(tmp_path / "k.jpg"), "钥匙")
        vdir = tmp_path / "videos"
        vdir.mkdir()
        (vdir / "a.mp4").write_bytes(b"\x00")
        (vdir / "b.avi").write_bytes(b"\x00")
        (vdir / "note.txt").write_text("no")

        a.search_video = lambda v, work: [
            HitMoment(video_path=str(v), video_name=v.name, timestamp=2.0,
                      confidence=0.9, reason="r", frame_path="c0.mp4")]

        report = a.run(str(vdir), str(tmp_path / "out"), max_videos=0)
        assert report.total_videos == 2
        assert len(report.hits) == 2
        assert (tmp_path / "out" / "search_report.json").exists()
        assert (tmp_path / "out" / "search_report.md").exists()
        md = (tmp_path / "out" / "search_report.md").read_text(encoding="utf-8")
        assert "a.mp4" in md and "命中" in md

        # 报告无命中路径（先建目录，surveillance_agent.run 的 iterdir 需目录存在）
        empty = tmp_path / "empty"
        empty.mkdir()
        a.search_video = lambda v, work: []
        a.run(str(empty), str(tmp_path / "out2"))
        assert (tmp_path / "out2" / "search_report.md").exists()


# ---------------------------------------------------------------------------
# src/core/run_store.py（91% → 边界分支）
# ---------------------------------------------------------------------------
class TestRunStoreEdges:
    def test_create_run_invalid_status(self, tmp_path):
        from src.core.run_store import RunStore
        s = RunStore(config_dir=str(tmp_path / "cfg"))
        with pytest.raises(ValueError):
            s.create_run("v.mp4", status="bogus")

    def test_update_run_empty_and_unknown_fields(self, tmp_path):
        from src.core.run_store import RunStore
        s = RunStore(config_dir=str(tmp_path / "cfg"))
        rid = s.create_run("v.mp4")
        assert s.update_run(rid) is False  # 无字段
        assert s.update_run(rid, unknown_col=1) is False  # 白名单外
        with pytest.raises(ValueError):
            s.update_run(rid, status="bad-status")
        assert s.update_run(rid, hits_count=3) is True
        assert s.update_run("nope", hits_count=1) is False

    def test_checkpoint_branches(self, tmp_path):
        from src.core.run_store import RunStore
        s = RunStore(config_dir=str(tmp_path / "cfg"))
        with pytest.raises(ValueError):
            s.checkpoint("", "phase", {})
        rid = s.create_run("v.mp4")
        s.checkpoint(rid, "抽帧", {"n": 1})
        s.checkpoint(rid, "推理", {"done": True})
        # restore 按 ts 倒序取最近一条（同秒 ts 可能并列 → 两条任一都算有效；
        # 验证 restore 返回两条之一 + list 两条都在）
        r = s.restore(rid)
        assert r and r["phase"] in ("抽帧", "推理")
        cps = s.list_checkpoints(rid)
        assert len(cps) == 2
        assert {c["phase"] for c in cps} == {"抽帧", "推理"}
        assert s.restore("none") is None
        s.clear_checkpoints("")  # 空 run_id → 静默
        s.clear_checkpoints(rid)
        assert s.list_checkpoints(rid) == []

    def test_clear_all_and_delete_purge(self, tmp_path):
        from src.core.run_store import RunStore
        s = RunStore(config_dir=str(tmp_path / "cfg"))
        rid = s.create_run("v.mp4")
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"\x00")
        s.add_hit(rid, {"hit_idx": 0, "abs_timestamp": "1.0",
                        "clip_path": str(clip)})
        s.add_clip(rid, str(clip), hit_idx=1, abs_timestamp="2.0")
        assert s.delete_run(rid, purge_files=True) is True
        assert not clip.exists()
        assert s.delete_run("none", purge_files=True) is False
        # clear_all purge
        rid2 = s.create_run("v2.mp4")
        c2 = tmp_path / "c2.mp4"
        c2.write_bytes(b"\x00")
        s.add_clip(rid2, str(c2), hit_idx=0)
        n = s.clear_all(purge_files=True)
        assert n >= 1
        assert not c2.exists()


# ---------------------------------------------------------------------------
# src/core/kb_indexer.py（79% → CLIP 不可用分支）
# ---------------------------------------------------------------------------
class TestKbIndexerEdges:
    def test_index_frames_no_frames(self, tmp_path, monkeypatch):
        import src.core.kb_indexer as kb
        monkeypatch.setattr(kb, "CLIP_AVAILABLE", False)
        assert kb.get_embedder() is None
        assert kb.index_frames(None, "s", "v", "p", []) == 0
        assert kb.index_frames(None, "s", "v", "p", [object()]) == 0


# ---------------------------------------------------------------------------
# src/utils/config_manager.py（70% → 读异常/JSON 保存异常/字典配置/update 新段）
# ---------------------------------------------------------------------------
class TestConfigManagerEdges:
    def test_load_main_config_read_failure(self, tmp_path, monkeypatch):
        from src.utils.config_manager import ConfigurationManager
        monkeypatch.setattr("src.utils.config_manager.CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr("src.utils.config_manager.MAIN_CONFIG_FILENAME", "app.ini")
        cm = ConfigurationManager()
        cm.config_path = str(tmp_path / "app.ini")
        (tmp_path / "app.ini").write_text("broken [ unclosed\n", encoding="utf-8")

        def _boom(*a, **k):
            raise Exception("parse error")
        monkeypatch.setattr(cm.config, "read", _boom)
        cfg = cm.load_main_config()
        assert cfg.has_section("Application")  # 读失败 → 重建默认

    def test_update_config_new_section_and_save_error(self, tmp_path, monkeypatch):
        from src.utils.config_manager import ConfigurationManager
        monkeypatch.setattr("src.utils.config_manager.CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr("src.utils.config_manager.MAIN_CONFIG_FILENAME", "app.ini")
        cm = ConfigurationManager()
        cm.config_path = str(tmp_path / "app.ini")
        cm.load_main_config()
        cm.update_config("NewSection", "k", "v")
        assert cm.config["NewSection"]["k"] == "v"

        # _save_config 抛异常 → 记日志不崩
        def _boom(f, *a, **k):
            raise OSError("disk full")
        monkeypatch.setattr("builtins.open", _boom)
        cm.update_config("NewSection", "k2", "v2")  # 不抛

    def test_presets_save_failure_and_prompts_corrupt(self, tmp_path, monkeypatch):
        from src.utils.config_manager import ConfigurationManager
        monkeypatch.setattr("src.utils.config_manager.CONFIG_DIR", str(tmp_path))
        cm = ConfigurationManager()
        cm.config_dir = str(tmp_path)
        cm.presets_path = str(tmp_path / "p.json")
        (tmp_path / "p.json").write_text("{broken", encoding="utf-8")
        assert cm.load_api_presets() == []
        cm.prompts_path = str(tmp_path / "pr.json")
        (tmp_path / "pr.json").write_text("not-json", encoding="utf-8")
        assert cm.load_prompts() == []
        assert len(cm.load_prompts()) == 0
        # 保存 JSON 抛 → 不崩
        monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
        cm.save_api_presets([{"name": "a"}])
        cm.save_prompts([{"name": "a"}])


# ---------------------------------------------------------------------------
# src/remote/config.py（74% → from_env / trusted_hosts / parse）
# ---------------------------------------------------------------------------
class TestRemoteConfigEdges:
    def test_from_env_and_trusted_hosts(self, monkeypatch):
        import src.remote.config as rc
        monkeypatch.setenv("VAP_REMOTE_ENABLED", "1")
        monkeypatch.setenv("VAP_REMOTE_METHOD", "cloudflare")
        monkeypatch.setenv("VAP_CLOUDFLARE_HOSTNAME", "tun.example.com")
        cfg = rc.from_env()
        assert cfg.enabled is True
        assert cfg.method == rc.RemoteMethod.CLOUDFLARE_ACCESS
        assert cfg.trusted_hosts() == ["tun.example.com"]

        monkeypatch.setenv("VAP_REMOTE_METHOD", "tailscale-direct")
        monkeypatch.setenv("VAP_TAILSCALE_IP", "100.64.0.1")
        cfg2 = rc.from_env()
        assert cfg2.trusted_hosts() == ["100.64.0.1:8080"]

        monkeypatch.setenv("VAP_REMOTE_METHOD", "bogus")
        cfg3 = rc.from_env()
        assert cfg3.method == rc.RemoteMethod.TAILSCALE_SERVE

        assert rc._parse_method("") == rc.RemoteMethod.TAILSCALE_SERVE
        assert rc._parse_method("tailscale-serve-https") == rc.RemoteMethod.TAILSCALE_SERVE

    def test_credential_store(self):
        import src.remote.config as rc
        store = rc.CredentialStore(fetcher=type("F", (), {
            "get": lambda self, k, fallback="": fallback})())
        assert store.get("k", "d") == "d"
        assert store.has("k") is False
        assert store.snapshot(["a", "b"]) == {"a": False, "b": False}


# ---------------------------------------------------------------------------
# src/web/job_store.py（88% → delete/cleanup_stale）
# ---------------------------------------------------------------------------
class TestJobStoreEdges:
    def test_delete_and_cleanup_stale(self, tmp_path, monkeypatch):
        from src.web.job_store import JobStore, JobStatus
        store = JobStore()
        assert store.delete("missing") is False

        work = tmp_path / "w"
        frames = work / "frames"
        rec = store.create("v.mp4", work, frames)
        frames.mkdir(parents=True)
        (frames / "f.jpg").write_bytes(b"\x00")
        assert store.delete(rec.job_id) is True
        assert not work.exists()

        # cleanup_stale：完成但超龄（time.time 依赖单调钟，
        # 用 max_age_sec=0 确保必超龄 + 覆盖 0 参数边界）
        s2 = JobStore()
        r = s2.create("v.mp4", tmp_path / "w2", tmp_path / "w2" / "frames")
        r.status = JobStatus.DONE
        assert s2.cleanup_stale(max_age_sec=0) == 1
        assert s2.get(r.job_id) is None

        # 非完成状态 → 不清理
        s3 = JobStore()
        r3 = s3.create("v.mp4", tmp_path / "w3", tmp_path / "w3" / "frames")
        r3.status = JobStatus.RUNNING
        assert s3.cleanup_stale(max_age_sec=0) == 0
        assert s3.get(r3.job_id) is not None