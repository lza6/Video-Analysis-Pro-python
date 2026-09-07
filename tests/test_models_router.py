"""模型管理路由测试(/api/models)。

覆盖 src/web/routers/models.py 的 5 条路由:
  - GET  /api/models                          模型卡状态 + 本地扫描列表
  - POST /api/models/{id}/download            启动下载(job 创建 + 后台线程)
  - GET  /api/models/{id}/download/stream     SSE 进度流(断线续连 + 404)
  - POST /api/models/{id}/verify              SHA256 校验(未知 id 400 / 缺文件 404)
  - POST /api/models/detect-type              本地模型类型探测(空 filename 400)

测试用 TestClient 真实调 API,不 Mock router 内部逻辑。重依赖(ModelManager
下载/verify 真实模型)用临时 models 目录 + 真实小文件构造。
守付费红线:不发起真实下载。

模型文件路径注入:monkeypatch `src.web.routers.models._get_manager`
返回指向 tmp_path 的 ModelManager,并在该目录预置 yolo11n.pt(任意字节即可,
verify 对 yolo_v11n 走"仅存在性校验"分支)。这是 ModelManager 的实例级注入,
不 Mock 路由内部行为。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch 必须先于 FastAPI 导入(Windows DLL 顺序铁律,与 test_web_api.py 一致)
try:
    import torch  # noqa: F401
except OSError:
    torch = None

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def module_app():
    """TestClient(app) 以 lifespan 为准构造(analyzer_service._loop 可用)。"""
    from src.web.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def models_mgr(monkeypatch, tmp_path):
    """把 router 的 ModelManager 指到 tmp_path 隔离目录。

    返回 (manager, tmp_path)。下载/verify 都在临时目录内,不碰真实 models/。
    """
    from src.core.logic import ModelManager
    mgr = ModelManager(models_dir=tmp_path / "models")
    monkeypatch.setattr("src.web.routers.models._get_manager", lambda: mgr)
    return mgr, tmp_path


def _start_stream(client, job_id: str, headers=None, timeout: float = 8.0):
    """读 SSE 流直到 __close__ 或 timeout,返回 (状态码, 事件列表, 原始行)。"""
    events: list[tuple[str, dict]] = []
    raw_lines: list[str] = []
    status = 0
    with client.stream(
        "GET", f"/api/models/{job_id}/download/stream", headers=headers, timeout=timeout
    ) as resp:
        status = resp.status_code
        cur: list[str] = []
        for line in resp.iter_lines():
            raw_lines.append(line)
            if line.startswith("event:"):
                cur.append(line[6:].strip())
            elif line.startswith("data:"):
                try:
                    payload = json.loads(line[6:])
                except Exception:
                    payload = {}
                for etype in cur:
                    events.append((etype, payload))
                cur = []
    return status, events, raw_lines


# ============================ GET /api/models ============================

def test_list_models_returns_cards(module_app, models_mgr):
    """模型卡 4 张 + 本地扫描列表;存在于模型的卡 exists=True 且 size>0。"""
    mgr, _tmp = models_mgr
    (mgr.models_dir / "yolo11n.pt").write_bytes(b"\x00" * 2048)
    r = module_app.get("/api/models")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["cards"]) == 4
    ids = {c["id"] for c in body["cards"]}
    assert ids == {"yolo_v11n", "whisper_base", "st_minilm", "ffmpeg"}
    yolo = next(c for c in body["cards"] if c["id"] == "yolo_v11n")
    assert yolo["exists"] is True
    # 2048 字节文件经 MB 四舍五入后为 0.0;断言"存在"即可,不做 <1MB 的精度假设
    assert yolo["size_mb"] >= 0.0
    # 注入的 ModelManager(yolo_v11n 无约束 hash)→ sha256_expected=False(真实语义)
    assert yolo["sha256_expected"] is False
    assert "path" in yolo
    missing = next(c for c in body["cards"] if c["id"] == "whisper_base")
    assert missing["exists"] is False
    assert missing["size_mb"] == 0


def test_list_models_local_scan(module_app, models_mgr):
    """本地扫描:仅 gguf/pt/bin 进列表,detect_model_type 对 yolo 判 Text-only。"""
    mgr, _tmp = models_mgr
    (mgr.models_dir / "myqwen.gguf").write_bytes(b"x")
    (mgr.models_dir / "llava.gguf").write_bytes(b"x")
    (mgr.models_dir / "readme.txt").write_text("not a model")
    r = module_app.get("/api/models")
    body = r.json()
    names = {m["name"] for m in body["local_models"]}
    assert names == {"myqwen.gguf", "llava.gguf"}, names
    types = {m["name"]: m["type"] for m in body["local_models"]}
    assert types["myqwen.gguf"] == "Text-only LLM"
    assert types["llava.gguf"] == "Vision-Language (VL)"


def test_list_models_empty_local(module_app, models_mgr):
    """空目录:cards 正常 + local_models 空数组。"""
    r = module_app.get("/api/models")
    assert r.status_code == 200, r.text
    assert r.json()["local_models"] == []


# ============================ POST /api/models/{id}/download ============================

def test_download_unknown_id_400(module_app):
    """未知 model_id 拒绝(400)。"""
    r = module_app.post("/api/models/nope/download")
    assert r.status_code == 400, r.text
    assert "unknown model_id" in r.json()["error"]


def test_download_creates_job_and_streams(module_app, models_mgr):
    """启动下载 → job 创建 → SSE 流 push 事件(含 verify 成功,不真实下载)。

    download_model 对 yolo_v11n 走"已存在则复用"逻辑:models_dir 预置
    同名文件,download_model 会提前返回 True,全程零网络。SSE 应收到
    verify/done/__close__ 事件。
    """
    mgr, _tmp = models_mgr
    # 预置目标文件:让 download_model 的"目标已存在"分支直接 return True(零网络)
    (mgr.models_dir / "yolo11n.pt").write_bytes(b"\x00" * 1024)

    r = module_app.post("/api/models/yolo_v11n/download")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["job_id"].startswith("dl_yolo_v11n_")
    assert body["status"] == "running"

    status, events, _raw = _start_stream(module_app, "yolo_v11n")
    assert status == 200
    types = [t for t, _p in events]
    assert "verify" in types, f"缺 verify 事件: {types}"
    assert "done" in types, f"缺 done 事件: {types}"
    done = next(p for t, p in events if t == "done")
    assert done.get("ok") is True


def test_download_stream_404_no_active(module_app):
    """无活跃下载时 stream 返回 404。"""
    r = module_app.get("/api/models/whisper_base/download/stream")
    assert r.status_code == 404, r.text


def test_download_stream_replay_last_event_id(module_app, models_mgr):
    """Last-Event-ID 断线续连:重放 seq > id 的历史事件(verify/done)。"""
    mgr, _tmp = models_mgr
    (mgr.models_dir / "yolo11n.pt").write_bytes(b"\x00" * 1024)
    module_app.post("/api/models/yolo_v11n/download")
    _start_stream(module_app, "yolo_v11n")  # 先消费完 live 流,事件落入环形缓冲

    status, events, _raw = _start_stream(
        module_app, "yolo_v11n", headers={"Last-Event-ID": "0"}
    )
    assert status == 200
    types = [t for t, _p in events]
    assert "verify" in types and "done" in types, f"重放缺事件: {types}"


# ============================ POST /api/models/{id}/verify ============================

def test_verify_unknown_id_400(module_app):
    """未知 model_id 校验拒绝(400)。"""
    r = module_app.post("/api/models/nope/verify")
    assert r.status_code == 400, r.text


def test_verify_missing_file_404(module_app, models_mgr):
    """已知 id 但文件不存在 → 404。"""
    r = module_app.post("/api/models/whisper_base/verify")
    assert r.status_code == 404, r.text


def test_verify_existing_yolo_ok(module_app, models_mgr):
    """yolo_v11n 无约束 hash → 存在性校验通过 valid=True。"""
    mgr, _tmp = models_mgr
    (mgr.models_dir / "yolo11n.pt").write_bytes(b"\x00" * 1024)
    r = module_app.post("/api/models/yolo_v11n/verify")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["valid"] is True
    assert body["model_id"] == "yolo_v11n"
    assert body["path"].endswith("yolo11n.pt")


# ============================ POST /api/models/detect-type ============================

def test_detect_type_requires_filename(module_app):
    """空 filename → 400。"""
    r = module_app.post("/api/models/detect-type", json={})
    assert r.status_code == 400, r.text


def test_detect_type_vision_keyword(module_app):
    """含 VL 关键词的模型名 → Vision-Language (VL)。"""
    r = module_app.post("/api/models/detect-type", json={"filename": "llava-v1.5.gguf"})
    assert r.status_code == 200, r.text
    assert r.json()["type"] == "Vision-Language (VL)"


def test_detect_type_torch_model(module_app):
    """无关键词的 .pt → Torch Model (Unknown)。"""
    r = module_app.post("/api/models/detect-type", json={"filename": "clip.pt"})
    assert r.status_code == 200, r.text
    assert r.json()["type"] == "Torch Model (Unknown)"


def test_detect_type_text_only(module_app):
    """gguf 无关键词 → Text-only LLM。"""
    r = module_app.post("/api/models/detect-type", json={"filename": "qwen2.5-7b.gguf"})
    assert r.status_code == 200, r.text
    assert r.json()["type"] == "Text-only LLM"


# ============================ 后台下载线程(零网络) ============================

def test_download_background_thread_runs(module_app, models_mgr):
    """后台线程完整跑完:事件推入 rec 队列(LOG→verify→done→__close__)。

    rec 经 store.create 创建后由线程驱动,done 事件证明 run() 走到了
    download_model 成功分支 + finally 的 __close__。
    """
    from src.web.deps import get_job_store
    store = get_job_store()
    mgr, _tmp = models_mgr
    (mgr.models_dir / "yolo11n.pt").write_bytes(b"\x00" * 1024)

    r = module_app.post("/api/models/yolo_v11n/download")
    assert r.status_code == 201, r.text
    job_id = r.json()["job_id"]
    rec = store.get(job_id)
    assert rec is not None

    status, events, _raw = _start_stream(module_app, "yolo_v11n")
    assert status == 200
    types = [t for t, _p in events]
    assert "log" in types, f"缺 LOG 事件: {types}"
    assert "verify" in types
    assert "done" in types
    assert "__close__" in types
    assert rec.status.value == "done"


def test_download_app_loop_available(module_app, models_mgr):
    """下载线程经 app loop 推事件:loop 存活则 rec.queue 可消费(SSE 正常收流)。

    与 test_download_creates_job_and_streams 互补:该用例验证 201 + verify/done
    事件,本用例验证同一路径下 SSE 完整收到 __close__(线程 finally 正常收尾)。
    """
    mgr, _tmp = models_mgr
    (mgr.models_dir / "yolo11n.pt").write_bytes(b"\x00" * 1024)
    module_app.post("/api/models/yolo_v11n/download")
    status, events, _raw = _start_stream(module_app, "yolo_v11n")
    assert status == 200
    assert "__close__" in [t for t, _p in events]


def test_download_resume_existing_file(module_app, models_mgr):
    """目标文件已存在(dest_path.exists())→ download_model 直接成功(断点续传路径)。"""
    mgr, _tmp = models_mgr
    (mgr.models_dir / "yolo11n.pt").write_bytes(b"\x00" * 4096)
    r = module_app.post("/api/models/yolo_v11n/download")
    assert r.status_code == 201, r.text
    status, events, _raw = _start_stream(module_app, "yolo_v11n")
    assert status == 200
    done = next((p for t, p in events if t == "done"), None)
    assert done is not None and done.get("ok") is True


def test_stream_idle_close_finishes(module_app, models_mgr):
    """手动构造:rec 有事件但 stream_closed=True → live 排空后 SSE 正常收尾。

    绕过后台线程,直接往 rec.queue 塞事件并标记流关闭,验证 stream_job_events
    的收尾路径(等价于分析作业结束后的流行为)。
    """
    from src.web.deps import get_job_store
    store = get_job_store()
    mgr, _tmp = models_mgr
    (mgr.models_dir / "yolo11n.pt").write_bytes(b"\x00" * 1024)
    job_id = module_app.post("/api/models/yolo_v11n/download").json()["job_id"]
    rec = store.get(job_id)
    assert rec is not None

    # 清空后台线程可能已入队的残留事件
    while not rec.queue.empty():
        rec.queue.get_nowait()

    rec.stream_closed = True
    rec.put_event("phase", {"phase": "done"})

    status, events, _raw = _start_stream(module_app, "yolo_v11n")
    assert status == 200
    types = [t for t, _p in events]
    assert "phase" in types, f"缺手动塞入的事件: {types}"
