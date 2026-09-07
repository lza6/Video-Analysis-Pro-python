"""Web 后端 API 测试(FastAPI TestClient)。

覆盖:
  - /api/health 能力矩阵
  - /api/analyze 创建作业 + SSE 流式(本机路径模式,跳过音频加速)
  - /api/jobs 列表 / 详情 / 帧图片
  - 路径遍历防护(../ 逃逸、非白名单扩展名)
  - Bearer Token 鉴权(配置 token 后未授权请求 401)

测试用真实小视频(ffmpeg 生成 3s testsrc),不依赖付费 API。
LLM 阶段在测试环境 Ollama 不可达时优雅降级(报告含 "LLM 阶段不可用"),
这是预期行为,不视为失败。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch 必须先于 FastAPI 导入(Windows DLL 顺序铁律,与 Qt 无关)
try:
    import torch  # noqa: F401
except OSError:
    torch = None

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def test_video(tmp_path_factory) -> Path:
    """生成 3 秒测试视频(ffmpeg testsrc)。无 ffmpeg 则跳过整模块。"""
    import imageio_ffmpeg
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pytest.skip("imageio-ffmpeg 不可用")
    out = tmp_path_factory.mktemp("videos") / "smoke.mp4"
    r = subprocess.run(
        [exe, "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        capture_output=True,
    )
    if r.returncode != 0 or not out.exists():
        pytest.skip("ffmpeg 生成测试视频失败")
    return out


@pytest.fixture(scope="module")
def client():
    from src.web.app import app
    with TestClient(app) as c:
        yield c


def _wait_done(client, job_id: str, timeout: float = 90.0) -> str:
    """轮询作业状态直到 done/failed 或超时。返回最终状态。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}", timeout=10)
        if r.status_code == 200:
            st = r.json()["status"]
            if st in ("done", "failed"):
                return st
        time.sleep(0.5)
    return "timeout"


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "capabilities" in body
    assert "disk_free_gb" in body
    assert isinstance(body["keyring_available"], bool)


def test_analyze_local_path_creates_job(client, test_video):
    cfg = {
        "density": 0.3,
        "smart_extraction": False,
        "enable_audio": False,
        "model": "test-model",
        "local_path": str(test_video),
    }
    r = client.post("/api/analyze", data={"config": json.dumps(cfg)})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["job_id"]
    assert body["video_name"] == "smoke.mp4"

    status = _wait_done(client, body["job_id"])
    assert status == "done", f"job did not finish: {status}"

    detail = client.get(f"/api/jobs/{body['job_id']}").json()
    assert detail["frame_count"] >= 3
    assert len(detail["frames"]) >= 3
    # 帧含 metrics
    assert "brightness" in detail["frames"][0]["metrics"]


def test_jobs_list_contains_created(client, test_video):
    cfg = {"density": 0.2, "smart_extraction": False, "enable_audio": False,
           "model": "m", "local_path": str(test_video)}
    r = client.post("/api/analyze", data={"config": json.dumps(cfg)})
    assert r.status_code == 201
    jid = r.json()["job_id"]
    _wait_done(client, jid)
    lst = client.get("/api/jobs").json()
    assert any(j["job_id"] == jid for j in lst)


def test_frame_image_served(client, test_video):
    cfg = {"density": 0.2, "smart_extraction": False, "enable_audio": False,
           "model": "m", "local_path": str(test_video)}
    jid = client.post("/api/analyze", data={"config": json.dumps(cfg)}).json()["job_id"]
    _wait_done(client, jid)
    frames = client.get(f"/api/jobs/{jid}/frames").json()
    assert frames
    img = client.get(frames[0]["url"])
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/jpeg"
    assert len(img.content) > 100


def test_path_traversal_blocked(client, test_video):
    cfg = {"density": 0.2, "smart_extraction": False, "enable_audio": False,
           "model": "m", "local_path": str(test_video)}
    jid = client.post("/api/analyze", data={"config": json.dumps(cfg)}).json()["job_id"]
    _wait_done(client, jid)
    # 逃逸 job 目录
    r = client.get(f"/api/jobs/{jid}/frames/..%2F..%2F..%2Fetc%2Fpasswd")
    assert r.status_code in (400, 404)
    # 非白名单扩展名
    r2 = client.get(f"/api/jobs/{jid}/frames/evil.exe")
    assert r2.status_code == 400
    assert "not allowed" in r2.text.lower()


def test_bad_ext_local_path_rejected(client, tmp_path):
    fake = tmp_path / "notvideo.exe"
    fake.write_bytes(b"MZ")
    cfg = {"density": 0.2, "local_path": str(fake)}
    r = client.post("/api/analyze", data={"config": json.dumps(cfg)})
    assert r.status_code == 400
    assert "unsupported" in r.text.lower()


def test_missing_source_rejected(client):
    cfg = {"density": 0.2}
    r = client.post("/api/analyze", data={"config": json.dumps(cfg)})
    assert r.status_code == 400


def test_sse_stream_emits_events(client, test_video):
    cfg = {"density": 0.2, "smart_extraction": False, "enable_audio": False,
           "model": "m", "local_path": str(test_video)}
    jid = client.post("/api/analyze", data={"config": json.dumps(cfg)}).json()["job_id"]
    with client.stream("GET", f"/api/jobs/{jid}/stream") as resp:
        assert resp.status_code == 200
        seen = set()
        for line in resp.iter_lines():
            if line.startswith("event:"):
                seen.add(line[6:].strip())
            if "done" in seen or "error" in seen:
                break
    assert "phase" in seen
    assert "frame" in seen or "done" in seen


def test_auth_enforced_when_token_set(monkeypatch, test_video):
    """配置 VAP_HEADLESS_TOKEN 后,无 Authorization 的请求应 401。"""
    monkeypatch.setenv("VAP_HEADLESS_TOKEN", "a-test-token-value-32-chars-xxxxxx")
    # 清 lru_cache 让 Settings 重读 env
    from src.web.config import get_settings
    get_settings.cache_clear()
    try:
        from src.web.app import app
        with TestClient(app) as c:
            r = c.get("/api/health")
            assert r.status_code == 401
            r2 = c.get("/api/health", headers={"Authorization": "Bearer a-test-token-value-32-chars-xxxxxx"})
            assert r2.status_code == 200
    finally:
        get_settings.cache_clear()
