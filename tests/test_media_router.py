# -*- coding: utf-8 -*-
"""media router 测试(/api/jobs/{id}/media)。

覆盖 src/web/routers/media.py 的三条路由:
  - POST /api/jobs/{id}/media            生成摘要媒体(monkeypatch moviepy 真实调用)
  - GET  /api/jobs/{id}/media            媒体产物列表
  - GET  /api/jobs/{id}/media/{name}     媒体文件下载(路径消毒 + 404)

守付费红线:不跑真实 moviepy 生成,monkeypatch create_summary_media_artifacts
返回固定产物;媒体文件用 tmp_path 真实小文件构造。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
except OSError:
    torch = None

from fastapi.testclient import TestClient  # noqa: E402

from src.web.deps import get_job_store  # noqa: E402
from src.web.job_store import JobRecord, JobStatus  # noqa: E402


@pytest.fixture
def app(monkeypatch, tmp_path):
    """TestClient(app) + 注入 tmp_path 的 JobStore 隔离。"""
    from src.web.app import app as real_app
    import src.web.deps as deps_mod

    new_store = type(get_job_store())()
    monkeypatch.setattr(deps_mod, "_job_store", new_store)
    with TestClient(real_app) as c:
        yield c, new_store


@pytest.fixture
def sample_job(tmp_path):
    """构造一个带 video_path + frames + workdir 的 JobRecord。"""
    workdir = tmp_path / "job_x"
    workdir.mkdir(parents=True, exist_ok=True)
    frames_dir = workdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # 造一个帧图文件供 frames 列表引用
    frame_file = frames_dir / "f001.jpg"
    frame_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 128)

    rec = JobRecord(
        job_id="job_x",
        video_name="demo.mp4",
        workdir=workdir,
        frames_dir=frames_dir,
        video_path=str(tmp_path / "demo.mp4"),
        status=JobStatus.DONE,
        duration=30.0,
        frames=[{"url": "/api/jobs/job_x/frames/f001.jpg", "timestamp": 5.0}],
    )
    return rec, workdir


def _seed(app_store, rec):
    app_store._jobs[rec.job_id] = rec
    return rec


# ============================ GET /api/jobs/{id}/media ============================


def test_list_media_404_unknown_job(app):
    client, store = app
    r = client.get("/api/jobs/nope/media")
    assert r.status_code == 404, r.text
    assert "job not found" in r.json()["detail"]["error"]


def test_list_media_empty(app, sample_job):
    client, store = app
    _seed(store, sample_job[0])
    r = client.get("/api/jobs/job_x/media")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == "job_x"
    assert body["clips"] == []
    assert body["summary_video"] is None
    assert body["gif"] is None


def test_list_media_with_artifacts(app, sample_job):
    """workdir 预置 clips/gif 文件后,_media_payload 生成真实 URL。"""
    client, store = app
    rec, workdir = sample_job
    # 预置产物文件(clips 存相对路径,payload 用 Path(...).name)
    clip1 = workdir / "clip_001.mp4"
    clip1.write_bytes(b"\x00" * 64)
    rec.media_clips = [str(clip1)]
    rec.media_summary_video = str(workdir / "summary.mp4")
    (workdir / "summary.mp4").write_bytes(b"\x00" * 64)
    rec.media_gif = str(workdir / "preview.gif")
    (workdir / "preview.gif").write_bytes(b"\x00" * 64)
    _seed(store, rec)

    r = client.get("/api/jobs/job_x/media")
    assert r.status_code == 200
    body = r.json()
    assert body["clips"] == ["/api/jobs/job_x/media/clip_001.mp4"]
    assert body["summary_video"] == "/api/jobs/job_x/media/summary.mp4"
    assert body["gif"] == "/api/jobs/job_x/media/preview.gif"


# ============================ GET /api/jobs/{id}/media/{name} ============================


def test_media_file_404_unknown_job(app):
    client, _ = app
    r = client.get("/api/jobs/nope/media/clip.mp4")
    assert r.status_code == 404


def test_media_file_served(app, sample_job):
    client, store = app
    rec, workdir = sample_job
    target = workdir / "clip_001.mp4"
    target.write_bytes(b"\x00" * 256)
    _seed(store, rec)

    r = client.get("/api/jobs/job_x/media/clip_001.mp4")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "video/mp4"
    assert len(r.content) == 256


def test_media_file_404_missing(app, sample_job):
    client, store = app
    _seed(store, sample_job[0])
    r = client.get("/api/jobs/job_x/media/nope.mp4")
    assert r.status_code == 404


def test_media_file_path_traversal_blocked(app, sample_job):
    """../ 逃逸 → 404(路径消毒);非白名单扩展名 → 400(file type not allowed)。"""
    client, store = app
    _seed(store, sample_job[0])
    r = client.get("/api/jobs/job_x/media/..%2F..%2Fsecret.mp4")
    assert r.status_code == 404, r.text
    r2 = client.get("/api/jobs/job_x/media/evil.exe")
    assert r2.status_code == 400, r2.text  # 真实行为:file type not allowed → 400
    assert "file type not allowed" in r2.json()["detail"]["error"]


def test_media_file_gif_mime(app, sample_job):
    client, store = app
    rec, workdir = sample_job
    target = workdir / "preview.gif"
    target.write_bytes(b"GIF89a" + b"\x00" * 16)
    _seed(store, rec)
    r = client.get("/api/jobs/job_x/media/preview.gif")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/gif"


# ============================ POST /api/jobs/{id}/media ============================


def test_generate_media_404_unknown_job(app):
    client, _ = app
    r = client.post("/api/jobs/nope/media", json={})
    assert r.status_code == 404


def test_generate_media_400_no_video(app, sample_job):
    client, store = app
    rec, _ = sample_job
    rec.video_path = None
    _seed(store, rec)
    r = client.post("/api/jobs/job_x/media", json={})
    assert r.status_code == 400


def test_generate_media_400_no_frames(app, sample_job):
    client, store = app
    rec, _ = sample_job
    rec.frames = []
    _seed(store, rec)
    r = client.post("/api/jobs/job_x/media", json={})
    assert r.status_code == 400
    assert "no frames" in r.json()["detail"]["error"]


def test_generate_media_success(app, sample_job, monkeypatch):
    """monkeypatch create_summary_media_artifacts 返回固定产物 → 200 + clips。"""
    client, store = app
    rec, workdir = sample_job
    _seed(store, rec)

    clips = [str(workdir / "clip_001.mp4"), str(workdir / "clip_002.mp4")]
    for c in clips:
        Path(c).write_bytes(b"\x00" * 64)
    summary = str(workdir / "summary.mp4")
    Path(summary).write_bytes(b"\x00" * 64)

    def _fake_create_summary(**kw):
        # 断言接收的关键参数
        assert kw["video_duration"] == 30.0
        assert kw["num_clips"] == 10
        assert kw["make_video"] is True
        return clips, [1, 2], summary, None

    monkeypatch.setattr(
        "src.core.logic.create_summary_media_artifacts",
        _fake_create_summary,
    )
    r = client.post("/api/jobs/job_x/media", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["clips"] == [
        "/api/jobs/job_x/media/clip_001.mp4",
        "/api/jobs/job_x/media/clip_002.mp4",
    ]
    assert body["summary_video"] == "/api/jobs/job_x/media/summary.mp4"
    assert rec.media_clips == clips


def test_generate_media_failure_500(app, sample_job, monkeypatch):
    """create_summary_media_artifacts 抛异常 → 500 + error 透传。"""
    def _boom(**kw):
        raise RuntimeError("moviepy boom")

    monkeypatch.setattr(
        "src.core.logic.create_summary_media_artifacts",
        _boom,
    )
    client, store = app
    _seed(store, sample_job[0])
    r = client.post("/api/jobs/job_x/media", json={})
    assert r.status_code == 500, r.text
    assert "moviepy boom" in r.json()["detail"]["error"]
