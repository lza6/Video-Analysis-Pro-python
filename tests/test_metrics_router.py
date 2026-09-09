# -*- coding: utf-8 -*-
"""metrics router 测试(/api/jobs/{id}/metrics + /api/runtime/gc)。

覆盖 src/web/routers/metrics.py 的四条路由:
  - GET  /api/jobs/{id}/metrics          avg + chart_url + generating
  - POST /api/jobs/{id}/metrics          触发后台生成(monkeypatch cv2/logic)
  - GET  /api/jobs/{id}/metrics/chart    图表 PNG(路径消毒 + 404)
  - GET  /api/runtime/gc                 GCGuard 状态

守付费红线:不跑真实 cv2 解码/matplotlib 绘图,monkeypatch
src.core.logic 的 get_frame_metrics / get_advanced_video_metrics。
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
    """TestClient(app) + 注入干净 JobStore。"""
    from src.web.app import app as real_app
    import src.web.deps as deps_mod

    new_store = type(get_job_store())()
    monkeypatch.setattr(deps_mod, "_job_store", new_store)
    with TestClient(real_app) as c:
        yield c, new_store


@pytest.fixture
def sample_job(tmp_path):
    workdir = tmp_path / "job_m"
    workdir.mkdir(parents=True, exist_ok=True)
    rec = JobRecord(
        job_id="job_m",
        video_name="demo.mp4",
        workdir=workdir,
        frames_dir=workdir / "frames",
        video_path=str(tmp_path / "demo.mp4"),
        status=JobStatus.DONE,
    )
    return rec, workdir


def _seed(store, rec):
    store._jobs[rec.job_id] = rec
    return rec


# ============================ GET /api/jobs/{id}/metrics ============================


def test_get_metrics_404_unknown_job(app):
    client, _ = app
    r = client.get("/api/jobs/nope/metrics")
    assert r.status_code == 404


def test_get_metrics_empty(app, sample_job):
    client, store = app
    _seed(store, sample_job[0])
    r = client.get("/api/jobs/job_m/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == "job_m"
    assert body["avg"] == {}
    assert body["chart_url"] is None
    assert body["generating"] is False


def test_get_metrics_with_avg_and_chart(app, sample_job):
    client, store = app
    rec, workdir = sample_job
    rec.metrics_avg = {"brightness": 0.6, "saturation": 0.4}
    rec.metrics_chart_path = str(workdir / "metrics_chart.png")
    (workdir / "metrics_chart.png").write_bytes(b"\x89PNG" + b"\x00" * 16)
    _seed(store, rec)

    r = client.get("/api/jobs/job_m/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["avg"]["brightness"] == 0.6
    assert body["chart_url"] == "/api/jobs/job_m/metrics/chart"


def test_get_metrics_generating_flag(app, sample_job):
    """RUNNING + 无 avg → generating=True。"""
    client, store = app
    rec, _ = sample_job
    rec.status = JobStatus.RUNNING
    _seed(store, rec)
    r = client.get("/api/jobs/job_m/metrics")
    assert r.status_code == 200
    assert r.json()["generating"] is True


# ============================ POST /api/jobs/{id}/metrics ============================


def test_generate_metrics_404(app):
    client, _ = app
    r = client.post("/api/jobs/nope/metrics")
    assert r.status_code == 404


def test_generate_metrics_400_no_video(app, sample_job):
    client, store = app
    rec, _ = sample_job
    rec.video_path = None
    _seed(store, rec)
    r = client.post("/api/jobs/job_m/metrics")
    assert r.status_code == 400


def test_generate_metrics_success(app, sample_job, monkeypatch):
    """monkeypatch cv2.VideoCapture + get_advanced_video_metrics → 200 + avg/series。"""
    import numpy as np

    client, store = app
    rec, workdir = sample_job
    _seed(store, rec)

    # --- 假 VideoCapture(按真实 CAP_PROP 常量识别属性)---
    import cv2

    class _FakeCap:
        def isOpened(self):
            return True

        def get(self, prop):
            # CAP_PROP_FRAME_COUNT=7(100 帧),CAP_PROP_POS_MSEC=0.0(每帧 0ms)
            return 100 if prop == cv2.CAP_PROP_FRAME_COUNT else 0.0

        def set(self, *a):
            return True

        def read(self):
            return True, np.zeros((64, 64, 3), dtype=np.uint8)

        def release(self):
            pass

    import src.core.logic as logic_mod

    def _fake_get_frame_metrics(frame):
        return {"brightness": 0.5, "saturation": 0.3, "sharpness": 0.4}

    def _fake_advanced(video_path, num_frames_to_sample=100):
        return {"brightness": 0.5, "saturation": 0.3, "sharpness": 0.4}, None

    monkeypatch.setattr("cv2.VideoCapture", lambda path: _FakeCap())
    monkeypatch.setattr(logic_mod, "get_frame_metrics", _fake_get_frame_metrics)
    monkeypatch.setattr(logic_mod, "get_advanced_video_metrics", _fake_advanced)

    r = client.post("/api/jobs/job_m/metrics")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["avg"]["brightness"] == 0.5
    assert body["series"]["timestamps"] != []
    assert body["chart_url"] is None  # fig=None → 不生成 chart


def test_generate_metrics_creates_chart(app, sample_job, monkeypatch):
    """fig 非 None → savefig 落盘 + chart_url 返回。"""
    client, store = app
    rec, workdir = sample_job
    _seed(store, rec)

    class _FakeFig:
        def savefig(self, path, **kw):
            Path(path).write_bytes(b"\x89PNG" + b"\x00" * 8)

    import src.core.logic as logic_mod

    monkeypatch.setattr("cv2.VideoCapture", lambda p: _NoOpCap())
    monkeypatch.setattr(logic_mod, "get_frame_metrics",
                        lambda f: {"brightness": 0.5, "saturation": 0.3, "sharpness": 0.4})
    monkeypatch.setattr(logic_mod, "get_advanced_video_metrics",
                        lambda *a, **k: ({"brightness": 0.5}, _FakeFig()))

    r = client.post("/api/jobs/job_m/metrics")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chart_url"] == "/api/jobs/job_m/metrics/chart"
    assert (workdir / "metrics_chart.png").exists()


class _NoOpCap:
    """无操作 VideoCapture:不开视频,直接返回没帧(series 空,不崩)。"""

    def isOpened(self):
        return False

    def get(self, prop):
        return 0

    def set(self, *a):
        return True

    def read(self):
        return False, None

    def release(self):
        pass


def test_generate_metrics_failure_500(app, sample_job, monkeypatch):
    """get_advanced_video_metrics 抛异常 → 500 + error。"""
    import src.core.logic as logic_mod

    monkeypatch.setattr("cv2.VideoCapture", lambda p: _NoOpCap())
    monkeypatch.setattr(logic_mod, "get_frame_metrics",
                        lambda f: {"brightness": 0.5, "saturation": 0.3, "sharpness": 0.4})

    def _boom(*a, **k):
        raise RuntimeError("metrics boom")

    monkeypatch.setattr(logic_mod, "get_advanced_video_metrics", _boom)
    client, store = app
    _seed(store, sample_job[0])
    r = client.post("/api/jobs/job_m/metrics")
    assert r.status_code == 500
    assert "metrics boom" in r.json()["detail"]["error"]


# ============================ GET /api/jobs/{id}/metrics/chart ============================


def test_chart_404_unknown_job(app):
    client, _ = app
    r = client.get("/api/jobs/nope/metrics/chart")
    assert r.status_code == 404


def test_chart_404_not_generated(app, sample_job):
    client, store = app
    _seed(store, sample_job[0])
    r = client.get("/api/jobs/job_m/metrics/chart")
    assert r.status_code == 404
    assert "not generated" in r.json()["detail"]["error"]


def test_chart_served(app, sample_job):
    client, store = app
    rec, workdir = sample_job
    rec.metrics_chart_path = str(workdir / "metrics_chart.png")
    (workdir / "metrics_chart.png").write_bytes(b"\x89PNG" + b"\x00" * 32)
    _seed(store, rec)
    r = client.get("/api/jobs/job_m/metrics/chart")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


# ============================ GET /api/runtime/gc ============================


def test_gc_status_disabled(app):
    """显式清掉 gc_guard → enabled=False。"""
    client, _ = app
    from src.web.app import app as real_app
    real_app.state.gc_guard = None
    r = client.get("/api/runtime/gc")
    assert r.status_code == 200
    assert r.json()["enabled"] is False


def test_gc_status_enabled(app, monkeypatch):
    """有 gc_guard → enabled=True + 关键字段。"""
    client, _ = app

    class _FakeGuard:
        def is_running(self):
            return True

        @property
        def threshold_mb(self):
            return 2048

        @property
        def interval_sec(self):
            return 60

        @property
        def baseline(self):
            class _B:
                def snapshot(self):
                    return {"rss_mb": 500}
            return _B()

        def get_memory_mb(self):
            return 600

    from src.web.app import app as real_app
    real_app.state.gc_guard = _FakeGuard()
    try:
        r = client.get("/api/runtime/gc")
        assert r.status_code == 200
        body = r.json()
        assert body["enabled"] is True
        assert body["threshold_mb"] == 2048
        assert body["baseline"]["rss_mb"] == 500
        assert body["current_mb"] == 600
    finally:
        real_app.state.gc_guard = None
