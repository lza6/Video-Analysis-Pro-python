"""元数据与画质路由。

  GET  /api/jobs/{id}/metrics     → 平均指标 + 时间序列(亮度/饱和度/清晰度)
  POST /api/jobs/{id}/metrics      → 触发 get_advanced_video_metrics 生成(图 + avg)
  GET  /api/jobs/{id}/metrics/chart → 图表 PNG(路径消毒)

复用 src/core/logic.get_advanced_video_metrics + get_frame_metrics。
"""
from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from ..deps import get_job_store
from ..job_store import JobStatus
from ..security import ALLOWED_IMAGE_EXTS, require_auth, resolve_within_root

log = logging.getLogger("web.metrics")

router = APIRouter(prefix="/api", tags=["metrics"])


@router.get("/jobs/{job_id}/metrics", dependencies=[Depends(require_auth)])
def get_metrics(job_id: str) -> dict:
    """返回该作业的 avg 指标 + 时间序列(若已生成)。"""
    store = get_job_store()
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail={"error": "job not found"})
    return {
        "job_id": job_id,
        "avg": rec.metrics_avg,
        "chart_url": f"/api/jobs/{job_id}/metrics/chart" if rec.metrics_chart_path else None,
        "generating": rec.status == JobStatus.RUNNING and not rec.metrics_avg,
    }


@router.post("/jobs/{job_id}/metrics", dependencies=[Depends(require_auth)])
def generate_metrics(job_id: str, request: Request) -> dict:
    """触发后台生成画质指标 + 图表。同步阻塞返回(本地小视频 <10s)。"""
    store = get_job_store()
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail={"error": "job not found"})
    if not rec.video_path:
        raise HTTPException(status_code=400, detail={"error": "video path unknown"})

    done = threading.Event()
    result_holder: dict = {}

    def run():
        try:
            from src.core.logic import get_advanced_video_metrics, get_frame_metrics
            import cv2

            # 时间序列:直接解码视频采样帧,与原 Qt 实现一致
            # (rec.frames 的 metrics 可能为空 — VideoProcessor 不一定填全字段)
            ts_data = {"timestamps": [], "brightness": [], "saturation": [], "sharpness": []}
            if rec.video_path:
                cap = cv2.VideoCapture(rec.video_path)
                if cap.isOpened():
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    n = min(50, max(1, total))
                    import numpy as np
                    for idx in np.linspace(0, max(0, total - 1), n, dtype=int) if total > 0 else []:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                        ok, frame = cap.read()
                        if not ok:
                            continue
                        m = get_frame_metrics(frame)
                        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        ts_data["timestamps"].append(float(ts))
                        ts_data["brightness"].append(m["brightness"])
                        ts_data["saturation"].append(m["saturation"])
                        ts_data["sharpness"].append(m["sharpness"])
                    cap.release()

            avg, fig = get_advanced_video_metrics(rec.video_path, num_frames_to_sample=50)
            rec.metrics_avg = avg or {}
            if fig is not None:
                chart_path = rec.workdir / "metrics_chart.png"
                fig.savefig(str(chart_path), dpi=120, bbox_inches="tight")
                rec.metrics_chart_path = str(chart_path)
            result_holder["avg"] = rec.metrics_avg
            result_holder["series"] = ts_data
            result_holder["ok"] = True
        except Exception as e:
            log.exception(f"metrics gen failed: {e}")
            result_holder["error"] = str(e)
            result_holder["ok"] = False
        finally:
            done.set()

    threading.Thread(target=run, daemon=True, name="vap-metrics").start()
    # 本地工具同步等待(给前端一个完整结果,避免轮询)
    done.wait(timeout=60)
    if not result_holder.get("ok"):
        raise HTTPException(status_code=500, detail={"error": result_holder.get("error", "metrics failed")})
    return {
        "job_id": job_id,
        "avg": result_holder["avg"],
        "series": result_holder["series"],
        "chart_url": f"/api/jobs/{job_id}/metrics/chart" if rec.metrics_chart_path else None,
    }


@router.get("/jobs/{job_id}/metrics/chart", dependencies=[Depends(require_auth)])
def metrics_chart(job_id: str):
    """图表 PNG。路径消毒到该 job 的 workdir。"""
    store = get_job_store()
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail={"error": "job not found"})
    if not rec.metrics_chart_path:
        raise HTTPException(status_code=404, detail={"error": "chart not generated"})
    full = resolve_within_root(rec.workdir, "metrics_chart.png", ALLOWED_IMAGE_EXTS)
    if not full.is_file():
        raise HTTPException(status_code=404, detail={"error": "chart file not found"})
    return FileResponse(full, media_type="image/png")
