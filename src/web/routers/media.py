"""摘要媒体路由。

  POST /api/jobs/{id}/media  → 生成高光集锦视频 + GIF(后台线程,同步返回)
  GET  /api/jobs/{id}/media  → 媒体产物列表(clips/summary_video/gif)
  GET  /api/jobs/{id}/media/{name} → 媒体文件(路径消毒)

复用 src/core/logic.create_summary_media_artifacts(moviepy 2.x)。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..deps import get_job_store
from ..security import (
    ALLOWED_MEDIA_EXTS,
    require_auth,
    resolve_within_root,
)

log = logging.getLogger("web.media")

router = APIRouter(prefix="/api", tags=["media"])


@router.post("/jobs/{job_id}/media", dependencies=[Depends(require_auth)])
def generate_media(job_id: str, payload: dict = None) -> dict:
    """生成摘要媒体。payload: make_video/make_gif/num_clips(可选)。"""
    payload = payload or {}
    store = get_job_store()
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail={"error": "job not found"})
    if not rec.video_path:
        raise HTTPException(status_code=400, detail={"error": "video path unknown"})
    if not rec.frames:
        raise HTTPException(status_code=400, detail={"error": "no frames; run analysis first"})

    from src.core.logic import create_summary_media_artifacts, Frame

    frames = [
        Frame(
            path=Path(_frame_url_to_path(f["url"], rec)),
            timestamp=f["timestamp"],
            metrics=f.get("metrics", {}),
        )
        for f in rec.frames
    ]

    done = threading.Event()
    holder: dict = {}

    def run():
        try:
            clips, selected, summary, gif = create_summary_media_artifacts(
                original_video_path=rec.video_path,
                video_duration=rec.duration or 0.0,
                frames=frames,
                output_dir=rec.workdir,
                video_stem=Path(rec.video_path).stem,
                num_clips=int(payload.get("num_clips", 10)),
                clip_duration_around_keyframe=float(payload.get("clip_duration", 5.0)),
                make_video=bool(payload.get("make_video", True)),
                make_gif=bool(payload.get("make_gif", False)),
                gif_resolution=str(payload.get("gif_resolution", "中")),
            )
            rec.media_clips = clips or []
            rec.media_summary_video = summary
            rec.media_gif = gif
            holder["ok"] = True
        except Exception as e:
            log.exception(f"media gen failed: {e}")
            holder["error"] = str(e)
            holder["ok"] = False
        finally:
            done.set()

    threading.Thread(target=run, daemon=True, name="vap-media").start()
    done.wait(timeout=120)

    if not holder.get("ok"):
        raise HTTPException(status_code=500, detail={"error": holder.get("error", "media failed")})
    return _media_payload(rec)


@router.get("/jobs/{job_id}/media", dependencies=[Depends(require_auth)])
def list_media(job_id: str) -> dict:
    store = get_job_store()
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail={"error": "job not found"})
    return _media_payload(rec)


@router.get("/jobs/{job_id}/media/{name}", dependencies=[Depends(require_auth)])
def media_file(job_id: str, name: str):
    store = get_job_store()
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail={"error": "job not found"})
    full = resolve_within_root(rec.workdir, name, ALLOWED_MEDIA_EXTS)
    if not full.is_file():
        raise HTTPException(status_code=404, detail={"error": "media not found"})
    return FileResponse(full, media_type=_guess_media_mime(full.suffix))


# ============================ helpers ============================

def _media_payload(rec) -> dict:
    base = f"/api/jobs/{rec.job_id}/media"
    return {
        "job_id": rec.job_id,
        "clips": [f"{base}/{Path(c).name}" for c in rec.media_clips],
        "summary_video": f"{base}/{Path(rec.media_summary_video).name}" if rec.media_summary_video else None,
        "gif": f"{base}/{Path(rec.media_gif).name}" if rec.media_gif else None,
    }


def _frame_url_to_path(url: str, rec) -> str:
    """从帧 url(/api/jobs/<id>/frames/<name>)反查磁盘绝对路径。"""
    name = url.rsplit("/", 1)[-1]
    candidate = rec.frames_dir / name
    return str(candidate) if candidate.exists() else ""


def _guess_media_mime(suffix: str) -> str:
    return {
        ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
        ".gif": "image/gif",
    }.get(suffix.lower(), "application/octet-stream")
