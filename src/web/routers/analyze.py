"""分析作业路由。

  POST /api/analyze              → 创建作业(上传视频 或 本机路径),返回 job_id
  GET  /api/jobs                 → 列出作业
  GET  /api/jobs/{id}            → 作业详情(非流式)
  DELETE /api/jobs/{id}          → 删除作业 + 清理工作目录
  GET  /api/jobs/{id}/stream     → SSE 事件流(phase/progress/frame/transcript/report-token/done/error)
  GET  /api/jobs/{id}/frames     → 帧列表 JSON
  GET  /api/jobs/{id}/frames/{name} → 帧图片(FileResponse,路径消毒)

本地部署:支持 local_path 直接读本机视频(绕过上传),契合"隐私不出本机"。
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from ..deps import get_job_store, web_jobs_root
from ..job_store import JobRecord
from ..schemas import AnalyzeRequest, FrameInfo, JobCreated, JobDetail, JobSummary
from ..security import (
    ALLOWED_IMAGE_EXTS,
    require_auth,
    require_rate_limit,
    resolve_within_root,
    sanitize_upload_filename,
)
from ..services.analyzer_service import AnalyzerService
from ..sse import stream_job_events

log = logging.getLogger("web.analyze")

router = APIRouter(prefix="/api", tags=["analyze"])

# 全局分析并发信号量(防 VRAM OOM)。lifespan 时按 settings 初始化。
_ANALYZE_SEM: asyncio.Semaphore | None = None


def init_analyze_semaphore(concurrency: int) -> None:
    global _ANALYZE_SEM
    _ANALYZE_SEM = asyncio.Semaphore(max(1, concurrency))


def _get_service(request: Request) -> AnalyzerService:
    """从 app.state 取 AnalyzerService(lifespan 注入,持有主 loop)。"""
    svc = getattr(request.app.state, "analyzer_service", None)
    if svc is None:
        raise HTTPException(status_code=500, detail={"error": "service not ready"})
    return svc


def _alloc_workdir(video_name: str) -> tuple[Path, Path]:
    """为作业分配独立工作目录:web_jobs_root/<job_uuid>/。"""
    job_uuid = uuid.uuid4().hex[:12]
    workdir = web_jobs_root() / job_uuid
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir, workdir / "frames"


@router.post(
    "/analyze",
    response_model=JobCreated,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_auth), Depends(require_rate_limit)],
)
async def create_analysis(
    request: Request,
    file: UploadFile | None = File(None),
    config: str = Form(...),  # AnalyzeRequest 的 JSON 字符串
) -> JobCreated:
    """创建分析作业。

    multipart: file=<视频二进制,可选>; config=<AnalyzeRequest JSON>
    若 config.local_path 提供且 file 为空 → 直接读本机路径。
    """
    try:
        cfg = AnalyzeRequest.model_validate(json.loads(config))
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"invalid config: {e}"},
        )

    svc = _get_service(request)

    # ── 解析视频来源 ──
    if file is not None:
        workdir, frames_dir = _alloc_workdir(cfg.local_path or "upload")
        safe_name = sanitize_upload_filename(file.filename or "upload.mp4")
        video_path = workdir / safe_name
        # 流式写盘,不整体读进内存(防大文件 OOM)
        max_bytes = _max_upload_bytes()
        written = 0
        with video_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    out.close()
                    shutil.rmtree(workdir, ignore_errors=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail={"error": f"upload exceeds {_max_upload_mb()} MB"},
                    )
                out.write(chunk)
        video_name = safe_name
    elif cfg.local_path:
        video_path = Path(cfg.local_path)
        if not video_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"local file not found: {cfg.local_path}"},
            )
        if video_path.suffix.lower() not in _ALLOWED_VIDEO_EXTS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"unsupported file type: {video_path.suffix}"},
            )
        workdir, frames_dir = _alloc_workdir(video_path.name)
        video_name = video_path.name
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "provide either an uploaded file or local_path"},
        )

    # ── 启动作业(帧产物写入受控 workdir,不污染用户视频目录)──
    rec = svc.create_job(video_path, video_name, cfg.model_dump(), workdir=workdir)
    log.info(f"analysis job started: {rec.job_id} ({video_name})")

    return JobCreated(
        job_id=rec.job_id,
        status=rec.status.value,
        video_name=video_name,
    )


@router.get("/jobs", response_model=list[JobSummary], dependencies=[Depends(require_auth)])
def list_jobs(limit: int = 50) -> list[JobSummary]:
    store = get_job_store()
    return [JobSummary(**r.summary()) for r in store.list_all(limit=limit)]


@router.get("/jobs/{job_id}", response_model=JobDetail, dependencies=[Depends(require_auth)])
def get_job(job_id: str) -> JobDetail:
    store = get_job_store()
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail={"error": "job not found"})
    return JobDetail(
        job_id=rec.job_id,
        video_name=rec.video_name,
        status=rec.status.value,
        duration=round(rec.duration, 2),
        frame_count=rec.frame_count,
        transcript=rec.transcript,
        report=rec.report,
        frames=[FrameInfo(**f) for f in rec.frames],
        error=rec.error,
    )


@router.delete("/jobs/{job_id}", dependencies=[Depends(require_auth)])
def delete_job(job_id: str) -> dict:
    store = get_job_store()
    if not store.delete(job_id):
        raise HTTPException(status_code=404, detail={"error": "job not found"})
    return {"ok": True}


@router.get("/jobs/{job_id}/stream", dependencies=[Depends(require_auth)])
async def job_stream(job_id: str, request: Request):
    """SSE 事件流。客户端断开时优雅退出。

    v10.2.0:支持断线续连。客户端重连带 Last-Event-ID header,
    服务端从 JobRecord.recent_events 重放 seq > last_event_id 的事件,
    再继续 live 流。EventSource 原生不支持自定义 header,前端用
    fetch + ReadableStream 实现 SSE 客户端(见 webapp/src/lib/sse.ts)。
    """
    store = get_job_store()
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail={"error": "job not found"})

    # Last-Event-ID header(HTTP 大小写不敏感,FastAPI headers case-insensitive)
    last_event_id: int | None = None
    raw = request.headers.get("last-event-id")
    if raw:
        try:
            last_event_id = int(raw)
        except ValueError:
            log.warning(f"invalid Last-Event-ID header ignored: {raw!r}")
            last_event_id = None

    from sse_starlette.sse import EventSourceResponse
    return EventSourceResponse(_guard_stream(rec, request, last_event_id))


async def _guard_stream(rec: JobRecord, request: Request, last_event_id: int | None = None):
    async for evt in stream_job_events(rec, last_event_id=last_event_id):
        if await request.is_disconnected():
            log.debug(f"client disconnected from job {rec.job_id}")
            return
        yield evt


@router.get("/jobs/{job_id}/frames", dependencies=[Depends(require_auth)])
def job_frames(job_id: str) -> list[dict]:
    store = get_job_store()
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail={"error": "job not found"})
    return list(rec.frames)


@router.get("/jobs/{job_id}/frames/{name}", dependencies=[Depends(require_auth)])
def job_frame_image(job_id: str, name: str):
    """返回帧图片。路径消毒:只允许该 job 的 frames_dir 内的白名单图片。"""
    store = get_job_store()
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail={"error": "job not found"})

    # 纯函数消毒:只允许该 job 的 frames_dir 内的白名单图片,无全局状态残留
    full = resolve_within_root(rec.frames_dir, name, ALLOWED_IMAGE_EXTS)
    if not full.is_file():
        raise HTTPException(status_code=404, detail={"error": "frame not found"})
    return FileResponse(full, media_type=_guess_image_mime(full.suffix))


# ============================ helpers ============================

_ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".wmv", ".ts"}


def _max_upload_mb() -> int:
    from ..config import get_settings
    return get_settings().max_upload_mb


def _max_upload_bytes() -> int:
    return _max_upload_mb() * 1024 * 1024


def _guess_image_mime(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
    }.get(suffix.lower(), "application/octet-stream")

