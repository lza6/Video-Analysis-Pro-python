"""批量处理路由(接 BatchRunner + RunStore)。

  POST   /api/batch/run      → 启动批量分析(后台线程跑 run_batch)
  POST   /api/batch/cancel  → 取消当前批量
  POST   /api/batch/resume  → 续跑未完成 run
  GET    /api/runs           → RunStore.list_runs
  GET    /api/runs/{id}      → RunStore.get_run(run+segments+clips)
  GET    /api/runs/{id}/progress → RunStore.get_progress
  DELETE /api/runs/{id}      → RunStore.delete_run
  DELETE /api/runs           → RunStore.clear_all
  GET    /api/batch/stream   → SSE 进度(run_started/video_started/segment_done/...)

复用 src/core/batch_runner.BatchRunner + src/core/run_store.RunStore。
BatchRunner 是 QObject + pyqtSignal:Web 无 Qt 事件循环,需 video_concurrency=1
(DirectConnection 即时投递)或外层 worker 线程轮询。本路由用 worker 线程 +
pyqtSignal.connect 回调 → asyncio.Queue → SSE 的桥接。
"""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..security import require_auth, require_rate_limit

log = logging.getLogger("web.batch")

router = APIRouter(prefix="/api", tags=["batch"])

# 进程级单例(类型用字符串前缀避免 import PyQt6 在模块加载期失败)
_runner = None
_run_store = None
_lock = threading.Lock()
# SSE 订阅者
_subscribers: list[asyncio.Queue] = []
_sub_lock = threading.Lock()


def _get_run_store():
    global _run_store
    if _run_store is None:
        with _lock:
            if _run_store is None:
                try:
                    from src.core.run_store import RunStore
                    _run_store = RunStore()
                except Exception as e:
                    log.warning(f"RunStore 不可用: {e}")
                    _run_store = None
    return _run_store


class BatchRunReq(BaseModel):
    video_dir: str = Field(..., description="视频目录绝对路径")
    key_item_image: str = ""
    item_description: str = "关键物品"
    segment_sec: int = 120
    fps_sample: float = 1.0
    clean_segments: bool = True
    resume: bool = True
    model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    enable_thinking: bool = True
    reasoning_budget: int = 8192
    max_tokens: int = 65536
    temperature: float = 0.2
    confidence_threshold: float = 0.7
    frame_change_pct: int = 20


@router.post(
    "/batch/run",
    status_code=201,
    dependencies=[Depends(require_auth), Depends(require_rate_limit)],
)
def start_batch(req: BatchRunReq, request: Request) -> dict:
    """启动批量分析。后台 worker 线程跑 BatchRunner.run_batch。"""
    global _runner
    store = _get_run_store()
    if store is None:
        raise HTTPException(status_code=500, detail={"error": "RunStore 不可用"})
    video_dir = Path(req.video_dir)
    if not video_dir.is_dir():
        raise HTTPException(status_code=400, detail={"error": f"video_dir 不存在: {req.video_dir}"})

    loop = request.app.state.analyzer_service._loop

    with _lock:
        # 取消旧 runner
        if _runner is not None:
            try:
                _runner.cancel()
            except Exception:
                pass
        _runner = _build_runner(req, store)

    videos = sorted([p for p in video_dir.iterdir()
                     if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")])
    if not videos:
        raise HTTPException(status_code=400, detail={"error": "目录无支持的视频文件"})

    def worker():
        try:
            _wire_signals(_runner, loop)
            done, hits = _runner.run_batch(videos)
            _broadcast(loop, {"type": "batch_finished", "data": {"done": done, "hits": hits}})
        except Exception as e:
            log.exception(f"batch run failed: {e}")
            _broadcast(loop, {"type": "error", "data": {"message": str(e)}})

    threading.Thread(target=worker, daemon=True, name="vap-batch").start()
    return {"status": "running", "video_count": len(videos), "video_dir": str(video_dir)}


@router.post("/batch/cancel", dependencies=[Depends(require_auth)])
def cancel_batch() -> dict:
    with _lock:
        if _runner is None:
            return {"status": "idle"}
        try:
            _runner.cancel()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    return {"status": "canceling"}


@router.post("/batch/resume", dependencies=[Depends(require_auth)])
def resume_batch(request: Request) -> dict:
    """续跑未完成 run。"""
    store = _get_run_store()
    runner = _runner
    if store is None or runner is None:
        raise HTTPException(status_code=400, detail={"error": "无 runner 可续跑,先 /batch/run"})
    loop = request.app.state.analyzer_service._loop

    def worker():
        try:
            done, hits = _runner.resume_batch()
            _broadcast(loop, {"type": "batch_finished", "data": {"done": done, "hits": hits}})
        except Exception as e:
            log.exception(f"resume failed: {e}")
            _broadcast(loop, {"type": "error", "data": {"message": str(e)}})

    threading.Thread(target=worker, daemon=True, name="vap-batch-resume").start()
    return {"status": "resuming"}


@router.get("/runs", dependencies=[Depends(require_auth)])
def list_runs(limit: int = 50, status: str | None = None) -> dict:
    store = _get_run_store()
    if store is None:
        return {"runs": [], "available": False}
    return {"runs": store.list_runs(limit=limit, status=status), "available": True}


@router.get("/runs/{run_id}", dependencies=[Depends(require_auth)])
def get_run(run_id: str) -> dict:
    store = _get_run_store()
    if store is None:
        raise HTTPException(status_code=500, detail={"error": "RunStore 不可用"})
    r = store.get_run(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail={"error": "run not found"})
    return r


@router.get("/runs/{run_id}/progress", dependencies=[Depends(require_auth)])
def run_progress(run_id: str) -> dict:
    store = _get_run_store()
    if store is None:
        raise HTTPException(status_code=500, detail={"error": "RunStore 不可用"})
    p = store.get_progress(run_id)
    if p is None:
        raise HTTPException(status_code=404, detail={"error": "run not found"})
    return p


@router.delete("/runs/{run_id}", dependencies=[Depends(require_auth)])
def delete_run(run_id: str, purge: bool = False) -> dict:
    store = _get_run_store()
    if store is None:
        raise HTTPException(status_code=500, detail={"error": "RunStore 不可用"})
    ok = store.delete_run(run_id, purge_files=purge)
    return {"ok": ok}


@router.delete("/runs", dependencies=[Depends(require_auth)])
def clear_runs(purge: bool = False) -> dict:
    store = _get_run_store()
    if store is None:
        raise HTTPException(status_code=500, detail={"error": "RunStore 不可用"})
    n = store.clear_all(purge_files=purge)
    return {"ok": True, "deleted": n}


@router.get("/batch/stream", dependencies=[Depends(require_auth)])
async def batch_stream(request: Request):
    """SSE 批量进度流。"""
    q: asyncio.Queue = asyncio.Queue()
    with _sub_lock:
        _subscribers.append(q)

    async def _gen():
        try:
            while True:
                if await request.is_disconnected():
                    return
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    import json
                    yield {
                        "event": ev["type"],
                        "data": json.dumps(ev["data"], ensure_ascii=False),
                    }
                except asyncio.TimeoutError:
                    yield {"comment": "keepalive"}
        finally:
            with _sub_lock:
                if q in _subscribers:
                    _subscribers.remove(q)

    return EventSourceResponse(_gen())


# ============================ helpers ============================

def _build_runner(req: BatchRunReq, store):
    """构造 BatchRunner + ProviderRouter。"""
    try:
        from src.core.batch_runner import BatchRunner, BatchConfig
        from src.core.provider_router import ProviderRouter, load_from_env, load_router_config_from_env
        nv_keys = [k for k in load_from_env() if k.provider == "nvidia"]
        if not nv_keys:
            raise RuntimeError("无 nvidia key(配 .env VAP_NV_API_KEYS)")
        pr = ProviderRouter(nv_keys, **load_router_config_from_env())
        cfg = BatchConfig(
            video_dir=req.video_dir,
            key_item_image=req.key_item_image,
            item_description=req.item_description,
            segment_sec=req.segment_sec,
            fps_sample=req.fps_sample,
            clean_segments=req.clean_segments,
            resume=req.resume,
            model=req.model,
            enable_thinking=req.enable_thinking,
            reasoning_budget=req.reasoning_budget,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            confidence_threshold=req.confidence_threshold,
            frame_change_pct=req.frame_change_pct,
            # 单视频串行:Web 无 Qt 事件循环,避免信号跨线程问题
            video_concurrency=1,
            concurrency_per_key=2,
        )
        return BatchRunner(cfg, store, pr)
    except Exception as e:
        log.exception(f"build runner failed: {e}")
        raise


def _wire_signals(runner, loop: asyncio.AbstractEventLoop) -> None:
    """把 BatchRunner 的 pyqtSignal 连到 SSE 广播。

    pyqtSignal.connect 在无 QCoreApplication 时也能工作(DirectConnection
    即时投递,不依赖 Qt 事件循环),因为 worker 线程里直接 emit 会同步调
    connected 的 Python 可调用对象。
    """
    try:
        runner.run_started.connect(
            lambda run_id, name: _broadcast(loop, {
                "type": "run_started", "data": {"run_id": run_id, "video_name": name}
            })
        )
        runner.video_started.connect(
            lambda run_id, name: _broadcast(loop, {
                "type": "video_started", "data": {"run_id": run_id, "video_name": name}
            })
        )
        runner.segment_done.connect(
            lambda run_id, seg, match, conf: _broadcast(loop, {
                "type": "segment_done",
                "data": {"run_id": run_id, "seg_idx": seg, "match": match, "conf": conf},
            })
        )
        runner.video_done.connect(
            lambda run_id, name, hits: _broadcast(loop, {
                "type": "video_done", "data": {"run_id": run_id, "video_name": name, "hits": hits}
            })
        )
        runner.batch_progress.connect(
            lambda done, total: _broadcast(loop, {
                "type": "batch_progress", "data": {"done": done, "total": total}
            })
        )
        runner.error.connect(
            lambda msg: _broadcast(loop, {"type": "error", "data": {"message": msg}})
        )
    except Exception as e:
        log.warning(f"wire signals failed (非 Qt 环境?): {e}")


def _broadcast(loop: asyncio.AbstractEventLoop, event: dict) -> None:
    """向所有 SSE 订阅者推事件。"""
    with _sub_lock:
        dead = []
        for q in _subscribers:
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)
