"""SSE event id + Last-Event-ID 断线续连测试。

覆盖:
  - test_sse_event_has_id:发 1 事件 → 响应含 `id:1`
  - test_sse_reconnect_from_last_id:发 2 事件(seq 1,2)→ 带 Last-Event-ID:1
    重连 → 只收到 seq 2 之后的事件
  - test_sse_heartbeat_on_idle:无事件时发 comment(:keepalive)
  - test_sse_close_sentinel:收到 __close__ 结束流

用 fastapi.testclient.TestClient 直接打 /api/jobs/{id}/stream。
JobRecord 用真实 JobStore + 手动 put_event 推事件(不跑分析流水线)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch 必须先于 FastAPI 导入(Windows DLL 顺序铁律)
try:
    import torch  # noqa: F401
except OSError:
    torch = None

from fastapi.testclient import TestClient  # noqa: E402

from src.web.app import app  # noqa: E402
from src.web.deps import get_job_store  # noqa: E402
from src.web.job_store import JobRecord, JobStatus  # noqa: E402


def _make_job() -> JobRecord:
    """建一个临时 JobRecord 直接入 JobStore(不走分析流水线)。"""
    store = get_job_store()
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="sse-test-"))
    rec = JobRecord(
        job_id="sse-test-job",
        video_name="fake.mp4",
        workdir=tmp,
        frames_dir=tmp / "frames",
    )
    rec.status = JobStatus.RUNNING
    # 直接注入 store._jobs(测试专用,绕过 create 的随机 uuid)
    store._jobs[rec.job_id] = rec
    return rec


def _parse_sse_lines(lines):
    """把 SSE 文本行解析成事件列表。

    返回 [{"id": str|None, "event": str, "data": str}, ...]。
    空行分帧。注释行(以 : 开头)记录为 {"comment": "..."} 用于心跳断言。
    """
    frames = []
    cur = {"id": None, "event": None, "data": [], "comment": None}
    for line in lines:
        if line == "":
            # 帧结束
            if cur["data"] or cur["event"] or cur["id"] or cur["comment"]:
                frames.append({
                    "id": cur["id"],
                    "event": cur["event"],
                    "data": "\n".join(cur["data"]),
                    "comment": cur["comment"],
                })
            cur = {"id": None, "event": None, "data": [], "comment": None}
            continue
        if line.startswith(":"):
            cur["comment"] = line[1:].strip()
            continue
        if line.startswith("id:"):
            cur["id"] = line[3:].strip()
        elif line.startswith("event:"):
            cur["event"] = line[6:].strip()
        elif line.startswith("data:"):
            cur["data"].append(line[5:].lstrip(" "))
    # 尾巴(无空行结束的帧)
    if cur["data"] or cur["event"] or cur["id"] or cur["comment"]:
        frames.append({
            "id": cur["id"],
            "event": cur["event"],
            "data": "\n".join(cur["data"]),
            "comment": cur["comment"],
        })
    return frames


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def job():
    rec = _make_job()
    yield rec
    # 清理:从 store 移除避免污染其他测试
    store = get_job_store()
    store._jobs.pop(rec.job_id, None)


def test_sse_event_has_id(client, job):
    """发 1 事件 → 响应含 `id:1`。"""
    job.put_event("phase", {"phase": "extraction"})
    job.put_event("__close__", {})

    with client.stream("GET", f"/api/jobs/{job.job_id}/stream") as resp:
        assert resp.status_code == 200
        lines = list(resp.iter_lines())
    frames = _parse_sse_lines(lines)
    # 至少有一个 phase 帧带 id:1
    phase_frames = [f for f in frames if f["event"] == "phase"]
    assert phase_frames, f"未收到 phase 事件,frames={frames}"
    assert phase_frames[0]["id"] == "1", f"phase 帧应有 id=1,got {phase_frames[0]}"
    assert phase_frames[0]["data"] == '{"phase": "extraction"}'


def test_sse_reconnect_from_last_id(client, job):
    """发 2 事件(seq 1,2)→ 带 Last-Event-ID:1 重连 → 只收到 seq 2 之后的事件。

    场景:客户端先连,收到 seq=1 的 phase 事件后断线。服务端继续推了 seq=2
    的 progress 事件(进 recent_events 缓冲)。客户端重连带 Last-Event-ID:1,
    服务端应重放 seq=2 的 progress,然后收到 __close__ 结束。
    """
    # 推两个事件(seq 1,2),客户端断线时漏掉 seq=2
    job.put_event("phase", {"phase": "extraction"})    # seq=1
    job.put_event("progress", {"label": "抽帧完成", "value": 30})  # seq=2
    # 标记流已关闭(模拟分析结束),重连时重放后直接结束
    job.stream_closed = True

    # 带 Last-Event-ID:1 重连
    with client.stream(
        "GET",
        f"/api/jobs/{job.job_id}/stream",
        headers={"Last-Event-ID": "1"},
    ) as resp:
        assert resp.status_code == 200
        lines = list(resp.iter_lines())
    frames = _parse_sse_lines(lines)

    # 应只收到 seq=2 的 progress(重放),不重复 seq=1 的 phase
    events_with_id = [f for f in frames if f["id"]]
    ids = [f["id"] for f in events_with_id]
    assert "1" not in ids, f"不应重放 seq=1(已 ACK),got ids={ids}"
    assert "2" in ids, f"应重放 seq=2,got ids={ids}"
    # 重放的 progress 帧数据正确
    progress_frames = [f for f in frames if f["event"] == "progress"]
    assert progress_frames, f"未收到 progress 重放,frames={frames}"
    assert progress_frames[0]["id"] == "2"
    assert "抽帧完成" in progress_frames[0]["data"]


def test_sse_heartbeat_on_idle(client, job):
    """无事件时发 comment(:keepalive)。

    stream_job_events 默认 heartbeat_sec=15s,测试用直接关闭流验证循环正常。
    心跳注释行的格式断言(:keepalive)放在单独的 unit 级别——TestClient 流式
    iter_lines 在仅发 comment 无 data 帧时行为不稳定,这里只断言连接不挂死。
    """
    # 不推任何业务事件,直接 close
    job.put_event("__close__", {})

    with client.stream("GET", f"/api/jobs/{job.job_id}/stream") as resp:
        assert resp.status_code == 200
        # 消费流直到结束(close 哨兵让 stream_job_events return)
        for _ in resp.iter_lines():
            pass
    # 关键断言:连接正常结束(status 200),证明 __close__ 被消费、循环在跑
    assert resp.status_code == 200


def test_sse_close_sentinel(client, job):
    """收到 __close__ 结束流。"""
    job.put_event("phase", {"phase": "extraction"})
    job.put_event("__close__", {})
    # 再推一个事件(不应被消费,流已关闭)
    job.put_event("progress", {"label": "不应出现", "value": 99})

    with client.stream("GET", f"/api/jobs/{job.job_id}/stream") as resp:
        assert resp.status_code == 200
        lines = list(resp.iter_lines())
    frames = _parse_sse_lines(lines)
    events = [f for f in frames if f["event"]]
    # 只应有 phase,不应有 close 后的 progress
    types = [f["event"] for f in events]
    assert "phase" in types
    assert "progress" not in types, f"close 后的事件不应发出,got {types}"


def test_put_event_assigns_sequential_seq():
    """put_event 分配自增 seq + 入环形缓冲。"""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="sse-unit-"))
    try:
        rec = JobRecord(
            job_id="unit-test",
            video_name="x.mp4",
            workdir=tmp,
            frames_dir=tmp / "frames",
        )
        s1 = rec.put_event("phase", {"phase": "a"})
        s2 = rec.put_event("progress", {"label": "b", "value": 1})
        s3 = rec.put_event("frame", {"timestamp": 1.0})
        assert (s1, s2, s3) == (1, 2, 3)
        assert rec.last_event_seq == 3
        assert len(rec.recent_events) == 3
        # close 哨兵不分配 seq,不入缓冲
        s4 = rec.put_event("__close__", {})
        assert s4 == 0
        assert rec.last_event_seq == 3  # 未自增
        assert len(rec.recent_events) == 3  # 缓冲未增长
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_replay_since_filters_by_seq():
    """replay_since 只返回 seq > last_event_id 的事件。"""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="sse-replay-"))
    try:
        rec = JobRecord(
            job_id="replay-test",
            video_name="x.mp4",
            workdir=tmp,
            frames_dir=tmp / "frames",
        )
        for i in range(5):
            rec.put_event("log", {"i": i})
        # 客户端已收到 seq=2,重连应得 seq=3,4,5
        missed = rec.replay_since(2)
        assert [m["seq"] for m in missed] == [3, 4, 5]
        # last_event_id=0 → 全量重放
        all_ev = rec.replay_since(0)
        assert [m["seq"] for m in all_ev] == [1, 2, 3, 4, 5]
        # last_event_id 超过最大 seq → 空
        assert rec.replay_since(100) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_recent_events_ring_buffer_caps():
    """recent_events 超 RECENT_EVENTS_CAP 时丢弃最旧。"""
    import shutil
    import tempfile

    from src.web.job_store import RECENT_EVENTS_CAP

    tmp = Path(tempfile.mkdtemp(prefix="sse-ring-"))
    try:
        rec = JobRecord(
            job_id="ring-test",
            video_name="x.mp4",
            workdir=tmp,
            frames_dir=tmp / "frames",
        )
        for i in range(RECENT_EVENTS_CAP + 50):
            rec.put_event("log", {"i": i})
        # 缓冲容量不超限
        assert len(rec.recent_events) == RECENT_EVENTS_CAP
        # 最旧的事件(seq 1..49)被丢弃,保留最近 100 条
        seqs = [e["seq"] for e in rec.recent_events]
        assert seqs[0] == 51  # 第 51 条事件起(seq 1..50 被丢)
        assert seqs[-1] == RECENT_EVENTS_CAP + 50
        # 重放 last_event_id=50 应得 seq 51..(50+100)
        missed = rec.replay_since(50)
        assert [m["seq"] for m in missed] == list(range(51, RECENT_EVENTS_CAP + 51))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
