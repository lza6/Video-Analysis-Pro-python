"""P1-7/P1-8 实时事件流与审批体验后端测试（v10.5.0）。

覆盖：
  - /run_stream 现在推送实时事件(assistant_delta / tool_start / tool_done /
    supervise),不再是"turn 结束后一次性补发"（hooks→event_queue 接线）
  - 每请求独立事件队列：两个 session 的流互不串台
  - 审批事件带 session_id 且 pending 端点可按会话过滤
  - APPROVAL 前端映射守护：新增写工具忘中文映射 → 红
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
except OSError:
    torch = None

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def react_client(monkeypatch, tmp_path):
    monkeypatch.setenv("VAP_AGENT_BACKEND", "react")
    from src.web.config import get_settings
    get_settings.cache_clear()
    try:
        from src.web.app import app
        from src.core.agent.session import SessionStore
        with TestClient(app) as c:
            app.state.session_store = SessionStore(str(tmp_path / "test_sessions.db"))
            yield c
    finally:
        get_settings.cache_clear()


def _stream_events(client, url: str) -> list[dict]:
    """消费一个 SSE 流,返回 [{event, data_obj}]。"""
    out: list[dict] = []
    with client.stream("GET", url) as r:
        assert r.status_code == 200, r.text
        event = None
        data = ""
        for line in r.iter_lines():
            if not line:
                if event is not None:
                    try:
                        obj = __import__("json").loads(data) if data else None
                    except Exception:
                        obj = None
                    out.append({"event": event, "data": obj})
                event = None
                data = ""
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
    return out


# ---------------------------------------------------------------------------
# 1) 实时事件流（无 LLM 凭据 → 降级适配器也走 hooks 事件）
# ---------------------------------------------------------------------------


def test_run_stream_emits_realtime_events(react_client):
    """P1-7: /run_stream 必须实时推送 assistant_delta + done(不再黑盒等待)。"""
    r = react_client.post("/api/agent/chat", json={"text": "实时流测试"})
    sid = r.json()["session_id"]

    events = _stream_events(react_client, f"/api/agent/run_stream?session_id={sid}&text=实时流测试")
    names = [e["event"] for e in events]

    assert "assistant_delta" in names, f"缺 assistant_delta,实际事件: {names}"
    assert "done" in names, f"缺 done,实际事件: {names}"
    done = next(e for e in events if e["event"] == "done")
    assert done["data"]["session_id"] == sid


def test_run_stream_sessions_isolated(react_client):
    """P1-7: 两个 session 各自的事件流互不串台(done 只带自己的 session_id)。"""
    sid1 = react_client.post("/api/agent/chat", json={"text": "甲"}).json()["session_id"]
    sid2 = react_client.post("/api/agent/chat", json={"text": "乙"}).json()["session_id"]

    ev1 = _stream_events(react_client, f"/api/agent/run_stream?session_id={sid1}&text=甲")
    ev2 = _stream_events(react_client, f"/api/agent/run_stream?session_id={sid2}&text=乙")

    done1 = next(e for e in ev1 if e["event"] == "done")
    done2 = next(e for e in ev2 if e["event"] == "done")
    assert done1["data"]["session_id"] == sid1
    assert done2["data"]["session_id"] == sid2
    # 任何事件都不得携带对方 session_id
    for e in ev1:
        if e["data"] and isinstance(e["data"], dict):
            assert e["data"].get("session_id") in (None, sid1), f"串台: {e}"
    for e in ev2:
        if e["data"] and isinstance(e["data"], dict):
            assert e["data"].get("session_id") in (None, sid2), f"串台: {e}"


# ---------------------------------------------------------------------------
# 2) 审批事件路由到本请求队列（多窗口隔离）+ pending 按会话过滤
# ---------------------------------------------------------------------------


def test_approval_fn_routes_to_own_queue(monkeypatch):
    """P1-7/P1-8: approval_fn 指定 approval_queue 时事件进该队列(带 session_id)。"""
    from types import SimpleNamespace
    from src.web.deps import ApprovalBus
    from src.web.routers.agent import _make_sse_emitting_approval_fn
    from src.core.tools.registry import ToolCall

    class FakeBus(ApprovalBus):
        async def wait_decision_async(self, pin, timeout=None):
            return False  # 不真等,立即"未决定"

    fake = FakeBus()
    monkeypatch.setattr("src.web.deps.get_approval_bus", lambda: fake)

    q1: asyncio.Queue = asyncio.Queue()
    fn = _make_sse_emitting_approval_fn(q1, "sess-aaa")
    call = ToolCall(name="delete_history", args={"job_id": "1"}, call_id="c1")
    sig = SimpleNamespace(prompt="危险写操作?", priority="dangerous_write")

    async def _drive():
        decided = await fn(call, sig)
        assert decided is False
        ev = q1.get_nowait()
        return ev

    ev = asyncio.run(_drive())
    assert "approval-request" in ev
    import json as _json
    payload = _json.loads(ev.split("data:", 1)[1])
    assert payload["session_id"] == "sess-aaa"
    assert payload["tool"] == "delete_history"
    assert payload["priority"] == "dangerous_write"


def test_approval_pending_filters_by_session(react_client):
    """P1-8: GET /approval/pending?session_id=… 只回该会话的待办。"""
    from src.web.deps import get_approval_bus

    bus = get_approval_bus()
    bus.request_approval({"tool": "a", "args": {}, "session_id": "s1"})
    bus.request_approval({"tool": "b", "args": {}, "session_id": "s2"})
    try:
        r = react_client.get("/api/agent/approval/pending?session_id=s1")
        assert r.status_code == 200, r.text
        items = r.json()["pending"]
        assert len(items) == 1 and items[0]["tool"] == "a"
        r2 = react_client.get("/api/agent/approval/pending")
        assert len(r2.json()["pending"]) >= 2
    finally:
        # 清掉测试注入的 pending,避免污染其它用例
        for it in bus.pending_requests():
            bus.decide(it["pin"], False)


# ---------------------------------------------------------------------------
# 3) 前端审批映射守护（新增写工具忘映射 → 红）
# ---------------------------------------------------------------------------


WRITE_TOOLS = [
    "delete_history", "delete_video", "highlight_cut", "trigger_batch",
    "start_rtsp_monitor", "generate_skill", "make_subtitle",
    "make_short_video", "make_voiceover", "create_cut_clip",
    "web_browser_trigger", "web_browser_update",
    "cdp_evaluate", "cdp_eval_write", "send_message",
]


def test_approval_maps_cover_all_write_tools():
    """P1-8: approvalMaps.ts 必须覆盖全部写/危险写工具(中文名 + 后果)。"""
    maps_path = Path(__file__).resolve().parents[1] / "webapp" / "src" / "lib" / "approvalMaps.ts"
    assert maps_path.is_file(), "approvalMaps.ts 不存在"
    text = maps_path.read_text(encoding="utf-8")

    missing_cn = [t for t in WRITE_TOOLS if f"{t}:" not in text]
    missing_cq = [t for t in WRITE_TOOLS if t not in text]
    assert not missing_cn, f"缺中文名映射: {missing_cn}"
    assert not missing_cq, f"缺后果文案: {missing_cq}"
    assert "Record<string, string>" in text