"""Agent 对话路由(ReAct 流式)。

  POST /api/agent/chat  → 同步返回意图解析 + 计划;若 GENERAL 走 LLM 流式(SSE token)

复用 src/core/agent_orchestrator.AgentOrchestrator(纯同步,无 QThread)。
LLM 走 src/core/logic.build_llm_client(ConfigManager),读 LastUsed 配置。

v10.2:Feature Flag `VAP_AGENT_BACKEND`(env,默认 `legacy` 零回归)。
  - `legacy`:走旧 AgentOrchestrator + _PLAN_CACHE(视频分析还在用)
  - `react`:走新 ReactLoopAgent + SessionStore(关窗不丢历史 + Turn 时间轴 API)
react 路径用 if 分支隔离,不改 legacy 函数体。SyncLLMClientAdapter
桥接现有同步 LLM cb 到 LLMClient Protocol(async stream)。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..deps import get_approval_bus, get_config_manager, get_job_store
from ..security import require_auth, require_rate_limit

log = logging.getLogger("web.agent")

router = APIRouter(prefix="/api/agent", tags=["agent"])


class AgentChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    job_id: str | None = Field(
        None, description="关联分析作业(工具调用需要 frames/video_path 时传)"
    )


class AgentRunRequest(BaseModel):
    job_id: str | None = None
    text: str | None = None  # 若已有 plan 可不传,复用上次


class ApprovalDecisionRequest(BaseModel):
    """前端对审批请求 pin 的决定(SSE approval-request 事件后回调)。"""

    allow: bool = Field(..., description="True=批准执行 / False=拒绝")


# 审批等待超时(秒)。工具 ask 后前端须在超时前调 decide;超时默认拒绝(安全侧)。
_APPROVAL_TIMEOUT_SEC = float(os.environ.get("VAP_AGENT_APPROVAL_TIMEOUT", "60"))

# 审批 SSE 事件环形缓冲(进程内,供 /run_stream 的 approval-request 事件复用)。
# key = pin;value = 已格式化的 SSE 事件文本。容量有限,decide 后由前端的
# /approval/pending 或直接回调消费。不参与业务决策(仅投递管道)。
_APPROVAL_SSE_BUFFER: dict[str, str] = {}
_APPROVAL_SSE_BUFFER_CAP = 64


def _buffer_approval_sse(pin: str, payload: dict) -> None:
    """把一次审批请求的 SSE 事件文本缓冲起来,供 /run_stream 推送。"""
    _APPROVAL_SSE_BUFFER[pin] = _sse(_APPROVAL_SSE_EVENT, payload)
    # 防内存泄漏:只留最近 _APPROVAL_SSE_BUFFER_CAP 条
    if len(_APPROVAL_SSE_BUFFER) > _APPROVAL_SSE_BUFFER_CAP:
        oldest = next(iter(_APPROVAL_SSE_BUFFER))
        _APPROVAL_SSE_BUFFER.pop(oldest, None)


# 会话级审批事件队列(react /run_stream 与 approval_fn 之间传递)。
# 懒创建、绑定当前运行 loop:审批事件由 approval_fn(在 run_turn task 中)put,
# run_stream 生成器 get 消费推送前端。同一进程只建一个,单用户桌面场景够用。
_APPROVAL_SSE_QUEUE: Any = None
_APPROVAL_QUEUE_LOCK: Any = None


def _approval_sse_queue() -> Any:
    """返回审批事件队列(懒创建,绑定当前运行 loop,线程安全)。"""
    import asyncio as _asyncio

    global _APPROVAL_SSE_QUEUE, _APPROVAL_QUEUE_LOCK
    q = _APPROVAL_SSE_QUEUE
    if q is not None:
        return q
    if _APPROVAL_QUEUE_LOCK is None:
        import threading as _threading
        _APPROVAL_QUEUE_LOCK = _threading.Lock()
    with _APPROVAL_QUEUE_LOCK:
        if _APPROVAL_SSE_QUEUE is None:
            # 当前线程未必是事件循环线程(run_stream 生成器在 uvicorn loop,
            # 测试线程可能无 loop)。直接构造,绑定不依赖 loop——Python
            # 3.10+ 的 asyncio.Queue 在首次 await 时自动绑定 running loop。
            _APPROVAL_SSE_QUEUE = _asyncio.Queue()
        return _APPROVAL_SSE_QUEUE


def _make_sse_emitting_approval_fn() -> Any:
    """构造「先 emit SSE 再 wait」的审批回调。

    MAJOR-1:把写工具触发 Ask 时的审批请求作为 SSE 事件推送。实现方式——
    用 asyncio.Queue 承载审批事件文本(run_stream 生成器消费推送),同时在
    模块级 ring buffer 留底(供事后重查 / 无 run_stream 时前端轮询 pending)。
    """

    async def _approval_fn(call, sig) -> bool:
        from src.web.deps import get_approval_bus

        bus = get_approval_bus()
        try:
            pin = bus.request_approval(
                {"tool": call.name, "args": call.args or {},
                 "reason": str(getattr(sig, "prompt", "") or ""),
                 "priority": getattr(sig, "priority", "write") or "write"})
        except Exception:  # noqa: BLE001 — 投递失败视为拒绝(安全侧)
            log.warning("ApprovalBus 投递失败,按拒绝处理: %s", call.name)
            return False
        payload = {
            "pin": pin,
            "tool": call.name,
            "args": call.args or {},
            "reason": str(getattr(sig, "prompt", "") or ""),
            "priority": getattr(sig, "priority", "write") or "write",
            "timeout": _APPROVAL_TIMEOUT_SEC,
        }
        sse_text = _sse(_APPROVAL_SSE_EVENT, payload)
        _buffer_approval_sse(pin, payload)
        # 推给正在监听的 /run_stream 生成器(无人消费则丢弃,超时 deny 兜底)
        try:
            _approval_sse_queue().put_nowait(sse_text)
        except Exception:  # noqa: BLE001
            pass
        decision = await bus.wait_decision_async(pin, timeout=_APPROVAL_TIMEOUT_SEC)
        return bool(decision)

    return _approval_fn


# ============================ 审批端点 (react 路径) ============================


@router.post(
    "/approval/{pin}/decide",
    dependencies=[Depends(require_auth)],
)
async def approval_decide(pin: str, req: ApprovalDecisionRequest) -> dict:
    """前端审批回调:对 SSE approval-request 事件里的 pin 做允许/拒绝。

    Args:
        pin: SSE 事件 approval-request.data.pin。
        req: {allow: bool}。

    Returns:
        {"decided": bool}:decided=True 表示首次成功(工具继续/被拒),
        False 表示 pin 未知或已决定(幂等,重复回调不生效)。
    """
    bus = get_approval_bus()
    decided = bus.decide(pin, req.allow)
    return {"decided": decided}


@router.get(
    "/approval/pending",
    dependencies=[Depends(require_auth)],
)
async def approval_pending() -> dict:
    """当前所有未决审批请求(供前端轮询 / SSE 首次连接推送)。

    Returns:
        {"pending": [{pin, tool, args, reason, priority?}, ...]}。
        每项字段来自 ApprovalBus.request_approval 登记的 ask 字典。
    """
    bus = get_approval_bus()
    return {"pending": bus.pending_requests()}


# ============================ backend 选择 ============================

def _get_backend(request: Request) -> str:
    """选择 agent 后端:header > env > 默认 legacy。

    header `x-agent-backend` 优先(便于测试与单请求切换),其次 env
    `VAP_AGENT_BACKEND`(默认 `legacy`,零回归)。非法值回退 legacy。
    """
    backend = request.headers.get("x-agent-backend", "").lower()
    if not backend:
        backend = os.environ.get("VAP_AGENT_BACKEND", "legacy").lower()
    return backend if backend in ("legacy", "react") else "legacy"


# ============================ SyncLLMClientAdapter ============================

# <think>...</think> 思考段 + <tool name="x">...</tool> 工具调用标签,
# 用于从 LLM 全文输出里剥出干净的面向用户的文本(delta_text)。
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_TAG_RE = re.compile(r'<tool name="\w+">.*?</tool>', re.DOTALL)


class SyncLLMClientAdapter:
    """桥接现有同步 LLM cb 到 LLMClient Protocol(async stream deltas)。

    现有 `_make_llm_callback` 返回的 sync_cb 签名是 `(prompt, images) -> str`
    (同步,阻塞)。ReactLoopAgent 期望 LLMClient.stream 是 async iterator
    产出 LLMChunk。本 adapter:
      1. 把 messages(OpenAI 格式 list[dict])拼成单 prompt 字符串
         (system + user 消息拼接,简化:tool/assistant 历史不进 prompt)
      2. 用 asyncio.to_thread 调 sync_cb(避免阻塞 event loop)
      3. 用 agent_orchestrator.parse_tool_call 解析 XML 工具调用
      4. 有工具调用:yield LLMChunk(delta_text=剥标签后的文本,
         final_tool_calls=[{id,name,args}], stop_reason=None)
      5. 无工具调用:yield LLMChunk(delta_text=全文, stop_reason="stop")
      6. sync_cb 为 None(无 LLM 凭据):yield 降级文本,不崩

    关键设计决策:
      - 单 prompt 拼接(非多轮 messages):现有 sync_cb 只接受 str prompt,
        无法接收结构化 messages 列表。v10.2 简化:只拼 system+user,
        tool_result/assistant 历史丢失。真实多轮 ReAct 推理留 v10.3
        (需 sync_cb 升级为 messages 列表签名,或改走 ProviderRouterClient)。
      - asyncio.to_thread 包同步 cb:NVIDIA router / Ollama client 都是
        同步阻塞 HTTP 调用,直接 await 会卡死 event loop。to_thread 丢
        线程池跑,与 adapter.py 的 adapt_legacy_tool 一致。
      - parse_tool_call 复用:与 legacy AgentOrchestrator 用同一个 XML
        解析器,保证两条路径的工具调用格式一致(单点维护)。
      - 降级而非崩:sync_cb 为 None 时 yield 固定降级文本,run_turn 正常
        结束(产出 assistant event + 持久化),前端拿到 reply 非空。
    """

    def __init__(self, sync_cb: Optional[Any]) -> None:
        self._sync_cb = sync_cb

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Any]:
        # 延迟 import 避免启动期强依赖(与 _build_orchestrator 风格一致),
        # 同时取 LLMChunk 供本方法内构造 chunk 用。
        from src.core.agent.loop import LLMChunk
        from src.core.agent_orchestrator import parse_tool_call

        # sync_cb 为 None:无 LLM 凭据,降级返回意图分析占位(不崩)
        if self._sync_cb is None:
            yield LLMChunk(
                delta_text="(未接入 LLM,仅返回意图分析)",
                stop_reason="stop",
            )
            return

        # 拼 prompt:system + user 消息(简化,见类 docstring)
        prompt = "\n".join(
            m.get("content", "")
            for m in messages
            if m.get("role") in ("system", "user")
        )

        # v10.2:tools 透传。有工具 schema 时传给 sync_cb(供 _nvidia_chat 塞
        # payload),无 tools 时零回归(行为与 v10.1.0 一致)。
        tools_for_cb = tools or None

        # 同步 cb 丢线程池,避免阻塞 event loop
        try:
            full_text = await asyncio.to_thread(
                self._sync_cb, prompt, images=[], tools=tools_for_cb
            )
        except Exception as e:  # noqa: BLE001 — LLM 调用失败不崩 agent
            log.warning("SyncLLMClientAdapter sync_cb 调用失败: %s", e)
            yield LLMChunk(
                delta_text=f"(LLM 调用失败,未发起真实付费请求: {e})",
                stop_reason="stop",
            )
            return

        # 解析 XML 工具调用(复用 legacy 路径的解析器,单点维护)
        parsed = parse_tool_call(full_text)
        if parsed:
            tool_name, args = parsed
            # 剥 <think>...</think> 与 <tool>...</tool> 标签,留干净文本
            cleaned = _THINK_TAG_RE.sub("", full_text)
            cleaned = _TOOL_TAG_RE.sub("", cleaned)
            yield LLMChunk(
                delta_text=cleaned.strip(),
                final_tool_calls=[{
                    "id": uuid.uuid4().hex,
                    "name": tool_name,
                    "args": args,
                }],
                stop_reason=None,
            )
        else:
            # 无工具调用:终答,stop
            yield LLMChunk(
                delta_text=full_text,
                final_tool_calls=None,
                stop_reason="stop",
            )


# ============================ /chat (legacy + react) ============================

@router.post(
    "/chat",
    dependencies=[Depends(require_auth), Depends(require_rate_limit)],
)
async def chat(req: AgentChatRequest, request: Request) -> dict:
    """处理用户消息:解析意图 → 选 skill → plan。

    legacy 路径(默认):AgentOrchestrator 同步处理,返回
    intent / skill_name / plan_steps / reply / auto_run。
    - 若 plan 有 steps 且 intent≠GENERAL → auto_run=true,前端据此自动循环
      调 /api/agent/run 逐步执行,无需用户二次确认(闭环执行)。
    - GENERAL 意图:走 LLM 对话(已有,保持),auto_run=false。
    - CONFIG/DOWNLOAD 引导文案,无 plan 步,auto_run=false。
    LLM 回调在 GENERAL 意图时调用,读 ConfigManager.LastUsed 配置;
    若未配 Provider,降级返回意图分析(不真实付费 API)。

    react 路径(VAP_AGENT_BACKEND=react):ReactLoopAgent 异步跑一个 turn,
    Session 持久化到 SessionStore(关窗不丢历史)。返回
    session_id / reply / intent="react" / plan_steps=[] / auto_run=false。
    前端拿到 session_id 后续 /run_stream 带此 id 续接历史。
    """
    backend = _get_backend(request)
    if backend == "react":
        return await _react_chat(req, request)

    # ---- legacy 路径(与改动前一致,零回归) ----
    cm = get_config_manager()
    orchestrator = _build_orchestrator(request, cm, req.job_id)
    result = orchestrator.handle_user_message(req.text)
    # 注入 auto_run 标记:有 plan 步且非 GENERAL → 自动执行
    has_steps = bool(result.get("plan_steps"))
    is_general = result.get("intent") == "general"
    result["auto_run"] = has_steps and not is_general
    # 把 plan 序列化进 state(供 run/run_stream 重建使用)
    _stash_plan(request, result.get("plan_steps") or [], req.job_id)
    return result


async def _react_chat(req: AgentChatRequest, request: Request) -> dict:
    """react 路径 /chat:跑一个 ReactLoopAgent turn + 持久化 Session。

    生成 session_id(uuid4 hex),前端拿到后续 /run_stream 带此 id 续接。
    无 LLM 凭据时 SyncLLMClientAdapter 降级返回意图分析占位,不崩。
    """
    session_id = uuid.uuid4().hex
    agent, session, session_store = _build_react_agent(
        request, session_id, req.job_id, req.text)
    result = await agent.run_turn(session, req.text)
    # 持久化(关窗不丢历史):run_turn 内部已 _persist,这里再 save 一次
    # 保证 final_text 落库(run_turn 的 _persist 在 final_text 设置后调用,
    # 但崩溃恢复路径下显式 save 更稳)
    session_store.save(session)
    return {
        "session_id": session_id,
        "reply": result.final_text or "",
        "intent": "react",
        "plan_steps": [],
        "auto_run": False,
    }


# ============================ /run (legacy only) ============================

@router.post("/run", dependencies=[Depends(require_auth), Depends(require_rate_limit)])
def run_step(req: AgentRunRequest, request: Request) -> dict:
    """执行当前计划的下一步工具调用。返回该步结果。

    用于前端 chat 后自动循环调用,直到 done=true 闭环。
    每次 run 重建 orchestrator + plan(从 chat stash),推进单步。

    注:react 路径不走此端点(react 用 /run_stream 一次跑完 + Session 持久化)。
    """
    cm = get_config_manager()
    orchestrator = _build_orchestrator(request, cm, req.job_id)
    _restore_plan(orchestrator, request, cm, req.job_id, req.text)
    step = orchestrator.run_plan()
    if step is None:
        return {"done": True, "step": None}
    return {
        "done": False,
        "step": {
            "description": getattr(step, "description", str(step)),
            "tool": getattr(step, "tool", None),
            "result": getattr(step, "result", None),
            "status": getattr(step, "status", None),
        },
    }


# ============================ /run_stream (legacy + react) ============================

@router.get(
    "/run_stream",
    dependencies=[Depends(require_auth), Depends(require_rate_limit)],
)
async def run_stream(request: Request, job_id: str | None = None) -> StreamingResponse:
    """SSE 流式自动执行整个 plan,逐步投递 step 事件直到 done。

    legacy 路径:前端一次性调用,后端循环 run_plan 直到完成,每步作为
    SSE event 投递。闭环执行:chat → run_stream → 每步实时显示。

    react 路径(VAP_AGENT_BACKEND=react):query param `session_id` 续接
    历史(无则新建)。简化方案:run_turn 整体 await(同步等完),结束前
    一次性投 `done` event + 全文。真正逐 phase 流式留 v10.3。
    """
    backend = _get_backend(request)
    if backend == "react":
        return await _react_run_stream(request, job_id)

    # ---- legacy 路径(与改动前一致,零回归) ----
    cm = get_config_manager()
    orchestrator = _build_orchestrator(request, cm, job_id)
    _restore_plan(orchestrator, request, cm, job_id, None)

    def gen():
        step_idx = 0
        while True:
            step = orchestrator.run_plan()
            if step is None:
                yield _sse("done", {"step": None, "total": step_idx})
                return
            step_idx += 1
            yield _sse("step", {
                "index": step_idx,
                "description": getattr(step, "description", str(step)),
                "tool": getattr(step, "tool", None),
                "result": getattr(step, "result", None),
                "status": getattr(step, "status", None),
            })
            # error 连续 3 次 stop(orchestrator.on_task_step_done 决策)
            decision = orchestrator.on_task_step_done(step)
            if decision == "stop":
                yield _sse("done", {"step": None, "total": step_idx,
                                    "reason": "error_limit"})
                return

    return StreamingResponse(gen(), media_type="text/event-stream")


async def _react_run_stream(
    request: Request, job_id: str | None,
) -> StreamingResponse:
    """react 路径 /run_stream:await run_turn,审批事件实时推 SSE。

    session_id 从 query param 取(无则新建),跑完 save 持久化。
    逐 phase 流式留 v10.3(需 TurnHooks + asyncio.Queue 投递)。

    v10.2 (B-FIN-2 MAJOR-1):run_turn 期间写工具触发 Ask 时,approval_fn
    会把 approval-request 事件 put 进审批队列。run_turn 在
    wait_decision_async 处让出事件循环,本 async 生成器在等待间隙轮询
    队列把事件推给前端 SSE。前端收到后调 approve 端点,decide() set
    Event,wait_decision_async 解除,run_turn 继续(工具执行 / 拒绝)。
    """
    import asyncio as _asyncio

    session_id = request.query_params.get("session_id") or uuid.uuid4().hex
    agent, session, session_store = _build_react_agent(
        request, session_id, job_id, "")
    queue = _approval_sse_queue()

    async def _run_turn_and_save() -> Any:
        result = await agent.run_turn(session, "")
        session_store.save(session)
        return result

    turn_task = _asyncio.create_task(_run_turn_and_save())

    async def gen() -> AsyncIterator[str]:
        while True:
            if turn_task.done():
                break
            try:
                # 要么取到新审批事件(前端弹审批 UI),要么 50ms 超时后
                # 重新检查 run_turn 是否已完成(stream 由此结束)。
                ev = await _asyncio.wait_for(queue.get(), timeout=0.05)
                yield ev
            except _asyncio.TimeoutError:
                continue
        # 终局排空(最后时刻 put 的审批事件)
        while not queue.empty():
            try:
                yield queue.get_nowait()
            except Exception:  # noqa: BLE001
                break
        result = turn_task.result()
        yield _sse("done", {
            "done": True,
            "session_id": session_id,
            "reply": result.final_text or "",
        })

    return StreamingResponse(gen(), media_type="text/event-stream")


# ============================ Session / Turn API (react only) ============================

@router.get(
    "/sessions",
    dependencies=[Depends(require_auth)],
)
async def list_sessions(request: Request) -> dict:
    """列出最近更新的 session(按 updated_at 倒序)。

    react 路径专属。空 store 返回空数组(友好,不 404)。
    """
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        return {"sessions": []}
    sessions = store.list_sessions(limit=100)
    return {"sessions": sessions}


@router.get(
    "/sessions/{session_id}",
    dependencies=[Depends(require_auth)],
)
async def get_session(session_id: str, request: Request) -> dict:
    """返回单个 session 的全部 events(append-only log)。

    react 路径专属。不存在返回 404。
    """
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "session store unavailable"},
        )
    session = store.load(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"session {session_id} not found"},
        )
    return {
        "session_id": session_id,
        "events": [e.to_dict() for e in session.events],
        "event_count": len(session.events),
    }


@router.get(
    "/sessions/{session_id}/turns",
    dependencies=[Depends(require_auth)],
)
async def get_session_turns(session_id: str, request: Request) -> dict:
    """返回 session 的 Turn 时间轴(按 user event 分组)。

    react 路径专属。分组规则:每个 user event 开启新 turn,后续
    assistant/tool_result/step 事件归入该 turn,直到下一个 user。
    每个 phase 节点含:phase 名 / ts / duration_ms / 可选 tool/content_preview。
    不存在返回 404。
    """
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "session store unavailable"},
        )
    session = store.load(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"session {session_id} not found"},
        )
    turns = _build_turn_timeline(session.events)
    return {
        "session_id": session_id,
        "turns": turns,
        "turn_count": len(turns),
    }


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_auth)],
)
async def delete_session(session_id: str, request: Request) -> None:
    """删除 session(幂等:不存在也 204)。

    react 路径专属。store 不可用或 session 不存在均返回 204(幂等语义)。
    """
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        return None
    store.delete(session_id)
    return None


# ============================ Turn 时间轴构建 ============================

def _build_turn_timeline(events: List[Any]) -> List[Dict[str, Any]]:
    """把 SessionEvent 列表按 user 分组为 Turn 时间轴。

    每个 turn:
      - turn_id: "turn-{idx}"
      - phases: [{phase, ts, duration_ms, tool?, content_preview?}]
    分组:system event 跳过(全局,不属任何 turn);user event 开新 turn;
    assistant/tool_result/step 归入当前 turn。
    duration_ms:当前 phase ts 到下一 phase ts 的差(末 phase 取自身 ts)。
    """
    turns: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    turn_idx = 0

    # 过滤 system(全局前缀,不属任何 turn)
    filtered = [e for e in events if e.type != "system"]
    for i, event in enumerate(filtered):
        if event.type == "user":
            if current is not None:
                turns.append(current)
            turn_idx += 1
            current = {"turn_id": f"turn-{turn_idx}", "phases": []}
        if current is None:
            # 无 user 前导的孤立 event(理论上不该有,防御性跳过)
            continue
        # duration:到下一 event 的差(末 event 取 0)
        next_ts = (filtered[i + 1].timestamp
                   if i + 1 < len(filtered) else event.timestamp)
        duration_ms = max(0.0, (next_ts - event.timestamp) * 1000)
        phase = _map_event_to_phase(event)
        phase["duration_ms"] = round(duration_ms, 1)
        current["phases"].append(phase)
    if current is not None:
        turns.append(current)
    return turns


def _map_event_to_phase(event: Any) -> Dict[str, Any]:
    """把 SessionEvent 映射成 Turn phase 节点。

    映射规则(与 loop.py run_turn 的 event append 对齐):
      - user        → claim/input(用户输入,turn 起点)
      - assistant   → 若有 tool_calls 是 tool/call;否则 assistant-stream(终答)
      - tool_result → tool/result(工具执行返回)
      - tool_call   → tool/call(兼容,run_turn 当前不产此 event 但防御性支持)
      - step/turn   → step/turn(元事件)
    """
    p = event.payload
    ts = event.timestamp
    etype = event.type
    if etype == "user":
        content = str(p.get("content", ""))
        return {
            "phase": "claim/input",
            "ts": ts,
            "content_preview": content[:100],
        }
    if etype == "assistant":
        tool_calls = p.get("tool_calls")
        if tool_calls:
            tc = tool_calls[0] if isinstance(tool_calls, list) else {}
            return {
                "phase": "tool/call",
                "ts": ts,
                "tool": tc.get("name", "") if isinstance(tc, dict) else "",
            }
        content = str(p.get("content", ""))
        return {
            "phase": "assistant-stream",
            "ts": ts,
            "content_preview": content[:100],
        }
    if etype == "tool_result":
        content = str(p.get("content", ""))
        return {
            "phase": "tool/result",
            "ts": ts,
            "content_preview": content[:100],
        }
    if etype == "tool_call":
        return {"phase": "tool/call", "ts": ts}
    if etype == "step":
        return {"phase": "step", "ts": ts}
    if etype == "turn":
        return {"phase": "turn", "ts": ts}
    return {"phase": etype, "ts": ts}


# ============================ react 路径:agent 构造 ============================

# 审批 SSE 事件名:前端监听 `event: approval-request` 后调
# POST /api/agent/approval/{pin}/decide 回调。
_APPROVAL_SSE_EVENT = "approval-request"


def _sse_approval(call: Any, sig: Any, pin: str) -> str:
    """把一次工具审批请求格式化成 SSE 事件(前端据此弹审批 UI)。

    data 字段:
      - pin: 前端 decide 回调必传的审批唯一 id
      - tool / args / reason: 展示给用户的工具名 / 参数 / 审批理由
      - priority: 审批优先级(scope_guard Ask.priority,缺省 "write")
      - timeout: 超时秒数(超时默认拒绝)
    """
    return _sse(_APPROVAL_SSE_EVENT, {
        "pin": pin,
        "tool": call.name,
        "args": call.args or {},
        "reason": str(getattr(sig, "prompt", "") or ""),
        "priority": getattr(sig, "priority", "write") or "write",
        "timeout": _APPROVAL_TIMEOUT_SEC,
    })


def _install_tool_guard(
    registry: Any,
    *,
    approval_fn: Any = None,
    sandbox: Any = None,
    enabled: bool = True,
    emit_sse_approval: bool = True,
) -> None:
    """把 ScopeGuard + approval + sandbox 装配到 react 路径的 ToolRegistry。

    延迟 import scope_guard(避免启动期强依赖)。`build_sandbox()` 在无
    pywin32/landlock 平台返回 NullSandbox(空操作,enter/run/exit 直通),
    install_tool_guard 在 sandbox 为 NullSandbox 时仍安全——registry 只在
    `_sandbox_enabled and _sandbox is not None` 时进沙箱,NullSandbox 的
    run(coro) 就是 `await coro`,不阻断工具执行(见 registry._execute_once)。

    v10.2 (B-FIN-2 MAJOR-1):approval_fn 为 None 时默认注入「先 emit SSE 再
    wait」的审批回调(经 asyncio.Queue 把 approval-request 事件推给
    /run_stream 生成器,同时缓冲留底)。emit_sse_approval=False 可显式
    关闭(测试隔离/纯默认 behavior 场景)。

    Args:
        registry: 已注册好工具的 ToolRegistry。
        approval_fn: 审批回调。None = 使用默认 SSE-emitting ApprovalBus
            实现(先 emit approval-request 事件,再异步等待;超时默认拒绝)。
        sandbox: build_sandbox() 产物(NullSandbox / WindowsJobSandbox /
            LinuxLandlockSandbox)。None = 不装配沙箱(零回归)。
        enabled: 是否启用沙箱(registry 层再受 VAP_SANDBOX_ENABLED 控制,
            见 set_sandbox 语义:enabled=False 时即便 sandbox 非 None 也不包)。
    """
    from src.core.tools.scope_guard import install_tool_guard

    if approval_fn is None and emit_sse_approval:
        approval_fn = _make_sse_emitting_approval_fn()

    install_tool_guard(
        registry,
        approval_fn=approval_fn,
        sandbox=sandbox,
        enabled=enabled,
    )


def _build_react_agent(
    request: Request,
    session_id: str,
    job_id: Optional[str],
    text: str = "",
) -> tuple:
    """构造 ReactLoopAgent + Session + SessionStore。

    返回 (agent, session, session_store)。
    - 从 app.state.session_store 取 SessionStore 单例;不存在则降级新建
    - 用 register_legacy_tools 注册 17 个老工具到新 ToolRegistry
    - v10.2 (B-ASSEMBLE):注入监督层 + 工具范围守卫(审批/sandbox)
        - 监督层:AgentConfig(supervisor=Supervisor.from_env()),
          `VAP_AGENT_SUPERVISOR` 默认关 → None → 行为零回归
        - 工具守卫:install_tool_guard,approval_fn 走 ApprovalBus
          (请求级 SSE 回调,超时默认拒绝);sandbox 用 build_sandbox()
          自动选型,`VAP_SANDBOX_ENABLED` 默认 false → 不进沙箱
    - SyncLLMClientAdapter 桥接 _make_llm_callback 的同步 cb
    - system_prompt 复用 build_agent_system_prompt(与 legacy 一致)
    - session 不存在则新建(带 system_prompt);存在则 load(历史 events 重建)
    """
    # 延迟 import 避免循环依赖与启动期强依赖
    from src.core.agent.loop import AgentConfig, ReactLoopAgent
    from src.core.agent.session import SessionStore
    from src.core.agent.supervisor import Supervisor
    from src.core.memory.connector import MemoryLayeredConnector
    from src.core.runtime.sandbox import build_sandbox
    from src.core.tools.adapter import register_legacy_tools
    from src.core.tools.registry import ToolRegistry
    from src.core.agent_orchestrator import build_agent_system_prompt

    # session_store:从 app.state 取(单例),不存在则降级新建(不崩)
    session_store = getattr(request.app.state, "session_store", None)
    if session_store is None:
        session_store = SessionStore()

    # 构造 ToolRegistry + 注册 17 个老工具
    registry = ToolRegistry()
    ctx = _AgentContext(request, job_id)
    register_legacy_tools(registry, lambda: ctx)

    # v10.2 (B-ASSEMBLE):工具范围守卫装配(审批走 ApprovalBus + SSE,
    # 超时默认拒绝;sandbox 自动选型,VAP_SANDBOX_ENABLED 默认 false)。
    # approval_fn 走 SSE-emitting 实现:写工具 Ask 时把 approval-request
    # 事件推给 /run_stream 生成器(前端弹审批 UI),前端调
    # POST /api/agent/approval/{pin}/decide 回调;超时默认 deny(安全侧)。
    # 移除 B-FIN-2 MAJOR-2 前阻塞事件循环的死锁:等待走 async Event。
    _install_tool_guard(
        registry,
        sandbox=build_sandbox(),
        enabled=(os.environ.get("VAP_SANDBOX_ENABLED", "false").strip().lower()
                 in ("1", "true", "yes", "on")),
    )

    # 工具描述(供 system_prompt 用)
    tool_descs = "\n".join(
        f"- {s['function']['name']}: {s['function']['description']}"
        for s in registry.schemas()
    )

    # active skills 可选加载(失败不阻断)。
    # v10.2 (B-FIN-2 MAJOR-4):`VAP_SKILLS_ROSTER=1` 时换用
    # src.skills.roster.resolve_skills_for_intent(渐进披露 + 领域语义路由);
    # 默认 0 保持旧 match_skills 行为(零回归)。
    active_skills: Optional[str] = None
    roster_skills: list[str] = []
    use_roster = os.environ.get("VAP_SKILLS_ROSTER", "0").strip().lower() in (
        "1", "true", "yes", "on")
    try:
        from src.skills import load_skills  # type: ignore
        from src.utils.constants import CONFIG_DIR
        from pathlib import Path as _Path
        skills = load_skills()
        if use_roster and text:
            try:
                from src.skills.roster import resolve_skills_for_intent
                roster_hits = resolve_skills_for_intent(
                    text, _Path(CONFIG_DIR) / "skills")
                roster_skills = [s.name for s in roster_hits]
            except Exception as e:  # noqa: BLE001 — roster 失败回退旧行为
                log.debug("roster 匹配失败,回退 match_skills: %s", e)
                roster_skills = []
        if roster_skills:
            active_skills = "、".join(roster_skills)
        elif skills and text:
            from src.core.agent_prompt import match_skills
            active_skills = match_skills(text, skills)
    except Exception as e:  # noqa: BLE001 — skills 可选,失败不阻断
        log.debug("skills 加载失败(react 路径,不阻断): %s", e)

    # system_prompt 复用 legacy 的 build_agent_system_prompt(单点维护)
    system_prompt = build_agent_system_prompt(
        tool_descriptions=tool_descs,
        context=None,
        active_skills=active_skills,
    )

    # LLM adapter:桥接现有同步 cb
    cm = get_config_manager()
    sync_cb = _make_llm_callback(cm, ctx)
    llm_client = SyncLLMClientAdapter(sync_cb)

    # v10.2:tools 透传——LLM 收到 registry 的工具 schema 时,
    # SyncLLMClientAdapter 需把它接出来传给 sync_cb 的 messages 列表,
    # 由 _make_llm_callback 的 cb 转发给 _nvidia_chat(tools=...)。
    # 无 tools 时零回归(行为与 v10.1.0 完全一致)。

    # load_session:不存在则新建(带 system_prompt);存在则恢复历史 events
    session = ReactLoopAgent.load_session(
        session_store, session_id, system_prompt=system_prompt)

    # v10.2 (B-FIN-2 MAJOR-3):装配分层记忆。MemoryLayeredConnector.from_env
    # 读 VAP_MEMORY_LAYERED(默认 1=开);=0 时返回 enabled=False 的实例
    # (working/experience/triples 全 None,record/inject 均为 no-op),
    # 行为与 None 等价。这里显式传 feature_flag 使测试可精确断言。
    memory_connector = MemoryLayeredConnector.from_env()
    if memory_connector is not None and not memory_connector.enabled:
        memory_connector = None

    agent = ReactLoopAgent(
        AgentConfig(supervisor=Supervisor.from_env(),
                    system_prompt=system_prompt,
                    memory=memory_connector,
                    max_steps=8),
        llm_client,
        registry,
        store=session_store,
    )
    return agent, session, session_store


# ============================ helpers ============================

def _sse(event: str, data: dict) -> str:
    """格式化 SSE event 行(text/event-stream)。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stash_key(job_id: str | None) -> str:
    """进程级 plan 缓存 key(按 job_id 隔离同一会话的 plan)。"""
    return f"agent_plan::{job_id or 'global'}"


# 进程级 plan 缓存(web 多 worker 下不共享,但单进程 dev 足够;
# 生产多 worker 部署时由前端循环调 /run 自带状态,不依赖此缓存)
_PLAN_CACHE: dict[str, list[dict]] = {}


def _stash_plan(request: Request, plan_steps: list[dict], job_id: str | None) -> None:
    """把 chat 产出的 plan 序列化缓存,供后续 /run 重建使用。

    仅缓存 plan 步描述+工具名+args,不含 LLM/敏感数据。
    """
    try:
        _PLAN_CACHE[_stash_key(job_id)] = plan_steps
        # 限制缓存大小防内存泄漏(只留最近 32 个会话)
        if len(_PLAN_CACHE) > 32:
            oldest = next(iter(_PLAN_CACHE))
            _PLAN_CACHE.pop(oldest, None)
    except Exception as e:
        log.warning(f"stash plan 失败(不影响 chat 返回): {e}")


def _restore_plan(orchestrator, request: Request, cm, job_id: str | None,
                  text: str | None) -> None:
    """从缓存重建 orchestrator._plan;无缓存时若 text 给定则重新 handle。

    run_plan 依赖 orchestrator._plan 已设;chat 已 build_plan 并 stash,
    run 端点重建 orchestrator(新实例)后从 stash 恢复 plan 状态机。
    """
    from src.core.agent_orchestrator import Intent, TaskPlan, TaskStep

    # 优先从 stash 恢复
    stashed = _PLAN_CACHE.get(_stash_key(job_id))
    if stashed:
        try:
            # 从 plan_steps 反推 intent(取首个 step 的 tool 推断)
            # 简化:重新 handle_user_message 重建 plan(纯逻辑,无副作用)
            if text:
                orchestrator.handle_user_message(text)
                return
            # 无 text 时用 stash 的 steps 重建 TaskPlan
            steps = [
                TaskStep(
                    step_id=s.get("step_id", f"s{i}"),
                    description=s.get("description", ""),
                    tool_name=s.get("tool") or "",
                    args=s.get("args") or {},
                )
                for i, s in enumerate(stashed)
            ]
            plan = TaskPlan(intent=Intent.GENERAL, steps=steps)
            orchestrator._plan = plan
            return
        except Exception as e:
            log.warning(f"restore plan 从 stash 重建失败,回退 handle: {e}")

    # 回退:有 text 就重新 handle,否则 plan 为空(run_plan 返回 None)
    if text:
        orchestrator.handle_user_message(text)


def _build_orchestrator(request: Request, cm, job_id: str | None):
    """构造 AgentOrchestrator + 注入 ToolRegistry + LLM 回调。"""
    from src.core.agent_orchestrator import AgentOrchestrator
    from src.core.agent_tools import ToolRegistry
    from src.core.agent_tools import (
        create_get_video_meta_tool,
        create_get_frame_details_tool,
        create_search_web_tool,
        create_highlight_cut_tool,
        create_kb_search_tool,
    )

    registry = ToolRegistry()

    # app_context_getter 返回一个轻量 context 对象,持有当前 job 的 frames/video_path
    ctx = _AgentContext(request, job_id)

    registry.register_tool(
        "get_video_meta", "获取当前视频元信息",
        create_get_video_meta_tool(lambda: ctx),
    )
    registry.register_tool(
        "get_frame_details", "按时间戳取帧详情",
        create_get_frame_details_tool(lambda: ctx),
    )
    registry.register_tool(
        "search_web", "网页搜索",
        create_search_web_tool(),
    )
    registry.register_tool(
        "highlight_cut", "按描述剪辑集锦",
        create_highlight_cut_tool(lambda: ctx),
    )
    registry.register_tool(
        "search_kb", "跨视频知识库搜索",
        create_kb_search_tool(lambda: ctx),
    )

    # LLM 回调:读 LastUsed 配置,构建 client;失败则 None(降级)
    llm_cb = _make_llm_callback(cm, ctx)

    # skills 可选加载(失败不阻断)
    skills = None
    try:
        from src.skills import load_skills  # type: ignore
        skills = load_skills()
    except Exception:
        skills = None

    return AgentOrchestrator(
        tool_registry=registry,
        llm_callback=llm_cb,
        skills=skills,
    )


class _AgentContext:
    """轻量 app context,供 agent_tools 各 create_*_tool(lambda: ctx) 使用。

    持有当前 job 的 frames / video_path / output_dir / history_manager。
    """
    def __init__(self, request: Request, job_id: str | None):
        self._request = request
        self._job_id = job_id
        self.video_path = None
        self.video_duration = 0.0
        self.frames = []
        self.output_dir = None
        self.history_manager = None
        self._load_job()

    def _load_job(self):
        if not self._job_id:
            return
        store = get_job_store()
        rec = store.get(self._job_id)
        if rec is None:
            return
        self.video_path = Path(rec.video_path) if rec.video_path else None
        self.video_duration = rec.duration
        self.output_dir = rec.workdir
        # frames 用 Frame 对象重建(工具内部访问 .path/.timestamp/.metrics/.vision_content)
        from src.core.logic import Frame
        self.frames = [
            Frame(
                path=rec.frames_dir / Path(f["url"]).name,
                timestamp=f["timestamp"],
                metrics=f.get("metrics", {}),
            )
            for f in rec.frames
        ]
        try:
            from src.core.history_manager import HistoryManager
            self.history_manager = HistoryManager()
        except Exception:
            self.history_manager = None

    def seek_video(self, ts: float):
        """Web 下 seek 由前端处理(后端无法直接控制播放器)。

        这里只记录意图,工具返回会提示前端跳转。真实跳转走 SSE 事件或返回值。
        """
        log.info(f"agent seek request: job={self._job_id} ts={ts}")


def _make_llm_callback(cm, ctx):
    """构造 LLM 同步回调 (prompt, images) -> str。

    优先级:
      1. 若 .env 配了 VAP_NV_API_KEYS → 走 ProviderRouter 多 key 路由(nvidia),
         用 nvidia_models.build_nvidia_payload + router.post_nvidia 流式。
      2. 否则读 LastUsed 单 provider → build_llm_client(llm_gateway)。

    缺凭据返回 None(降级为意图分析,不真实付费 API)。
    """
    # (1) NVIDIA 多 key 路由
    try:
        from src.core.provider_router import ProviderRouter, load_from_env, load_router_config_from_env
        nv_keys = [k for k in load_from_env() if k.provider == "nvidia"]
        if nv_keys:
            router = ProviderRouter(nv_keys, **load_router_config_from_env())

            def cb(prompt, images=None, tools=None):  # type: ignore[no-redef]
                return _nvidia_chat(router, prompt, images, tools=tools)

            return cb
    except Exception as e:
        log.info(f"nvidia router 不可用,回退单 provider: {e}")

    # (2) 单 provider(llm_gateway)
    try:
        from src.core.logic import build_llm_client
        client = build_llm_client(cm)

        def cb(prompt, images=None, tools=None):  # type: ignore[no-redef]
            image_paths = images or None
            chunks = []
            for chunk in client.chat_stream(
                client.model if hasattr(client, "model") else "",
                prompt,
                image_paths,
            ):
                if chunk.startswith("__"):
                    continue
                chunks.append(chunk)
            return "".join(chunks)

        return cb
    except Exception as e:
        log.info(f"LLM callback 不可用(降级为意图分析): {e}")
        return None


def _nvidia_chat(router, prompt, images=None, tools=None) -> str:
    """用 ProviderRouter 发 NVIDIA chat/completions,流式拼接纯文本。

    prompt 支持 str 或 messages 列表(多轮 ReAct)。images 是帧 path 列表,
    NVIDIA 视频模型走 frames 字段(由 build_nvidia_payload 处理),这里只
    转成 messages 不塞图(视频分析用 frames 流)。tools 是 OpenAI 兼容
    工具 schema 列表,非 None 时透传进 payload(ReactLoopAgent 工具调用)。
    """
    import os
    from src.core.nvidia_models import build_nvidia_payload
    model = os.environ.get(
        "VAP_NV_VIDEO_MODEL",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    )

    if isinstance(prompt, list):
        messages = [dict(m) for m in prompt]
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    payload = build_nvidia_payload(
        model_id=model,
        messages=messages,
        enable_thinking=True,
        stream=False,
        max_tokens=int(os.environ.get("VAP_NV_MAX_TOKENS", "65536")),
        tools=tools or None,
    )
    try:
        resp = router.post_nvidia(payload, timeout=120)
        choices = resp.get("choices", []) if isinstance(resp, dict) else []
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        parts = []
        if reasoning:
            parts.append(f"💭{reasoning}\n\n")
        if content:
            parts.append(content)
        return "".join(parts)
    except Exception as e:
        return f"[NVIDIA 调用失败: {e}]"
