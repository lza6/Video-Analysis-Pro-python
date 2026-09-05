"""Agent 对话路由(ReAct 流式)。

  POST /api/agent/chat  → 同步返回意图解析 + 计划;若 GENERAL 走 LLM 流式(SSE token)

复用 src/core/agent_orchestrator.AgentOrchestrator(纯同步,无 QThread)。
LLM 走 src/core/logic.build_llm_client(ConfigManager),读 LastUsed 配置。
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..deps import get_config_manager, get_job_store
from ..security import require_auth, require_rate_limit

log = logging.getLogger("web.agent")

router = APIRouter(prefix="/api/agent", tags=["agent"])


class AgentChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    job_id: str | None = Field(
        None, description="关联分析作业(工具调用需要 frames/video_path 时传)"
    )


@router.post(
    "/chat",
    dependencies=[Depends(require_auth), Depends(require_rate_limit)],
)
def chat(req: AgentChatRequest, request: Request) -> dict:
    """处理用户消息:解析意图 → 选 skill → plan。

    返回: intent / skill_name / plan_steps / reply。
    不真实跑长程任务(run_plan 由前端二次触发 /api/agent/run)。
    LLM 回调在 GENERAL 意图时调用,读 ConfigManager.LastUsed 配置;
    若未配 Provider,降级返回意图分析(不真实付费 API)。
    """
    cm = get_config_manager()
    orchestrator = _build_orchestrator(request, cm, req.job_id)
    result = orchestrator.handle_user_message(req.text)
    return result


class AgentRunRequest(BaseModel):
    job_id: str | None = None
    text: str | None = None  # 若已有 plan 可不传,复用上次


@router.post("/run", dependencies=[Depends(require_auth)])
def run_step(req: AgentRunRequest, request: Request) -> dict:
    """执行当前计划的下一步工具调用。返回该步结果。"""
    cm = get_config_manager()
    orchestrator = _build_orchestrator(request, cm, req.job_id)
    if req.text:
        orchestrator.handle_user_message(req.text)
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


# ============================ helpers ============================

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

            def cb(prompt, images=None):  # type: ignore[no-redef]
                return _nvidia_chat(router, prompt, images)

            return cb
    except Exception as e:
        log.info(f"nvidia router 不可用,回退单 provider: {e}")

    # (2) 单 provider(llm_gateway)
    try:
        from src.core.logic import build_llm_client
        client = build_llm_client(cm)

        def cb(prompt, images=None):  # type: ignore[no-redef]
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


def _nvidia_chat(router, prompt, images=None) -> str:
    """用 ProviderRouter 发 NVIDIA chat/completions,流式拼接纯文本。

    prompt 支持 str 或 messages 列表(多轮 ReAct)。images 是帧 path 列表,
    NVIDIA 视频模型走 frames 字段(由 build_nvidia_payload 处理),这里只
    转成 messages 不塞图(视频分析用 frames 流)。
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
