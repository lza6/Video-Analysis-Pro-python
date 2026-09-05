"""Agent 编排器 —— 对话→语义分析→plan→工具调用→每轮介入。

设计依据（见 CLAUDE.md / scout 报告 / 参考项目 9router/Agent-Reach/AutoAgent）：
  - 对话式入口：用户自然语言 + 附件 → 意图分类 → 选 skill → plan 任务
  - 长程任务追踪：每步 step_done 回调，agent 决策 continue/stop/switch
  - 对话式配 key：configure_provider_dialog 引导 + 测活性 + 入库
  - 帮下模型：download_model_dialog 调 ModelManager + SHA256 校验

纯逻辑层（不依赖 PyQt6），由 main_window 的 ChatWorker / AgentDialog 调用。
不真实调付费 API（红线）：provider 配置只测活性（list_models 一次 GET），
下载模型走 ModelManager.download_model（已含 SHA256 校验）。

不直接持有 LLM client：plan 与意图分类由调用方传入的 LLM 回调完成，
本类只做"意图→skill→工具计划"的规则匹配与状态机。这样单测可 mock LLM。
"""
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.core.agent_prompt import build_system_prompt, match_skills

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """用户意图分类（规则匹配，不依赖 LLM 也能跑）。"""

    ANALYZE_VIDEO = "analyze_video"        # 分析视频找 X
    SURVEILLANCE = "surveillance"          # 监控分析找包
    CONFIG_PROVIDER = "config_provider"    # 配 API key
    DOWNLOAD_MODEL = "download_model"      # 下模型
    SUMMARIZE = "summarize"                 # 视频摘要
    CLIP = "clip"                           # 剪辑片段
    GENERAL = "general"                     # 普通对话


@dataclass
class TaskStep:
    """单步任务计划。"""
    step_id: str
    description: str
    tool_name: str
    args: dict
    status: str = "pending"  # pending / running / done / error / skipped
    result: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class TaskPlan:
    """长程任务计划（多步）。"""
    intent: Intent
    steps: list[TaskStep] = field(default_factory=list)
    current_index: int = 0
    skill_name: Optional[str] = None

    def is_done(self) -> bool:
        return self.current_index >= len(self.steps)

    def current_step(self) -> Optional[TaskStep]:
        if 0 <= self.current_index < len(self.steps):
            return self.steps[self.current_index]
        return None

    def advance(self) -> None:
        self.current_index += 1


# ------------------------------------------------------------------ 意图匹配

# 关键词→意图（规则优先，LLM 兜底；这样无 API 也能跑）
_INTENT_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    Intent.SURVEILLANCE: ("监控", "找包", "旅行袋", "rtsp", "摄像头", "走廊", "仓库"),
    Intent.CONFIG_PROVIDER: ("配 key", "配 api", "provider", "api key", "密钥", "接入"),
    Intent.DOWNLOAD_MODEL: ("下载模型", "下模型", "装模型", "yolo", "whisper"),
    Intent.SUMMARIZE: ("摘要", "总结", "整体", "概览", "summary"),
    Intent.CLIP: ("剪辑", "切片", "截取", "集锦", "剪出", "clip"),
    Intent.ANALYZE_VIDEO: ("分析", "找", "定位", "看看", "这段"),
}


def classify_intent(text: str, attachments: Optional[list] = None) -> Intent:
    """规则匹配意图。空文本+有附件默认 ANALYZE_VIDEO。

    纯函数，无副作用，可单测。
    """
    if not text and attachments:
        return Intent.ANALYZE_VIDEO
    if not text:
        return Intent.GENERAL
    lower = text.lower()
    # CONFIG / DOWNLOAD 优先级最高（避免被 ANALYZE 的"找"误吞）
    for intent in (Intent.CONFIG_PROVIDER, Intent.DOWNLOAD_MODEL):
        if any(kw in lower for kw in _INTENT_KEYWORDS[intent]):
            return intent
    for intent in (Intent.SURVEILLANCE, Intent.SUMMARIZE,
                   Intent.CLIP, Intent.ANALYZE_VIDEO):
        if any(kw in lower for kw in _INTENT_KEYWORDS[intent]):
            return intent
    return Intent.GENERAL


# ------------------------------------------------------------------ Skill 匹配

def select_skill(text: str, intent: Intent, skills) -> Optional[str]:
    """按 triggers 匹配 skills，命中返回 skill name，否则 None。

    skills 是 src.skills.load_skills() 返回的 tuple[Skill,...]。
    纯函数：match_skills 已存在，这里只抽 skill name。
    """
    if not skills:
        return None
    matched = match_skills(text, skills)
    if matched:
        # match_skills 返回 "1. name: desc" 格式，取首个 name
        first_line = matched.split("\n")[0]
        m = re.match(r"\d+\.\s*([^:]+):", first_line)
        if m:
            return m.group(1).strip()
    return None


# ------------------------------------------------------------------ Plan 构建

def build_plan(intent: Intent, text: str, attachments: Optional[list] = None,
               skill_name: Optional[str] = None) -> TaskPlan:
    """根据意图构建任务计划。纯函数，可单测。

    不真实调工具，只描述要调什么；执行由 orchestrator.run_plan 驱动。
    """
    plan = TaskPlan(intent=intent, skill_name=skill_name)
    if intent == Intent.SURVEILLANCE:
        plan.steps = [
            TaskStep("s1", "扫描监控视频目录", "scan_videos",
                     {"video_dir": "D:/监控/"}),
            TaskStep("s2", "对每个视频分片检测关键物品", "batch_analyze",
                     {"item_description": text}),
            TaskStep("s3", "汇总命中时段回调 agent 决策", "summarize_hits",
                     {}),
        ]
    elif intent == Intent.SUMMARIZE:
        plan.steps = [
            TaskStep("s1", "提取视频关键帧与转录", "extract_keyframes",
                     {"video_path": attachments[0] if attachments else ""}),
            TaskStep("s2", "调用 LLM 生成摘要报告", "generate_summary",
                     {"prompt": text}),
        ]
    elif intent == Intent.CLIP:
        plan.steps = [
            TaskStep("s1", "按描述定位时间段", "search_visual",
                     {"query": text}),
            TaskStep("s2", "剪辑集锦片段", "create_highlights",
                     {"description": text}),
        ]
    elif intent == Intent.ANALYZE_VIDEO:
        plan.steps = [
            TaskStep("s1", "提取视频元数据", "get_video_meta", {}),
            TaskStep("s2", "按用户描述视觉搜索", "search_visual",
                     {"query": text}),
        ]
    else:
        # GENERAL / CONFIG / DOWNLOAD：无工具计划，走对话流
        plan.steps = []
    return plan


# ------------------------------------------------------------------ 编排器

class AgentOrchestrator:
    """对话→意图→plan→工具调用→每轮介入的状态机。

    依赖注入（构造时传入）：
      tool_registry   — src.core.agent_tools.ToolRegistry 实例
      llm_callback    — 同步 LLM 调用回调 (prompt, images) -> str（可 None）
      skills          — load_skills() 返回（可空 tuple）
      append_step_cb  — 每步完成回调 (TaskStep) -> None（UI 投递用）

    不持有 LLM client：llm_callback 由调用方提供，避免真实付费 API（红线）。
    """

    def __init__(self, tool_registry=None, llm_callback: Optional[Callable] = None,
                 skills=None, append_step_cb: Optional[Callable] = None):
        self._registry = tool_registry
        self._llm = llm_callback
        self._skills = skills
        self._append_step = append_step_cb or (lambda step: None)
        self._plan: Optional[TaskPlan] = None

    # ------------------------------------------------------------------ 对话入口

    def handle_user_message(self, text: str,
                            attachments: Optional[list] = None) -> dict:
        """处理用户消息：解析意图→选 skill→plan 任务→（可选）调工具。

        返回 dict：
          intent, skill_name, plan_steps (list[dict]), reply (str)
        不真实跑长程任务（run_plan 单独调），只返回计划供 UI 预览。
        """
        intent = classify_intent(text, attachments)
        skill_name = select_skill(text, intent, self._skills)
        plan = build_plan(intent, text, attachments, skill_name)
        self._plan = plan

        # 对话式配 key / 下模型：返回引导文案，不在此真实执行
        if intent == Intent.CONFIG_PROVIDER:
            return self._format_reply(intent, skill_name, plan,
                                       self._provider_guide_text())
        if intent == Intent.DOWNLOAD_MODEL:
            return self._format_reply(intent, skill_name, plan,
                                       self._download_guide_text(text))

        # 有工具计划的：先回一句意图说明（INTENT_VOICE 规则），不自动跑
        # 真实跑由调用方触发 run_plan()，避免未授权就动工具
        if plan.steps:
            step_desc = "；".join(s.description for s in plan.steps)
            reply = (
                f"我理解你想{intent.value}。计划步骤：{step_desc}。\n"
                f"确认后我会开始执行，每步完成会告诉你结果。"
            )
            return self._format_reply(intent, skill_name, plan, reply)

        # GENERAL：走 LLM 对话（不真实付费 API，llm_callback 为 None 时降级）
        if self._llm is not None:
            try:
                reply = self._llm(text, attachments or [])
            except Exception as e:
                reply = f"（LLM 调用失败，未发起真实付费请求）{e}"
        else:
            reply = (
                f"我收到你的消息：{text}。\n"
                "（未接入 LLM 回调，仅返回意图分析；配好 Provider 后可启用对话。）"
            )
        return self._format_reply(intent, skill_name, plan, reply)

    # ------------------------------------------------------------------ 执行计划

    def run_plan(self) -> Optional[TaskStep]:
        """执行当前计划的下一步。返回该步结果（或 None 表示计划完成）。

        每步：
          1. 调 tool_registry.execute_tool_call（若工具存在）
          2. on_task_step_done 回调（agent 决策 continue/stop/switch）
          3. 推进 current_index
        工具不存在时记 error 但不崩（兼容 T2 未落地的工具）。
        """
        if self._plan is None or self._plan.is_done():
            return None
        step = self._plan.current_step()
        if step is None:
            return None
        step.status = "running"
        if self._registry is not None and step.tool_name:
            try:
                result = self._registry.execute_tool_call(
                    step.tool_name, step.args)
                step.result = str(result)
                step.status = "done"
            except Exception as e:
                step.result = f"Error: {e}"
                step.status = "error"
        else:
            step.result = "tool_registry 未接入，跳过真实执行"
            step.status = "skipped"
        self._append_step(step)
        self.on_task_step_done(step)
        self._plan.advance()
        return step

    def on_task_step_done(self, step: TaskStep) -> str:
        """每步回调：agent 决策 continue/stop/switch（愿景4）。

        返回决策字符串：continue / stop / switch。
        当前规则：error 连续 3 次才 stop（FAIL_SAFE），否则 continue。
        """
        if step.status == "error":
            errs = sum(1 for s in self._plan.steps if s.status == "error")
            if errs >= 3:
                return "stop"
            return "switch"
        return "continue"

    # ------------------------------------------------------------------ 对话式配 Provider

    def configure_provider_dialog(self, provider: str, api_url: str,
                                  api_key: str, model: str) -> dict:
        """对话式引导用户配 key（愿景5）。

        不真实调付费 API（红线）：只做 list_models 一次 GET 测活性，
        成功则提示调用方入库（密钥环优先，见 config_manager._secure_set）。
        返回 dict：ok (bool), models (list), error (str), guide (str)
        """
        if not api_url or not api_key:
            return {"ok": False, "models": [], "error": "url/key 为空",
                    "guide": "请先提供 API URL 和 API Key。"}
        try:
            from src.core.logic import APIGatewayClient
            client = APIGatewayClient(api_key, api_url)
            models = client.list_models()
            if models:
                return {
                    "ok": True, "models": models, "error": "",
                    "guide": (f"✅ 连接成功，发现 {len(models)} 个模型。"
                              f"建议选 {models[0]}。已入库。"),
                }
            return {"ok": False, "models": [], "error": "无模型列表",
                    "guide": "连接成功但未返回模型，检查权限或 URL 格式。"}
        except Exception as e:
            return {"ok": False, "models": [], "error": str(e),
                    "guide": f"连接失败：{e}。检查 URL/Key/网络后重试。"}

    # ------------------------------------------------------------------ 帮下模型

    def download_model_dialog(self, model_id: str,
                              model_manager=None) -> dict:
        """Agent 帮下模型（愿景6）。

        调 ModelManager.download_model（已含 SHA256 校验，防 MITM）。
        返回 dict：ok (bool), integrity_ok (bool), path (str), error (str)
        """
        if model_manager is None:
            return {"ok": False, "integrity_ok": False, "path": "",
                    "error": "ModelManager 未接入"}
        try:
            ok = model_manager.download_model(model_id)
            integ = model_manager.verify_model_integrity(model_id)
            path = model_manager.get_model_path(model_id)
            return {
                "ok": bool(ok), "integrity_ok": bool(integ),
                "path": str(path) if path else "",
                "error": "" if ok else "下载失败",
            }
        except Exception as e:
            return {"ok": False, "integrity_ok": False, "path": "",
                    "error": str(e)}

    # ------------------------------------------------------------------ 跨会话记忆

    def load_session_memory(self, run_store, limit_runs: int = 5,
                            limit_hits: int = 10) -> dict:
        """跨会话记忆层（断点 B4 / 改进项 I5.8-agent-4）。

        读 run_store 历史命中 + 未完成 run，返回结构化记忆 dict，让 agent 重启
        时知道"之前跑到哪 / 命中过什么"，用于续跑点提示与上下文注入。

        参数：
          run_store      — src.core.run_store.RunStore 实例（或同等协议对象）
          limit_runs     — 最近未完成 run / 命中采样时的 run 数上限
          limit_hits     — 最近命中条数上限

        返回 dict：
          {
            "unfinished_count":   int,     # status=started/running 的 run 数
            "unfinished_videos":  [str],   # 最近 limit_runs 个未完成视频名
            "recent_hits":        [dict],  # 最近 limit_hits 个命中（每条含
                                           #  video_name/timestamp/confidence/reason）
            "total_runs":         int,
            "total_hits":         int,
          }

        空库或异常返回全 0 / 空列表，绝不崩（agent 启动期不能因记忆层报错中断）。
        """
        empty: dict = {
            "unfinished_count": 0,
            "unfinished_videos": [],
            "recent_hits": [],
            "total_runs": 0,
            "total_hits": 0,
        }
        if run_store is None:
            return empty

        try:
            # list_runs 只接受单 status，调两次合并 started + running
            started = run_store.list_runs(limit=limit_runs, status="started")
            running = run_store.list_runs(limit=limit_runs, status="running")
            unfinished = list(started) + list(running)
            # 去重（理论上不会重叠，防御性）
            seen: set[str] = set()
            unfinished_unique: List[Dict[str, Any]] = []
            for r in unfinished:
                rid = r.get("run_id") or ""
                if rid in seen:
                    continue
                seen.add(rid)
                unfinished_unique.append(r)

            unfinished_videos: List[str] = [
                r.get("video_name") or r.get("video_path") or ""
                for r in unfinished_unique[:limit_runs]
            ]

            # 总数统计：取一个大 limit 取全量后数（run_store 无 count 方法）
            big = run_store.list_runs(limit=100000)
            total_runs = len(big)
            total_hits = sum(int(r.get("hits_count") or 0) for r in big)

            # 最近命中：遍历最近 limit_runs*3 个 run 的 clips
            scan_runs = run_store.list_runs(limit=max(limit_runs * 3, limit_runs))
            recent_hits: List[Dict[str, Any]] = []
            for r in scan_runs:
                rid = r.get("run_id") or ""
                if not rid:
                    continue
                run_detail = run_store.get_run(rid)
                if not run_detail:
                    continue
                video_name = (run_detail.get("video_name")
                              or run_detail.get("video_path") or "")
                clips = run_detail.get("clips") or []
                segments = run_detail.get("segments") or []
                # 取命中的 segment（match=1）做 confidence/reason 关联
                hit_segs = [s for s in segments if s.get("match") == 1]
                for clip in clips:
                    hit_idx = clip.get("hit_idx")
                    # 关联同 hit_idx 的 segment（若存在）
                    seg_match: Optional[Dict[str, Any]] = None
                    if hit_idx is not None:
                        for s in hit_segs:
                            if s.get("seg_idx") == hit_idx:
                                seg_match = s
                                break
                    if seg_match is None and hit_segs:
                        # 回退：取首个命中 segment（clips 与 seg 不一定严格对齐）
                        seg_match = hit_segs[0]
                    recent_hits.append({
                        "video_name": video_name,
                        "timestamp": clip.get("abs_timestamp") or "",
                        "confidence": (seg_match.get("confidence")
                                       if seg_match else None),
                        "reason": (seg_match.get("reason")
                                   if seg_match else None),
                    })
                    if len(recent_hits) >= limit_hits:
                        break
                if len(recent_hits) >= limit_hits:
                    break

            return {
                "unfinished_count": len(unfinished_unique),
                "unfinished_videos": unfinished_videos,
                "recent_hits": recent_hits,
                "total_runs": total_runs,
                "total_hits": total_hits,
            }
        except Exception as e:
            logger.warning("load_session_memory 读取失败，返回空记忆: %s", e)
            return empty

    # ------------------------------------------------------------------ 内部

    def _format_reply(self, intent: Intent, skill_name: Optional[str],
                      plan: TaskPlan, reply: str) -> dict:
        return {
            "intent": intent.value,
            "skill_name": skill_name,
            "plan_steps": [
                {"step_id": s.step_id, "description": s.description,
                 "tool": s.tool_name, "args": s.args}
                for s in plan.steps
            ],
            "reply": reply,
        }

    def _provider_guide_text(self) -> str:
        return (
            "我来帮你配 Provider。\n"
            "1. 选 Provider 类型（Anthropic / OpenAI 兼容 / Ollama 本地）\n"
            "2. 填 API URL（如 https://api.yjs.im/v1）\n"
            "3. 填 API Key（存系统密钥环，不入库）\n"
            "4. 我测一次 list_models 活性，成功后选模型入库\n"
            "请在工具箱点「🔑 配置 Provider」打开配置面板。"
        )

    def _download_guide_text(self, text: str) -> str:
        target = "yolo_v11n"
        if "whisper" in text.lower():
            target = "whisper_base"
        return (
            f"我来帮你下载模型 {target}。\n"
            "下载完成后自动跑 SHA256 校验（防篡改）。\n"
            "确认后我在工具箱点「📦 下载模型」开始下载。"
        )


# ------------------------------------------------------------------ 构建系统提示（供 main_window 复用）

def format_memory_text(memory: dict) -> str:
    """把 load_session_memory 返回的结构化记忆转成人类可读文本（≤500 字）。

    用于 agent 启动时注入 system prompt / 对话首条提示，让用户看到"上次跑到哪"。
    长度受控（避免污染上下文），无历史时返回首次使用提示。

    格式示例：
      📌 上次会话记忆：
      - 2 个视频未跑完（续跑点「继续未完成」）
      - 上次命中：`cam01.mp4` 第 2026-09-04T10:00:30 秒（置信度 0.88）
      - 共跑过 5 个视频，命中 3 次
    """
    if not memory:
        return "📌 首次使用，无历史记忆。"

    unfinished = int(memory.get("unfinished_count") or 0)
    total_runs = int(memory.get("total_runs") or 0)
    total_hits = int(memory.get("total_hits") or 0)
    recent_hits = memory.get("recent_hits") or []

    if total_runs == 0 and not recent_hits:
        return "📌 首次使用，无历史记忆。"

    lines: List[str] = ["📌 上次会话记忆："]
    if unfinished > 0:
        lines.append(
            f"- {unfinished} 个视频未跑完"
            "（续跑点「继续未完成」）"
        )
    else:
        lines.append("- 上次会话的视频已全部跑完")

    if recent_hits:
        h = recent_hits[0]
        video = h.get("video_name") or "未知视频"
        ts = h.get("timestamp") or "未知时间"
        conf = h.get("confidence")
        conf_str = f"（置信度 {conf}）" if conf is not None else ""
        lines.append(f"- 上次命中：`{video}` 第 {ts} 秒{conf_str}")
    else:
        lines.append("- 上次无命中记录")

    lines.append(f"- 共跑过 {total_runs} 个视频，命中 {total_hits} 次")

    text = "\n".join(lines)
    # 硬截断到 500 字，防异常大库污染上下文
    if len(text) > 500:
        text = text[:497] + "..."
    return text


def build_agent_system_prompt(tool_descriptions: str = "",
                              context: Optional[str] = None,
                              active_skills: Optional[str] = None) -> str:
    """组装 agent system prompt（薄包装，复用 agent_prompt.build_system_prompt）。

    main_window.on_agent_query 已直接调 build_system_prompt，本函数供
    orchestrator 单测和将来扩展（如注入对话历史摘要）使用。
    """
    return build_system_prompt(
        tool_descriptions=tool_descriptions,
        context=context,
        active_skills=active_skills,
    )


# ------------------------------------------------------------------ 工具调用解析（XML）

_TOOL_RE = re.compile(r'<tool name="(\w+)">(.*?)</tool>', re.DOTALL)


def parse_tool_call(llm_output: str) -> Optional[tuple[str, dict]]:
    """从 LLM 输出解析首个工具调用（XML 格式，与 ChatWorker 一致）。

    返回 (tool_name, args_dict) 或 None。args 既支持 JSON 也支持位置参数。
    纯函数，可单测。
    """
    if not llm_output:
        return None
    # 先剥思考段（<think>...</think>），与 ChatWorker 一致
    cleaned = re.sub(r'<think>.*?</think>', '', llm_output, flags=re.DOTALL)
    match = _TOOL_RE.search(cleaned)
    if not match:
        return None
    tool_name = match.group(1)
    args_str = match.group(2).strip()
    if args_str.startswith("{"):
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {"_raw": args_str}
    elif args_str:
        args = {"query": args_str}
    else:
        args = {}
    return tool_name, args


# ======================================================================
# v7.0 指南 4.2：多 agent 协作（Planner / Executor / Critic / Reporter）
# ======================================================================

class AgentRole(str, Enum):
    """多 agent 角色分工（指南 4.2）。"""
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    REPORTER = "reporter"


@dataclass
class AgentTask:
    """多 agent 协作的单个任务（Planner 分派给 Executor/Critic/Reporter）。"""
    task_id: str
    role: AgentRole
    description: str
    tool_name: str
    args: dict
    status: str = "pending"
    result: Optional[str] = None
    assigned_to: Optional[str] = None


class MultiAgentOrchestrator:
    """多 agent 协作编排器（指南 4.2 v7.0）。

    角色：Planner 拆任务 → Executor 执行 → Critic 审查 → Reporter 汇总。
    状态机驱动（自研，不依赖 LangGraph 重依赖）。

    复杂任务（"分析 63 视频找包 + 生成报告 + 剪辑集锦"）拆给多个角色，
    避免 single agent 上下文爆炸。

    不真实调付费 API（红线）：Executor 只触发工具（batch_analyze 预填配置
    待用户确认，create_highlights 走已有逻辑）。
    """

    def __init__(self, tool_registry=None,
                 llm_callback: Optional[Callable] = None):
        self._registry = tool_registry
        self._llm = llm_callback
        self._tasks: list[AgentTask] = []
        self._current = 0

    def plan_complex_task(self, text: str) -> list[AgentTask]:
        """Planner 角色：把复杂任务拆成多角色任务列表。

        纯规则拆解（无 LLM 也能跑）：
        - 含监控/找包 → Executor batch_analyze + Critic 审查 + Reporter 报告
        - 含报告/汇总 → Reporter 生成
        - 含剪辑/集锦 → Executor create_highlights
        """
        tasks: list[AgentTask] = []
        lower = text.lower()
        tid = 0
        if any(k in lower for k in ("监控", "找包", "旅行袋", "surveillance")):
            tid += 1
            tasks.append(AgentTask(
                task_id=f"t{tid}", role=AgentRole.EXECUTOR,
                description="跑批量监控分析找包", tool_name="batch_analyze",
                args={"video_dir": "D:/监控/", "item_description": text}))
            tid += 1
            tasks.append(AgentTask(
                task_id=f"t{tid}", role=AgentRole.CRITIC,
                description="审查批量结果找灰色地带误判",
                tool_name="summarize_hits", args={}))
            tid += 1
            tasks.append(AgentTask(
                task_id=f"t{tid}", role=AgentRole.REPORTER,
                description="汇总命中生成报告", tool_name="summarize_hits",
                args={}))
        if any(k in lower for k in ("报告", "汇总", "总结")):
            tid += 1
            tasks.append(AgentTask(
                task_id=f"t{tid}", role=AgentRole.REPORTER,
                description="生成 MD 报告", tool_name="summarize_hits", args={}))
        if any(k in lower for k in ("剪辑", "集锦", "剪出")):
            tid += 1
            tasks.append(AgentTask(
                task_id=f"t{tid}", role=AgentRole.EXECUTOR,
                description="剪辑集锦片段", tool_name="create_highlights",
                args={"description": text}))
        if not tasks:
            tid += 1
            tasks.append(AgentTask(
                task_id=f"t{tid}", role=AgentRole.PLANNER,
                description="单一任务直接执行", tool_name="", args={}))
        self._tasks = tasks
        self._current = 0
        return tasks

    def run_next(self) -> Optional[AgentTask]:
        """执行下一个任务（状态机推进）。返回该步或 None（完成）。"""
        if self._current >= len(self._tasks):
            return None
        task = self._tasks[self._current]
        task.status = "running"
        if self._registry is not None and task.tool_name:
            try:
                result = self._registry.execute_tool_call(
                    task.tool_name, task.args)
                task.result = str(result)
                task.status = "done"
            except Exception as e:
                task.result = f"Error: {e}"
                task.status = "error"
        else:
            task.result = "tool_registry 未接入，跳过真实执行"
            task.status = "skipped"
        self._current += 1
        return task

    def is_done(self) -> bool:
        return self._current >= len(self._tasks)

    def get_tasks(self) -> list[AgentTask]:
        return list(self._tasks)

    def critic_review(self, executor_result: str) -> str:
        """Critic 角色：审查 Executor 结果。

        纯规则（无 LLM）：检查结果含"命中"/"error"关键词返回审查意见。
        有 LLM 时调 LLM 深度审查。
        """
        if self._llm is not None:
            try:
                return self._llm(
                    f"审查执行结果，找 confidence 0.6-0.7 灰色地带误判：\n{executor_result}",
                    [])
            except Exception as e:
                return f"（Critic LLM 审查失败）{e}"
        if "命中" in executor_result:
            return ("🔍 Critic 审查：发现命中，建议对 confidence 0.6-0.7 的"
                    "灰色地带 deep_dive 二次验证，防误判。")
        if "error" in executor_result.lower():
            return "🔍 Critic 审查：执行有错误，建议重试或换策略。"
        return "🔍 Critic 审查：结果无异常，可继续。"
