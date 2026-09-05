"""AgentOrchestrator 纯函数逻辑测试（指南 7.2 要求）。

记忆层（load_session_memory / format_memory_text）已有
test_agent_orchestrator_memory.py 覆盖；本文件补意图分类 / skill 匹配 /
plan 构建 / handle_user_message / run_plan / on_task_step_done /
parse_tool_call 这些纯函数与状态机逻辑。

设计原则：
  - 纯函数直接调，无 LLM / 无 PyQt6 / 无真实工具
  - AgentOrchestrator 用 mock tool_registry（FakeRegistry）注入，避免真实工具
  - skills 用真实 load_skills()（读 config/skills）；环境无 skill 时兼容空 tuple
  - 不发起任何真实付费 API（llm_callback 一律不传或用 lambda mock）
"""
from pathlib import Path

import pytest

from src.core.agent_orchestrator import (
    AgentOrchestrator,
    Intent,
    TaskPlan,
    TaskStep,
    build_plan,
    classify_intent,
    parse_tool_call,
    select_skill,
)
from src.skills.loader import load_skills
from src.skills.schema import Skill


# ----------------------------------------------------------------------
# 测试用 skills fixture：用真实 load_skills()，环境无 skill 时回退构造
# 2 个固定 Skill，保证 select_skill 测试可复现
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def skills():
    real = load_skills()
    if real:
        return real
    # 回退：构造 2 个 Skill 用于 select_skill 测试（环境无 skills 目录时）
    return (
        Skill(
            name="surveillance-sparse-corridor",
            description="稀疏走廊监控找包算法",
            triggers=("走廊", "监控", "找包", "rtsp"),
            path=Path("."),
            enabled=True,
        ),
        Skill(
            name="video-summary",
            description="视频摘要总结工作流",
            triggers=("摘要", "总结", "summary"),
            path=Path("."),
            enabled=True,
        ),
    )


# ----------------------------------------------------------------------
# FakeRegistry：mock tool_registry，run_plan 测试用
# ----------------------------------------------------------------------
class FakeRegistry:
    """假 ToolRegistry：execute_tool_call 返回固定结果，不调真实工具。"""

    def __init__(self, result: str = "ok", exc: Exception | None = None):
        self._result = result
        self._exc = exc
        self.calls: list[tuple[str, dict]] = []

    def execute_tool_call(self, tool_name: str, args: dict) -> str:
        self.calls.append((tool_name, dict(args)))
        if self._exc is not None:
            raise self._exc
        return self._result


# ======================================================================
# classify_intent：7 类规则匹配 + 优先级 + 附件兜底
# ======================================================================
def test_classify_intent_surveillance():
    assert classify_intent("分析监控找包") == Intent.SURVEILLANCE


def test_classify_intent_config_provider():
    assert classify_intent("帮我配 key") == Intent.CONFIG_PROVIDER


def test_classify_intent_download_model():
    assert classify_intent("下载 yolo 模型") == Intent.DOWNLOAD_MODEL


def test_classify_intent_summarize():
    assert classify_intent("视频摘要总结") == Intent.SUMMARIZE


def test_classify_intent_clip():
    assert classify_intent("剪辑精彩片段") == Intent.CLIP


def test_classify_intent_analyze():
    assert classify_intent("找视频里的人") == Intent.ANALYZE_VIDEO


def test_classify_intent_general():
    assert classify_intent("你好") == Intent.GENERAL


def test_classify_intent_empty_with_attachment():
    # 空文本 + 有附件 → ANALYZE_VIDEO（直接走附件兜底）
    assert classify_intent("", attachments=["a.mp4"]) == Intent.ANALYZE_VIDEO


def test_classify_intent_priority_config_over_analyze():
    # "配 key 找包"：CONFIG 优先级高于 SURVEILLANCE/ANALYZE
    assert classify_intent("配 key 找包") == Intent.CONFIG_PROVIDER


def test_classify_intent_empty_no_attachment():
    assert classify_intent("") == Intent.GENERAL


# ======================================================================
# select_skill：按 triggers 命中
# ======================================================================
def test_select_skill_hit(skills):
    name = select_skill("走廊里找包", Intent.SURVEILLANCE, skills)
    assert name is not None
    # 命中的 skill 名应在 skills 集合中
    all_names = [s.name for s in skills]
    assert name in all_names


def test_select_skill_no_hit(skills):
    # "你好" 不匹配任何 trigger
    assert select_skill("你好", Intent.GENERAL, skills) is None


def test_select_skill_empty_skills():
    assert select_skill("监控找包", Intent.SURVEILLANCE, ()) is None


# ======================================================================
# build_plan：按意图构建多步计划
# ======================================================================
def test_build_plan_surveillance():
    plan = build_plan(Intent.SURVEILLANCE, "找包", None, "surveillance-sparse-corridor")
    assert isinstance(plan, TaskPlan)
    assert plan.intent == Intent.SURVEILLANCE
    assert plan.skill_name == "surveillance-sparse-corridor"
    tool_names = [s.tool_name for s in plan.steps]
    assert tool_names == ["scan_videos", "batch_analyze", "summarize_hits"]
    assert len(plan.steps) == 3
    # batch_analyze 的 args 应携带用户描述
    assert plan.steps[1].args["item_description"] == "找包"


def test_build_plan_summarize():
    plan = build_plan(Intent.SUMMARIZE, "总结", ["v1.mp4"])
    assert len(plan.steps) == 2
    tool_names = [s.tool_name for s in plan.steps]
    assert tool_names == ["extract_keyframes", "generate_summary"]
    # extract_keyframes 的 args 应携带附件视频路径
    assert plan.steps[0].args["video_path"] == "v1.mp4"
    # generate_summary 的 args 应携带用户文本
    assert plan.steps[1].args["prompt"] == "总结"


def test_build_plan_summarize_no_attachment():
    plan = build_plan(Intent.SUMMARIZE, "总结", None)
    # 无附件时 video_path 为空串
    assert plan.steps[0].args["video_path"] == ""


def test_build_plan_clip():
    plan = build_plan(Intent.CLIP, "剪辑精彩片段")
    assert [s.tool_name for s in plan.steps] == ["search_visual", "create_highlights"]


def test_build_plan_analyze_video():
    plan = build_plan(Intent.ANALYZE_VIDEO, "找视频里的人")
    assert [s.tool_name for s in plan.steps] == ["get_video_meta", "search_visual"]


def test_build_plan_general_empty_steps():
    # GENERAL / CONFIG / DOWNLOAD 走对话流，无工具计划
    plan = build_plan(Intent.GENERAL, "你好")
    assert plan.steps == []
    plan_cfg = build_plan(Intent.CONFIG_PROVIDER, "配 key")
    assert plan_cfg.steps == []
    plan_dl = build_plan(Intent.DOWNLOAD_MODEL, "下载模型")
    assert plan_dl.steps == []


# ======================================================================
# AgentOrchestrator.handle_user_message：意图→skill→plan→reply
# ======================================================================
def test_handle_user_message_returns_dict(skills):
    orch = AgentOrchestrator(tool_registry=None, skills=skills)
    result = orch.handle_user_message("监控走廊找包", None)
    assert isinstance(result, dict)
    # 必填字段齐
    assert set(result.keys()) >= {"intent", "skill_name", "plan_steps", "reply"}
    assert result["intent"] == "surveillance"
    assert isinstance(result["plan_steps"], list)
    # 有计划时 reply 非空
    assert result["reply"]


def test_handle_user_message_general_no_llm(skills):
    orch = AgentOrchestrator(tool_registry=None, skills=skills,
                             llm_callback=None)
    result = orch.handle_user_message("你好", None)
    assert result["intent"] == "general"
    assert result["plan_steps"] == []
    # 无 LLM 时降级提示
    assert "未接入 LLM" in result["reply"]


def test_handle_user_message_config_provider_guide(skills):
    orch = AgentOrchestrator(tool_registry=None, skills=skills)
    result = orch.handle_user_message("帮我配 key", None)
    assert result["intent"] == "config_provider"
    # 配 key 走引导文案，无 plan 步
    assert result["plan_steps"] == []
    assert "Provider" in result["reply"] or "配" in result["reply"]


def test_handle_user_message_download_model_guide(skills):
    orch = AgentOrchestrator(tool_registry=None, skills=skills)
    result = orch.handle_user_message("下载 yolo 模型", None)
    assert result["intent"] == "download_model"
    assert result["plan_steps"] == []
    assert "yolo" in result["reply"].lower() or "下载" in result["reply"]


def test_handle_user_message_with_llm_callback(skills):
    # llm_callback 用 lambda mock，不真实付费调用
    calls = []

    def fake_llm(text, attachments):
        calls.append((text, attachments))
        return f"mock reply to {text}"

    orch = AgentOrchestrator(tool_registry=None, skills=skills,
                             llm_callback=fake_llm)
    result = orch.handle_user_message("你好", None)
    assert calls, "llm_callback 应被调用一次"
    assert result["reply"] == "mock reply to 你好"


# ======================================================================
# AgentOrchestrator.run_plan：执行下一步
# ======================================================================
def test_run_plan_executes_step(skills):
    reg = FakeRegistry(result="ok")
    orch = AgentOrchestrator(tool_registry=reg, skills=skills)
    # 先 build_plan 设 self._plan
    orch._plan = build_plan(Intent.SURVEILLANCE, "找包", None, "test-skill")
    step = orch.run_plan()
    assert step is not None
    assert step.status == "done"
    assert step.result == "ok"
    # FakeRegistry 应记录调用
    assert reg.calls and reg.calls[0][0] == "scan_videos"
    # plan 推进一步
    assert orch._plan.current_index == 1


def test_run_plan_no_registry_skips(skills):
    orch = AgentOrchestrator(tool_registry=None, skills=skills)
    orch._plan = build_plan(Intent.SURVEILLANCE, "找包", None, None)
    step = orch.run_plan()
    assert step is not None
    assert step.status == "skipped"
    assert "未接入" in step.result
    assert orch._plan.current_index == 1


def test_run_plan_error_status(skills):
    reg = FakeRegistry(exc=RuntimeError("boom"))
    orch = AgentOrchestrator(tool_registry=reg, skills=skills)
    orch._plan = build_plan(Intent.SURVEILLANCE, "找包", None, None)
    step = orch.run_plan()
    assert step.status == "error"
    assert "boom" in step.result


def test_run_plan_done_returns_none(skills):
    orch = AgentOrchestrator(tool_registry=None, skills=skills)
    # 空 plan（GENERAL）→ is_done 立即为真
    orch._plan = build_plan(Intent.GENERAL, "你好")
    assert orch.run_plan() is None


def test_run_plan_no_plan_returns_none(skills):
    orch = AgentOrchestrator(tool_registry=None, skills=skills)
    assert orch.run_plan() is None


# ======================================================================
# AgentOrchestrator.on_task_step_done：continue / stop / switch
# ======================================================================
def test_on_task_step_done_continue(skills):
    orch = AgentOrchestrator(tool_registry=None, skills=skills)
    orch._plan = build_plan(Intent.SURVEILLANCE, "找包", None, None)
    done_step = TaskStep("s1", "ok", "scan_videos", {}, status="done")
    assert orch.on_task_step_done(done_step) == "continue"


def test_on_task_step_done_switch_on_first_error(skills):
    orch = AgentOrchestrator(tool_registry=None, skills=skills)
    orch._plan = build_plan(Intent.SURVEILLANCE, "找包", None, None)
    err_step = TaskStep("s1", "err", "scan_videos", {}, status="error")
    # 第一次 error 不到 3 次 → switch
    assert orch.on_task_step_done(err_step) == "switch"


def test_on_task_step_done_stop_after_3_errors(skills):
    orch = AgentOrchestrator(tool_registry=None, skills=skills)
    orch._plan = build_plan(Intent.SURVEILLANCE, "找包", None, None)
    # 预置 3 个 error 步（含当前步本身）
    orch._plan.steps = [
        TaskStep("s1", "e1", "t1", {}, status="error"),
        TaskStep("s2", "e2", "t2", {}, status="error"),
        TaskStep("s3", "e3", "t3", {}, status="error"),
    ]
    cur = orch._plan.steps[-1]
    assert orch.on_task_step_done(cur) == "stop"


# ======================================================================
# parse_tool_call：XML 解析 + 思考段剥离
# ======================================================================
def test_parse_tool_call_xml():
    out = '<tool name="search">{"query":"猫"}</tool>'
    result = parse_tool_call(out)
    assert result is not None
    tool_name, args = result
    assert tool_name == "search"
    assert args == {"query": "猫"}


def test_parse_tool_call_thinking():
    out = (
        "</think>我来搜索一下。\n"
        '<tool name="search">{"query":"找包"}</tool>'
    )
    result = parse_tool_call(out)
    assert result is not None
    tool_name, args = result
    assert tool_name == "search"
    assert args == {"query": "找包"}


def test_parse_tool_call_no_tool():
    assert parse_tool_call("普通文本，没有工具调用") is None


def test_parse_tool_call_empty():
    assert parse_tool_call("") is None
    assert parse_tool_call(None) is None  # type: ignore[arg-type]


def test_parse_tool_call_positional_arg():
    # 非 JSON 参数 → 包成 {"query": ...}
    out = '<tool name="search">猫</tool>'
    result = parse_tool_call(out)
    assert result is not None
    tool_name, args = result
    assert tool_name == "search"
    assert args == {"query": "猫"}


def test_parse_tool_call_empty_args():
    out = '<tool name="list_models"></tool>'
    result = parse_tool_call(out)
    assert result is not None
    tool_name, args = result
    assert tool_name == "list_models"
    assert args == {}


def test_parse_tool_call_invalid_json_fallback_raw():
    # JSON 解析失败 → 走 _raw 回退
    out = '<tool name="search">{invalid json}</tool>'
    result = parse_tool_call(out)
    assert result is not None
    tool_name, args = result
    assert tool_name == "search"
    assert args == {"_raw": "{invalid json}"}


def test_parse_tool_call_picks_first_when_multiple():
    out = (
        '<tool name="search">{"query":"a"}</tool>'
        '<tool name="other">{"x":1}</tool>'
    )
    result = parse_tool_call(out)
    assert result is not None
    tool_name, args = result
    assert tool_name == "search"
    assert args == {"query": "a"}
