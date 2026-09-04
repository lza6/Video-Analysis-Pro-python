"""AgentDialog + AgentOrchestrator 冒烟测试（offscreen）。

不发起真实付费 API / 真实模型下载——只验证：
  1. AgentDialog 构造不崩（PyQt6 控件树正常建立）
  2. 工具箱 + 消息列表 + 输入区控件存在
  3. add_tool_page 添加工具页后 list_tools 增项
  4. append_user_message / append_agent_message 添加气泡
  5. message_sent 信号在 _on_send 时触发
  6. AgentOrchestrator.classify_intent 规则匹配
  7. AgentOrchestrator.build_plan 各意图步骤数
  8. AgentOrchestrator.handle_user_message 返回结构正确
  9. AgentOrchestrator.run_plan mock 工具调用推进步骤
  10. parse_tool_call XML 解析
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from src.ui.agent_dialog import AgentDialog
from src.core.agent_orchestrator import (
    AgentOrchestrator,
    Intent,
    TaskStep,
    build_plan,
    classify_intent,
    parse_tool_call,
    select_skill,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------- AgentDialog

class TestAgentDialogConstruct:
    def test_construct_does_not_crash(self, qapp):
        dlg = AgentDialog()
        assert dlg.__class__.__name__ == "AgentDialog"

    def test_key_controls_exist(self, qapp):
        dlg = AgentDialog()
        assert dlg.list_tools is not None
        assert dlg.scroll is not None
        assert dlg.input_msg is not None
        assert dlg.btn_send is not None
        assert dlg.btn_upload is not None
        assert dlg.thinking_widget is not None
        assert dlg.stacked is not None

    def test_default_toolbox_has_dialog_entry(self, qapp):
        dlg = AgentDialog()
        assert dlg.list_tools.count() >= 1
        assert dlg.stacked.count() == 1  # 只对话页


class TestAgentDialogAddToolPage:
    def test_add_tool_page_increases_list(self, qapp):
        dlg = AgentDialog()
        before = dlg.list_tools.count()
        dlg.add_tool_page("🎥", "监控分析", QWidget(), "surveillance")
        assert dlg.list_tools.count() == before + 1
        assert dlg.stacked.count() == 2  # 对话页 + 工具页

    def test_add_tool_page_emits_tool_requested(self, qapp):
        dlg = AgentDialog()
        dlg.add_tool_page("🎥", "监控分析", QWidget(), "surveillance")
        received = []
        dlg.tool_requested.connect(lambda tid: received.append(tid))
        # row 1 是新加的工具项（row 0 是对话首页）
        dlg.list_tools.setCurrentRow(1)
        assert received == ["surveillance"]


class TestAgentDialogMessages:
    def test_append_user_message_adds_bubble(self, qapp):
        dlg = AgentDialog()
        # chat_layout 末尾是 stretch，count >= 1
        before = dlg.chat_layout.count()
        dlg.append_user_message("你好")
        assert dlg.chat_layout.count() == before + 1

    def test_append_agent_message_sets_last_bubble(self, qapp):
        dlg = AgentDialog()
        dlg.append_agent_message("我在", model_name="test-model")
        assert dlg._last_bubble is not None

    def test_update_last_bubble_appends(self, qapp):
        dlg = AgentDialog()
        dlg.append_agent_message("hello")
        dlg.update_last_bubble(" world")
        assert "hello" in dlg._last_bubble.full_text
        assert "world" in dlg._last_bubble.full_text

    def test_clear_messages_resets(self, qapp):
        dlg = AgentDialog()
        dlg.append_user_message("x")
        dlg.append_agent_message("y")
        dlg.clear_messages()
        # 只剩 stretch
        assert dlg.chat_layout.count() == 1
        assert dlg._last_bubble is None

    def test_append_tool_call_shows_tool_and_result(self, qapp):
        dlg = AgentDialog()
        dlg.append_tool_call("search_visual", {"query": "包"}, "时间点 12.0s")
        assert dlg._last_bubble is not None
        assert "search_visual" in dlg._last_bubble.full_text
        assert "12.0s" in dlg._last_bubble.full_text


class TestAgentDialogSend:
    def test_send_emits_message_sent(self, qapp):
        dlg = AgentDialog()
        received = []
        dlg.message_sent.connect(
            lambda text, atts: received.append((text, atts)))
        dlg.input_msg.setPlainText("分析监控找包")
        dlg._on_send()
        assert len(received) == 1
        assert received[0][0] == "分析监控找包"
        assert received[0][1] == []

    def test_send_empty_no_emit(self, qapp):
        dlg = AgentDialog()
        received = []
        dlg.message_sent.connect(
            lambda text, atts: received.append((text, atts)))
        dlg._on_send()
        assert received == []


class TestAgentDialogAttachments:
    def test_add_attachment_appears_in_area(self, qapp, tmp_path):
        f = tmp_path / "x.mp4"
        f.write_text("dummy")
        dlg = AgentDialog()
        dlg.add_attachment(str(f))
        assert len(dlg.get_attachments()) == 1
        assert dlg.attach_area.count() == 1


# ---------------------------------------------------------------- Orchestrator

class TestClassifyIntent:
    def test_surveillance(self):
        assert classify_intent("分析监控找包") == Intent.SURVEILLANCE

    def test_config_provider(self):
        assert classify_intent("帮我配 api key") == Intent.CONFIG_PROVIDER

    def test_download_model(self):
        assert classify_intent("下载模型 yolo") == Intent.DOWNLOAD_MODEL

    def test_summarize(self):
        assert classify_intent("给我一个整体摘要") == Intent.SUMMARIZE

    def test_clip(self):
        assert classify_intent("剪出这段话") == Intent.CLIP

    def test_analyze_with_attachments_only(self):
        assert classify_intent("", ["x.mp4"]) == Intent.ANALYZE_VIDEO

    def test_general_empty(self):
        assert classify_intent("") == Intent.GENERAL


class TestBuildPlan:
    def test_surveillance_plan_has_three_steps(self):
        plan = build_plan(Intent.SURVEILLANCE, "找包")
        assert len(plan.steps) == 3
        assert plan.steps[0].tool_name == "scan_videos"

    def test_clip_plan_has_two_steps(self):
        plan = build_plan(Intent.CLIP, "剪出这段")
        assert len(plan.steps) == 2
        assert plan.steps[1].tool_name == "create_highlights"

    def test_general_plan_empty(self):
        plan = build_plan(Intent.GENERAL, "你好")
        assert plan.steps == []


class TestSelectSkill:
    def test_no_skills_returns_none(self):
        assert select_skill("anything", Intent.GENERAL, None) is None

    def test_match_skill_name(self):
        from src.skills.schema import Skill
        from pathlib import Path
        sk = Skill(name="funclip-clip", description="剪辑",
                   triggers=("剪辑", "切片"), path=Path("x"), enabled=True)
        name = select_skill("剪辑这段", Intent.CLIP, (sk,))
        assert name == "funclip-clip"


class TestHandleUserMessage:
    def test_surveillance_returns_plan_and_reply(self):
        orch = AgentOrchestrator(tool_registry=None, llm_callback=None)
        result = orch.handle_user_message("分析监控找包")
        assert result["intent"] == "surveillance"
        assert len(result["plan_steps"]) == 3
        assert "计划步骤" in result["reply"]

    def test_config_provider_returns_guide(self):
        orch = AgentOrchestrator()
        result = orch.handle_user_message("帮我配 api key")
        assert result["intent"] == "config_provider"
        assert "Provider" in result["reply"]

    def test_general_with_llm_callback(self):
        def fake_llm(text, atts):
            return f"echo:{text}"
        orch = AgentOrchestrator(llm_callback=fake_llm)
        result = orch.handle_user_message("你好")
        assert result["intent"] == "general"
        assert result["reply"] == "echo:你好"

    def test_general_without_llm_callback_degrades(self):
        orch = AgentOrchestrator(llm_callback=None)
        result = orch.handle_user_message("你好")
        assert "未接入 LLM" in result["reply"]


class TestRunPlan:
    def test_run_plan_advances_steps(self):
        reg = MagicMock()
        reg.execute_tool_call.return_value = "ok"
        orch = AgentOrchestrator(tool_registry=reg)
        orch._plan = build_plan(Intent.ANALYZE_VIDEO, "找包")
        step1 = orch.run_plan()
        assert step1.status == "done"
        assert step1.result == "ok"
        step2 = orch.run_plan()
        assert step2.step_id == "s2"
        # 第三次：计划完成
        assert orch.run_plan() is None

    def test_run_plan_error_handled(self):
        reg = MagicMock()
        reg.execute_tool_call.side_effect = RuntimeError("boom")
        orch = AgentOrchestrator(tool_registry=reg)
        orch._plan = build_plan(Intent.ANALYZE_VIDEO, "找包")
        step = orch.run_plan()
        assert step.status == "error"
        assert "boom" in step.result

    def test_on_task_step_done_error_returns_switch(self):
        orch = AgentOrchestrator()
        orch._plan = build_plan(Intent.ANALYZE_VIDEO, "找包")
        step = TaskStep("s1", "x", "y", {})
        step.status = "error"
        assert orch.on_task_step_done(step) == "switch"

    def test_on_task_step_done_ok_returns_continue(self):
        orch = AgentOrchestrator()
        orch._plan = build_plan(Intent.ANALYZE_VIDEO, "找包")
        step = TaskStep("s1", "x", "y", {})
        step.status = "done"
        assert orch.on_task_step_done(step) == "continue"


class TestConfigureProvider:
    def test_empty_url_returns_not_ok(self):
        orch = AgentOrchestrator()
        r = orch.configure_provider_dialog("openai", "", "key", "model")
        assert r["ok"] is False
        assert "为空" in r["error"]


class TestDownloadModel:
    def test_no_manager_returns_error(self):
        orch = AgentOrchestrator()
        r = orch.download_model_dialog("yolo_v11n", model_manager=None)
        assert r["ok"] is False
        assert "未接入" in r["error"]

    def test_mock_manager_success(self):
        mm = MagicMock()
        mm.download_model.return_value = True
        mm.verify_model_integrity.return_value = True
        mm.get_model_path.return_value = "models/yolo11n.pt"
        orch = AgentOrchestrator()
        r = orch.download_model_dialog("yolo_v11n", model_manager=mm)
        assert r["ok"] is True
        assert r["integrity_ok"] is True
        assert r["path"] == "models/yolo11n.pt"


class TestParseToolCall:
    def test_xml_json_args(self):
        out = '<tool name="get_frame_details">{"seconds": 10.5}</tool>'
        r = parse_tool_call(out)
        assert r is not None
        assert r[0] == "get_frame_details"
        assert r[1] == {"seconds": 10.5}

    def test_xml_positional_args(self):
        out = '<tool name="search_visual">黑色包</tool>'
        r = parse_tool_call(out)
        assert r is not None
        assert r[1] == {"query": "黑色包"}

    def test_thought_tag_stripped(self):
        out = '</think>思考中...<tool name="x">{}</tool>'
        r = parse_tool_call(out)
        assert r is not None
        assert r[0] == "x"

    def test_no_tool_returns_none(self):
        assert parse_tool_call("普通回复") is None
        assert parse_tool_call("") is None
