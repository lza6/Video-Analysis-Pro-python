"""agent_prompt v5.2 增补段 + skills 触发匹配单测。

覆盖 v5.2 八项 CL4R1T4S 增补（AGENT_LOOP/THOUGHT/CLARIFY_GATE/CITATION/
FAIL_SAFE/NOTIFY_ASK/PARALLEL/INTENT_VOICE）+ build_skills 注入 + match_skills
命中规则（正向/反向/大小写/禁用/无 triggers/空输入）。
"""
from dataclasses import dataclass
from pathlib import Path

from src.core.agent_prompt import (build_agent_loop, build_citation,
                                   build_clarify_gate, build_fail_safe,
                                   build_intent_voice, build_notify_ask,
                                   build_parallel, build_skills,
                                   build_system_prompt, build_thought,
                                   match_skills)


class TestNewPromptSections:
    def test_agent_loop_six_steps(self):
        s = build_agent_loop()
        assert "# AGENT_LOOP" in s
        for step in ("分析", "选工具", "等待", "迭代", "提交", "待命"):
            assert step in s

    def test_thought_uses_think_tag(self):
        s = build_thought()
        assert "<think>" in s and "</think>" in s

    def test_clarify_gate_requires_ambiguity_clarification(self):
        s = build_clarify_gate()
        assert "澄清" in s and "2-3 个选项" in s

    def test_citation_forbids_fabricated_timestamps(self):
        s = build_citation()
        assert "【帧" in s and "不许编时间" in s

    def test_fail_safe_three_strikes(self):
        s = build_fail_safe()
        assert "3 次" in s and "换思路" in s

    def test_notify_ask_distinction(self):
        s = build_notify_ask()
        assert "notify" in s and "ask" in s and "不许自作主张" in s

    def test_parallel_dependency_rule(self):
        s = build_parallel()
        assert "互不依赖" in s and "有依赖" in s

    def test_intent_voice_hides_tool_names(self):
        s = build_intent_voice()
        assert "不要说出工具英文名" in s


class TestSystemPromptAssembly:
    def test_all_sections_present_in_order(self):
        p = build_system_prompt(tool_descriptions="- t: d",
                                context="ctx",
                                active_skills="1. s: desc")
        order = ["# IDENTITY", "# CAPABILITIES", "# AGENT_LOOP", "# THOUGHT",
                 "# RULES", "# CLARIFY_GATE", "# TOOL_USE", "# FAIL_SAFE",
                 "# NOTIFY_ASK", "# PARALLEL", "# INTENT_VOICE", "# CONTEXT",
                 "# SKILLS", "# CITATION", "# OUTPUT"]
        positions = [p.index(seg) for seg in order]
        assert positions == sorted(positions), "段落顺序错乱"

    def test_no_skills_section_when_none(self):
        p = build_system_prompt(tool_descriptions="- t: d")
        assert "# SKILLS" not in p

    def test_no_context_section_when_none(self):
        p = build_system_prompt(tool_descriptions="- t: d")
        assert "# CONTEXT" not in p

    def test_backward_compat_signature(self):
        """旧调用（只有 tool_descriptions/context）不报错。"""
        p = build_system_prompt(tool_descriptions="- t: d", context="c")
        assert "# IDENTITY" in p and "# CONTEXT" in p

    def test_prompt_size_within_small_model_budget(self):
        """本地小模型上下文预算：全段+样例工具描述应 <4000 字符。"""
        p = build_system_prompt(tool_descriptions="- " + "x" * 300)
        assert len(p) < 4000


@dataclass(frozen=True)
class _FakeSkill:
    name: str
    description: str
    triggers: tuple
    path: Path = Path(".")
    enabled: bool = True


class TestBuildSkills:
    def test_empty_returns_empty_string(self):
        assert build_skills(None) == ""
        assert build_skills("") == ""

    def test_nonempty_formats(self):
        s = build_skills("1. a: b")
        assert s.startswith("# SKILLS") and "1. a: b" in s


class TestMatchSkills:
    def _skills(self, enabled=True):
        return (
            _FakeSkill("funclip-clip", "按文本剪辑视频",
                       ("剪辑", "切片", "说话人"), enabled=enabled),
            _FakeSkill("luxtts-voiceover", "本地 TTS 配音",
                       ("配音", "旁白"), enabled=enabled),
        )

    def test_hit_by_trigger_in_text(self):
        out = match_skills("帮我剪辑一下这段视频", self._skills())
        assert out is not None and "funclip-clip" in out

    def test_hit_case_insensitive(self):
        sk = (_FakeSkill("s", "d", ("Summary",)),)
        assert match_skills("please summary this", sk) is not None

    def test_no_hit_returns_none(self):
        assert match_skills("今天天气怎么样", self._skills()) is None

    def test_disabled_skill_skipped(self):
        assert match_skills("帮我剪辑", self._skills(enabled=False)) is None

    def test_skill_without_triggers_skipped(self):
        sk = (_FakeSkill("s", "d", ()),)
        assert match_skills("剪辑", sk) is None

    def test_empty_inputs(self):
        assert match_skills("", self._skills()) is None
        assert match_skills("剪辑", ()) is None
        assert match_skills("剪辑", None) is None

    def test_numbered_output(self):
        sk = (_FakeSkill("a", "d1", ("剪辑",)), _FakeSkill("b", "d2", ("配音",)))
        out = match_skills("剪辑并配音", sk)
        assert out is not None
        assert out.startswith("1. a") and "2. b" in out
