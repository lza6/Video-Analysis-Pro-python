"""M4 监控分析 skills 自动路由单测。

验证 match_skills 在用户用口语化描述时（"分析监控找包""商场人流分析"）
能正确路由到对应 surveillance skill，而非默认不命中或命中错误 skill。
"""
from dataclasses import dataclass
from pathlib import Path

from src.core.agent_prompt import match_skills


@dataclass(frozen=True)
class _FakeSkill:
    name: str
    description: str
    triggers: tuple
    path: Path = Path(".")
    enabled: bool = True


def _surveillance_skills() -> tuple:
    """复刻 .claude/skills/ 下两个 surveillance skill 的 frontmatter。"""
    return (
        _FakeSkill(
            name="surveillance-sparse-corridor",
            description=(
                "稀疏走廊/楼梯口监控分析（长时间无人），"
                "1fps抽帧+场景检测+帧差分+昼夜自适应，只送变化时段给AI，省90%调用"
            ),
            triggers=(),  # 真实 SKILL.md 未列 triggers，靠语义路由
        ),
        _FakeSkill(
            name="surveillance-crowded-scene",
            description="人多密集场景监控分析（商场/路口），YOLO目标追踪，后续实现",
            triggers=(),
        ),
        # 对照组：带显式 triggers 的普通 skill，验证显式命中优先
        _FakeSkill(
            name="funclip-clip",
            description="按文本剪辑视频",
            triggers=("剪辑", "切片"),
        ),
    )


class TestSurveillanceRouting:
    def test_sparse_corridor_by_finding_bag(self):
        """'分析监控找包' → 命中 sparse-corridor（走廊/稀疏场景语义）。"""
        out = match_skills("分析监控找包", _surveillance_skills())
        assert out is not None
        assert "surveillance-sparse-corridor" in out

    def test_sparse_corridor_by_corridor(self):
        """'走廊监控夜里有没有人经过' → 命中 sparse-corridor。"""
        out = match_skills("走廊监控夜里有没有人经过", _surveillance_skills())
        assert out is not None
        assert "surveillance-sparse-corridor" in out
        assert "surveillance-crowded-scene" not in out

    def test_sparse_corridor_by_stairs(self):
        """'楼梯口有没有人路过' → 命中 sparse-corridor。"""
        out = match_skills("楼梯口有没有人路过", _surveillance_skills())
        assert out is not None
        assert "surveillance-sparse-corridor" in out

    def test_crowded_scene_by_mall(self):
        """'商场人流分析' → 命中 crowded-scene（密集场景语义）。"""
        out = match_skills("商场人流分析", _surveillance_skills())
        assert out is not None
        assert "surveillance-crowded-scene" in out
        assert "surveillance-sparse-corridor" not in out

    def test_crowded_scene_by_density(self):
        """'路口拥挤检测' → 命中 crowded-scene。"""
        out = match_skills("路口拥挤检测", _surveillance_skills())
        assert out is not None
        assert "surveillance-crowded-scene" in out

    def test_crowded_overrides_sparse_when_both_match(self):
        """'商场里的走廊' 同时含商场+走廊 → 密集优先（密集场景算法更合适）。"""
        out = match_skills("商场里的走廊人流", _surveillance_skills())
        assert out is not None
        assert "surveillance-crowded-scene" in out
        assert "surveillance-sparse-corridor" not in out

    def test_no_hit_for_unrelated_text(self):
        """'今天天气怎么样' → 不命中任何 surveillance skill。"""
        out = match_skills("今天天气怎么样", _surveillance_skills())
        assert out is None

    def test_explicit_trigger_takes_precedence(self):
        """显式 triggers 命中时不走语义路由（避免重复或冲突注入）。"""
        out = match_skills("帮我剪辑一下", _surveillance_skills())
        assert out is not None
        assert "funclip-clip" in out
        # 不应把 surveillance skill 也塞进来
        assert "surveillance-sparse-corridor" not in out
        assert "surveillance-crowded-scene" not in out

    def test_disabled_skill_not_routed(self):
        """禁用的 surveillance skill 不参与语义路由。"""
        skills = (
            _FakeSkill(
                name="surveillance-sparse-corridor",
                description="稀疏走廊监控",
                triggers=(),
                enabled=False,
            ),
        )
        assert match_skills("分析监控找包", skills) is None

    def test_empty_inputs(self):
        skills = _surveillance_skills()
        assert match_skills("", skills) is None
        assert match_skills("分析监控找包", ()) is None
        assert match_skills("分析监控找包", None) is None

    def test_numbered_output_format(self):
        """命中后输出格式为 'N. name: description'。"""
        out = match_skills("分析监控找包", _surveillance_skills())
        assert out is not None
        assert out.startswith("1. surveillance-sparse-corridor:")
