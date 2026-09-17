"""eli5 工具面覆盖契约测试（v10.4.0 P0-5）。

为什么必须存在
--------------
v10.3.1 的 eli5 有两类真实缺陷：

1. **模板名与工具名对不上**：模板匹配 ``create_highlights`` / ``run_ocr`` /
   ``point_and_jump`` / ``delete_this_history``，而注册表里的真实名字是
   ``highlight_cut`` / ``ocr_frame`` / ``point_at_object`` / ``delete_history``
   —— 这 4 条模板从未命中过。
2. **覆盖不完整**：32 个工具只有 9 个有模板。

本文件用**从工具注册表动态枚举**的方式守护覆盖度：将来新增工具却没补模板时，
``test_all_registered_tools_have_eli5_template`` 会立刻变红，而不是等用户看到
"返回了 N 字符的结果"才发现。
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from src.core.eli5 import (
    DANGEROUS_TOOLS,
    WRITE_TOOLS,
    _FALLBACK_MARKER,
    _TEMPLATES,
    explain_tool_call,
)
from src.core.tools.adapter import (
    register_legacy_tools,
    register_media_gen_tools,
    register_web_auto_tools,
)
from src.core.tools.registry import ToolRegistry
from src.core.tools.scope_guard import ScopeGuard


def _build_full_registry() -> ToolRegistry:
    """按生产装配顺序建出完整工具面（与 src/web/routers/agent.py 一致）。"""
    registry = ToolRegistry()
    register_legacy_tools(registry, lambda: None)
    register_media_gen_tools(registry)
    register_web_auto_tools(registry)
    return registry


@pytest.fixture(scope="module")
def tool_names() -> List[str]:
    return sorted(_build_full_registry().list_names())


def test_full_tool_surface_is_32(tool_names: List[str]):
    """工具面规模守卫：与 v10.3.1 发布时的 32 个工具一致。"""
    assert len(tool_names) == 32, f"工具面变成 {len(tool_names)} 个，需同步补 eli5 模板"


def test_all_registered_tools_have_eli5_template(tool_names: List[str]):
    """每个注册工具都必须有专属模板（走兜底即失败）。"""
    missing = [n for n in tool_names if n not in _TEMPLATES]
    assert not missing, (
        f"以下工具缺少 eli5 人话模板（会退化成兜底文案）: {missing}；"
        "请在 src/core/eli5.py 的 _TEMPLATES 里补上"
    )


def test_no_registered_tool_falls_back_to_generic(tool_names: List[str]):
    """正向契约：32 个工具渲染结果都不得含兜底标记。"""
    offenders = []
    for name in tool_names:
        text = explain_tool_call(name, {}, "正常返回内容")
        if _FALLBACK_MARKER in text:
            offenders.append((name, text))
    assert not offenders, f"这些工具走了兜底文案: {offenders}"


def test_eli5_templates_have_no_stale_tool_names():
    """反向断言：历史上 4 个错误模板名不得再出现。

    create_highlights / run_ocr / point_and_jump / delete_this_history
    是真实注册表里不存在的名字（P0-5 修正）。
    """
    stale = {"create_highlights", "run_ocr", "point_and_jump", "delete_this_history"}
    assert not (stale & set(_TEMPLATES)), (
        f"_TEMPLATES 里仍有对不上的工具名: {sorted(stale & set(_TEMPLATES))}"
    )


def test_unknown_tool_fallback_warns_explicitly():
    """未知工具必须**明确告知缺口**，不许假装正常。"""
    text = explain_tool_call("definitely_not_a_tool", {}, "whatever")
    assert _FALLBACK_MARKER in text
    assert "definitely_not_a_tool" in text


def test_fallback_does_not_claim_success():
    """兜底文案不得包含"返回了 N 字符"这类假装正常的旧措辞。"""
    text = explain_tool_call("definitely_not_a_tool", {}, "x" * 500)
    assert "返回了" not in text
    assert "字符" not in text


# --------------------------------------------------------------------------
# 风险标注与 scope_guard 分类一致（防止"危险的没标危险"）
# --------------------------------------------------------------------------


def test_eli5_risk_matches_scope_guard(tool_names: List[str]):
    """eli5 的 DANGEROUS/WRITE 集合必须与 scope_guard 的分类完全一致。"""
    guard = ScopeGuard()
    expected_dangerous = {n for n in tool_names if guard.classify(n) == "dangerous_write"}
    expected_write = {n for n in tool_names if guard.classify(n) == "write"}

    assert set(DANGEROUS_TOOLS) == expected_dangerous, (
        f"危险写工具集合漂移: eli5={sorted(DANGEROUS_TOOLS)} "
        f"scope_guard={sorted(expected_dangerous)}"
    )
    assert set(WRITE_TOOLS) == expected_write, (
        f"写工具集合漂移: eli5={sorted(WRITE_TOOLS)} "
        f"scope_guard={sorted(expected_write)}"
    )


def test_dangerous_write_tools_say_dangerous(tool_names: List[str]):
    """危险写工具的文案必须显式含"危险"。"""
    guard = ScopeGuard()
    for name in tool_names:
        if guard.classify(name) != "dangerous_write":
            continue
        text = explain_tool_call(name, {}, "ok")
        assert "危险" in text, f"{name} 是危险写，但文案没标危险: {text}"


def test_write_tools_mention_confirmation(tool_names: List[str]):
    """一般写工具的文案必须提示需要用户确认。"""
    guard = ScopeGuard()
    for name in tool_names:
        if guard.classify(name) != "write":
            continue
        text = explain_tool_call(name, {}, "ok")
        assert "确认" in text, f"{name} 是写操作，但文案没提示确认: {text}"


def test_read_tools_have_no_risk_prefix(tool_names: List[str]):
    """读工具不得被误加风险前缀（避免吓到用户）。"""
    guard = ScopeGuard()
    for name in tool_names:
        if guard.classify(name) != "read":
            continue
        text = explain_tool_call(name, {}, "ok")
        assert "危险操作" not in text
        assert "需要你确认" not in text


# --------------------------------------------------------------------------
# 鲁棒性：永不抛异常、参数缺失也能出话
# --------------------------------------------------------------------------


@pytest.mark.parametrize("args", [{}, {"seconds": 12}, {"query": "   "}, {"url": None}])
def test_eli5_never_raises_for_any_tool(tool_names: List[str], args: Dict):
    """任意入参组合下，32 个工具都返回非空字符串且不抛异常。"""
    for name in tool_names:
        text = explain_tool_call(name, args, "some result")
        assert isinstance(text, str) and text, f"{name} 在 args={args} 下返回空"


@pytest.mark.parametrize(
    "result",
    [None, 123, ["a", "b"], {"k": "v"}, "x" * 10000, b"bytes"],
)
def test_eli5_handles_non_string_results(tool_names: List[str], result):
    """非 str 结果（None/int/list/dict/超长/bytes）都不得导致异常。"""
    for name in tool_names:
        text = explain_tool_call(name, {}, result)
        assert isinstance(text, str) and text


def test_eli5_exception_result_reports_error():
    """Exception 结果走错误分支，不暴露堆栈细节。"""
    text = explain_tool_call("search_web", {"query": "a"}, RuntimeError("boom"))
    assert "出错" in text
    assert "boom" in text


def test_eli5_failure_result_adds_warning(tool_names: List[str]):
    """结果含失败信号时，文案附加"似乎没有成功"提示。"""
    text = explain_tool_call("highlight_cut", {"description": "进球"}, "未找到足够的相关片段进行剪辑。")
    assert "没有成功" in text


def test_eli5_parses_visual_search_timestamp():
    """已确认的结果格式必须被正确解析（防静默退化）。"""
    head = "时间点 12.34s (匹配度: 0.85)\n时间点 30.00s (匹配度: 0.80)"
    text = explain_tool_call("search_visual", {"query": "红色汽车"}, head)
    assert "12.34" in text and "0.85" in text


def test_eli5_visual_search_no_fabricated_score_on_empty():
    """0 命中时不得编造相似度分数。"""
    text = explain_tool_call("search_visual", {"query": "不存在的东西"}, "没有找到相似画面")
    assert "相似度" not in text or "没有找到" in text
