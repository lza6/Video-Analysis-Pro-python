
"""P1-1 术语气泡 + P1-6 全局 Toast 前端契约守护（v10.6.0）。

防止:词表被误删、页面停止引用 GlossaryText/Toast —— 小白友好能力静默回退。
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8", errors="ignore")


REQUIRED_TERMS = ["抽帧", "智能抽帧", "帧差", "转录", "YOLO", "向量知识库",
                  "CLIP", "RTSP", "批处理", "断点续跑", "SSE", "token",
                  "上下文", "审批", "会话", "skill", "skill 蒸馏", "监督层"]


def test_glossary_map_has_required_terms():
    """术语表必须覆盖必备词(新增术语忘进表 → 红)。"""
    text = _read("webapp/src/lib/glossary.ts")
    missing = [t for t in REQUIRED_TERMS if t not in text]
    assert not missing, f"术语表缺: {missing}"


def test_glossary_component_references_map():
    """GlossaryTerm 必须引用词表且未收录词原样渲染(不炸)。"""
    text = _read("webapp/src/components/ui/GlossaryTerm.tsx")
    assert "GLOSSARY[term]" in text
    assert "GLOSSARY" in text and "abbr" in text


def test_pages_use_glossary_and_toast():
    """分析页与 Agent 页必须引用 GlossaryText;布局必须挂 ToastProvider。"""
    home = _read("webapp/src/app/(app)/page.tsx")
    agent = _read("webapp/src/app/(app)/agent/page.tsx")
    layout = _read("webapp/src/app/(app)/layout.tsx")
    toast = _read("webapp/src/components/ui/Toast.tsx")
    assert "GlossaryText" in home, "分析页未用术语气泡"
    assert "GlossaryText" in agent, "Agent 页未用术语气泡"
    assert "ToastProvider" in layout, "布局未挂 ToastProvider"
    assert "useToast" in toast and "aria-live" in toast


def test_api_has_timeout_and_retry():
    """api.ts 必须有统一超时与幂等重试(P1-6 前端可靠性)。"""
    text = _read("webapp/src/lib/api.ts")
    assert "AbortController" in text
    assert "请求超时" in text
    assert "withRetry" in text
