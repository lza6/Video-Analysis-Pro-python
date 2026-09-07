"""Agent 经验沉淀（v9 自进化 —《改进指南》2.5）。

从历史 Session 提取 Experience，沉淀到 SQLite + WAL，
同类 intent 出现 ≥3 次且 tool_chain 稳定时建议生成新 skill。

红线：纯 stdlib + 规则版，不调付费 LLM（skill 草稿走 skill_generator
规则模板，与 skill_generator.py 一脉相承）。本模块只新增文件，
不改 loop.py / skill_generator.py / self_check.py，接入主控后续做。
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from src.core.agent.session import Session


@dataclass(frozen=True)
class Experience:
    """单次 agent 经验。"""
    intent: str
    tool_chain: List[str]
    success: bool
    quality_score: float
    timestamp: float
    session_id: str


@dataclass(frozen=True)
class SkillSuggestion:
    """skill 建议草稿。"""
    intent: str
    recommended_tool_chain: List[str]
    sample_count: int
    draft_skill_md: str


def _extract_intent(content: str) -> str:
    """从 user content 提取 intent 关键词指纹（规则版，无 LLM）。

    取前 60 字符 + lower + strip，作为模糊匹配 token；
    与 skill_generator.detect_scene 一脉相承：纯文本规则。
    """
    if not content:
        return ""
    return content.strip().lower()[:60]


def _tool_result_success(content: str) -> bool:
    """tool_result content 是否成功（loop._stringify_tool_output 错误时 "Error:" 前缀）。"""
    if not content:
        return False
    return not content.lstrip().startswith("Error:")


class ExperienceExtractor:
    """从 SessionEvent log 提取 Experience。"""

    @staticmethod
    def from_session(session: "Session") -> Optional[Experience]:
        """session 为空或无 user event 返回 None。"""
        events = session.events
        if not events:
            return None
        intent = ""
        for e in events:
            if e.type == "user":
                intent = _extract_intent(str(e.payload.get("content", "")))
                break
        if not intent:
            return None
        # tool_chain：所有 assistant event 的 tool_calls[].name 顺序去重
        tool_chain: List[str] = []
        for e in events:
            if e.type == "assistant":
                for tc in e.payload.get("tool_calls") or []:
                    name = tc.get("name", "")
                    if name and name not in tool_chain:
                        tool_chain.append(name)
        # success：最后一个 assistant 是否有非空 content 且无 error
        success = False
        assistants = [e for e in events if e.type == "assistant"]
        if assistants:
            content = str(assistants[-1].payload.get("content", "")).strip()
            success = bool(content) and not content.lower().startswith("error:")
        # quality_score：tool_result success 比例（无 tool_result 给中性 0.5）
        tr_events = [e for e in events if e.type == "tool_result"]
        if tr_events:
            ok = sum(
                1 for e in tr_events
                if _tool_result_success(str(e.payload.get("content", "")))
            )
            quality_score = ok / len(tr_events)
        else:
            quality_score = 0.5
        return Experience(
            intent=intent,
            tool_chain=tool_chain,
            success=success,
            quality_score=quality_score,
            timestamp=time.time(),
            session_id=session.session_id,
        )


class ExperienceStore:
    """SQLite + WAL 持久化 Experience（同 SessionStore 风格）。"""

    def __init__(self, db_path: str = "config/agent_experiences.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent        TEXT NOT NULL,
                    tool_chain    TEXT NOT NULL,
                    success       INTEGER NOT NULL,
                    quality_score REAL NOT NULL,
                    timestamp     REAL NOT NULL,
                    session_id    TEXT NOT NULL
                )
            """)

    def record(self, exp: Experience) -> int:
        """写一条经验，返回自增 id。"""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO experiences "
                "(intent, tool_chain, success, quality_score, timestamp, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (exp.intent,
                 json.dumps(exp.tool_chain, ensure_ascii=False),
                 1 if exp.success else 0,
                 exp.quality_score,
                 exp.timestamp,
                 exp.session_id),
            )
            return int(cur.lastrowid)

    def find_similar(self, intent: str, top_n: int = 10) -> List[Experience]:
        """按 intent LIKE 模糊匹配 + quality_score 降序。"""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT intent, tool_chain, success, quality_score, "
                "timestamp, session_id FROM experiences "
                "WHERE intent LIKE ? "
                "ORDER BY quality_score DESC LIMIT ?",
                (f"%{intent}%", top_n),
            )
            rows = cur.fetchall()
        return [Experience(
            intent=r[0],
            tool_chain=json.loads(r[1]),
            success=bool(r[2]),
            quality_score=r[3],
            timestamp=r[4],
            session_id=r[5],
        ) for r in rows]

    def count_by_intent(self, intent: str) -> int:
        """同类 intent 出现次数（LIKE 模糊匹配）。"""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM experiences WHERE intent LIKE ?",
                (f"%{intent}%",),
            )
            return int(cur.fetchone()[0])

    def should_suggest_skill(self, intent: str, min_count: int = 3) -> bool:
        """同类 intent ≥min_count 次且 tool_chain 稳定（去重后链相同）→ True。"""
        if self.count_by_intent(intent) < min_count:
            return False
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT tool_chain FROM experiences WHERE intent LIKE ?",
                (f"%{intent}%",),
            )
            chains = [json.loads(r[0]) for r in cur.fetchall()]
        uniq = {tuple(c) for c in chains if c}
        return len(uniq) == 1


class SkillAdvisor:
    """聚合 Experience + 调 skill_generator 生成 SkillSuggestion（守红线：不调 LLM）。"""

    @staticmethod
    def suggest(store: ExperienceStore, intent: str,
                skill_generator=None) -> Optional[SkillSuggestion]:
        """should_suggest_skill 为 True 时返回建议，否则 None。

        skill_generator 为 None 时只给 tool_chain 不生成 md（守红线：
        skill_generator 本身是规则版，不真实调付费 LLM）。
        """
        if not store.should_suggest_skill(intent):
            return None
        similars = store.find_similar(intent, top_n=50)
        if not similars:
            return None
        chain = similars[0].tool_chain  # 已稳定（should_suggest_skill 校验过）
        draft_md = ""
        if skill_generator is not None:
            try:
                scene = skill_generator.detect_scene(intent)
                draft = skill_generator.draft_skill_from_scene(scene, intent) if scene else None
                if draft is not None:
                    draft_md = skill_generator.render_skill_md(draft)
            except Exception:  # noqa: BLE001
                draft_md = ""
        return SkillSuggestion(
            intent=intent,
            recommended_tool_chain=list(chain),
            sample_count=len(similars),
            draft_skill_md=draft_md,
        )
