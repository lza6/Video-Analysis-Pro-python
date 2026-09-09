"""Agent 经验沉淀（v9 自进化 —《改进指南》2.5 + v10.2 P1-1 语义层升级）。

从历史 Session 提取 Experience，沉淀到 SQLite + WAL，
同类 intent 出现 ≥3 次且 tool_chain 稳定时建议生成新 skill。

v10.2 P1-1 升级（语义层）：
  - 新增 FTS5 全文副表 experience_fts（intent + tool_chain 可搜索字段），
    查询从「LIKE %q%」改为「FTS5 关键词分 + quality_score + 时间衰减」混合评分。
  - tokenize='trigram'：SQLite 内置三字符 n-gram tokenizer，无需 ICU 扩展，
    支持中文/多字节子串匹配（'停车场' 可召回 '我要分析停车场视频'）。
  - **不删除旧表**（experiences 保持原结构），新增 fts 副表，upgrade 兼容旧库。
  - `find_similar` / `count_by_intent` / `should_suggest_skill` 返回结构与语义
    保持兼容，现有调用不破坏。
  - FTS5 触发词：若平台 Python sqlite3 无 fts5（pragma compile_options
    无 FTS5），建表抛 OperationalError，捕获后降级为无 FTS 的 LIKE + 评分，
    并打 warning（行为仍可用，只是召回不优于 LIKE）。

红线：纯 stdlib + 规则版，不调付费 LLM（skill 草稿走 skill_generator
规则模板，与 skill_generator.py 一脉相承）。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from src.core.agent.session import Session

log = logging.getLogger("core.agent.experience")

# 混合评分权重（FTS5 关键词 / quality_score / 时间衰减）
_W_FTS = 0.3
_W_QUALITY = 0.4
_W_RECENCY = 0.3


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


def _fts5_available(conn: sqlite3.Connection) -> bool:
    """探测当前 SQLite 是否编译了 FTS5（编译选项含 FTS5）。"""
    rows = conn.execute("pragma compile_options").fetchall()
    return any("FTS5" in (r[0] or "") for r in rows)


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
    """SQLite + WAL 持久化 Experience（同 SessionStore 风格）。

    v10.2 P1-1：新增 FTS5 副表 experience_fts，find_similar 用混合评分
    （FTS5 关键词分 + quality_score + 时间衰减）。FTS5 不可用时降级 LIKE。
    """

    def __init__(self, db_path: str = "config/agent_experiences.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fts_available = False
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
            self.fts_available = _fts5_available(conn)
            if self.fts_available:
                try:
                    # tokenize='trigram'：三字符 n-gram，中文子串可匹配
                    conn.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS experience_fts
                        USING fts5(intent, tool_chain,
                                   content='experiences',
                                   content_rowid='id',
                                   tokenize='trigram')
                    """)
                    self._recreate_triggers(conn)
                    # 存量数据回填（upgrade 兼容旧库：预已有 experiences 旧行）
                    conn.execute(
                        "INSERT INTO experience_fts(rowid, intent, tool_chain) "
                        "SELECT id, intent, tool_chain FROM experiences "
                        "WHERE id NOT IN (SELECT rowid FROM experience_fts)")
                except sqlite3.OperationalError as exc:  # pragma: no cover
                    self.fts_available = False
                    log.warning(
                        "experience FTS5 不可用，降级 LIKE+评分: %s", exc)
            if not self.fts_available:
                log.info("experience store FTS5 未启用，find_similar 走 LIKE")

    def _recreate_triggers(self, conn: sqlite3.Connection) -> None:
        """FTS5 外部内容表同步触发器（幂等：先 drop 再建）。"""
        conn.execute("DROP TRIGGER IF EXISTS experience_fts_ai")
        conn.execute("DROP TRIGGER IF EXISTS experience_fts_ad")
        conn.execute("DROP TRIGGER IF EXISTS experience_fts_au")
        conn.execute("""
            CREATE TRIGGER experience_fts_ai AFTER INSERT ON experiences BEGIN
                INSERT INTO experience_fts(rowid, intent, tool_chain)
                VALUES (new.id, new.intent, new.tool_chain);
            END
        """)
        conn.execute("""
            CREATE TRIGGER experience_fts_ad AFTER DELETE ON experiences BEGIN
                INSERT INTO experience_fts(experience_fts, rowid, intent, tool_chain)
                VALUES('delete', old.id, old.intent, old.tool_chain);
            END
        """)
        conn.execute("""
            CREATE TRIGGER experience_fts_au AFTER UPDATE ON experiences BEGIN
                INSERT INTO experience_fts(experience_fts, rowid, intent, tool_chain)
                VALUES('delete', old.id, old.intent, old.tool_chain);
                INSERT INTO experience_fts(rowid, intent, tool_chain)
                VALUES (new.id, new.intent, new.tool_chain);
            END
        """)

    def record(self, exp: Experience) -> int:
        """写一条经验，返回自增 id。

        幂等保护：同 (intent, session_id) 已存在则跳过（不产生重复行）。
        防重复接入：run_turn 每次结束都调 record，同 session 同 intent
        重复调用不污染 count_by_intent / should_suggest_skill。
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            row = conn.execute(
                "SELECT id FROM experiences WHERE intent = ? AND session_id = ?",
                (exp.intent, exp.session_id)).fetchone()
            if row is not None:
                return int(row[0])
            cur = conn.execute(
                "INSERT INTO experiences "
                "(intent, tool_chain, success, quality_score, timestamp, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (exp.intent,
                 json.dumps(exp.tool_chain, ensure_ascii=False),
                 1 if exp.success else 0,
                 exp.quality_score,
                 exp.timestamp,
                 exp.session_id))
            return int(cur.lastrowid)

    # -- 读取 --------------------------------------------------------------

    def find_similar(self, intent: str, top_n: int = 10) -> List[Experience]:
        """按 intent 混合评分召回相似经验。

        v10.2 P1-1 新语义：FTS5 关键词分 + quality_score + 时间衰减。
        评分 = 0.3×fts_norm + 0.4×quality + 0.3×recency_norm。
        FTS5 不可用时降级为 LIKE %q%（保留 v9 语义）。
        """
        if not intent:
            return []
        if getattr(self, "fts_available", False):
            try:
                rows = self._find_similar_fts(intent)
                if rows:
                    return rows[:top_n]
                # FTS 无命中时回退 LIKE（拼接场景，如 intent 过短）
            except sqlite3.OperationalError:
                log.warning("experience FTS5 查询失败，降级 LIKE")
        return self._find_similar_like(intent, top_n)

    def find_all(self, limit: int = 1000) -> List[Experience]:
        """读取全部经验（按时间倒序）。供 skills 蒸馏聚合使用。"""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT intent, tool_chain, success, quality_score, "
                "timestamp, session_id FROM experiences "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,))
            rows = cur.fetchall()
        return [self._row_to_exp(r) for r in rows]

    def _find_similar_like(self, intent: str, top_n: int) -> List[Experience]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT intent, tool_chain, success, quality_score, "
                "timestamp, session_id FROM experiences "
                "WHERE intent LIKE ? "
                "ORDER BY quality_score DESC LIMIT ?",
                (f"%{intent}%", top_n))
            rows = cur.fetchall()
        return [self._row_to_exp(r) for r in rows]

    def _find_similar_fts(self, intent: str) -> List[Experience]:
        """FTS5 trigram 匹配 + 混合评分（已在调用方 ensure fts_available）。"""
        main_cols = ("e.intent, e.tool_chain, e.success, e.quality_score, "
                     "e.timestamp, e.session_id")
        with sqlite3.connect(self.db_path) as conn:
            rows_all = conn.execute("SELECT intent, timestamp FROM experiences").fetchall()
            if not rows_all:
                return []
            ts_list = [r[1] for r in rows_all]
            max_ts, min_ts = max(ts_list), min(ts_list)
            span = max(max_ts - min_ts, 1e-9)
            cur = conn.execute(
                f"SELECT {main_cols}, bm25(experience_fts) AS fts_raw "
                "FROM experiences e "
                "JOIN experience_fts fts ON fts.rowid = e.id "
                "WHERE experience_fts MATCH ?",
                (f'"{intent}"',))
            rows_matched = cur.fetchall()
        scored: List[tuple[float, float, float, Experience]] = []
        for r in rows_matched:
            exp = self._row_to_exp(r[:6])
            bm25 = r[6]
            # bm25 越小越好（负值），归一化到 [0,1]
            fts_norm = max(0.0, min(1.0, -bm25 / 5.0))
            recency = (exp.timestamp - min_ts) / span
            total = (_W_FTS * fts_norm + _W_QUALITY * exp.quality_score
                     + _W_RECENCY * recency)
            # quality_score 主键 + ts 次级，使同分时 quality 高的排前面
            scored.append((total, exp.quality_score, exp.timestamp, exp))
        # 总分降序；同分时 quality + ts 降序（确定性，供测试稳定断言）
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return [item[3] for item in scored]

    @staticmethod
    def _row_to_exp(r) -> Experience:
        return Experience(
            intent=r[0],
            tool_chain=json.loads(r[1]),
            success=bool(r[2]),
            quality_score=r[3],
            timestamp=r[4],
            session_id=r[5],
        )

    def count_by_intent(self, intent: str) -> int:
        """同类 intent 出现次数（LIKE 模糊匹配，兼容旧行为）。"""
        if not intent:
            return 0
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM experiences WHERE intent LIKE ?",
                (f"%{intent}%",))
            return int(cur.fetchone()[0])

    def should_suggest_skill(self, intent: str, min_count: int = 3) -> bool:
        """同类 intent ≥min_count 次且 tool_chain 稳定（去重后链相同）→ True。"""
        if self.count_by_intent(intent) < min_count:
            return False
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT tool_chain FROM experiences WHERE intent LIKE ?",
                (f"%{intent}%",))
            chains = [json.loads(r[0]) for r in cur.fetchall()]
        uniq = {tuple(c) for c in chains if c}
        return len(uniq) == 1

    # -- FTS 辅助（测试/诊断） ---------------------------------------------

    @property
    def has_fts(self) -> bool:
        """当前 store 是否启用 FTS5（True=混合评分，False=LIKE 降级）。"""
        return getattr(self, "fts_available", False)


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