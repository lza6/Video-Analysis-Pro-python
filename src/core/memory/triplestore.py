"""时序/图记忆（triplestore）— 实体关系三元组表 + 简单图谱。

《改进指南》P1-1 三层记忆第 3 层：从 experience/decision 事件抽取实体关系
（意图→工具→结果），支持 query_subject / query_between（时序过滤）。

参考 graphiti/plur 范式但保持最小：纯 stdlib SQLite，(subject, predicate,
object, ts) 四列 + ts 上建索引，支持时间窗口过滤。不做节点持久化（边
即可推导节点），无第三方依赖。
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

_DEFAULT_DB = "config/agent_triples.db"


@dataclass(frozen=True)
class Triple:
    """单条三元组。"""

    subject: str
    predicate: str
    object: str
    ts: float


def _norm(value: str) -> str:
    return (value or "").strip().lower()


class TripleStore:
    """SQLite 三元组表：(subject, predicate, object, ts)。

    幂等 upsert：同 (subject, predicate, object) 只保留最新 ts（时间感知，
    避免查询看到过期快照）。
    """

    def __init__(self, db_path: str = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS triples (
                    subject   TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object    TEXT NOT NULL,
                    ts        REAL NOT NULL,
                    PRIMARY KEY (subject, predicate, object)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_triples_subject "
                "ON triples (subject)")

    def record(self, subject: str, predicate: str, object: str,
               ts: float | None = None) -> None:
        """写一条三元组（幂等 upsert，保留最新 ts）。"""
        stamp = time.time() if ts is None else ts
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                "INSERT INTO triples "
                "(subject, predicate, object, ts) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(subject, predicate, object) "
                "DO UPDATE SET ts = excluded.ts",
                (_norm(subject), _norm(predicate), _norm(object), stamp))

    def record_experience(self, intent: str, tool_chain: List[str],
                          success: bool,
                          ts: float | None = None) -> None:
        """从 Experience 抽取三元组：意图 → 工具 → 结果。

        链式抽取：intent →tool→ t0；t_i →next_tool→ t_{i+1}；末工具
        →result→ success/failure。为空 intent 时 no-op。
        """
        intent_n = _norm(intent)
        if not intent_n:
            return
        stamp = time.time() if ts is None else ts
        chain = [_norm(t) for t in tool_chain if _norm(t)]
        if not chain:
            self.record(intent_n, "result",
                        "success" if success else "failure", stamp)
            return
        for i, tool in enumerate(chain):
            if i == 0:
                self.record(intent_n, "tool", tool, stamp)
            else:
                self.record(chain[i - 1], "next_tool", tool, stamp)
        self.record(chain[-1], "result",
                    "success" if success else "failure", stamp)

    def query_subject(self, subject: str,
                      start_ts: float | None = None,
                      end_ts: float | None = None) -> List[Triple]:
        """按 subject 查三元组，可选时间窗口过滤。"""
        return self._query(_norm(subject), start_ts, end_ts)

    def query_between(self, subject: str, object_: str,
                      start_ts: float | None = None,
                      end_ts: float | None = None) -> List[Triple]:
        """按 (subject, object) 查三元组（任意 predicate），时序过滤。"""
        return self._query(_norm(subject), start_ts, end_ts,
                           object_=object_)

    def query_object(self, object_: str,
                     start_ts: float | None = None,
                     end_ts: float | None = None) -> List[Triple]:
        """按 object 反查三元组，时序过滤。"""
        return self._query(None, start_ts, end_ts, object_=object_)

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM triples")
            return int(cur.fetchone()[0])

    def _query(self, subject: str | None,
               start_ts: float | None, end_ts: float | None,
               object_: str | None = None) -> List[Triple]:
        clauses: List[str] = []
        params: List[float | str] = []
        if subject is not None:
            clauses.append("subject = ?")
            params.append(subject)
        if object_ is not None:
            clauses.append("object = ?")
            params.append(_norm(object_))
        if start_ts is not None:
            clauses.append("ts >= ?")
            params.append(start_ts)
        if end_ts is not None:
            clauses.append("ts <= ?")
            params.append(end_ts)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT subject, predicate, object, ts FROM triples "
                f"{where} ORDER BY ts", tuple(params))
            rows = cur.fetchall()
        return [Triple(subject=r[0], predicate=r[1],
                       object=r[2], ts=r[3]) for r in rows]
