"""Working Memory 热层 — 最近会话关键事实。

《改进指南》P1-1 三层记忆第 1 层：最近会话关键事实（用户意图 + 工具链 +
时间戳）存 SQLite 热层表。turn 开始时按 TTL（默认 24h）注入为一条附加
system 记忆消息，TTL 过期自动清除。

与 ExperienceStore（语义层）不同，本表按 (session_id, intent) 幂等 upsert，
同一事实只保留最新 ts。纯 stdlib + SQLite，无三方依赖。
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_TTL_SEC = 86400  # 24h
_DEFAULT_DB = "config/agent_working_memory.db"


@dataclass(frozen=True)
class WorkingFact:
    """单条热层记忆。"""

    session_id: str
    intent: str
    tool_chain: List[str]
    ts: float


class WorkingMemory:
    """SQLite 热层：record_intent / recent / forget_old / inject_system_messages。"""

    def __init__(self, db_path: str = _DEFAULT_DB,
                 ttl_sec: float = DEFAULT_TTL_SEC) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_sec = ttl_sec
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS working_memory (
                    session_id  TEXT NOT NULL,
                    intent      TEXT NOT NULL,
                    tool_chain  TEXT NOT NULL,
                    ts          REAL NOT NULL,
                    PRIMARY KEY (session_id, intent)
                )
            """)

    def record_intent(self, session_id: str, intent: str,
                      tool_chain: List[str],
                      ts: float | None = None) -> None:
        """记录/刷新一条热层事实（幂等 upsert，同 session 同 intent 只留最新）。"""
        if not intent:
            return
        stamp = time.time() if ts is None else ts
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO working_memory "
                "(session_id, intent, tool_chain, ts) VALUES (?, ?, ?, ?)",
                (session_id, intent,
                 json.dumps(tool_chain, ensure_ascii=False), stamp))

    def recent(self, ttl_sec: float | None = None,
               now: float | None = None) -> List[WorkingFact]:
        """返回未过期（ts >= now - ttl）的热层事实，按 ts 倒序。"""
        ttl = self.ttl_sec if ttl_sec is None else ttl_sec
        stamp = time.time() if now is None else now
        cutoff = stamp - ttl
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT session_id, intent, tool_chain, ts "
                "FROM working_memory").fetchall()
        facts = [WorkingFact(
            session_id=r[0], intent=r[1],
            tool_chain=json.loads(r[2]), ts=r[3]) for r in rows]
        fresh = [f for f in facts if f.ts >= cutoff]
        fresh.sort(key=lambda f: f.ts, reverse=True)
        return fresh

    def forget_old(self, ttl_sec: float | None = None,
                   now: float | None = None) -> int:
        """删除过期事实，返回删除条数。"""
        ttl = self.ttl_sec if ttl_sec is None else ttl_sec
        stamp = time.time() if now is None else now
        cutoff = stamp - ttl
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT session_id, intent FROM working_memory").fetchall()
            deleted = 0
            for sid, intent in rows:
                row = conn.execute(
                    "SELECT ts FROM working_memory "
                    "WHERE session_id = ? AND intent = ?",
                    (sid, intent)).fetchone()
                if row is not None and row[0] < cutoff:
                    conn.execute(
                        "DELETE FROM working_memory "
                        "WHERE session_id = ? AND intent = ?",
                        (sid, intent))
                    deleted += 1
            conn.commit()
            return deleted

    def inject_system_messages(
        self, messages: List[Dict[str, Any]],
        ttl_sec: float | None = None,
        now: float | None = None) -> List[Dict[str, Any]]:
        """把热层事实注入为一条附加 system 段（不污染 system_prompt）。

        返回新列表（不可变）：无可注入事实时返回原列表副本。
        system_prompt（若存在）保持第一位，记忆段紧随其后。
        """
        facts = self.recent(ttl_sec, now)
        out = list(messages)
        if not facts:
            return out
        lines = ["[Working Memory] 最近会话关键事实（供本轮参考）:"]
        for f in facts:
            chain = ", ".join(f.tool_chain) or "(无工具调用)"
            lines.append(
                f"- 意图: {f.intent} | 工具链: [{chain}] | session: {f.session_id}")
        system_msg = {"role": "system", "content": "\n".join(lines)}
        idx = 0
        while idx < len(out) and out[idx].get("role") == "system":
            idx += 1
        out.insert(idx, system_msg)
        return out
