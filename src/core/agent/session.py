"""SessionEvent append-only log + Session。

参考 DSH `docs/subsystems/core.md:9-18`：
  - Session 维护 events: list[SessionEvent]（append-only）
  - derive_messages() 投影成 LLM 消息列表（不维护消息列表）
  - serialize() 崩溃可恢复（JSON 存 SQLite）

事件类型：system / user / assistant / tool_call / tool_result / step / turn。
每条事件不可变（frozen dataclass），带 type/timestamp/payload。

derive_messages 规则：
  - system → role=system
  - user → role=user
  - assistant → role=assistant（带 tool_calls 时附 tool_calls 字段）
  - tool_result → role=tool，tool_call_id 关联
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SessionEvent:
    """不可变事件记录。append-only log 的单条。"""

    type: str  # system | user | assistant | tool_call | tool_result | step | turn
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionEvent":
        return cls(
            type=d["type"],
            timestamp=d["timestamp"],
            payload=d.get("payload", {}),
            event_id=d.get("event_id", uuid.uuid4().hex),
        )


class Session:
    """append-only SessionEvent log + 消息投影。

    用法：
        s = Session("sess-1")
        s.append(SessionEvent("user", time.time(), {"content": "你好"}))
        msgs = s.derive_messages()
    """

    def __init__(self, session_id: str, *,
                 system_prompt: Optional[str] = None) -> None:
        self.session_id = session_id
        self._events: List[SessionEvent] = []
        if system_prompt:
            self.append(SessionEvent(
                "system", time.time(), {"content": system_prompt}))

    @property
    def events(self) -> List[SessionEvent]:
        return list(self._events)

    def append(self, event: SessionEvent) -> None:
        self._events.append(event)

    def derive_messages(self) -> List[Dict[str, Any]]:
        """把 events 投影成 LLM 消息列表。

        - system → {role: system, content}
        - user → {role: user, content}
        - assistant → {role: assistant, content?, tool_calls?}
        - tool_result → {role: tool, tool_call_id, content}
        - step/turn → 元事件，不进 LLM 消息（skip）
        """
        messages: List[Dict[str, Any]] = []
        for e in self._events:
            p = e.payload
            if e.type == "system":
                messages.append({"role": "system", "content": p.get("content", "")})
            elif e.type == "user":
                messages.append({"role": "user", "content": p.get("content", "")})
            elif e.type == "assistant":
                m: Dict[str, Any] = {"role": "assistant"}
                if "content" in p:
                    m["content"] = p["content"]
                if p.get("tool_calls"):
                    m["tool_calls"] = p["tool_calls"]
                messages.append(m)
            elif e.type == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": p.get("tool_call_id", ""),
                    "content": p.get("content", ""),
                })
            # step/turn 不进 LLM 消息
        return messages

    def serialize(self) -> str:
        return json.dumps({
            "session_id": self.session_id,
            "events": [e.to_dict() for e in self._events],
        }, ensure_ascii=False)

    @classmethod
    def deserialize(cls, blob: str) -> "Session":
        d = json.loads(blob)
        s = cls(d["session_id"])
        s._events = [SessionEvent.from_dict(e) for e in d.get("events", [])]
        return s


class SessionStore:
    """SQLite 持久化 Session。

    复用 `src/core/run_store.py` 的 sqlite + WAL 模式（同库不同表 sessions）。
    崩溃可恢复：append 时写库，load 时读库重建 Session。
    """

    def __init__(self, db_path: str = "config/agent_sessions.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    blob        TEXT NOT NULL,
                    updated_at  TEXT
                )
            """)

    def save(self, session: Session) -> None:
        from datetime import datetime
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, blob, updated_at) "
                "VALUES (?, ?, ?)",
                (session.session_id, session.serialize(),
                 datetime.now().isoformat(timespec="seconds")),
            )

    def load(self, session_id: str) -> Optional[Session]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT blob FROM sessions WHERE session_id = ?",
                (session_id,))
            row = cur.fetchone()
        if not row:
            return None
        return Session.deserialize(row[0])

    def delete(self, session_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """列出最近更新的 session（按 updated_at 倒序）。

        返回 [{session_id, updated_at, size}]，size 为 blob 字符长度。
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT session_id, updated_at, LENGTH(blob) AS size "
                "FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,))
            rows = cur.fetchall()
        return [
            {"session_id": row[0], "updated_at": row[1], "size": row[2]}
            for row in rows
        ]

    def search_sessions(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """搜索 payload.content 含 query 的 session。

        SQL 用 blob LIKE %query% 预筛，再 Python 精筛 events 的 payload.content。
        返回 [{session_id, updated_at, snippet}]，snippet 为匹配位置前后片段。
        """
        like = f"%{query}%"
        results: List[Dict[str, Any]] = []
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT session_id, blob, updated_at FROM sessions "
                "WHERE blob LIKE ? LIMIT ?",
                (like, limit))
            rows = cur.fetchall()
        for session_id, blob, updated_at in rows:
            try:
                session = Session.deserialize(blob)
            except (json.JSONDecodeError, KeyError):
                continue
            snippet = ""
            for event in session.events:
                content = event.payload.get("content")
                if isinstance(content, str) and query in content:
                    idx = content.find(query)
                    start = max(0, idx - 20)
                    end = min(len(content), idx + len(query) + 20)
                    snippet = content[start:end]
                    break
            if snippet:
                results.append({
                    "session_id": session_id,
                    "updated_at": updated_at,
                    "snippet": snippet,
                })
        return results
