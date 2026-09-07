"""凭据审计日志：记录 key 的 read/write/rotate 行为。

支撑《改进指南》6.3 凭据轮换与审计日志。本期只落地独立模块 + 测试，
暂不接入主控（key.py / settings 页保持不变，后续再做）。

设计参考：
  - src/core/im_gateway/mailbox.py 的 SQLite 短连接 + WAL 模式
  - src/core/credentials/key.py 的不可变 dataclass + 中文 docstring 风格

线程安全：sqlite3 短连接 + WAL（读不阻塞写），写入用 threading.Lock
串行化，避免多线程并发 INSERT 时 SQLITE_BUSY。

表 audit_log：id INTEGER PK / ts TEXT / key_name TEXT / action TEXT /
caller TEXT / trace_id TEXT / success INTEGER。
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


def _now_iso() -> str:
    """ISO8601 时间戳（毫秒精度，UTC 带 +00:00，便于跨时区比较）。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class AuditEntry:
    """审计日志不可变快照（查询返回值）。

    frozen=True 保证日志在写入后不被意外修改；如需"修改"走
    CredentialAudit.log 追加新行，而非改本对象。
    """

    ts: str
    key_name: str
    action: str
    caller: str
    trace_id: Optional[str] = None
    success: bool = True


class CredentialAudit:
    """凭据审计日志持久层。

    线程安全（sqlite3 短连接 + WAL + threading.Lock 串行化写入）。
    本类方法同步，便于复用 mailbox 的连接模式；async 调用方应把阻塞
    的 sqlite 调用放到 asyncio.to_thread 里跑。
    """

    def __init__(self, db_path: str = "config/cred_audit.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """建表 + 开 WAL（沿用 mailbox 模式）。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            cur = conn.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL, key_name TEXT NOT NULL,
                    action TEXT NOT NULL, caller TEXT NOT NULL,
                    trace_id TEXT, success INTEGER NOT NULL
                )"""
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_key ON audit_log(key_name)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """短连接工厂（WAL 下读不阻塞写）。"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def log(
        self,
        key_name: str,
        action: str,
        caller: str,
        trace_id: Optional[str] = None,
        success: bool = True,
    ) -> None:
        """写一条审计日志（失败也记录，success=False）。"""
        ts = _now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO audit_log
                       (ts, key_name, action, caller, trace_id, success)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ts, key_name, action, caller, trace_id, 1 if success else 0),
                )
                conn.commit()

    def list(
        self,
        key_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """查询审计日志（按 id 倒序，最近优先）。

        Args:
            key_name: 仅查此 key（None = 全部）。
            limit: 最多返回条数（默认 100）。
        """
        with self._connect() as conn:
            if key_name is not None:
                rows = conn.execute(
                    """SELECT ts, key_name, action, caller, trace_id, success
                       FROM audit_log WHERE key_name=?
                       ORDER BY id DESC LIMIT ?""",
                    (key_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT ts, key_name, action, caller, trace_id, success
                       FROM audit_log ORDER BY id DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [
                AuditEntry(
                    ts=r["ts"], key_name=r["key_name"], action=r["action"],
                    caller=r["caller"], trace_id=r["trace_id"],
                    success=bool(r["success"]),
                )
                for r in rows
            ]
