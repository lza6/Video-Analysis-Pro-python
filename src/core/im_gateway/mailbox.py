"""IM mailbox：SQLite 存入站/出站消息，lease/ack 投递语义。

参考：
  - Minke packages/im-gateway/src/contract.ts (IPC 契约)
  - Minke packages/im-gateway/src/agent-route.ts GatewayAgentMailboxPort
  - 项目 src/core/run_store.py 的 SQLite 短连接 + WAL 模式

投递语义：
  - lease(lease_holder): agent 处理前先 lease（领消息，设 lease_expires_at
    与 status=processing）。被 lease 的消息不会被别的 lease 重复领。
  - ack(msg_id): agent 处理完调 ack，status=done，清 lease_expires_at。
  - 超时未 ack 的消息（lease_expires_at < now）可被重新 lease
    （reclaim_stale + lease 时跳过 done 行）。

状态机：pending → processing(leased) → done | failed

表设计（im_messages）：
  msg_id        : uuid4 hex 主键
  direction     : 'inbound' / 'outbound'
  channel       : 'wechat' / 'telegram' / 'discord'
  peer          : 对端用户/群标识（如 wechat wxid / tg chat_id）
  content       : 文本内容
  status        : pending / processing / done / failed
  leased_by     : 当前 lease 持有者（agent 实例标识），done 后清空
  lease_expires_at : lease 超时时间 ISO，超时后可被重 lease
  created_at    : 入库时间
  processed_at  : ack/failed 时间
  error         : failed 时的原因
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 状态枚举集中声明，避免拼写漂移
_STATUS_VALUES = ("pending", "processing", "done", "failed")
_DIRECTION_VALUES = ("inbound", "outbound")

_DEFAULT_LEASE_TTL_SEC = 30.0


def _now_iso() -> str:
    """ISO8601 时间戳（毫秒精度，UTC 带 +00:00，便于跨时区比较）。

    用毫秒而非秒：lease TTL 可能 < 1s（测试用 0.001s 验证超时重 lease），
    秒精度会让 expires_at 与 now 截断到同一秒导致 '<' 比较失败。
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _new_id() -> str:
    """uuid4 hex，全表唯一，同秒批量创建不冲突（学 run_store 教训）。"""
    return uuid.uuid4().hex


@dataclass(frozen=True)
class IMMessage:
    """IM 消息不可变快照（mailbox 查询返回值）。

    frozen=True 保证消息在 lease→ack 之间不会被意外修改；如需"修改"
    （如改 status），用 mailbox 的写入方法，而非改本对象。
    """

    msg_id: str
    direction: str
    channel: str
    peer: str
    content: str
    status: str
    leased_by: Optional[str] = None
    lease_expires_at: Optional[str] = None
    created_at: str = ""
    processed_at: Optional[str] = None
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row | Dict[str, Any]) -> "IMMessage":
        """从 sqlite Row / dict 构造（忽略未知键，extra 存剩余字段）。"""
        d = dict(row)
        known = {
            "msg_id", "direction", "channel", "peer", "content", "status",
            "leased_by", "lease_expires_at", "created_at", "processed_at",
            "error",
        }
        known_vals = {k: d.get(k) for k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(extra=extra, **known_vals)  # type: ignore[arg-type]


class IMMailbox:
    """IM 消息持久层 + lease/ack 投递语义。

    线程安全（sqlite3 短连接 + WAL）。async 调用方应把阻塞的 sqlite 调用
    放到 asyncio.to_thread 里跑；本类方法本身同步，便于复用 run_store
    的连接模式。
    """

    def __init__(self, config_dir: str = "config", db_filename: str = "im_gateway.db"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.config_dir / db_filename
        self._init_db()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        """建表 + 开 WAL + 索引（沿用 run_store 模式）。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS im_messages (
                    msg_id            TEXT PRIMARY KEY,
                    direction         TEXT NOT NULL,
                    channel           TEXT NOT NULL,
                    peer              TEXT NOT NULL,
                    content           TEXT NOT NULL DEFAULT '',
                    status            TEXT NOT NULL DEFAULT 'pending',
                    leased_by         TEXT,
                    lease_expires_at  TEXT,
                    created_at        TEXT NOT NULL,
                    processed_at      TEXT,
                    error             TEXT,
                    extra_json        TEXT
                )
                """
            )
            # lease 高频路径：找 pending 或 stale 的入站消息
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_im_status "
                "ON im_messages(status, direction)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_im_lease "
                "ON im_messages(lease_expires_at)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_im_channel_peer "
                "ON im_messages(channel, peer)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """短连接工厂（WAL 下读不阻塞写）。"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    # ------------------------------------------------------------------
    # 写入：入站 / 出站
    # ------------------------------------------------------------------
    def put_inbound(
        self,
        *,
        channel: str,
        peer: str,
        content: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """记录一条入站消息（来自 IM adapter.receive），返回 msg_id。

        入站消息初始 status=pending，等待 gateway lease 后交给 agent。
        """
        import json as _json

        msg_id = _new_id()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO im_messages
                   (msg_id, direction, channel, peer, content, status,
                    created_at, extra_json)
                   VALUES (?, 'inbound', ?, ?, ?, 'pending', ?, ?)""",
                (
                    msg_id, channel, peer, content, _now_iso(),
                    _json.dumps(extra, ensure_ascii=False) if extra else None,
                ),
            )
            conn.commit()
        logger.info(
            "inbound put: msg_id=%s channel=%s peer=%s len=%d",
            msg_id, channel, peer, len(content),
        )
        return msg_id

    def put_outbound(
        self,
        *,
        channel: str,
        peer: str,
        content: str,
        reply_to: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """记录一条出站消息（gateway 调 adapter.send 后写入），返回 msg_id。

        出站消息 status=done（已发送即终态），reply_to 存 extra.reply_to
        以便后续追溯。出站消息不参与 lease 语义。
        """
        import json as _json

        msg_id = _new_id()
        extra_payload: Dict[str, Any] = dict(extra or {})
        if reply_to:
            extra_payload["reply_to"] = reply_to
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO im_messages
                   (msg_id, direction, channel, peer, content, status,
                    created_at, processed_at, extra_json)
                   VALUES (?, 'outbound', ?, ?, ?, 'done', ?, ?, ?)""",
                (
                    msg_id, channel, peer, content, _now_iso(), _now_iso(),
                    _json.dumps(extra_payload, ensure_ascii=False),
                ),
            )
            conn.commit()
        logger.info(
            "outbound put: msg_id=%s channel=%s peer=%s reply_to=%s",
            msg_id, channel, peer, reply_to,
        )
        return msg_id

    # ------------------------------------------------------------------
    # lease / ack / fail（投递语义核心）
    # ------------------------------------------------------------------
    def lease(
        self,
        *,
        lease_holder: str,
        lease_ttl_sec: float = _DEFAULT_LEASE_TTL_SEC,
    ) -> Optional[IMMessage]:
        """领一条入站待处理消息（原子 UPDATE...WHERE status='pending'）。

        被领的消息 status='processing'，leased_by=lease_holder，
        lease_expires_at = now + ttl。返回 IMMessage 或 None（无可用消息）。

        原子性：单条 UPDATE 的 WHERE 同时要求 status='pending'，多 lease
        并发不会重复领同一条（sqlite 行锁保证）。

        超时重 lease：本方法只领 pending；stale（processing 且超时）的
        消息由 reclaim_stale 转回 pending 后才能被 lease。调用方应在
        lease 前调 reclaim_stale() 让超时消息重回队列。
        """
        # 先选一条候选（pending 中最早的）
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM im_messages
                   WHERE direction='inbound' AND status='pending'
                   ORDER BY created_at ASC LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            msg_id = row["msg_id"]
            expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=lease_ttl_sec)
            ).isoformat(timespec="milliseconds")
            # 原子占用：只在仍是 pending 时改（防并发竞争）
            cur = conn.execute(
                """UPDATE im_messages
                   SET status='processing', leased_by=?, lease_expires_at=?
                   WHERE msg_id=? AND status='pending'""",
                (lease_holder, expires_at, msg_id),
            )
            conn.commit()
            if cur.rowcount == 0:
                # 被别的 lease 抢先了，递归再试一次（边界情况，极少）
                return self.lease(lease_holder=lease_holder, lease_ttl_sec=lease_ttl_sec)
            # 重新读取最新行返回
            row = conn.execute(
                "SELECT * FROM im_messages WHERE msg_id=?", (msg_id,)
            ).fetchone()
            return IMMessage.from_row(row)

    def ack(self, msg_id: str) -> bool:
        """确认处理完成：status='done'，清 leased_by/lease_expires_at。

        返回是否实际更新了一行（msg_id 不存在或非 processing 时返回 False）。
        """
        with self._connect() as conn:
            cur = conn.execute(
                """UPDATE im_messages
                   SET status='done', leased_by=NULL, lease_expires_at=NULL,
                       processed_at=?
                   WHERE msg_id=? AND status='processing'""",
                (_now_iso(), msg_id),
            )
            conn.commit()
            ok = cur.rowcount > 0
        if ok:
            logger.info("ack: msg_id=%s", msg_id)
        return ok

    def fail(self, msg_id: str, error: str) -> bool:
        """标记处理失败：status='failed'，error 记录原因。

        failed 是终态，不会自动重 lease（需显式 requeue_failed 才能回 pending）。
        """
        with self._connect() as conn:
            cur = conn.execute(
                """UPDATE im_messages
                   SET status='failed', leased_by=NULL, lease_expires_at=NULL,
                       processed_at=?, error=?
                   WHERE msg_id=? AND status='processing'""",
                (_now_iso(), error, msg_id),
            )
            conn.commit()
            ok = cur.rowcount > 0
        if ok:
            logger.warning("fail: msg_id=%s error=%s", msg_id, error)
        return ok

    def reclaim_stale(self, now: Optional[datetime] = None) -> int:
        """把超时未 ack 的 processing 消息转回 pending（可被重新 lease）。

        判定：status='processing' AND lease_expires_at < now。
        返回回收条数。lease 前应先调本方法清理超时租约。
        """
        now = now or datetime.now(timezone.utc)
        # 毫秒精度：lease_expires_at 存毫秒，比较时 now 也要毫秒，
        # 否则秒级 now 可能大于 expires 但被截断到同一秒导致 '<' 判定失败
        # （测试用 0.001s ttl 时尤其敏感）。
        now_iso = now.isoformat(timespec="milliseconds")
        with self._connect() as conn:
            cur = conn.execute(
                """UPDATE im_messages
                   SET status='pending', leased_by=NULL, lease_expires_at=NULL
                   WHERE status='processing'
                     AND lease_expires_at IS NOT NULL
                     AND lease_expires_at < ?""",
                (now_iso,),
            )
            conn.commit()
            n = cur.rowcount
        if n:
            logger.warning("reclaim_stale: %d stale messages -> pending", n)
        return n

    def requeue_failed(self, msg_id: str) -> bool:
        """把 failed 消息转回 pending（手动重试）。"""
        with self._connect() as conn:
            cur = conn.execute(
                """UPDATE im_messages
                   SET status='pending', leased_by=NULL, lease_expires_at=NULL,
                       processed_at=NULL, error=NULL
                   WHERE msg_id=? AND status='failed'""",
                (msg_id,),
            )
            conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get(self, msg_id: str) -> Optional[IMMessage]:
        """按主键取一条消息。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM im_messages WHERE msg_id=?", (msg_id,)
            ).fetchone()
            return IMMessage.from_row(row) if row else None

    def list_pending(self, limit: int = 50) -> List[IMMessage]:
        """列出 pending 入站消息（按 created_at 升序）。"""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM im_messages
                   WHERE direction='inbound' AND status='pending'
                   ORDER BY created_at ASC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [IMMessage.from_row(r) for r in rows]

    def list_outbound_for_peer(
        self, channel: str, peer: str, limit: int = 50
    ) -> List[IMMessage]:
        """列出某对端的出站消息（gateway 回复追溯）。"""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM im_messages
                   WHERE direction='outbound' AND channel=? AND peer=?
                   ORDER BY created_at DESC LIMIT ?""",
                (channel, peer, limit),
            ).fetchall()
            return [IMMessage.from_row(r) for r in rows]
