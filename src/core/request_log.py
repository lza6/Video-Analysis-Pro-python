"""LLM 请求日志 + token 消耗统计 (F7)

记录每次 LLM 请求的元数据（provider/model/key_id/状态码/耗时/token 用量/
请求预览/响应预览），支持按 provider 聚合的 token 统计。

设计要点
  - 纯 sqlite3，不引入 ORM（与 run_store.py 保持一致）
  - WAL 模式：worker 线程写、UI/API 线程读，并发不互斥
  - 所有写操作走 context manager（with sqlite3.connect(...)），commit 自动
  - log_id 用 uuid4 hex（同秒批量创建不冲突，学 history_manager 教训）
  - 数据库文件 config/request_logs.db（已 gitignore，不入库）
  - 请求/响应 preview 截断前 500 字符，防 SQLite 膨胀
  - 密钥不记录：key_id 只记 id 不记 key 本身
  - 查询用参数化占位符，防 SQL 注入
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)

# preview 截断长度（前 N 字符，防 SQLite 膨胀）
PREVIEW_LIMIT = 500


def _now_iso() -> str:
    """ISO8601 时间戳（秒精度，人读友好）。"""
    return datetime.now().isoformat(timespec="seconds")


def _new_id() -> str:
    """uuid4 hex，全表唯一，同秒批量创建不冲突。"""
    return uuid.uuid4().hex


def _truncate(text: Optional[str], limit: int = PREVIEW_LIMIT) -> str:
    """截断到前 limit 字符，None 返回空串。"""
    if not text:
        return ""
    return text[:limit]


@dataclass
class RequestLog:
    """单次 LLM 请求的日志条目（内存 DTO，落库由 RequestLogStore 完成）。

    字段映射表 request_logs 列：
      - timestamp: ISO8601 创建时刻
      - provider: "nvidia" | "kilo" | "openai-compatible" | "ollama" | ...
      - model: 模型名（如 nvidia/nemotron-3-nano-omni-30b-a3b-reasoning）
      - key_id: 路由器内 key 标识（如 "nvidia-1"），不记 key 本身
      - status_code: HTTP 状态码；网络异常时 None
      - latency_ms: 端到端耗时（含重试），毫秒
      - prompt_tokens / completion_tokens / total_tokens: usage 字段，缺失时 0
      - error: 错误简述（如 "HTTP 503 Worker 16/16" 或 "ConnectionError"）
      - request_preview / response_preview: 截断后的请求/响应预览
    """
    timestamp: str = ""
    provider: str = ""
    model: str = ""
    key_id: str = ""
    status_code: Optional[int] = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str = ""
    request_preview: str = ""
    response_preview: str = ""
    # 内存态：落库时由 store 生成 log_id，此处仅供测试构造完整对象用
    log_id: str = field(default_factory=_new_id)


class RequestLogStore:
    """LLM 请求日志持久层（append-only + 查询 + token 聚合）。

    所有方法线程安全（sqlite3 短连接 + WAL）。
    """

    def __init__(self, config_dir: str = "config") -> None:
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.config_dir / "request_logs.db"
        self._init_db()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        """建表 + 开 WAL。WAL 让读连接不阻塞写连接。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    log_id            TEXT PRIMARY KEY,
                    timestamp         TEXT,
                    provider          TEXT,
                    model             TEXT,
                    key_id            TEXT,
                    status_code       INTEGER,
                    latency_ms        REAL,
                    prompt_tokens     INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens      INTEGER DEFAULT 0,
                    error             TEXT,
                    request_preview   TEXT,
                    response_preview  TEXT
                )
            """)
            # 索引：按 provider 查 + 按时间倒序列出是最高频路径
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_request_logs_provider "
                "ON request_logs(provider)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_request_logs_ts "
                "ON request_logs(timestamp DESC)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # 连接工厂（短连接，每次调用新建，WAL 下安全）
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def log_request(self, entry: RequestLog) -> str:
        """追加一条请求日志，返回 log_id。

        preview 在落库前再截断一次（调用方可能传超长字符串）。
        """
        if not entry.timestamp:
            entry.timestamp = _now_iso()
        log_id = entry.log_id or _new_id()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO request_logs
                   (log_id, timestamp, provider, model, key_id, status_code,
                    latency_ms, prompt_tokens, completion_tokens, total_tokens,
                    error, request_preview, response_preview)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    log_id, entry.timestamp, entry.provider, entry.model,
                    entry.key_id, entry.status_code, entry.latency_ms,
                    entry.prompt_tokens, entry.completion_tokens,
                    entry.total_tokens, entry.error,
                    _truncate(entry.request_preview),
                    _truncate(entry.response_preview),
                ),
            )
            conn.commit()
        return log_id

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def list_logs(
        self,
        limit: int = 100,
        provider: Optional[str] = None,
        since: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """按时间倒序列出请求日志。

        - limit: 最多返回 N 条（默认 100）
        - provider: 过滤指定 provider（None = 全部）
        - since: ISO8601 起始时刻（含），只返回该时刻之后的记录
        """
        clauses: list[str] = []
        params: list[Any] = []
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            cur = conn.execute(
                f"""SELECT * FROM request_logs {where}
                    ORDER BY timestamp DESC LIMIT ?""",
                params,
            )
            return [dict(r) for r in cur.fetchall()]

    def token_stats(
        self,
        provider: Optional[str] = None,
        since: Optional[str] = None,
    ) -> dict[str, dict[str, Any]]:
        """按 provider 聚合 token 用量 / 请求数 / 平均延迟。

        返回 {provider: {requests, prompt_tokens, completion_tokens,
                         total_tokens, avg_latency_ms, error_count}}。
        provider 参数非 None 时只返回该 provider 的统计（仍以 dict 形式返回，
        key 为 provider 名）。
        """
        clauses: list[str] = []
        params: list[Any] = []
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            cur = conn.execute(
                f"""SELECT
                       provider,
                       COUNT(*) AS requests,
                       COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
                       COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                       SUM(CASE
                           WHEN status_code IS NULL
                             OR status_code < 200
                             OR status_code >= 300
                           THEN 1 ELSE 0 END) AS error_count
                   FROM request_logs {where}
                   GROUP BY provider""",
                params,
            )
            rows = cur.fetchall()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            p = r["provider"] or "(unknown)"
            out[p] = {
                "requests": int(r["requests"] or 0),
                "prompt_tokens": int(r["prompt_tokens"] or 0),
                "completion_tokens": int(r["completion_tokens"] or 0),
                "total_tokens": int(r["total_tokens"] or 0),
                "avg_latency_ms": float(r["avg_latency_ms"] or 0.0),
                "error_count": int(r["error_count"] or 0),
            }
        return out

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------
    def clear_all(self) -> int:
        """一键清理：删所有请求日志。返回删除的行数。"""
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) AS c FROM request_logs")
            count = int(cur.fetchone()["c"])
            conn.execute("DELETE FROM request_logs")
            conn.commit()
        logger.info("clear_all: removed %d request logs", count)
        return count


# ---- 进程级单例（线程安全，懒加载）----
_store: Optional[RequestLogStore] = None
_store_lock = Lock()


def get_store() -> RequestLogStore:
    """进程级单例 RequestLogStore。

    路由层（src/web/routers/requests.py）和 ProviderRouter 接入点共用同一实例，
    保证读写落到同一文件。测试用 RequestLogStore(config_dir=tmp_path) 独立隔离。
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = RequestLogStore()
    return _store


def reset_store() -> None:
    """重置单例（测试隔离用：每个测试重置后 get_store 会新建）。"""
    global _store
    with _store_lock:
        _store = None


# ---- usage 解析工具（ProviderRouter 接入用）----
def parse_usage(resp_json: Any) -> tuple[int, int, int]:
    """从 LLM 响应 JSON 抽 (prompt_tokens, completion_tokens, total_tokens)。

    兼容 OpenAI/NVIDIA 格式 {"usage": {...}} 与缺失 usage 的场景。
    返回三元组，缺失字段记 0。
    """
    if not isinstance(resp_json, dict):
        return (0, 0, 0)
    usage = resp_json.get("usage")
    if not isinstance(usage, dict):
        return (0, 0, 0)
    pt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    ct = (usage.get("completion_tokens")
          or usage.get("output_tokens") or 0)
    tt = usage.get("total_tokens") or (int(pt) + int(ct))
    return (int(pt) or 0, int(ct) or 0, int(tt) or 0)
