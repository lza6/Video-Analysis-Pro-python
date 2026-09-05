"""监控视频分析的运行记录数据库层 (RunStore)。

独立于 history_manager.py（跨视频知识库），本模块专职记录"批量监控视频分析"
的运行元数据：每个视频一条 run，下挂分片 (segments) 与命中片段 (clips)。

设计要点
  - 纯 sqlite3，不引入 ORM（项目未使用 ORM）
  - WAL 模式：UI 线程读、worker 线程写，并发不互斥
  - 所有写操作走 context manager（with sqlite3.connect(...)），commit 自动
  - run_id 用 uuid4 hex（同秒创建多个 run 不冲突，学 history_manager 教训）
  - seg_id / clip_id 同理用 uuid4 hex
  - 数据库文件 config/runs.db（已 gitignore，不入库）
  - delete_run 支持可选删磁盘 clip 文件（purge_files=True），默认只删 DB 行
"""
from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 运行状态枚举（避免拼写漂移，集中声明）
_RUN_STATUS_VALUES = ("started", "running", "done", "failed")
_SEG_STATUS_VALUES = ("pending", "ok", "failed", "skipped")


def _now_iso() -> str:
    """ISO8601 时间戳（秒精度，人读友好）。"""
    return datetime.now().isoformat(timespec="seconds")


def _new_id() -> str:
    """uuid4 hex，全表唯一，同秒批量创建不冲突。"""
    return uuid.uuid4().hex


class RunStore:
    """监控视频分析运行记录持久层。

    三张表：
      runs      : 一视频一 run，记录总进度/命中数/耗时/模型信息
      segments  : 一 run 多分片，每分片的匹配状态/置信度/原因/耗时/usage
      clips     : 一 run 多命中片段，记录 hit_idx/时间戳/clip 文件路径

    所有方法线程安全（sqlite3 短连接 + WAL）。get_run 聚合三表返回嵌套 dict。
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.config_dir / "runs.db"
        self._init_db()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        """建表 + 开 WAL + 外键。

        WAL 模式让读连接不阻塞写连接，UI 查进度时 worker 可继续写。
        外键 ON 保证 delete_run 级联清理 segments/clips。
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id             TEXT PRIMARY KEY,
                    video_path         TEXT,
                    video_name         TEXT,
                    duration_sec       REAL,
                    status             TEXT DEFAULT 'started',
                    started_at         TEXT,
                    finished_at        TEXT,
                    hits_count         INTEGER DEFAULT 0,
                    segments_total     INTEGER DEFAULT 0,
                    segments_ok        INTEGER DEFAULT 0,
                    segments_failed    INTEGER DEFAULT 0,
                    vlm_elapsed_sec    REAL,
                    total_elapsed_sec  REAL,
                    model              TEXT,
                    provider           TEXT,
                    mode               TEXT,
                    strip_path         TEXT
                )
            """)
            # v5.7：旧库补 strip_path 列（长图证据路径）
            self._ensure_column(conn, "runs", "strip_path", "TEXT")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS segments (
                    seg_id        TEXT PRIMARY KEY,
                    run_id        TEXT NOT NULL,
                    seg_idx       INTEGER,
                    start_sec    REAL,
                    dur_sec       REAL,
                    status        TEXT DEFAULT 'pending',
                    match         INTEGER DEFAULT 0,
                    confidence    REAL,
                    reason        TEXT,
                    abs_timestamp TEXT,
                    attempts      INTEGER DEFAULT 0,
                    elapsed_sec   REAL,
                    first_token_ms INTEGER,
                    usage_json    TEXT,
                    error         TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                )
            """)
            # 兼容旧库：若 segments 表已存在但缺 first_token_ms 列则补列。
            # sqlite 的 PRAGMA table_info 可探测列是否存在，缺失则 ALTER TABLE ADD COLUMN。
            self._ensure_column(conn, "segments", "first_token_ms",
                                "INTEGER", default=None)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clips (
                    clip_id       TEXT PRIMARY KEY,
                    run_id        TEXT NOT NULL,
                    hit_idx       INTEGER,
                    abs_timestamp TEXT,
                    clip_path     TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                )
            """)
            # 索引：按 run_id 查 segments/clips 是最高频路径
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_segments_run_id "
                "ON segments(run_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_clips_run_id "
                "ON clips(run_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_status "
                "ON runs(status)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # 连接工厂（短连接，每次调用新建，WAL 下安全）
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str,
                      column: str, decl: str, *,
                      default: Any = None) -> bool:
        """兼容旧库：若 table 缺 column 列，则 ALTER TABLE ADD COLUMN。

        sqlite 不支持 IF NOT EXISTS 于 ADD COLUMN，必须先 PRAGMA table_info 探测。
        返回是否真的新增了列（已存在则返回 False）。

        default 非 None 时，新列声明拼 DEFAULT（用于 NOT NULL 列回填老行）。
        本方法只用于"加可空字段"场景（如 first_token_ms），default 通常留 None。
        """
        cur = conn.execute(f"PRAGMA table_info({table})")
        existing = {r[1] for r in cur.fetchall()}  # row[1]=name
        if column in existing:
            return False
        ddl = f"ALTER TABLE {table} ADD COLUMN {column} {decl}"
        if default is not None:
            ddl += f" DEFAULT {default}"
        conn.execute(ddl)
        conn.commit()
        logger.info("schema 升级：表 %s 新增列 %s %s", table, column, decl)
        return True

    # ------------------------------------------------------------------
    # 写入：run
    # ------------------------------------------------------------------
    def create_run(
        self,
        video_path: str,
        *,
        duration_sec: Optional[float] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        mode: Optional[str] = None,
        status: str = "started",
    ) -> str:
        """创建一条 run 记录，返回 run_id。

        语义：批量跑开始处理一个视频时调用。segments_total 初始 0，后续
        add_segment 时由调用方 update_run 更新计数（避免 store 内隐式聚合
        带来并发竞争）。
        """
        if status not in _RUN_STATUS_VALUES:
            raise ValueError(
                f"status 必须是 {_RUN_STATUS_VALUES} 之一，得到 {status!r}"
            )
        run_id = _new_id()
        video_name = Path(video_path).name if video_path else ""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO runs
                   (run_id, video_path, video_name, duration_sec, status,
                    started_at, hits_count, segments_total, segments_ok,
                    segments_failed, model, provider, mode)
                   VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 0, ?, ?, ?)""",
                (
                    run_id, str(video_path), video_name, duration_sec, status,
                    _now_iso(), model, provider, mode,
                ),
            )
            conn.commit()
        logger.info("run created: run_id=%s video=%s", run_id, video_name)
        return run_id

    def update_run(self, run_id: str, **fields: Any) -> bool:
        """部分更新 run 字段。任意 fields 键值对，仅更新非 None 字段。

        返回是否实际更新了一行。finished_at 不自动写——调用方显式传
        status='done'/'failed' 时建议同时传 finished_at（本方法不耦合语义）。
        """
        if not fields:
            return False
        # 白名单：只允许更新 schema 中存在的列，防 SQL 注入与拼写错误
        allowed = {
            "video_path", "video_name", "duration_sec", "status",
            "started_at", "finished_at", "hits_count", "segments_total",
            "segments_ok", "segments_failed", "vlm_elapsed_sec",
            "total_elapsed_sec", "model", "provider", "mode", "strip_path",
        }
        if "status" in fields and fields["status"] not in _RUN_STATUS_VALUES:
            raise ValueError(
                f"status 必须是 {_RUN_STATUS_VALUES} 之一，得到 {fields['status']!r}"
            )
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        params = list(updates.values()) + [run_id]
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE runs SET {set_clause} WHERE run_id = ?", params
            )
            conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # 写入：segment
    # ------------------------------------------------------------------
    def add_segment(self, run_id: str, seg_data: Dict[str, Any]) -> str:
        """插入一条分片记录。seg_data 为字段名→值的字典。

        必须含 run_id（参数传入）；seg_id 自动生成。未知键被忽略（白名单过滤），
        避免调用方塞入 schema 外字段导致 SQL 错误。

        first_token_ms（首字耗时，毫秒）由 M1 batch_runner 在分片结果中填入，
        旧库通过 _ensure_column 自动补列，老行回填 NULL。
        """
        seg_id = _new_id()
        allowed = {
            "seg_idx", "start_sec", "dur_sec", "status", "match",
            "confidence", "reason", "abs_timestamp", "attempts",
            "elapsed_sec", "first_token_ms", "usage_json", "error",
        }
        cols = ["seg_id", "run_id"]
        vals: List[Any] = [seg_id, run_id]
        for k, v in seg_data.items():
            if k in allowed:
                cols.append(k)
                vals.append(v)
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO segments ({col_names}) VALUES ({placeholders})",
                vals,
            )
            conn.commit()
        return seg_id

    # ------------------------------------------------------------------
    # 写入：hit + clip
    # ------------------------------------------------------------------
    def add_hit(self, run_id: str, hit_data: Dict[str, Any]) -> str:
        """记录一条命中（hit）。hit_idx / abs_timestamp 等字段在 hit_data 中。

        命中本身没有独立表（hit 信息落在 segments.match=1 上），但 clips 表
        保存命中的片段文件路径。本方法同时把命中片段写入 clips 表，并递增
        runs.hits_count，保持计数一致。
        """
        clip_id = _new_id()
        hit_idx = hit_data.get("hit_idx")
        abs_ts = hit_data.get("abs_timestamp")
        clip_path = hit_data.get("clip_path")
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO clips (clip_id, run_id, hit_idx, abs_timestamp, clip_path)
                   VALUES (?, ?, ?, ?, ?)""",
                (clip_id, run_id, hit_idx, abs_ts, clip_path),
            )
            conn.execute(
                "UPDATE runs SET hits_count = hits_count + 1 WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()
        return clip_id

    def add_clip(
        self,
        run_id: str,
        clip_path: str,
        *,
        hit_idx: Optional[int] = None,
        abs_timestamp: Optional[str] = None,
    ) -> str:
        """单独追加一条 clip（不递增 hits_count，用于补录片段文件路径）。"""
        clip_id = _new_id()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO clips (clip_id, run_id, hit_idx, abs_timestamp, clip_path)
                   VALUES (?, ?, ?, ?, ?)""",
                (clip_id, run_id, hit_idx, abs_timestamp, str(clip_path)),
            )
            conn.commit()
        return clip_id

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def list_runs(
        self, limit: int = 50, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """按 started_at 倒序列出 run。可按 status 过滤。"""
        with self._connect() as conn:
            if status:
                cur = conn.execute(
                    """SELECT * FROM runs WHERE status = ?
                       ORDER BY started_at DESC LIMIT ?""",
                    (status, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                    (limit,),
                )
            return [dict(r) for r in cur.fetchall()]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """聚合查询：run + segments + clips（嵌套 dict）。

        UI 详情面板调用此方法渲染单条 run 的全部信息。
        """
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
            if row is None:
                return None
            run = dict(row)
            cur = conn.execute(
                "SELECT * FROM segments WHERE run_id = ? ORDER BY seg_idx ASC",
                (run_id,),
            )
            run["segments"] = [dict(r) for r in cur.fetchall()]
            cur = conn.execute(
                "SELECT * FROM clips WHERE run_id = ? ORDER BY hit_idx ASC",
                (run_id,),
            )
            run["clips"] = [dict(r) for r in cur.fetchall()]
        return run

    def get_progress(self, run_id: str) -> Optional[Dict[str, Any]]:
        """UI 实时刷新进度用：{total, done, hits, failed}。

        done = segments_ok + segments_failed + segments_skipped（但 schema 用
        ok/failed 计数，skipped 归入 failed 计数由调用方维护）。这里直接读
        runs 表的聚合字段，最快（不扫 segments 表）。
        """
        with self._connect() as conn:
            cur = conn.execute(
                """SELECT segments_total, segments_ok, segments_failed, hits_count
                   FROM runs WHERE run_id = ?""",
                (run_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "total": int(row["segments_total"] or 0),
                "done": int(row["segments_ok"] or 0) + int(row["segments_failed"] or 0),
                "hits": int(row["hits_count"] or 0),
                "failed": int(row["segments_failed"] or 0),
            }

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------
    def delete_run(
        self, run_id: str, *, purge_files: bool = False
    ) -> bool:
        """删除一条 run 及其 segments/clips（外键级联）。

        purge_files=True 时同时删磁盘上的 clip 文件（一键清理按钮可能传 True，
        普通删除记录但保留产物）。返回是否删了 run 行。
        """
        deleted = False
        with self._connect() as conn:
            if purge_files:
                cur = conn.execute(
                    "SELECT clip_path FROM clips WHERE run_id = ?", (run_id,)
                )
                for r in cur.fetchall():
                    p = r["clip_path"]
                    if not p:
                        continue
                    try:
                        fp = Path(p)
                        if fp.exists() and fp.is_file():
                            os.remove(fp)
                    except Exception as e:
                        logger.warning("删除 clip 文件失败 %s: %s", p, e)

            cur = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
            deleted = cur.rowcount > 0
        if deleted:
            logger.info("run deleted: run_id=%s purge_files=%s", run_id, purge_files)
        return deleted

    def clear_all(self, *, purge_files: bool = False) -> int:
        """一键清理：删所有 run + 关联 segments/clips。返回删除的 run 数。

        purge_files 语义同 delete_run。clear_all 用 TRUNCATE 风格的 DELETE，
        比逐条 delete_run 快得多（无外键逐条级联开销）。
        """
        count = 0
        with self._connect() as conn:
            if purge_files:
                cur = conn.execute("SELECT clip_path FROM clips WHERE clip_path IS NOT NULL")
                for r in cur.fetchall():
                    p = r["clip_path"]
                    try:
                        fp = Path(p)
                        if fp.exists() and fp.is_file():
                            os.remove(fp)
                    except Exception as e:
                        logger.warning("删除 clip 文件失败 %s: %s", p, e)

            cur = conn.execute("SELECT COUNT(*) AS c FROM runs")
            count = int(cur.fetchone()["c"])
            # 外键级联会自动清空 segments/clips
            conn.execute("DELETE FROM runs")
            conn.commit()
        logger.info("clear_all: removed %d runs (purge_files=%s)", count, purge_files)
        return count
