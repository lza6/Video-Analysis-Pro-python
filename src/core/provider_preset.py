"""多 Provider 预设管理 (ProviderPreset + ProviderPresetStore)。

cc-switch 风格:用户可新增任意 provider (OpenAI / Anthropic / Gemini / Ollama /
自定义) + 切换活跃。SQLite 持久化 preset 元数据,api_key 走 OS keyring (复用
config_manager._secure_set / _secure_get),DB 只存 key_ref 标记 (不落明文)。

设计要点 (学 run_store.py / history_manager.py 的既有约定)
  - 纯 sqlite3,不引入 ORM (项目未使用 ORM)
  - WAL 模式:UI 线程读、worker 线程写,并发不互斥
  - 所有写走 context manager (with sqlite3.connect),commit 自动
  - preset_id 用 uuid4 hex (同秒批量创建不冲突,学 history_manager 教训)
  - 参数化查询 (? 占位) 防注入
  - DB 文件 config/provider_presets.db (已 gitignore,不入库)
  - set_active 事务内先全置 0 再置目标 1,保证同时只有一个活跃
  - api_key_ref 是 keyring 中的 key 名 (非明文),形如 "vap_provider:{id}"

本模块是 config router (LastUsed 单 provider) 的补充,不改 config.py。
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.constants import CONFIG_DIR

log = logging.getLogger(__name__)

# 与 config_manager._KEYRING_SERVICE 保持一致 (硬编码,避免 import 私有名)
_KEYRING_SERVICE = "VideoAnalysisPro"


def _now_iso() -> str:
    """ISO8601 时间戳 (秒精度,人读友好)。"""
    return datetime.now().isoformat(timespec="seconds")


def _new_id() -> str:
    """uuid4 hex,全表唯一,同秒批量创建不冲突。"""
    return uuid.uuid4().hex


def _key_ref(preset_id: str) -> str:
    """preset 在 keyring 中的 key 名 (service=VideoAnalysisPro, username=this)。

    非明文 key,只是 keyring 查找用的引用标记。
    """
    return f"vap_provider:{preset_id}"


@dataclass
class ProviderPreset:
    """一个 provider 预设 (cc-switch 的一条配置)。"""

    id: str
    name: str
    provider_type: str = "custom"  # openai/anthropic/gemini/ollama/custom
    base_url: str = ""
    model: str = ""
    api_key_ref: str = ""  # keyring 中的 key 名,非明文
    enabled: bool = True
    created_at: str = field(default_factory=_now_iso)
    is_active: bool = False

    def to_dict(self) -> dict:
        """转 dict (含 api_key_ref,内部用;router 层会 pop 掉只回 has_key)。"""
        return asdict(self)


class ProviderPresetStore:
    """SQLite 持久化的 provider preset 仓库 (CRUD + active 切换 + keyring 存取)。"""

    def __init__(self, config_dir: str | None = None) -> None:
        # 动态解析 CONFIG_DIR (不绑默认值),便于测试 monkeypatch
        self.config_dir = Path(config_dir or CONFIG_DIR)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.config_dir / "provider_presets.db"
        self._init_db()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        """建表 + 开 WAL + 唯一索引 (name 不重复,防 cc-switch 重名混淆)。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS provider_presets (
                    id             TEXT PRIMARY KEY,
                    name           TEXT NOT NULL,
                    provider_type  TEXT NOT NULL DEFAULT 'custom',
                    base_url       TEXT DEFAULT '',
                    model          TEXT DEFAULT '',
                    api_key_ref    TEXT DEFAULT '',
                    enabled        INTEGER NOT NULL DEFAULT 1,
                    created_at     TEXT NOT NULL,
                    is_active      INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pp_active ON provider_presets(is_active)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_pp_name ON provider_presets(name)"
            )

    def _row_to_preset(self, row: sqlite3.Row) -> ProviderPreset:
        return ProviderPreset(
            id=row["id"],
            name=row["name"],
            provider_type=row["provider_type"],
            base_url=row["base_url"] or "",
            model=row["model"] or "",
            api_key_ref=row["api_key_ref"] or "",
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            is_active=bool(row["is_active"]),
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def list_presets(self) -> list[ProviderPreset]:
        """列所有 preset,按 created_at 升序 (稳定展示)。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM provider_presets ORDER BY created_at ASC"
            )
            return [self._row_to_preset(r) for r in cur.fetchall()]

    def get_preset(self, preset_id: str) -> Optional[ProviderPreset]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM provider_presets WHERE id = ?", (preset_id,)
            )
            row = cur.fetchone()
            return self._row_to_preset(row) if row else None

    def add_preset(self, p: ProviderPreset) -> ProviderPreset:
        """新增 preset。id/created_at/api_key_ref 未给则自动生成。"""
        if not p.id:
            p.id = _new_id()
        if not p.created_at:
            p.created_at = _now_iso()
        if not p.api_key_ref:
            p.api_key_ref = _key_ref(p.id)
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    """INSERT INTO provider_presets
                       (id, name, provider_type, base_url, model, api_key_ref,
                        enabled, created_at, is_active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (p.id, p.name, p.provider_type, p.base_url, p.model,
                     p.api_key_ref, int(p.enabled), p.created_at, int(p.is_active)),
                )
            except sqlite3.IntegrityError as e:
                raise ValueError(f"preset name 冲突或 id 重复: {e}") from e
        return p

    def update_preset(
        self, preset_id: str, p: ProviderPreset
    ) -> Optional[ProviderPreset]:
        """部分更新 (未给字段保留原值)。api_key_ref / is_active 不可此处改。

        api_key_ref 由 add 时固定;is_active 由 set_active 改;都拒绝通过 update 改。
        """
        existing = self.get_preset(preset_id)
        if existing is None:
            return None
        merged = ProviderPreset(
            id=preset_id,
            name=p.name or existing.name,
            provider_type=p.provider_type or existing.provider_type,
            base_url=p.base_url if p.base_url is not None else existing.base_url,
            model=p.model if p.model is not None else existing.model,
            api_key_ref=existing.api_key_ref,
            enabled=p.enabled if p.enabled is not None else existing.enabled,
            created_at=existing.created_at,
            is_active=existing.is_active,
        )
        with sqlite3.connect(self.db_path) as conn:
            try:
                cur = conn.execute(
                    """UPDATE provider_presets
                       SET name=?, provider_type=?, base_url=?, model=?, enabled=?
                       WHERE id=?""",
                    (merged.name, merged.provider_type, merged.base_url,
                     merged.model, int(merged.enabled), preset_id),
                )
            except sqlite3.IntegrityError as e:
                raise ValueError(f"preset name 冲突: {e}") from e
        if cur.rowcount == 0:
            return None
        return merged

    def delete_preset(self, preset_id: str) -> bool:
        """删 preset + best-effort 清 keyring 中的 key。"""
        existing = self.get_preset(preset_id)
        if existing is None:
            return False
        self._delete_keyring_key(existing.api_key_ref)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM provider_presets WHERE id = ?", (preset_id,)
            )
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # active 切换
    # ------------------------------------------------------------------
    def set_active(self, preset_id: str) -> Optional[ProviderPreset]:
        """设为活跃 (同时只一个)。事务内先全置 0 再置目标 1。"""
        existing = self.get_preset(preset_id)
        if existing is None:
            return None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE provider_presets SET is_active = 0")
            conn.execute(
                "UPDATE provider_presets SET is_active = 1 WHERE id = ?",
                (preset_id,),
            )
        existing.is_active = True
        return existing

    def get_active(self) -> Optional[ProviderPreset]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM provider_presets WHERE is_active = 1 LIMIT 1"
            )
            row = cur.fetchone()
            return self._row_to_preset(row) if row else None

    # ------------------------------------------------------------------
    # keyring 凭据 (复用 config_manager._secure_set/_secure_get,只读 import)
    # ------------------------------------------------------------------
    def set_api_key(self, preset_id: str, api_key: str) -> bool:
        """api_key 存 keyring。失败返回 False (调用方决定是否告警)。"""
        existing = self.get_preset(preset_id)
        if existing is None:
            return False
        try:
            from src.utils.config_manager import _secure_set
            _secure_set(existing.api_key_ref, api_key)
            return True
        except Exception as e:
            log.warning(f"keyring 写入失败 (preset={preset_id}): {e}")
            return False

    def get_api_key(self, preset_id: str) -> str:
        """从 keyring 取明文 key (仅内部 / 真正发起调用时用,REST 不暴露)。"""
        existing = self.get_preset(preset_id)
        if existing is None:
            return ""
        try:
            from src.utils.config_manager import _secure_get
            return _secure_get(existing.api_key_ref, "") or ""
        except Exception:
            return ""

    def has_api_key(self, preset_id: str) -> bool:
        """是否已存 key (不取明文,只判断存在性)。"""
        return bool(self.get_api_key(preset_id))

    def _delete_keyring_key(self, key_ref: str) -> None:
        """best-effort 删 keyring 中的 key (config_manager 未暴露 delete,自实现)。"""
        if not key_ref:
            return
        try:
            import keyring
            keyring.delete_password(_KEYRING_SERVICE, key_ref)
        except Exception:
            # key 不存在或 keyring 不可用,忽略 (DB 行已删,keyref 失效)
            pass
