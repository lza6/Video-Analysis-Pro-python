"""RunStore Repository 接口抽象（v6.x 指南 4.4 分布式前置）。

指南 4.4：分布式批量分发要把 SQLite 替换 PostgreSQL，让多 worker 节点共享。
当前 RunStore 是具体 SQLite 实现，本模块抽出 Repository 协议（Protocol，
duck typing），为将来 PostgreSQL/远程实现留接口，**不改 RunStore 现有实现**。

RunStore 已满足该协议（list_runs/get_run/create_run/update_run/add_segment/
add_hit/add_clip/delete_run/get_progress/clear_all 全有），所以它天然是
RunStoreRepository 的实现——无需改 RunStore，只需让调用方按协议编程。

用法（调用方从具体类解耦）：
    from src.core.run_store_repository import RunStoreRepository
    def process(repo: RunStoreRepository):
        runs = repo.list_runs(limit=10)
        ...

    # 本地：RunStore（已有）
    # 将来分布式：PostgresRunStore（待实现）替换即可
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class RunStoreRepository(Protocol):
    """运行记录存储仓库协议（duck typing，RunStore 天然满足）。

    协议方法签名与 src/core/run_store.py 的 RunStore 保持一致，调用方
    按此协议编程即可在 SQLite/PostgreSQL/远程实现间切换。

    将来实现 PostgresRunStore 时，只需满足这些方法签名，调用方零改动。
    """

    def create_run(self, video_path: str, *, duration_sec: Optional[float] = None,
                   model: Optional[str] = None, provider: Optional[str] = None,
                   mode: Optional[str] = None, status: str = "started") -> str:
        ...

    def update_run(self, run_id: str, **fields: Any) -> bool:
        ...

    def list_runs(self, limit: int = 50,
                  status: Optional[str] = None) -> List[Dict[str, Any]]:
        ...

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        ...

    def get_progress(self, run_id: str) -> Optional[Dict[str, Any]]:
        ...

    def add_segment(self, run_id: str, seg_data: Dict[str, Any]) -> str:
        ...

    def add_hit(self, run_id: str, hit_data: Dict[str, Any]) -> str:
        ...

    def add_clip(self, run_id: str, hit_idx: int, abs_timestamp: str,
                 clip_path: str) -> str:
        ...

    def delete_run(self, run_id: str, purge_files: bool = False) -> bool:
        ...

    def clear_all(self, purge_files: bool = False) -> int:
        ...


def is_repository(obj: Any) -> bool:
    """运行时检查 obj 是否满足 RunStoreRepository 协议。

    用于依赖注入校验：调用方传 repo 前 assert is_repository(repo)，
    不满足早抛 TypeError 而非运行时 AttributeError。
    """
    required = ("create_run", "update_run", "list_runs", "get_run",
               "get_progress", "add_segment", "add_hit", "add_clip",
               "delete_run", "clear_all")
    return all(hasattr(obj, m) for m in required)
