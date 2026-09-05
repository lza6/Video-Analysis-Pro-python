"""RunStoreRepository 协议测试（v6.x 指南 4.4 分布式前置）。

验证 RunStore 天然满足 RunStoreRepository 协议（duck typing），
为将来 PostgreSQL/远程实现替换留接口。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest


def test_runstore_satisfies_repository_protocol() -> None:
    """RunStore 天然满足 RunStoreRepository 协议（所有方法都有）。"""
    from src.core.run_store import RunStore
    from src.core.run_store_repository import is_repository
    tmp = Path(tempfile.mkdtemp()) / "test.db"
    rs = RunStore(str(tmp))
    assert is_repository(rs) is True


def test_is_repository_rejects_non_repository() -> None:
    """非 repository 对象被拒绝。"""
    from src.core.run_store_repository import is_repository
    assert is_repository(object()) is False
    assert is_repository("not a repo") is False
    assert is_repository(None) is False


def test_repository_protocol_methods_exist_on_runstore() -> None:
    """RunStore 有协议要求的全部方法。"""
    from src.core.run_store import RunStore
    tmp = Path(tempfile.mkdtemp()) / "test.db"
    rs = RunStore(str(tmp))
    # Protocol 是 duck typing，检查方法存在
    for m in ("create_run", "update_run", "list_runs", "get_run",
              "get_progress", "add_segment", "add_hit", "add_clip",
              "delete_run", "clear_all"):
        assert hasattr(rs, m), f"RunStore 缺方法 {m}"


def test_repository_dependency_injection() -> None:
    """调用方按协议编程：传 RunStore 实例给接受 Repository 的函数。"""
    from src.core.run_store import RunStore
    from src.core.run_store_repository import is_repository
    tmp = Path(tempfile.mkdtemp()) / "test.db"
    rs = RunStore(str(tmp))

    def count_runs(repo) -> int:
        """接受任意 Repository 实现的函数。"""
        assert is_repository(repo), "repo 必须满足 RunStoreRepository 协议"
        return len(repo.list_runs(limit=1000))

    assert count_runs(rs) == 0
    rs.create_run("test.mp4", status="done")
    assert count_runs(rs) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
