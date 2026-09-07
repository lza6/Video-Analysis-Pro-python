"""CredentialAudit + KeyRotator 测试。

覆盖：
  - log 一条 + list 查到
  - list 按 key_name 过滤
  - list limit 生效
  - read/write/rotate 三 action 都记录
  - 多线程并发 log 不丢（10 线程 × 10 条 = 100）
  - KeyRotator.rotate 写新值 + 记 audit rotate
  - KeyRotator.revoke 标记后 is_revoked True

测试用临时 db（tmp_path fixture），不污染 config/。
"""
from __future__ import annotations

import threading

from src.core.credentials.audit import CredentialAudit
from src.core.credentials.key import (
    CredentialResolver,
    EnvCredentialSource,
    KeyringCredentialSource,
    IniCredentialSource,
)
from src.core.credentials.rotation import KeyRotator


def _make_resolver(env=None, keyring=None, ini=None) -> CredentialResolver:
    """三源分层（env > keyring > ini）。"""
    return CredentialResolver(
        env=env if env is not None else EnvCredentialSource(),
        keyring=keyring if keyring is not None else KeyringCredentialSource(),
        ini=ini if ini is not None else IniCredentialSource(),
    )


# --------------------------------------------------------------------------
# CredentialAudit
# --------------------------------------------------------------------------
def test_log_and_list(tmp_path):
    """log 一条 + list 查到。"""
    audit = CredentialAudit(db_path=str(tmp_path / "audit.db"))
    audit.log(key_name="nvidia:k1", action="read", caller="test")
    entries = audit.list()
    assert len(entries) == 1
    e = entries[0]
    assert e.key_name == "nvidia:k1"
    assert e.action == "read"
    assert e.caller == "test"
    assert e.success is True
    assert e.trace_id is None
    assert e.ts  # 非空


def test_list_filter_by_key_name(tmp_path):
    """list 按 key_name 过滤。"""
    audit = CredentialAudit(db_path=str(tmp_path / "audit.db"))
    audit.log(key_name="nvidia:k1", action="read", caller="t")
    audit.log(key_name="openai:k2", action="write", caller="t")
    audit.log(key_name="nvidia:k1", action="rotate", caller="t")

    only_k1 = audit.list(key_name="nvidia:k1")
    assert len(only_k1) == 2
    assert all(e.key_name == "nvidia:k1" for e in only_k1)

    only_k2 = audit.list(key_name="openai:k2")
    assert len(only_k2) == 1
    assert only_k2[0].key_name == "openai:k2"


def test_list_limit(tmp_path):
    """list limit 生效。"""
    audit = CredentialAudit(db_path=str(tmp_path / "audit.db"))
    for i in range(10):
        audit.log(key_name="k", action="read", caller="t", trace_id=str(i))
    assert len(audit.list(limit=5)) == 5
    assert len(audit.list(limit=100)) == 10


def test_three_actions_recorded(tmp_path):
    """read/write/rotate 三 action 都记录。"""
    audit = CredentialAudit(db_path=str(tmp_path / "audit.db"))
    audit.log(key_name="k", action="read", caller="t")
    audit.log(key_name="k", action="write", caller="t")
    audit.log(key_name="k", action="rotate", caller="t")
    actions = {e.action for e in audit.list()}
    assert actions == {"read", "write", "rotate"}


def test_concurrent_log_no_loss(tmp_path):
    """多线程并发 log 不丢（10 线程 × 10 条 = 100）。"""
    audit = CredentialAudit(db_path=str(tmp_path / "audit.db"))

    def worker(tid: int) -> None:
        for i in range(10):
            audit.log(
                key_name=f"k_{tid}",
                action="read",
                caller=f"thread-{tid}",
                trace_id=f"{tid}-{i}",
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(audit.list(limit=1000)) == 100


# --------------------------------------------------------------------------
# KeyRotator
# --------------------------------------------------------------------------
def test_rotator_rotate_writes_and_audits(tmp_path):
    """KeyRotator.rotate 写新值 + 记 audit rotate。"""
    audit = CredentialAudit(db_path=str(tmp_path / "audit.db"))
    rotator = KeyRotator(audit=audit)
    resolver = _make_resolver()

    # env 源可写（os.environ），store 会写回 env
    result = rotator.rotate("nvidia:k1", "new-secret-value", resolver)
    assert result.new_key_set is True
    assert result.old_revoked is False

    # audit 记了一条 rotate
    entries = audit.list(key_name="nvidia:k1")
    assert len(entries) == 1
    assert entries[0].action == "rotate"
    assert entries[0].caller == "KeyRotator"
    assert entries[0].success is True


def test_rotator_revoke_marks_revoked(tmp_path):
    """KeyRotator.revoke 标记后 is_revoked True。"""
    audit = CredentialAudit(db_path=str(tmp_path / "audit.db"))
    rotator = KeyRotator(audit=audit)
    resolver = _make_resolver()

    # 先 rotate 写一个值
    rotator.rotate("nvidia:k2", "original-value", resolver)
    assert rotator.is_revoked("nvidia:k2", resolver) is False

    # revoke 追加后缀标记
    ok = rotator.revoke("nvidia:k2", resolver)
    assert ok is True
    assert rotator.is_revoked("nvidia:k2", resolver) is True

    # audit 记了 2 条（rotate + revoke 的 rotate action）
    entries = audit.list(key_name="nvidia:k2")
    assert len(entries) == 2
    # revoke 的 success=False
    assert entries[0].success is False  # 最近一条（revoke）
    assert entries[1].success is True   # 之前一条（rotate）
