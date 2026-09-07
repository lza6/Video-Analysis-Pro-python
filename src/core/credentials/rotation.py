"""凭据轮换：rotate / revoke / is_revoked。

支撑《改进指南》6.3 凭据轮换与审计日志。本期只落地独立模块 + 测试，
暂不接入主控（key.py / settings 页保持不变，后续再做）。

设计原则：
  - rotate 写新值走 resolver.store（env 优先，env 不可写则 keyring/ini）
  - revoke 不真删 env（env 是进程级全局，删了影响其他调用方），只在
    keyring 值后追加 `_revoked_<ts>` 后缀标记，is_revoked 检查后缀
  - 所有操作记 audit（action=rotate），便于审计追溯
  - rotation.py 顶部 `from .key import CredentialResolver` 只作类型注解
    （TYPE_CHECKING），不真实调——避免循环依赖

注意：revoke 标记只对 keyring 值生效；env 值无法标记（env 不可写后缀），
is_revoked 对 env 源返回 False（env 值视为"有效但无法吊销"，调用方应在
rotate 时用新 key_name 而非 revoke 旧 env）。
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, NamedTuple, Optional

from src.core.credentials.audit import CredentialAudit

if TYPE_CHECKING:
    # 仅作类型注解，避免循环依赖（rotation 不应在 key.py 之外被 key.py import）
    from src.core.credentials.key import CredentialResolver


# keyring 值的吊销后缀：`<原值>_revoked_<ISO时间>`
_REVOKED_SUFFIX = "_revoked_"


def _now_iso() -> str:
    """ISO8601 时间戳（毫秒精度，UTC）。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _parse_key(key_name: str) -> "tuple[str, str]":
    """拆 "provider:key_name" → (provider, kn)，兼容裸 key_name。"""
    if ":" in key_name:
        provider, kn = key_name.split(":", 1)
    else:
        provider, kn = "default", key_name
    return provider, kn


class RotatedKey(NamedTuple):
    """轮换结果状态。

    Attributes:
        key_name: 被轮换的 key 名。
        old_revoked: 旧 key 是否已标记吊销。
        new_key_set: 新 key 是否已写入。
    """

    key_name: str
    old_revoked: bool
    new_key_set: bool


class KeyRotator:
    """凭据轮换器。

    所有操作都记 audit（action=rotate）。线程安全（revoke 写 keyring
    用 threading.Lock 串行化，避免并发追加后缀时丢失）。

    注意：rotation.py 顶部 `from .key import CredentialResolver` 只作
    类型注解（TYPE_CHECKING），不真实调——避免循环依赖。
    """

    def __init__(self, audit: Optional[CredentialAudit] = None) -> None:
        self._audit = audit
        self._lock = threading.Lock()

    def rotate(
        self,
        key_name: str,
        new_value: str,
        resolver: "CredentialResolver",
    ) -> RotatedKey:
        """轮换 key：写新值 + 记 audit。

        调用 resolver.store(ck, new_value) 写新值（env 优先，env 不可写
        则 keyring/ini）。记 audit（action=rotate, caller=KeyRotator）。
        """
        from src.core.credentials.key import CredentialKey

        provider, kn = _parse_key(key_name)
        ck = CredentialKey(provider=provider, key_name=kn)

        new_key_set = True
        try:
            resolver.store(ck, new_value)
        except Exception:
            new_key_set = False

        if self._audit is not None:
            self._audit.log(
                key_name=key_name, action="rotate",
                caller="KeyRotator", success=new_key_set,
            )
        return RotatedKey(
            key_name=key_name, old_revoked=False, new_key_set=new_key_set,
        )

    def revoke(self, key_name: str, resolver: "CredentialResolver") -> bool:
        """吊销 key：在 keyring 值后追加 `_revoked_<ts>` 后缀标记。

        不真删 env（env 是进程级全局，删了影响其他调用方）。
        记 audit（action=rotate, success=False, 标记旧 key revoked）。

        Returns:
            是否成功追加吊销后缀（读到值且未吊销才追加，否则 False）。
        """
        from src.core.credentials.key import CredentialKey

        provider, kn = _parse_key(key_name)
        ck = CredentialKey(provider=provider, key_name=kn)

        try:
            current = resolver.resolve(ck)
        except Exception:
            current = None

        ok = False
        if current and _REVOKED_SUFFIX not in current:
            marked = f"{current}{_REVOKED_SUFFIX}{_now_iso()}"
            with self._lock:
                try:
                    resolver.store(ck, marked)
                    ok = True
                except Exception:
                    ok = False

        if self._audit is not None:
            self._audit.log(
                key_name=key_name, action="rotate",
                caller="KeyRotator",
                success=False,  # 标记旧 key revoked
            )
        return ok

    def is_revoked(self, key_name: str, resolver: "CredentialResolver") -> bool:
        """检查 key 是否已被吊销（读 resolver.resolve 检查后缀）。"""
        from src.core.credentials.key import CredentialKey

        provider, kn = _parse_key(key_name)
        ck = CredentialKey(provider=provider, key_name=kn)

        try:
            current = resolver.resolve(ck)
        except Exception:
            return False
        if not current:
            return False
        return _REVOKED_SUFFIX in current
