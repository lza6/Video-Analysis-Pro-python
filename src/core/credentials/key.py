"""CredentialKey 分层 + 多源解析。

参考 DSH 凭据分层概念：credential key 不是单一字符串，而是分 provider/model
的多维键，可来自 env / ini / keyring 多源，按优先级回退。

复用现有 `src/utils/config_manager.py` 的密钥环实现：KeyringCredentialSource
内部调用 `config_manager._secure_get`（只 import，不改 config_manager）。

支撑 NVIDIA 11 key + 未来 OpenAI/Anthropic 多 provider key 路由。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Protocol


class CredentialNotFound(Exception):
    """凭据未在任何源找到。"""


class CredentialSource(Protocol):
    """凭据源协议：按 (provider, key_name) 取真实 key。"""

    def get(self, provider: str, key_name: str) -> Optional[str]:
        ...


    def set(self, provider: str, key_name: str, value: str) -> None:
        ...


@dataclass(frozen=True)
class CredentialKey:
    """凭据键的不可变描述。

    Attributes:
        provider: "nvidia" | "openai" | "anthropic" | "ollama" | ...
        key_name: 该 provider 下的 key 名（如 "NVIDIA_API_KEY_1" 或 "default"）。
        env_var: 对应的环境变量名（可空）。
        ini_section: ini 配置段名（可空）。
        keyring_id: 密钥环条目 id（可空）。
    """

    provider: str
    key_name: str = "default"
    env_var: Optional[str] = None
    ini_section: Optional[str] = None
    keyring_id: Optional[str] = None


class EnvCredentialSource:
    """环境变量源：os.environ.get(env_var)。"""

    def get(self, provider: str, key_name: str) -> Optional[str]:
        # 约定 env_var = f"{PROVIDER}_{KEY_NAME}_API_KEY".upper()
        var = f"{provider}_{key_name}_API_KEY".upper()
        return os.environ.get(var)

    def set(self, provider: str, key_name: str, value: str) -> None:
        os.environ[f"{provider}_{key_name}_API_KEY".upper()] = value


class IniCredentialSource:
    """ini 源：从 app_config.ini 读 LastUsed.api_key（标记位或明文）。

    复用 config_manager 的 configparser 实例（只读 import）。
    """

    def __init__(self, config_manager=None) -> None:
        self._cm = config_manager

    def get(self, provider: str, key_name: str) -> Optional[str]:
        if self._cm is None:
            return None
        try:
            cfg = getattr(self._cm, "config", None)
            if cfg is None or not hasattr(cfg, "has_section"):
                return None
            if not cfg.has_section("LastUsed"):
                return None
            # 只在 key_name == "default" 时返回 ini 的 api_key
            if key_name != "default":
                return None
            v = cfg.get("LastUsed", "api_key", fallback="")
            # "__keyring__" 标记位 = 实际 key 在密钥环，不在 ini
            if v in ("", "__keyring__"):
                return None
            return v
        except Exception:
            return None

    def set(self, provider: str, key_name: str, value: str) -> None:
        if self._cm is None:
            return
        try:
            self._cm.update_config("LastUsed", "api_key", value)
        except Exception:
            pass


class KeyringCredentialSource:
    """密钥环源：复用 config_manager._secure_get / _secure_set。

    **只 import config_manager，不改它**。
    """

    def get(self, provider: str, key_name: str) -> Optional[str]:
        try:
            from src.utils import config_manager as cm
        except Exception:
            return None
        try:
            return cm._secure_get(f"{provider}:{key_name}") or None
        except Exception:
            return None

    def set(self, provider: str, key_name: str, value: str) -> None:
        try:
            from src.utils import config_manager as cm
            cm._secure_set(f"{provider}:{key_name}", value)
        except Exception:
            pass


@dataclass
class CredentialResolver:
    """多源分层解析器。

    按优先级顺序查 sources，首个命中返回。全 miss 抛 CredentialNotFound。

    默认优先级：env > keyring > ini（env 最高，keyring 次之，ini 兜底）。
    """

    sources: List[CredentialSource] = field(default_factory=list)

    def __init__(
        self,
        *,
        env: Optional[CredentialSource] = None,
        keyring: Optional[CredentialSource] = None,
        ini: Optional[CredentialSource] = None,
        extra: Optional[List[CredentialSource]] = None,
    ) -> None:
        self.sources = []
        if env is not None:
            self.sources.append(env)
        if keyring is not None:
            self.sources.append(keyring)
        if ini is not None:
            self.sources.append(ini)
        if extra:
            self.sources.extend(extra)

    @classmethod
    def default(cls, config_manager=None) -> "CredentialResolver":
        """默认三源分层（env > keyring > ini）。"""
        return cls(
            env=EnvCredentialSource(),
            keyring=KeyringCredentialSource(),
            ini=IniCredentialSource(config_manager),
        )

    def resolve(self, key: CredentialKey) -> str:
        """按优先级查 sources，首个命中返回。全 miss 抛 CredentialNotFound。"""
        for src in self.sources:
            try:
                v = src.get(key.provider, key.key_name)
            except Exception:
                v = None
            if v:
                return v
        raise CredentialNotFound(
            f"No credential for provider={key.provider} key={key.key_name}")

    def store(self, key: CredentialKey, value: str) -> None:
        """写所有源（keyring 优先，env/ini 兜底）。"""
        for src in self.sources:
            try:
                src.set(key.provider, key.key_name, value)
            except Exception:
                pass
