"""网关凭据 AES 加解密。

凭据（微信/TG/Discord token）落盘前必须加密。本模块提供 GatewayCipher：

依赖决策（Builder-C 收口项）：
  - 优先用 `cryptography`（Fernet = AES-128-CBC + HMAC-SHA256，真 AES）
  - 项目 venv 当前**未安装** cryptography / pycryptodome（2026-09-06 探测）
  - 降级路径：用标准库 hashlib + hmac + secrets 实现 XOR 流密码 + HMAC
    完整性标签。**这不是真 AES**，仅做凭据落盘的最低限度可逆保护，
    阻止肉眼读 token。主控收口时应 `pip install cryptography` 并切到 Fernet。

设计：
  - encrypt(plaintext: str) -> str : 返回 urlsafe-b64 密文（含 nonce+tag+ct）
  - decrypt(ciphertext: str) -> str : 校验 HMAC 后还原
  - 密钥来自 master_key（调用方提供，建议从 OS 密钥环 / env 取）
  - 整个类不依赖 config_manager.py，是纯变换层；config_manager 只负责
    把加密后的密文写入密钥环/ini（由调用方编排）。

注意：降级模式的 XOR 流密码在已知明文攻击下不抗破解，仅用于"无 cryptography
依赖时让网关能跑"。生产凭据保护必须切到 Fernet（cryptography 依赖）。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets

logger = logging.getLogger(__name__)

# 探测 cryptography 是否可用（真 AES 路径）
try:
    from cryptography.fernet import Fernet  # type: ignore[import-untyped]
    _CRYPTOGRAPHY_AVAILABLE = True
except Exception:  # ImportError / 缺 DLL 等
    _CRYPTOGRAPHY_AVAILABLE = False

# 降级模式：XOR 流密钥长度（HMAC-SHA256 输出 32 字节，按 32 字节块）
_FALLBACK_KEY_LEN = 32
_FALLBACK_NONCE_LEN = 16
_FALLBACK_TAG_LEN = 32
# 版本前缀：区分 Fernet 与降级密文，便于 decrypt 时分流
_PREFIX_FERNET = "v1:"
_PREFIX_FALLBACK = "v0:"


class GatewayCipher:
    """网关凭据加解密。

    用法：
        cipher = GatewayCipher(master_key="...32+ bytes...")
        ct = cipher.encrypt("wx_token_xxx")
        # 落盘 ct（密文）到密钥环/ini
        pt = cipher.decrypt(ct)  # 还原明文 token

    master_key 派生：本类用 HKDF-like（PBKDF2-HMAC-SHA256, 100k 轮）从
    任意长度 master_key 派生定长 key，调用方无需关心密钥长度。
    """

    def __init__(self, master_key: str):
        if not master_key:
            raise ValueError("master_key 不能为空（凭据加密主密钥缺失）")
        self._master_key = master_key
        # 派生定长 key（Fernet 需 32 字节 urlsafe-b64；降级用 32 字节 raw）
        self._derived = self._derive_key(master_key)
        if _CRYPTOGRAPHY_AVAILABLE:
            self._fernet = Fernet(self._fernet_key(self._derived))
            logger.info("GatewayCipher: cryptography 可用，使用 Fernet（真 AES）")
        else:
            self._fernet = None
            logger.warning(
                "GatewayCipher: cryptography 未安装，降级到 XOR 流密码"
                "（非真 AES，仅做最低限度凭据混淆；主控应 pip install cryptography）"
            )

    # ------------------------------------------------------------------
    # 密钥派生
    # ------------------------------------------------------------------
    @staticmethod
    def _derive_key(master_key: str) -> bytes:
        """PBKDF2-HMAC-SHA256 派生 32 字节 key（固定 salt 仅做定长派生，
        不抗彩虹表——salt 真实用途由调用方在 master_key 层面保证熵）。"""
        return hashlib.pbkdf2_hmac(
            "sha256", master_key.encode("utf-8"),
            salt=b"vap-im-gateway-v1", iterations=100_000, dklen=32,
        )

    @staticmethod
    def _fernet_key(derived: bytes) -> bytes:
        """Fernet 需要 urlsafe-b64 的 32 字节 key。"""
        return base64.urlsafe_b64encode(derived)

    # ------------------------------------------------------------------
    # 加解密
    # ------------------------------------------------------------------
    def encrypt(self, plaintext: str) -> str:
        """加密明文 token，返回带版本前缀的密文（urlsafe-b64）。"""
        if not _CRYPTOGRAPHY_AVAILABLE or self._fernet is None:
            return self._encrypt_fallback(plaintext)
        ct = self._fernet.encrypt(plaintext.encode("utf-8"))
        return _PREFIX_FERNET + ct.decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        """解密密文，返回明文 token。HMAC 校验失败抛 ValueError。"""
        if ciphertext.startswith(_PREFIX_FERNET):
            if not _CRYPTOGRAPHY_AVAILABLE or self._fernet is None:
                # 密文是 Fernet 格式但当前环境无 cryptography → 无法解
                raise ValueError(
                    "密文为 Fernet 格式但 cryptography 不可用，无法解密"
                )
            ct = ciphertext[len(_PREFIX_FERNET):].encode("ascii")
            return self._fernet.decrypt(ct).decode("utf-8")
        if ciphertext.startswith(_PREFIX_FALLBACK):
            return self._decrypt_fallback(ciphertext)
        raise ValueError(f"无法识别的密文版本前缀: {ciphertext[:4]!r}")

    # ------------------------------------------------------------------
    # 降级实现：XOR 流密码 + HMAC-SHA256 完整性标签
    # ------------------------------------------------------------------
    def _encrypt_fallback(self, plaintext: str) -> str:
        """降级加密：nonce(16) + ct + tag(32)，整体 urlsafe-b64。"""
        nonce = secrets.token_bytes(_FALLBACK_NONCE_LEN)
        pt_bytes = plaintext.encode("utf-8")
        # 流密钥：HMAC(master_key, nonce) 循环异或
        stream = self._keystream(nonce, len(pt_bytes))
        ct = bytes(p ^ s for p, s in zip(pt_bytes, stream))
        # HMAC(nonce || ct) 防篡改
        tag = hmac.new(self._derived, nonce + ct, hashlib.sha256).digest()
        blob = nonce + ct + tag
        return _PREFIX_FALLBACK + base64.urlsafe_b64encode(blob).decode("ascii")

    def _decrypt_fallback(self, ciphertext: str) -> str:
        """降级解密：校验 HMAC 后还原明文。"""
        raw = base64.urlsafe_b64decode(ciphertext[len(_PREFIX_FALLBACK):])
        if len(raw) < _FALLBACK_NONCE_LEN + _FALLBACK_TAG_LEN:
            raise ValueError("降级密文长度不足")
        nonce = raw[:_FALLBACK_NONCE_LEN]
        tag = raw[-_FALLBACK_TAG_LEN:]
        ct = raw[_FALLBACK_NONCE_LEN:-_FALLBACK_TAG_LEN]
        # 先验 HMAC（常数时间比较，防时序侧信道）
        expected = hmac.new(self._derived, nonce + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("降级密文 HMAC 校验失败（凭据被篡改或 key 错误）")
        stream = self._keystream(nonce, len(ct))
        pt = bytes(c ^ s for c, s in zip(ct, stream))
        return pt.decode("utf-8")

    def _keystream(self, nonce: bytes, length: int) -> bytes:
        """从 HMAC-SHA256(nonce, counter) 拼出 length 字节流密钥。"""
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hmac.new(
                self._derived, nonce + counter.to_bytes(4, "big"),
                hashlib.sha256,
            ).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])


def is_cryptography_available() -> bool:
    """暴露 cryptography 是否可用，供网关启动时状态检查 / 告警。"""
    return _CRYPTOGRAPHY_AVAILABLE


def generate_master_key() -> str:
    """生成一个随机 master_key（用于初始化或轮换）。

    返回 urlsafe-b64 的 32 字节随机串，适合存入 OS 密钥环或 env 变量。
    """
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")
