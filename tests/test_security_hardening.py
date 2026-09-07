"""v6.0 安全强化测试（6.1 密钥环告警 + 6.2 headless 鉴权/限流 + 6.3 SHA256）。

- 6.1：is_keyring_available 探测 + audit_ini_key_cleared 明文残留检查
- 6.2：headless IP 限流（10 req/min 滑动窗口）+ 弱 Token 警告
- 6.3：verify_model_integrity 校验失败自动删除 + 通过写 .sha256 记录
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---- 6.1 密钥环 ----

def test_is_keyring_available_returns_bool() -> None:
    """is_keyring_available 返回 bool，不崩。"""
    from src.utils.config_manager import is_keyring_available
    r = is_keyring_available()
    assert isinstance(r, bool)


def test_audit_ini_key_cleared_empty(tmp_path: Path) -> None:
    """空 api_key 字段 = 已清空（安全）。"""
    from src.utils.config_manager import audit_ini_key_cleared
    ini = tmp_path / "app_config.ini"
    ini.write_text("[LastUsed]\napi_key=\nclient_type=1\n", encoding="utf-8")
    assert audit_ini_key_cleared(str(ini)) is True


def test_audit_ini_key_cleared_marker(tmp_path: Path) -> None:
    """__keyring__ 标记位 = 已清空（安全）。"""
    from src.utils.config_manager import audit_ini_key_cleared
    ini = tmp_path / "app_config.ini"
    ini.write_text("[LastUsed]\napi_key=__keyring__\n", encoding="utf-8")
    assert audit_ini_key_cleared(str(ini)) is True


def test_audit_ini_key_plaintext_leak(tmp_path: Path) -> None:
    """明文 api_key 残留 = 不安全（返回 False）。"""
    from src.utils.config_manager import audit_ini_key_cleared
    ini = tmp_path / "app_config.ini"
    ini.write_text("[LastUsed]\napi_key=sk-xxx-secret\n", encoding="utf-8")
    assert audit_ini_key_cleared(str(ini)) is False


def test_audit_ini_no_file(tmp_path: Path) -> None:
    """无 ini 文件 = 无残留（安全）。"""
    from src.utils.config_manager import audit_ini_key_cleared
    assert audit_ini_key_cleared(str(tmp_path / "no.ini")) is True


# ---- 6.2 headless 限流 ----

def test_ip_rate_limit_under_threshold() -> None:
    """同 IP 10 次内不限流。"""
    import importlib
    import src.server.headless as h
    importlib.reload(h)
    h._ip_request_times.clear()
    h._IP_RATE_LIMIT_PER_MIN = 10
    for _ in range(9):
        assert h._ip_rate_limited("1.2.3.4") is False


def test_ip_rate_limit_over_threshold() -> None:
    """同 IP 第 11 次被限流（返回 True）。"""
    import importlib
    import src.server.headless as h
    importlib.reload(h)
    h._ip_request_times.clear()
    h._IP_RATE_LIMIT_PER_MIN = 3
    for _ in range(3):
        assert h._ip_rate_limited("5.6.7.8") is False
    # 第 4 次应被限流
    assert h._ip_rate_limited("5.6.7.8") is True


def test_ip_rate_limit_disabled() -> None:
    """VAP_IP_RATE_LIMIT_PER_MIN=0 禁用限流。"""
    import importlib
    import src.server.headless as h
    importlib.reload(h)
    h._IP_RATE_LIMIT_PER_MIN = 0
    # 任意次都不限流
    for _ in range(100):
        assert h._ip_rate_limited("9.9.9.9") is False


def test_ip_rate_limit_different_ips_independent() -> None:
    """不同 IP 限流独立计数（limit=2 → 各 IP 第 3 次才限）。"""
    import importlib
    import src.server.headless as h
    importlib.reload(h)
    h._ip_request_times.clear()
    h._IP_RATE_LIMIT_PER_MIN = 2
    # 各 IP 前 2 次都不限
    assert h._ip_rate_limited("1.1.1.1") is False
    assert h._ip_rate_limited("2.2.2.2") is False
    assert h._ip_rate_limited("1.1.1.1") is False  # 1.1.1.1 第 2 次
    assert h._ip_rate_limited("2.2.2.2") is False  # 2.2.2.2 第 2 次
    # 各自第 3 次都应被限
    assert h._ip_rate_limited("1.1.1.1") is True
    assert h._ip_rate_limited("2.2.2.2") is True


# ---- 6.3 SHA256 强化 ----

def test_verify_model_integrity_delete_on_fail(tmp_path: Path) -> None:
    """校验失败自动删除被篡改模型文件。"""
    from src.core.logic import ModelManager
    mm = ModelManager(models_dir=tmp_path)
    fake_model = tmp_path / "fake_model.pt"
    fake_model.write_bytes(b"tampered content")
    mm.EXPECTED_SHA256 = {"fake_model": "0" * 64}
    mm.get_model_path = lambda mid: fake_model if mid == "fake_model" else None
    assert fake_model.exists()
    result = mm.verify_model_integrity("fake_model")
    assert result is False
    assert not fake_model.exists()


def test_verify_model_integrity_writes_sha256_record(tmp_path: Path) -> None:
    """校验通过写 .sha256 记录文件。"""
    from src.core.logic import ModelManager
    mm = ModelManager(models_dir=tmp_path)
    fake_model = tmp_path / "good_model.pt"
    content = b"good content"
    fake_model.write_bytes(content)
    actual_hash = hashlib.sha256(content).hexdigest()
    mm.EXPECTED_SHA256 = {"good_model": actual_hash}
    mm.get_model_path = lambda mid: fake_model if mid == "good_model" else None
    result = mm.verify_model_integrity("good_model")
    assert result is True
    record = fake_model.with_suffix(".pt.sha256")
    assert record.exists()
    assert actual_hash in record.read_text()


def test_verify_model_integrity_no_hash_constraint(tmp_path: Path) -> None:
    """EXPECTED_SHA256[id]=None 时仅存在性校验（不删文件）。"""
    from src.core.logic import ModelManager
    mm = ModelManager(models_dir=tmp_path)
    fake_model = tmp_path / "yolo_v11n.pt"
    fake_model.write_bytes(b"any")
    mm.EXPECTED_SHA256 = {"yolo_v11n": None}
    mm.get_model_path = lambda mid: fake_model if mid == "yolo_v11n" else None
    assert mm.verify_model_integrity("yolo_v11n") is True
    assert fake_model.exists()


def test_verify_model_integrity_missing_file(tmp_path: Path) -> None:
    """模型文件不存在 → False 不崩。"""
    from src.core.logic import ModelManager
    mm = ModelManager(models_dir=tmp_path)
    mm.EXPECTED_SHA256 = {"missing": "0" * 64}
    mm.get_model_path = lambda mid: tmp_path / "noexist.pt"
    assert mm.verify_model_integrity("missing") is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
