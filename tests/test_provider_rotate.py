"""Provider rotate / revoke 端点测试。

覆盖:
  - POST /api/providers/{id}/rotate 写新 key + 返回 fingerprint (明文不回传)
  - rotate 拒绝弱 key (min_length=8 → 422)
  - rotate 不存在 preset → 404
  - POST /api/providers/{id}/revoke 标记 revoked
  - revoke 幂等 (连续两次都 200)
  - 响应 dict 不含明文 key 字段

测试用独立 keyring 内存表 + 独立 preset DB + 独立 audit DB (tmp 路径),
不污染项目 config/。keyring 用 monkeypatch 替换 config_manager._secure_set
/_secure_get,不依赖真实 OS keyring 后端。CredentialResolver 只用 keyring 源
(env/ini 不参与,避免污染进程 env,且与 set_api_key 写入的 keyring 条目一致)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch 必须先于 FastAPI 导入 (Windows DLL 顺序铁律)
try:
    import torch  # noqa: F401
except OSError:
    torch = None

from fastapi.testclient import TestClient  # noqa: E402

# 测试用 keyring 内存表 (替代真实 OS keyring)
_KEYRING_MEM: dict[str, str] = {}


def _fake_set(key: str, value: str):
    if value:
        _KEYRING_MEM[key] = value
        return None
    _KEYRING_MEM.pop(key, None)
    return None


def _fake_get(key: str, fallback: str = "") -> str:
    return _KEYRING_MEM.get(key, fallback)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """每个测试用独立 keyring 内存表 + 独立 preset DB + 独立 audit DB。

    - keyring: monkeypatch config_manager._secure_set/_secure_get → 内存表
    - preset DB: monkeypatch provider_preset.CONFIG_DIR → tmp/cfg
    - audit DB: monkeypatch providers._get_rotator → 用 tmp/audit.db 构造的实例
    - resolver: monkeypatch providers._get_resolver → 只 keyring 源 (不污染 env)
    """
    _KEYRING_MEM.clear()
    monkeypatch.setattr(
        "src.utils.config_manager._secure_set", _fake_set, raising=True
    )
    monkeypatch.setattr(
        "src.utils.config_manager._secure_get", _fake_get, raising=True
    )
    monkeypatch.setattr(
        "src.core.provider_preset._KEYRING_SERVICE", "VideoAnalysisProTest"
    )
    monkeypatch.setattr(
        "src.core.provider_preset.CONFIG_DIR", str(tmp_path / "cfg")
    )
    # provider_preset._delete_keyring_key 直接调 keyring.delete_password
    class _KrStub:
        @staticmethod
        def delete_password(service: str, key: str) -> None:
            _KEYRING_MEM.pop(key, None)
    monkeypatch.setitem(sys.modules, "keyring", _KrStub())

    # reset router 单例 (store / rotator / audit / resolver)
    import src.web.routers.providers as pr
    monkeypatch.setattr(pr, "_store", None)
    monkeypatch.setattr(pr, "_rotator", None)
    monkeypatch.setattr(pr, "_audit", None)
    monkeypatch.setattr(pr, "_resolver", None)

    # 让 _get_rotator / _get_resolver 用 tmp audit + keyring-only resolver
    from src.core.credentials.audit import CredentialAudit
    from src.core.credentials.key import (
        CredentialResolver,
        KeyringCredentialSource,
    )
    from src.core.credentials.rotation import KeyRotator
    _audit_inst = CredentialAudit(db_path=str(tmp_path / "audit.db"))
    _rotator_inst = KeyRotator(audit=_audit_inst)
    _resolver_inst = CredentialResolver(keyring=KeyringCredentialSource())
    monkeypatch.setattr(pr, "_get_rotator", lambda: _rotator_inst)
    monkeypatch.setattr(pr, "_get_resolver", lambda: _resolver_inst)

    yield
    _KEYRING_MEM.clear()


@pytest.fixture
def client():
    """FastAPI TestClient (不污染项目 config/)。"""
    from src.web.app import app
    with TestClient(app) as c:
        yield c


def _create_preset(client, name="p", api_key="sk-init-12345") -> str:
    """建一个 preset (带 api_key) 返回 id。"""
    r = client.post("/api/providers", json={
        "name": name, "provider_type": "openai",
        "api_key": api_key,
    })
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ============================ rotate 端点 ============================

def test_rotate_endpoint_sets_new_key(client):
    """POST /rotate 带 new_key → 200 + fingerprint 非空 + 明文 key 不在响应。"""
    pid = _create_preset(client)
    r = client.post(f"/api/providers/{pid}/rotate", json={
        "new_key": "sk-rotated-12345",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["key_name"] == f"vap_provider:{pid}"
    assert body["new_key_set"] is True
    assert body["fingerprint"]
    assert len(body["fingerprint"]) == 8
    # 明文 key 不在响应
    assert "new_key" not in body
    assert "sk-rotated-12345" not in r.text


def test_rotate_endpoint_rejects_short_key(client):
    """new_key='123' → 422 (min_length=8 校验)。"""
    pid = _create_preset(client)
    r = client.post(f"/api/providers/{pid}/rotate", json={"new_key": "123"})
    assert r.status_code == 422


def test_rotate_endpoint_404_for_missing_preset(client):
    """不存在的 preset_id → 404。"""
    r = client.post("/api/providers/nonexistent/rotate", json={
        "new_key": "sk-rotated-12345",
    })
    assert r.status_code == 404


def test_rotate_response_no_plaintext_key(client):
    """响应 dict 不含 new_key / api_key / 明文 key 字段。"""
    pid = _create_preset(client)
    new_key = "sk-supersecret-9999"
    r = client.post(f"/api/providers/{pid}/rotate", json={"new_key": new_key})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "new_key" not in body
    assert "api_key" not in body
    assert "plaintext" not in body
    # 明文 key 不出现在任何响应文本
    assert new_key not in r.text
    assert new_key not in r.content.decode("utf-8", errors="ignore")


def test_rotate_writes_to_keyring(client):
    """rotate 后 keyring 中是新 key (与 set_api_key 同一 keyring 条目)。"""
    pid = _create_preset(client, api_key="sk-old-12345")
    r = client.post(f"/api/providers/{pid}/rotate", json={
        "new_key": "sk-new-12345678",
    })
    assert r.status_code == 200
    # keyring 内存表中,key 是 "vap_provider:{id}",值应是新 key
    assert _KEYRING_MEM.get(f"vap_provider:{pid}") == "sk-new-12345678"
    # 旧 key 不再存在 (被覆盖)
    assert _KEYRING_MEM.get(f"vap_provider:{pid}") != "sk-old-12345"


# ============================ revoke 端点 ============================

def test_revoke_endpoint_marks_revoked(client):
    """POST /revoke → 200 + ok=True + revoked=True。"""
    pid = _create_preset(client, api_key="sk-init-12345")
    r = client.post(f"/api/providers/{pid}/revoke")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["revoked"] is True
    assert body["key_name"] == f"vap_provider:{pid}"


def test_revoke_endpoint_idempotent(client):
    """连续 revoke 两次都 200 (幂等)。"""
    pid = _create_preset(client, api_key="sk-init-12345")
    r1 = client.post(f"/api/providers/{pid}/revoke")
    assert r1.status_code == 200
    assert r1.json()["revoked"] is True
    # 再吊销一次仍 200 (幂等)
    r2 = client.post(f"/api/providers/{pid}/revoke")
    assert r2.status_code == 200
    assert r2.json()["revoked"] is True


def test_revoke_endpoint_404_for_missing_preset(client):
    """不存在的 preset_id → 404。"""
    r = client.post("/api/providers/nonexistent/revoke")
    assert r.status_code == 404


def test_revoke_marks_keyring_value_with_suffix(client):
    """revoke 后 keyring 值含 _revoked_ 后缀 (is_revoked True)。"""
    pid = _create_preset(client, api_key="sk-init-12345")
    r = client.post(f"/api/providers/{pid}/revoke")
    assert r.status_code == 200
    val = _KEYRING_MEM.get(f"vap_provider:{pid}")
    assert val is not None
    assert "_revoked_" in val
