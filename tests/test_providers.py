"""多 Provider 预设 CRUD + active 切换 + keyring 存取 单测。

覆盖:
  - ProviderPresetStore: CRUD + set_active + keyring 存取 (独立 DB)
  - REST router: GET/POST/PUT/DELETE + activate + active
  - api_key 永不回传 (GET 只返回 has_key 布尔)
  - 重名 409 / 不存在 404 / 参数化查询 (无注入)

测试用独立 DB (tmp 路径),不污染项目 config/。
keyring 用 monkeypatch 替换 config_manager._secure_set/_secure_get,
不依赖真实 OS keyring 后端。
"""
from __future__ import annotations

import os
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
def _isolate_keyring(monkeypatch):
    """每个测试用独立 keyring 内存表 + 独立 DB,互不污染。

    provider_preset._delete_keyring_key 直接调 keyring.delete_password
    (config_manager 未暴露 delete helper)。keyring 可能未安装
    (config_manager 内部 try/except 守护),故用 sys.modules 注入 stub,
    保证 _delete_keyring_key 的 `import keyring` 能成功。
    """
    import sys
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
    # provider_preset._delete_keyring_key 用 keyring.delete_password,
    # 替换为内存表删除 (与 _fake_set/_fake_get 同一表)
    class _KrStub:
        @staticmethod
        def delete_password(service: str, key: str) -> None:
            _KEYRING_MEM.pop(key, None)
    stub = _KrStub()
    monkeypatch.setitem(sys.modules, "keyring", stub)
    yield
    _KEYRING_MEM.clear()


@pytest.fixture
def store(tmp_path):
    """独立 DB 的 ProviderPresetStore (tmp 路径,测试结束自动清理)。"""
    from src.core.provider_preset import ProviderPresetStore
    return ProviderPresetStore(config_dir=str(tmp_path / "cfg"))


# ============================ Store 单测 ============================

def test_add_and_get_preset(store):
    from src.core.provider_preset import ProviderPreset
    p = ProviderPreset(
        id="", name="my-openai", provider_type="openai",
        base_url="https://api.openai.com/v1", model="gpt-4o",
    )
    created = store.add_preset(p)
    assert created.id
    assert created.api_key_ref == f"vap_provider:{created.id}"
    assert created.created_at

    got = store.get_preset(created.id)
    assert got is not None
    assert got.name == "my-openai"
    assert got.provider_type == "openai"
    assert got.is_active is False


def test_list_presets_order(store):
    from src.core.provider_preset import ProviderPreset
    a = store.add_preset(ProviderPreset(id="", name="a"))
    b = store.add_preset(ProviderPreset(id="", name="b"))
    ids = [p.id for p in store.list_presets()]
    assert ids == [a.id, b.id]  # created_at 升序


def test_duplicate_name_rejected(store):
    from src.core.provider_preset import ProviderPreset
    store.add_preset(ProviderPreset(id="", name="dup"))
    with pytest.raises(ValueError):
        store.add_preset(ProviderPreset(id="", name="dup"))


def test_update_preset_partial(store):
    from src.core.provider_preset import ProviderPreset
    created = store.add_preset(
        ProviderPreset(id="", name="orig", provider_type="custom",
                       base_url="http://x", model="m1")
    )
    updated = store.update_preset(
        created.id,
        ProviderPreset(
            id=created.id, name="orig", provider_type="openai",
            base_url=None, model=None, enabled=None,
        ),
    )
    assert updated is not None
    assert updated.provider_type == "openai"
    assert updated.base_url == "http://x"  # 保留
    assert updated.model == "m1"  # 保留


def test_update_preset_does_not_change_active(store):
    """update_preset 不可改 is_active (只能 set_active 改)。"""
    from src.core.provider_preset import ProviderPreset
    created = store.add_preset(ProviderPreset(id="", name="p"))
    store.set_active(created.id)
    store.update_preset(
        created.id,
        ProviderPreset(id=created.id, name="p2", provider_type="custom",
                       base_url=None, model=None, enabled=None),
    )
    got = store.get_preset(created.id)
    assert got.is_active is True  # 仍是 active


def test_delete_preset(store):
    from src.core.provider_preset import ProviderPreset
    created = store.add_preset(ProviderPreset(id="", name="to-del"))
    assert store.delete_preset(created.id) is True
    assert store.get_preset(created.id) is None
    assert store.delete_preset(created.id) is False  # 二次删


def test_set_active_exclusive(store):
    """同时只一个 active;set_active 新的旧的自动失活。"""
    from src.core.provider_preset import ProviderPreset
    a = store.add_preset(ProviderPreset(id="", name="a"))
    b = store.add_preset(ProviderPreset(id="", name="b"))
    store.set_active(a.id)
    assert store.get_active().id == a.id
    store.set_active(b.id)
    assert store.get_active().id == b.id
    assert store.get_preset(a.id).is_active is False


def test_set_active_unknown_returns_none(store):
    assert store.set_active("nonexistent") is None


def test_keyring_set_get(store):
    """api_key 存 keyring,DB 不落明文。"""
    from src.core.provider_preset import ProviderPreset
    created = store.add_preset(
        ProviderPreset(id="", name="k", provider_type="openai")
    )
    assert store.has_api_key(created.id) is False
    assert store.set_api_key(created.id, "sk-secret-123") is True
    assert store.has_api_key(created.id) is True
    assert store.get_api_key(created.id) == "sk-secret-123"
    # DB 行本身不含明文 key
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        cur = conn.execute("SELECT api_key_ref FROM provider_presets WHERE id=?",
                           (created.id,))
        row = cur.fetchone()
        assert "sk-secret-123" not in row[0]


def test_delete_clears_keyring(store):
    """删 preset 时 best-effort 清 keyring 中的 key。"""
    from src.core.provider_preset import ProviderPreset
    created = store.add_preset(ProviderPreset(id="", name="k2"))
    store.set_api_key(created.id, "sk-will-vanish")
    assert store.has_api_key(created.id) is True
    store.delete_preset(created.id)
    assert _KEYRING_MEM.get(created.api_key_ref) is None


# ============================ SQL 注入防护 ============================

def test_sql_injection_safe(store):
    """参数化查询:name 字段含 SQL 特殊字符不应破坏查询。"""
    from src.core.provider_preset import ProviderPreset
    p = ProviderPreset(
        id="", name="'; DROP TABLE provider_presets; --",
        provider_type="custom",
    )
    created = store.add_preset(p)
    # 表仍在
    assert store.get_preset(created.id) is not None
    assert store.list_presets()  # 表没被 DROP


# ============================ REST router 集成 ============================

@pytest.fixture
def client(tmp_path, monkeypatch):
    """独立 DB 的 FastAPI TestClient (不污染项目 config/)。"""
    # 指向 tmp 的 config_dir,store 单例惰性构造前先 monkeypatch CONFIG_DIR
    monkeypatch.setattr(
        "src.core.provider_preset.CONFIG_DIR", str(tmp_path / "cfg")
    )
    # reset router 单例 store (lru_cache / 模块全局 _store)
    import src.web.routers.providers as pr
    pr._store = None
    from src.web.app import app
    with TestClient(app) as c:
        yield c
    pr._store = None


def test_rest_create_and_list(client):
    r = client.post("/api/providers", json={
        "name": "openai-prod",
        "provider_type": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": "sk-rest-123",
        "enabled": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "openai-prod"
    assert body["has_key"] is True
    assert "api_key" not in body  # 明文不回传
    assert "api_key_ref" not in body  # keyring 引用也不暴露

    r = client.get("/api/providers")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1
    assert items[0]["name"] == "openai-prod"


def test_rest_get_detail(client):
    r = client.post("/api/providers", json={"name": "p1"})
    pid = r.json()["id"]
    r = client.get(f"/api/providers/{pid}")
    assert r.status_code == 200
    assert r.json()["name"] == "p1"


def test_rest_get_404(client):
    r = client.get("/api/providers/nonexistent")
    assert r.status_code == 404


def test_rest_update(client):
    r = client.post("/api/providers", json={
        "name": "orig", "provider_type": "custom",
        "base_url": "http://a", "model": "m1",
    })
    pid = r.json()["id"]
    r = client.put(f"/api/providers/{pid}", json={
        "provider_type": "ollama", "base_url": "http://b",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["provider_type"] == "ollama"
    assert body["base_url"] == "http://b"
    assert body["model"] == "m1"  # 保留


def test_rest_update_api_key(client):
    """PUT 带 api_key 时更新 keyring。"""
    r = client.post("/api/providers", json={"name": "k", "api_key": "sk-old"})
    pid = r.json()["id"]
    r = client.put(f"/api/providers/{pid}", json={"api_key": "sk-new"})
    assert r.status_code == 200
    # 验证 keyring 中是新 key
    import src.web.routers.providers as pr
    s = pr._get_store()
    assert s.get_api_key(pid) == "sk-new"


def test_rest_delete(client):
    r = client.post("/api/providers", json={"name": "del", "api_key": "sk-x"})
    pid = r.json()["id"]
    r = client.delete(f"/api/providers/{pid}")
    assert r.status_code == 200
    r = client.get(f"/api/providers/{pid}")
    assert r.status_code == 404


def test_rest_activate_and_active(client):
    a = client.post("/api/providers", json={"name": "a"}).json()
    b = client.post("/api/providers", json={"name": "b"}).json()

    r = client.post(f"/api/providers/{a['id']}/activate")
    assert r.status_code == 200
    assert r.json()["is_active"] is True

    r = client.get("/api/providers/active")
    assert r.status_code == 200
    assert r.json()["id"] == a["id"]

    # 切换到 b,a 失活
    client.post(f"/api/providers/{b['id']}/activate")
    r = client.get("/api/providers/active")
    assert r.json()["id"] == b["id"]
    # a 不再 active
    r = client.get(f"/api/providers/{a['id']}")
    assert r.json()["is_active"] is False


def test_rest_activate_404(client):
    r = client.post("/api/providers/nonexistent/activate")
    assert r.status_code == 404


def test_rest_active_empty_when_none(client):
    """无活跃 preset 时返回 is_active=False 的空壳 (不 404)。"""
    r = client.get("/api/providers/active")
    assert r.status_code == 200
    body = r.json()
    assert body["is_active"] is False
    assert body["id"] is None


def test_rest_duplicate_name_409(client):
    client.post("/api/providers", json={"name": "dup"})
    r = client.post("/api/providers", json={"name": "dup"})
    assert r.status_code == 409


def test_rest_no_key_has_key_false(client):
    """不给 api_key (如 Ollama 本地) 时 has_key=False。"""
    r = client.post("/api/providers", json={
        "name": "ollama-local", "provider_type": "ollama",
        "base_url": "http://localhost:11434",
    })
    assert r.status_code == 200
    assert r.json()["has_key"] is False
