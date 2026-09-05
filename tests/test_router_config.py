"""ProviderRouter 调优参数可配置性单测（I5.8-router-1/2）。

验证：
1. `ProviderRouter.__init__` 接受 `backoff_sec` / `same_key_retries` 参数，
   存为 `self._backoff_sec` / `self._same_key_retries`。
2. `post_nvidia` 的 503 短退避逻辑用实例参数而非模块级常量
   （构造 backoff_sec=0.1/same_key_retries=1，monkeypatch time.sleep，
   断言重试间隔=0.1s、同 key 重试次数=1）。
3. `load_router_config_from_env` 从 .env 读三个变量，默认值 1.5/2/2，
   非法值回退默认。
"""
import json
from unittest.mock import patch

import pytest

from src.core.provider_router import (
    DEFAULT_MAX_CONCURRENT_PER_KEY,
    ProviderKey,
    ProviderRouter,
    RateLimiter,
    SERVER_ERROR_BACKOFF_SEC,
    SERVER_ERROR_SAME_KEY_RETRIES,
    load_router_config_from_env,
)


def _nv_key(name: str, api_key: str, priority: int = 0,
            active: bool = True) -> ProviderKey:
    return ProviderKey(
        id=name, name=name, provider="nvidia", api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        priority=priority, isActive=active)


@pytest.fixture
def two_keys():
    return [_nv_key("nv-1", "nvapi-A", priority=10),
            _nv_key("nv-2", "nvapi-B", priority=5)]


class FakeResp:
    def __init__(self, code=200, body="{\"ok\":true}"):
        self.status_code = code
        self._body = body

    @property
    def text(self):
        return self._body

    def json(self):
        return json.loads(self._body)

    def close(self):
        pass


# ---- I5.8-router-1：__init__ 参数化 ----
class TestBackoffParamConfigurable:
    def test_default_uses_module_constants(self, two_keys):
        """不传参时实例参数应等于模块级默认常量（向后兼容）。"""
        r = ProviderRouter(two_keys)
        assert r._backoff_sec == SERVER_ERROR_BACKOFF_SEC
        assert r._same_key_retries == SERVER_ERROR_SAME_KEY_RETRIES

    def test_custom_backoff_stored(self, two_keys):
        r = ProviderRouter(two_keys, backoff_sec=0.1, same_key_retries=1)
        assert r._backoff_sec == pytest.approx(0.1)
        assert r._same_key_retries == 1

    def test_negative_backoff_clamped_to_zero(self, two_keys):
        """负值应被 max(0, ...) 夹紧到 0，避免 sleep 负数。"""
        r = ProviderRouter(two_keys, backoff_sec=-1.0, same_key_retries=-3)
        assert r._backoff_sec == 0.0
        assert r._same_key_retries == 0

    def test_503_uses_custom_backoff_and_retries(self, two_keys):
        """构造 backoff_sec=0.1/same_key_retries=1，503 时应：
        - 同 key 重试 1 次（共 2 次请求同一 key：初始 + 1 次重试）
        - sleep 调用出现 0.1s（不是模块级 1.5s）
        - 第 1 次重试仍 503 后切下一个 key
        """
        r = ProviderRouter(two_keys, backoff_sec=0.1, same_key_retries=1)
        r._limiter = RateLimiter(limit_per_min=1000)
        seen_keys = []  # 每次请求的 Bearer token
        sleeps = []

        def fake_post(url, headers=None, **kw):
            seen_keys.append(headers["Authorization"])
            # 前 2 次（nv-1: 初始+1 重试）503；第 3 次（nv-2）200
            return FakeResp(503) if len(seen_keys) < 3 else FakeResp(200)

        with patch.object(r._session, "post", side_effect=fake_post), \
             patch("time.sleep", side_effect=lambda s: sleeps.append(s)):
            out = r.post_nvidia({"model": "m", "messages": []})
        assert out == {"ok": True}
        # 前 2 次同 key（初始 + 1 次同 key 重试），第 3 次切到另一 key
        assert len(seen_keys) == 3
        assert seen_keys[0] == seen_keys[1]
        assert seen_keys[2] != seen_keys[0]
        # 退避用 0.1s（自定义），不应出现模块级 1.5s
        assert any(abs(s - 0.1) < 1e-9 for s in sleeps), (
            f"应出现 0.1s 自定义退避，sleeps={sleeps}")
        assert not any(abs(s - SERVER_ERROR_BACKOFF_SEC) < 1e-9
                       for s in sleeps), (
            f"不应出现模块级 1.5s 退避，sleeps={sleeps}")
        # 同 key 重试只应有 1 次 sleep（same_key_retries=1）
        backoff_sleeps = [s for s in sleeps if abs(s - 0.1) < 1e-9]
        assert len(backoff_sleeps) == 1, (
            f"同 key 重试 1 次应只 sleep 1 次，sleeps={sleeps}")

    def test_503_zero_retries_switches_immediately(self, two_keys):
        """same_key_retries=0：503 不重试同 key，立即切下一个 key。"""
        r = ProviderRouter(two_keys, backoff_sec=0.1, same_key_retries=0)
        r._limiter = RateLimiter(limit_per_min=1000)
        seen_keys = []
        sleeps = []

        def fake_post(url, headers=None, **kw):
            seen_keys.append(headers["Authorization"])
            # 前 2 次（nv-1, nv-2）503；第 3 次（nv-1 重新轮换）200
            return FakeResp(503) if len(seen_keys) < 3 else FakeResp(200)

        with patch.object(r._session, "post", side_effect=fake_post), \
             patch("time.sleep", side_effect=lambda s: sleeps.append(s)):
            out = r.post_nvidia({"model": "m", "messages": []})
        assert out == {"ok": True}
        # 前 2 次应是不同 key（不重试同 key）
        assert seen_keys[0] != seen_keys[1]
        # 不应出现自定义退避 0.1s（same_key_retries=0 不 sleep 退避）
        assert not any(abs(s - 0.1) < 1e-9 for s in sleeps), (
            f"same_key_retries=0 不应有退避 sleep，sleeps={sleeps}")


# ---- I5.8-router-2：load_router_config_from_env ----
class TestLoadRouterConfig:
    def test_defaults_when_no_env(self, monkeypatch, tmp_path):
        """无 .env 文件且无进程环境变量时返回模块级默认值。"""
        monkeypatch.delenv("VAP_NV_BACKOFF_SEC", raising=False)
        monkeypatch.delenv("VAP_NV_SAME_KEY_RETRIES", raising=False)
        monkeypatch.delenv("VAP_NV_MAX_CONCURRENT_PER_KEY", raising=False)
        cfg = load_router_config_from_env(str(tmp_path / "nope.env"))
        assert cfg == {
            "backoff_sec": SERVER_ERROR_BACKOFF_SEC,
            "same_key_retries": SERVER_ERROR_SAME_KEY_RETRIES,
            "max_concurrent_per_key": DEFAULT_MAX_CONCURRENT_PER_KEY,
        }

    def test_reads_three_vars_from_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VAP_NV_BACKOFF_SEC", raising=False)
        monkeypatch.delenv("VAP_NV_SAME_KEY_RETRIES", raising=False)
        monkeypatch.delenv("VAP_NV_MAX_CONCURRENT_PER_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text(
            "VAP_NV_BACKOFF_SEC=0.7\n"
            "VAP_NV_SAME_KEY_RETRIES=5\n"
            "VAP_NV_MAX_CONCURRENT_PER_KEY=3\n",
            encoding="utf-8")
        cfg = load_router_config_from_env(str(env))
        assert cfg == {
            "backoff_sec": 0.7,
            "same_key_retries": 5,
            "max_concurrent_per_key": 3,
        }

    def test_process_env_overrides_file(self, monkeypatch, tmp_path):
        """进程环境变量优先于 .env 文件同名变量。"""
        env = tmp_path / ".env"
        env.write_text(
            "VAP_NV_BACKOFF_SEC=0.7\n"
            "VAP_NV_SAME_KEY_RETRIES=5\n"
            "VAP_NV_MAX_CONCURRENT_PER_KEY=3\n",
            encoding="utf-8")
        monkeypatch.setenv("VAP_NV_BACKOFF_SEC", "2.5")
        cfg = load_router_config_from_env(str(env))
        assert cfg["backoff_sec"] == 2.5
        # 文件值仍生效（未覆盖的变量）
        assert cfg["same_key_retries"] == 5
        assert cfg["max_concurrent_per_key"] == 3

    def test_quotes_stripped(self, monkeypatch, tmp_path):
        """带引号的值应被剥离引号。"""
        monkeypatch.delenv("VAP_NV_BACKOFF_SEC", raising=False)
        env = tmp_path / ".env"
        env.write_text('VAP_NV_BACKOFF_SEC="0.9"\n',
                       encoding="utf-8")
        cfg = load_router_config_from_env(str(env))
        assert cfg["backoff_sec"] == 0.9

    def test_invalid_backoff_falls_back(self, monkeypatch, tmp_path):
        """非法 float 值应回退默认并日志告警。"""
        monkeypatch.delenv("VAP_NV_BACKOFF_SEC", raising=False)
        env = tmp_path / ".env"
        env.write_text("VAP_NV_BACKOFF_SEC=not-a-number\n",
                       encoding="utf-8")
        cfg = load_router_config_from_env(str(env))
        assert cfg["backoff_sec"] == SERVER_ERROR_BACKOFF_SEC

    def test_invalid_same_key_retries_falls_back(self, monkeypatch, tmp_path):
        """非法 int 值应回退默认。"""
        monkeypatch.delenv("VAP_NV_SAME_KEY_RETRIES", raising=False)
        env = tmp_path / ".env"
        env.write_text("VAP_NV_SAME_KEY_RETRIES=abc\n",
                       encoding="utf-8")
        cfg = load_router_config_from_env(str(env))
        assert cfg["same_key_retries"] == SERVER_ERROR_SAME_KEY_RETRIES

    def test_negative_backoff_falls_back(self, monkeypatch, tmp_path):
        """负值应回退默认（不允许负退避）。"""
        monkeypatch.delenv("VAP_NV_BACKOFF_SEC", raising=False)
        env = tmp_path / ".env"
        env.write_text("VAP_NV_BACKOFF_SEC=-2.0\n",
                       encoding="utf-8")
        cfg = load_router_config_from_env(str(env))
        assert cfg["backoff_sec"] == SERVER_ERROR_BACKOFF_SEC

    def test_negative_retries_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VAP_NV_SAME_KEY_RETRIES", raising=False)
        env = tmp_path / ".env"
        env.write_text("VAP_NV_SAME_KEY_RETRIES=-1\n",
                       encoding="utf-8")
        cfg = load_router_config_from_env(str(env))
        assert cfg["same_key_retries"] == SERVER_ERROR_SAME_KEY_RETRIES

    def test_comment_and_blank_lines_skipped(self, monkeypatch, tmp_path):
        """注释行和空行应被跳过，不报错。"""
        monkeypatch.delenv("VAP_NV_BACKOFF_SEC", raising=False)
        env = tmp_path / ".env"
        env.write_text(
            "# 这是注释\n"
            "\n"
            "VAP_NV_BACKOFF_SEC=1.0\n"
            "   # 缩进注释\n",
            encoding="utf-8")
        cfg = load_router_config_from_env(str(env))
        assert cfg["backoff_sec"] == 1.0

    def test_no_path_reads_process_env(self, monkeypatch):
        """env_path=None 时只读进程环境变量。"""
        monkeypatch.setenv("VAP_NV_BACKOFF_SEC", "3.3")
        monkeypatch.setenv("VAP_NV_SAME_KEY_RETRIES", "7")
        monkeypatch.setenv("VAP_NV_MAX_CONCURRENT_PER_KEY", "4")
        cfg = load_router_config_from_env(None)
        assert cfg == {
            "backoff_sec": 3.3,
            "same_key_retries": 7,
            "max_concurrent_per_key": 4,
        }

    def test_config_dict_keys_stable(self, monkeypatch, tmp_path):
        """返回的 dict 必须包含三个固定键（调用方依赖这些键名）。"""
        monkeypatch.delenv("VAP_NV_BACKOFF_SEC", raising=False)
        monkeypatch.delenv("VAP_NV_SAME_KEY_RETRIES", raising=False)
        monkeypatch.delenv("VAP_NV_MAX_CONCURRENT_PER_KEY", raising=False)
        cfg = load_router_config_from_env(str(tmp_path / "nope.env"))
        assert set(cfg.keys()) == {
            "backoff_sec", "same_key_retries", "max_concurrent_per_key"}
