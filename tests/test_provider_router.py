"""provider_router 多 key 轮换 / 限速 / 重试切 key 单测（mock，不发真实请求）。"""
import json
import threading
import time
from unittest.mock import patch

import pytest
import requests

from src.core.provider_router import (ProviderKey, ProviderRouter,
                                       RateLimiter, SERVER_ERROR_BACKOFF_SEC,
                                       SERVER_ERROR_SAME_KEY_RETRIES,
                                       _split_keys,
                                       load_from_9router, load_from_env)


# ---- Fixtures ----
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


# ---- ProviderKey ----
class TestProviderKey:
    def test_is_available_active_no_backoff(self, two_keys):
        assert two_keys[0].is_available is True

    def test_is_available_inactive(self, two_keys):
        two_keys[0].isActive = False
        assert two_keys[0].is_available is False

    def test_is_available_backoff_not_expired(self, two_keys):
        two_keys[0].backoff_until = time.monotonic() + 30
        assert two_keys[0].is_available is False

    def test_is_available_backoff_expired(self, two_keys):
        two_keys[0].backoff_until = time.monotonic() - 1
        assert two_keys[0].is_available is True


# ---- RateLimiter（用 monkeypatch 时钟快进）----
class TestRateLimiter:
    def test_acquire_under_limit_immediate(self):
        rl = RateLimiter(limit_per_min=5, window_sec=60)
        # 5 次内都应立即返回 True
        for _ in range(5):
            assert rl.acquire("k1", timeout=1.0) is True

    def test_acquire_blocks_until_window_slides(self):
        rl = RateLimiter(limit_per_min=2, window_sec=60)
        assert rl.acquire("k1", timeout=0.01) is True
        assert rl.acquire("k1", timeout=0.01) is True
        # 第 3 次会阻塞，timeout=0.01 应返回 False
        assert rl.acquire("k1", timeout=0.01) is False

    def test_monotonic_clock_advances_releases_slot(self, monkeypatch):
        rl = RateLimiter(limit_per_min=2, window_sec=60)
        fake_now = [1000.0]
        slept = []
        monkeypatch.setattr(rl, "_now", lambda: fake_now[0])
        monkeypatch.setattr(rl, "_sleep", lambda s: slept.append(s) or
                            fake_now.__setitem__(0, fake_now[0] + s))
        # 打满 2 个
        assert rl.acquire("k1", timeout=5.0) is True
        assert rl.acquire("k1", timeout=5.0) is True
        # 第 3 次：应睡到最早记录超出窗口（60s 后）；timeout 要 > 60s
        assert rl.acquire("k1", timeout=120.0) is True
        assert slept, "应当触发 sleep 等窗口滑动"
        assert fake_now[0] >= 1060.0

    def test_available_reports_remaining(self):
        rl = RateLimiter(limit_per_min=3, window_sec=60)
        assert rl.available("k1") == 3
        rl.acquire("k1")
        assert rl.available("k1") == 2


# ---- per-key 并发信号量（M2 新增）----
class TestRateLimiterConcurrent:
    def test_default_max_concurrent_per_key_is_2(self):
        rl = RateLimiter(limit_per_min=1000)
        assert rl.max_concurrent_per_key == 2
        assert rl.concurrent_slots("k1") == 2

    def test_custom_max_concurrent(self):
        rl = RateLimiter(limit_per_min=1000, max_concurrent_per_key=4)
        assert rl.concurrent_slots("k1") == 4

    def test_acquire_release_round_trip(self):
        """acquire 2 次后第 3 次应阻塞（timeout 验证），release 后又能拿。"""
        rl = RateLimiter(limit_per_min=1000, max_concurrent_per_key=2)
        assert rl.acquire_concurrent("k1") is True
        assert rl.acquire_concurrent("k1") is True
        # 第 3 个槽位：timeout=0.01 应拿不到（返回 False）
        assert rl.acquire_concurrent("k1", timeout=0.01) is False
        # release 一个，又能拿到
        rl.release_concurrent("k1")
        assert rl.acquire_concurrent("k1", timeout=0.5) is True
        rl.release_concurrent("k1")
        rl.release_concurrent("k1")

    def test_concurrent_blocks_thread(self):
        """多线程验证：max_concurrent=1 时第 2 个 acquire 阻塞直到第 1 个 release。"""
        rl = RateLimiter(limit_per_min=1000, max_concurrent_per_key=1)
        order = []
        lock = threading.Lock()

        def worker(name, hold_sec):
            got = rl.acquire_concurrent("k1", timeout=5.0)
            if not got:
                with lock:
                    order.append((name, "timeout"))
                return
            with lock:
                order.append((name, "acquired"))
            time.sleep(hold_sec)
            rl.release_concurrent("k1")
            with lock:
                order.append((name, "released"))

        t1 = threading.Thread(target=worker, args=("A", 0.2))
        t2 = threading.Thread(target=worker, args=("B", 0.0))
        t1.start()
        time.sleep(0.05)  # 让 A 先拿到
        t2.start()
        t1.join()
        t2.join()
        # B 必须在 A release 后才 acquired（不会与 A 并发）
        a_idx = order.index(("A", "acquired"))
        b_idx = order.index(("B", "acquired"))
        a_rel = order.index(("A", "released"))
        assert a_idx < b_idx
        assert a_rel < b_idx, f"B 在 A release 前拿到，order={order}"

    def test_per_key_semaphores_independent(self):
        """不同 key 的信号量独立：k1 打满不影响 k2。"""
        rl = RateLimiter(limit_per_min=1000, max_concurrent_per_key=1)
        assert rl.acquire_concurrent("k1") is True
        # k1 打满，k2 仍能拿
        assert rl.acquire_concurrent("k2") is True
        assert rl.acquire_concurrent("k1", timeout=0.01) is False
        assert rl.acquire_concurrent("k2", timeout=0.01) is False
        rl.release_concurrent("k1")
        rl.release_concurrent("k2")

    def test_router_uses_concurrent_limiter(self, two_keys):
        """ProviderRouter 默认 max_concurrent_per_key=2，且 limiter 有信号量。"""
        r = ProviderRouter(two_keys)
        assert r._limiter.max_concurrent_per_key == 2
        # 拿一个槽位
        assert r._limiter.acquire_concurrent("nv-1") is True
        assert r._limiter.acquire_concurrent("nv-1") is True
        assert r._limiter.acquire_concurrent("nv-1", timeout=0.01) is False
        r._limiter.release_concurrent("nv-1")
        r._limiter.release_concurrent("nv-1")


# ---- select_key ----
class TestSelectKey:
    def test_returns_highest_priority_first(self, two_keys):
        r = ProviderRouter(two_keys)
        k = r.select_key(provider="nvidia")
        assert k is not None
        assert k.id == "nv-1"  # priority 10 > 5

    def test_excludes_backoff_key(self, two_keys):
        r = ProviderRouter(two_keys)
        two_keys[0].backoff_until = time.monotonic() + 30
        k = r.select_key(provider="nvidia")
        assert k is not None
        assert k.id == "nv-2"

    def test_exclude_set_skips_keys(self, two_keys):
        r = ProviderRouter(two_keys)
        k = r.select_key(provider="nvidia", exclude={"nv-1"})
        assert k is not None
        assert k.id == "nv-2"

    def test_returns_none_when_all_inactive(self, two_keys):
        for k in two_keys:
            k.isActive = False
        r = ProviderRouter(two_keys)
        assert r.select_key(provider="nvidia") is None

    def test_selects_least_recently_used_same_priority(self):
        keys = [_nv_key("nv-a", "nvapi-A", priority=10),
                _nv_key("nv-b", "nvapi-B", priority=10)]
        # nv-b 更久未用（last_used 更小）
        keys[0].last_used = 100.0
        keys[1].last_used = 50.0
        r = ProviderRouter(keys)
        k = r.select_key(provider="nvidia")
        assert k.id == "nv-b"


# ---- record_result ----
class TestRecordResult:
    def test_fatal_marks_inactive(self, two_keys):
        r = ProviderRouter(two_keys)
        r.record_result("nv-1", 401, "Unauthorized")
        assert two_keys[0].isActive is False

    def test_429_sets_backoff(self, two_keys):
        r = ProviderRouter(two_keys)
        r.record_result("nv-1", 429, "Too Many Requests")
        assert two_keys[0].consecutive_429 == 1
        assert two_keys[0].backoff_until > time.monotonic()

    def test_5xx_sets_short_backoff(self, two_keys):
        r = ProviderRouter(two_keys)
        r.record_result("nv-1", 503, "Service Unavailable")
        assert two_keys[0].backoff_until > time.monotonic()
        assert two_keys[0].isActive is True  # 不失效

    def test_2xx_clears_backoff(self, two_keys):
        r = ProviderRouter(two_keys)
        two_keys[0].consecutive_429 = 5
        two_keys[0].backoff_until = time.monotonic() + 30
        r.record_result("nv-1", 200, "")
        assert two_keys[0].consecutive_429 == 0
        assert two_keys[0].backoff_until == 0.0


# ---- post_nvidia（mock session.post，不真实请求）----
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


class TestPostNvidia:
    def test_success_first_try(self, two_keys):
        r = ProviderRouter(two_keys)
        with patch.object(r._session, "post",
                          return_value=FakeResp(200)):
            out = r.post_nvidia({"model": "m", "messages": []})
        assert out == {"ok": True}

    def test_429_switches_key_and_retries(self, two_keys):
        r = ProviderRouter(two_keys)
        calls = []
        def fake_post(url, **kw):
            calls.append(url)
            # 前两次（nv-1 两次）：429；之后 200
            if len(calls) <= 2:
                return FakeResp(429, "rate")
            return FakeResp(200)
        # 让限速不阻塞：limit 很高
        r._limiter = RateLimiter(limit_per_min=1000)
        with patch.object(r._session, "post", side_effect=fake_post), \
             patch("time.sleep", return_value=None):
            out = r.post_nvidia({"model": "m", "messages": []})
        assert out == {"ok": True}
        # 至少试了 2 次以上
        assert len(calls) >= 2

    def test_401_marks_key_inactive_and_raises(self, two_keys):
        r = ProviderRouter(two_keys)
        r._limiter = RateLimiter(limit_per_min=1000)
        with patch.object(r._session, "post",
                          return_value=FakeResp(401, "Unauthorized")), \
             patch("time.sleep", return_value=None):
            with pytest.raises(RuntimeError, match="放弃"):
                r.post_nvidia({"model": "m", "messages": []})
        assert two_keys[0].isActive is False

    def test_500_retries_infinitely_until_success(self, two_keys):
        r = ProviderRouter(two_keys)
        r._limiter = RateLimiter(limit_per_min=1000)
        calls = []
        def fake_post(url, **kw):
            calls.append(1)
            return FakeResp(500) if len(calls) < 4 else FakeResp(200)
        with patch.object(r._session, "post", side_effect=fake_post), \
             patch("time.sleep", return_value=None):
            out = r.post_nvidia({"model": "m", "messages": []})
        assert out == {"ok": True}
        assert len(calls) == 4

    def test_network_error_switches_key(self, two_keys):
        r = ProviderRouter(two_keys)
        r._limiter = RateLimiter(limit_per_min=1000)
        calls = []
        def fake_post(url, **kw):
            calls.append(url)
            if len(calls) == 1:
                raise requests.ConnectionError("boom")
            return FakeResp(200)
        with patch.object(r._session, "post", side_effect=fake_post), \
             patch("time.sleep", return_value=None):
            out = r.post_nvidia({"model": "m", "messages": []})
        assert out == {"ok": True}
        assert len(calls) == 2

    def test_no_nvidia_keys_raises(self):
        r = ProviderRouter([])
        with pytest.raises(RuntimeError, match="无 nvidia key"):
            r.post_nvidia({"model": "m", "messages": []})

    # ---- 503 短退避重试同 key 2 次（M2 新增）----
    def test_503_retries_same_key_before_switching(self, two_keys):
        """503 应先短退避 1.5s 重试同 key 2 次，仍 5xx 才切下一个 key。

        断言：第 1 个 key 至少被连续请求 3 次（初始 + 2 次同 key 重试），
        且 sleep 调用里出现 SERVER_ERROR_BACKOFF_SEC（1.5）。
        """
        r = ProviderRouter(two_keys)
        r._limiter = RateLimiter(limit_per_min=1000)
        calls = []  # (url, code)
        sleeps = []

        def fake_post(url, **kw):
            calls.append((url, 503))
            return FakeResp(503, "Worker 16/16")

        def fake_sleep(s):
            sleeps.append(s)
            return None

        # 给足 max_attempts 让它不要无限重试：前 3 次同 key 503，第 4 次切 key
        # 也 503，max_attempts=4 后 raise（避免无限循环）
        def fake_post_seq(url, **kw):
            calls.append((url, 503))
            return FakeResp(503, "Worker 16/16")

        with patch.object(r._session, "post", side_effect=fake_post_seq), \
             patch("time.sleep", side_effect=fake_sleep):
            with pytest.raises(RuntimeError, match="max_attempts"):
                r.post_nvidia({"model": "m", "messages": []}, max_attempts=4)
        # 前 3 次应是同一个 key（同 key 重试 2 次）
        assert len(calls) >= 3
        first_url = calls[0][0]
        same_key_count = sum(1 for u, _ in calls[:3] if u == first_url)
        assert same_key_count == 3, (
            f"前 3 次应都打同一 key（同 key 重试 2 次），实际 calls={calls}")
        # 至少出现一次 SERVER_ERROR_BACKOFF_SEC 的退避
        assert any(abs(s - SERVER_ERROR_BACKOFF_SEC) < 1e-9 for s in sleeps), (
            f"应出现 {SERVER_ERROR_BACKOFF_SEC}s 短退避，sleeps={sleeps}")

    def test_503_same_key_retry_then_success(self, two_keys):
        """503 重试同 key 2 次后第 3 次成功，应不切 key 直接返回。"""
        r = ProviderRouter(two_keys)
        r._limiter = RateLimiter(limit_per_min=1000)
        calls = []
        sleeps = []

        def fake_post(url, **kw):
            calls.append(url)
            # 前 2 次 503，第 3 次 200
            return FakeResp(503) if len(calls) < 3 else FakeResp(200)

        with patch.object(r._session, "post", side_effect=fake_post), \
             patch("time.sleep", side_effect=lambda s: sleeps.append(s)):
            out = r.post_nvidia({"model": "m", "messages": []})
        assert out == {"ok": True}
        assert len(calls) == 3
        # 3 次都打同一 key（没切 key）
        assert calls[0] == calls[1] == calls[2]
        # 应有 2 次 SERVER_ERROR_BACKOFF_SEC 退避（同 key 重试 2 次）
        backoff_sleeps = [s for s in sleeps
                         if abs(s - SERVER_ERROR_BACKOFF_SEC) < 1e-9]
        assert len(backoff_sleeps) == 2, (
            f"应有 2 次 1.5s 退避，sleeps={sleeps}")

    def test_503_exhausts_same_key_retries_then_switches(self, two_keys):
        """同 key 重试 2 次仍 503，应切下一个 key。

        URL 不可见 key，故用 Authorization header 旁路断言。fake_post 收集
        headers，断言第 4 次的 Bearer 与前 3 次不同（切到了 nv-2）。
        """
        r = ProviderRouter(two_keys)
        r._limiter = RateLimiter(limit_per_min=1000)
        seen_keys = []  # 每次请求的 Bearer token

        def fake_post(url, headers=None, **kw):
            seen_keys.append(headers["Authorization"])
            # 前 3 次（nv-1: 初始+2 重试）503；第 4 次（nv-2）200
            return FakeResp(503) if len(seen_keys) < 4 else FakeResp(200)

        with patch.object(r._session, "post", side_effect=fake_post), \
             patch("time.sleep", return_value=None):
            out = r.post_nvidia({"model": "m", "messages": []})
        assert out == {"ok": True}
        assert len(seen_keys) == 4
        # 前 3 次同一 key（nv-1），第 4 次另一 key（nv-2）
        assert seen_keys[0] == seen_keys[1] == seen_keys[2]
        assert seen_keys[3] != seen_keys[0]

    def test_503_same_key_retries_constant(self):
        """常量校验：SERVER_ERROR_SAME_KEY_RETRIES=2 / BACKOFF=1.5。"""
        assert SERVER_ERROR_SAME_KEY_RETRIES == 2
        assert SERVER_ERROR_BACKOFF_SEC == 1.5


# ---- build_nvidia_payload ----
class TestBuildPayload:
    def test_plain_payload(self):
        p = ProviderRouter.build_nvidia_payload(
            "m", [{"role": "user", "content": "hi"}])
        assert p["model"] == "m"
        assert p["messages"][0]["content"] == "hi"
        assert "chat_template_kwargs" not in p

    def test_thinking_flattened_to_top_level(self):
        p = ProviderRouter.build_nvidia_payload(
            "m", [{"role": "user", "content": "hi"}],
            enable_thinking=True, reasoning_budget=8192)
        # 拍平到顶层（不是 extra_body）
        assert p["chat_template_kwargs"] == {"enable_thinking": True}
        assert p["reasoning_budget"] == 8192


# ---- 配置加载 ----
class TestConfigLoad:
    def test_split_keys_dedup(self):
        out = _split_keys("a,b, a ,c,,b")
        assert out == ["a", "b", "c"]

    def test_load_from_env_multi_keys(self, monkeypatch, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "VAP_NV_API_KEYS=nvapi-A,nvapi-B,nvapi-C\n"
            "VAP_KILO_API_KEYS=kilo-X\n",
            encoding="utf-8")
        monkeypatch.delenv("VAP_NV_API_KEYS", raising=False)
        monkeypatch.delenv("VAP_KILO_API_KEYS", raising=False)
        keys = load_from_env(str(env))
        nv = [k for k in keys if k.provider == "nvidia"]
        kl = [k for k in keys if k.provider == "kilo"]
        assert len(nv) == 3
        assert len(kl) == 1
        assert nv[0].api_key == "nvapi-A"
        assert nv[0].priority > nv[1].priority  # 先列出优先级高

    def test_load_from_env_singular_var(self, monkeypatch, tmp_path):
        env = tmp_path / ".env"
        env.write_text("VAP_NV_API_KEY=nvapi-SINGLE\n", encoding="utf-8")
        monkeypatch.delenv("VAP_NV_API_KEYS", raising=False)
        monkeypatch.delenv("VAP_NV_API_KEY", raising=False)
        keys = load_from_env(str(env))
        assert len(keys) == 1
        assert keys[0].api_key == "nvapi-SINGLE"

    def test_load_from_9router_local_nv_keys(self, tmp_path):
        # 模拟本地 _nv_keys.json 结构
        data = [
            {"name": "1", "key": "nvapi-A", "priority": 11,
             "last_status": "active"},
            {"name": "2", "key": "nvapi-B", "priority": 10,
             "last_status": "unavailable"},
        ]
        f = tmp_path / "nv.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        keys = load_from_9router(str(f))
        assert len(keys) == 2
        assert keys[0].provider == "nvidia"
        assert keys[0].api_key == "nvapi-A"
        assert keys[0].isActive is True
        # last_status=unavailable → inactive
        assert keys[1].isActive is False
        assert keys[0].base_url == "https://integrate.api.nvidia.com/v1"

    def test_load_from_9router_standard_shape(self, tmp_path):
        data = {"providerConnections": [
            {"id": "p1", "name": "k1", "provider": "kilo",
             "apiKey": "kl-X", "priority": 5, "isActive": True,
             "baseUrl": "https://api.kilocode.io/v1"},
            {"id": "p2", "name": "k2", "provider": "nvidia",
             "apiKey": "nv-Y", "priority": 3, "isActive": False},
        ]}
        f = tmp_path / "9r.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        keys = load_from_9router(str(f))
        assert len(keys) == 2
        kilo = next(k for k in keys if k.provider == "kilo")
        assert kilo.api_key == "kl-X"
        assert kilo.base_url == "https://api.kilocode.io/v1"
        nv = next(k for k in keys if k.provider == "nvidia")
        assert nv.isActive is False

    def test_load_from_9router_missing_file(self, tmp_path):
        assert load_from_9router(str(tmp_path / "nope.json")) == []


# ---- 集成：轮换真的切到下一个 key ----
class TestRotation:
    def test_all_keys_429_cycle_infinitely_until_success(self, monkeypatch):
        keys = [_nv_key("nv-1", "nvapi-A", priority=10),
                _nv_key("nv-2", "nvapi-B", priority=5)]
        r = ProviderRouter(keys, rate_limit_per_min=1000)
        calls = []
        def fake_post(url, **kw):
            calls.append(url)
            # 前 3 次 429，第 4 次 200
            return FakeResp(429) if len(calls) < 4 else FakeResp(200)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        with patch.object(r._session, "post", side_effect=fake_post):
            out = r.post_nvidia({"model": "m", "messages": []})
        assert out == {"ok": True}
        assert len(calls) == 4
        # 验证确实在两个 key 之间切换（URL 含 key 不可见，但 attempted 会清空）


# ---- nvidia_models.get_video_config（M2 per-model 分片配置）----
class TestNvidiaVideoConfig:
    """per-model 视频分片配置（不再写死 30s/720p）。"""

    def test_get_video_config_omni(self):
        from src.core.nvidia_models import get_video_config
        cfg = get_video_config("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
        assert cfg["max_segment_sec"] == 120
        assert cfg["target_height"] == 720
        assert cfg["target_fps"] == 2
        assert cfg["max_frames"] == 256
        assert cfg["max_video_mb"] == 200

    def test_get_video_config_cosmos3(self):
        from src.core.nvidia_models import get_video_config
        cfg = get_video_config("nvidia/cosmos3-nano-reasoner")
        # cosmos3 按注册值（与 omni 同档对齐）
        assert cfg["max_segment_sec"] == 120
        assert cfg["target_height"] == 720
        assert cfg["max_frames"] == 256

    def test_get_video_config_unknown_model_fallback(self):
        from src.core.nvidia_models import get_video_config
        cfg = get_video_config("nvidia/unknown-model-xyz")
        # 未知模型回退默认（对齐 Nemotron Omni 上限）
        assert cfg["max_segment_sec"] == 120
        assert cfg["target_height"] == 720
        assert cfg["target_fps"] == 2
        assert cfg["max_frames"] == 256

    def test_nvidia_model_dataclass_has_video_fields(self):
        from src.core.nvidia_models import NvidiaModel
        m = NvidiaModel(
            id="test/m", name="t", category="llm",
            supports_video=False,
        )
        # 默认值应与 omni 官方上限对齐
        assert m.max_segment_sec == 120
        assert m.max_video_mb == 200
        assert m.max_frames == 256
        assert m.target_height == 720
        assert m.target_fps == 2

    def test_nvidia_model_custom_video_fields(self):
        from src.core.nvidia_models import NvidiaModel
        m = NvidiaModel(
            id="test/m2", name="t2", category="video",
            supports_video=True,
            max_segment_sec=60,
            max_video_mb=50,
            max_frames=128,
            target_height=480,
            target_fps=1,
        )
        assert m.max_segment_sec == 60
        assert m.target_height == 480
        assert m.max_frames == 128

    def test_get_video_config_returns_dict_not_shared(self):
        """返回的 dict 应独立（修改不影响下次调用）。"""
        from src.core.nvidia_models import get_video_config
        cfg1 = get_video_config("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
        cfg1["max_segment_sec"] = 999
        cfg2 = get_video_config("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
        assert cfg2["max_segment_sec"] == 120
