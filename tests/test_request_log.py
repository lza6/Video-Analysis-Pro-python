"""RequestLog + RequestLogStore + parse_usage 单测（tmp_path 隔离）。

AAA 模式：Arrange 准备 fixture / Act 调用方法 / Assert 验证字段、聚合、解析。
"""
from __future__ import annotations

import json

import pytest

from src.core.request_log import (
    PREVIEW_LIMIT,
    RequestLog,
    RequestLogStore,
    _truncate,
    get_store,
    parse_usage,
    reset_store,
)


# ----------------------------------------------------------------------
# fixture：每个测试独立 tmp_path，互不污染
# ----------------------------------------------------------------------
def _make_store(tmp_path) -> RequestLogStore:
    return RequestLogStore(str(tmp_path / "cfg"))


def _entry(
    *,
    provider: str = "nvidia",
    model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    key_id: str = "nvidia-1",
    status_code: int = 200,
    latency_ms: float = 123.4,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    total_tokens: int = 150,
    error: str = "",
    request_preview: str = '{"model":"m"}',
    response_preview: str = '{"ok":true}',
) -> RequestLog:
    return RequestLog(
        provider=provider,
        model=model,
        key_id=key_id,
        status_code=status_code,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        error=error,
        request_preview=request_preview,
        response_preview=response_preview,
    )


# ----------------------------------------------------------------------
# _truncate
# ----------------------------------------------------------------------
class TestTruncate:
    def test_none_returns_empty(self):
        assert _truncate(None) == ""

    def test_empty_returns_empty(self):
        assert _truncate("") == ""

    def test_short_passes_through(self):
        assert _truncate("hello", limit=10) == "hello"

    def test_long_truncated_to_limit(self):
        s = "x" * 1000
        out = _truncate(s, limit=50)
        assert len(out) == 50
        assert out == "x" * 50

    def test_default_limit_is_preview_limit(self):
        s = "y" * (PREVIEW_LIMIT + 100)
        out = _truncate(s)
        assert len(out) == PREVIEW_LIMIT


# ----------------------------------------------------------------------
# log_request + list_logs
# ----------------------------------------------------------------------
class TestLogAndList:
    def test_log_request_returns_log_id(self, tmp_path):
        store = _make_store(tmp_path)
        eid = store.log_request(_entry())
        assert eid and len(eid) == 32  # uuid4 hex

    def test_log_request_persists_all_fields(self, tmp_path):
        store = _make_store(tmp_path)
        eid = store.log_request(_entry(latency_ms=250.5,
                                       prompt_tokens=10,
                                       completion_tokens=20,
                                       total_tokens=30))
        items = store.list_logs(limit=10)
        assert len(items) == 1
        row = items[0]
        assert row["log_id"] == eid
        assert row["provider"] == "nvidia"
        assert row["model"] == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        assert row["key_id"] == "nvidia-1"
        assert row["status_code"] == 200
        assert row["latency_ms"] == 250.5
        assert row["prompt_tokens"] == 10
        assert row["completion_tokens"] == 20
        assert row["total_tokens"] == 30
        assert row["error"] == ""
        assert row["request_preview"] == '{"model":"m"}'
        assert row["response_preview"] == '{"ok":true}'
        assert row["timestamp"]  # 自动填了 ISO8601

    def test_log_request_truncates_long_preview(self, tmp_path):
        store = _make_store(tmp_path)
        long_body = "z" * 2000
        store.log_request(_entry(request_preview=long_body,
                                  response_preview=long_body))
        row = store.list_logs(limit=1)[0]
        assert len(row["request_preview"]) == PREVIEW_LIMIT
        assert len(row["response_preview"]) == PREVIEW_LIMIT

    def test_list_logs_ordered_desc_by_timestamp(self, tmp_path):
        store = _make_store(tmp_path)
        # 三条，时间戳字符串字典序与时间顺序一致（ISO8601 秒精度）
        e1 = _entry()
        e1.timestamp = "2026-09-05T10:00:00"
        e2 = _entry()
        e2.timestamp = "2026-09-05T11:00:00"
        e3 = _entry()
        e3.timestamp = "2026-09-05T12:00:00"
        store.log_request(e1)
        store.log_request(e2)
        store.log_request(e3)
        items = store.list_logs(limit=100)
        assert len(items) == 3
        # 倒序：最新的在前
        assert items[0]["timestamp"] == "2026-09-05T12:00:00"
        assert items[2]["timestamp"] == "2026-09-05T10:00:00"

    def test_list_logs_provider_filter(self, tmp_path):
        store = _make_store(tmp_path)
        store.log_request(_entry(provider="nvidia"))
        store.log_request(_entry(provider="kilo"))
        store.log_request(_entry(provider="nvidia"))
        nv = store.list_logs(limit=100, provider="nvidia")
        assert len(nv) == 2
        assert all(r["provider"] == "nvidia" for r in nv)
        kl = store.list_logs(limit=100, provider="kilo")
        assert len(kl) == 1

    def test_list_logs_since_filter(self, tmp_path):
        store = _make_store(tmp_path)
        e1 = _entry(); e1.timestamp = "2026-09-05T10:00:00"
        e2 = _entry(); e2.timestamp = "2026-09-05T11:00:00"
        e3 = _entry(); e3.timestamp = "2026-09-05T12:00:00"
        store.log_request(e1)
        store.log_request(e2)
        store.log_request(e3)
        out = store.list_logs(limit=100, since="2026-09-05T11:00:00")
        assert len(out) == 2
        assert all(r["timestamp"] >= "2026-09-05T11:00:00" for r in out)

    def test_list_logs_limit_caps_results(self, tmp_path):
        store = _make_store(tmp_path)
        for i in range(5):
            store.log_request(_entry())
        out = store.list_logs(limit=3)
        assert len(out) == 3


# ----------------------------------------------------------------------
# token_stats
# ----------------------------------------------------------------------
class TestTokenStats:
    def test_empty_returns_empty_dict(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.token_stats() == {}

    def test_aggregates_per_provider(self, tmp_path):
        store = _make_store(tmp_path)
        # nvidia: 2 个请求，prompt=100+200=300, completion=50+50=100
        store.log_request(_entry(provider="nvidia", prompt_tokens=100,
                                  completion_tokens=50, total_tokens=150,
                                  latency_ms=100.0, status_code=200))
        store.log_request(_entry(provider="nvidia", prompt_tokens=200,
                                  completion_tokens=50, total_tokens=250,
                                  latency_ms=300.0, status_code=200))
        # kilo: 1 个请求
        store.log_request(_entry(provider="kilo", prompt_tokens=10,
                                  completion_tokens=5, total_tokens=15,
                                  latency_ms=50.0, status_code=200))
        stats = store.token_stats()
        assert "nvidia" in stats
        assert "kilo" in stats
        nv = stats["nvidia"]
        assert nv["requests"] == 2
        assert nv["prompt_tokens"] == 300
        assert nv["completion_tokens"] == 100
        assert nv["total_tokens"] == 400
        # 平均延迟 = (100+300)/2 = 200
        assert abs(nv["avg_latency_ms"] - 200.0) < 1e-9
        assert nv["error_count"] == 0
        kl = stats["kilo"]
        assert kl["requests"] == 1
        assert kl["total_tokens"] == 15

    def test_error_count_counts_non_2xx(self, tmp_path):
        store = _make_store(tmp_path)
        store.log_request(_entry(status_code=200))
        store.log_request(_entry(status_code=503))
        store.log_request(_entry(status_code=429))
        store.log_request(_entry(status_code=None))  # 网络异常
        stats = store.token_stats(provider="nvidia")
        nv = stats["nvidia"]
        assert nv["requests"] == 4
        assert nv["error_count"] == 3  # 503 + 429 + None

    def test_provider_filter_returns_only_that_provider(self, tmp_path):
        store = _make_store(tmp_path)
        store.log_request(_entry(provider="nvidia", total_tokens=100))
        store.log_request(_entry(provider="kilo", total_tokens=50))
        stats = store.token_stats(provider="nvidia")
        assert list(stats.keys()) == ["nvidia"]
        assert stats["nvidia"]["total_tokens"] == 100

    def test_since_filter(self, tmp_path):
        store = _make_store(tmp_path)
        e1 = _entry(total_tokens=100)
        e1.timestamp = "2026-09-05T10:00:00"
        e2 = _entry(total_tokens=200)
        e2.timestamp = "2026-09-05T12:00:00"
        store.log_request(e1)
        store.log_request(e2)
        stats = store.token_stats(since="2026-09-05T11:00:00")
        assert stats["nvidia"]["requests"] == 1
        assert stats["nvidia"]["total_tokens"] == 200


# ----------------------------------------------------------------------
# clear_all
# ----------------------------------------------------------------------
class TestClearAll:
    def test_clear_all_removes_all_rows(self, tmp_path):
        store = _make_store(tmp_path)
        for _ in range(3):
            store.log_request(_entry())
        assert len(store.list_logs(limit=100)) == 3
        count = store.clear_all()
        assert count == 3
        assert store.list_logs(limit=100) == []

    def test_clear_all_on_empty_returns_zero(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.clear_all() == 0


# ----------------------------------------------------------------------
# parse_usage
# ----------------------------------------------------------------------
class TestParseUsage:
    def test_standard_openai_usage(self):
        resp = {"usage": {"prompt_tokens": 12,
                          "completion_tokens": 34,
                          "total_tokens": 46}}
        pt, ct, tt = parse_usage(resp)
        assert (pt, ct, tt) == (12, 34, 46)

    def test_missing_usage_returns_zeros(self):
        assert parse_usage({"choices": []}) == (0, 0, 0)

    def test_non_dict_returns_zeros(self):
        assert parse_usage(None) == (0, 0, 0)
        assert parse_usage("not a dict") == (0, 0, 0)
        assert parse_usage([]) == (0, 0, 0)

    def test_usage_not_dict_returns_zeros(self):
        assert parse_usage({"usage": "nope"}) == (0, 0, 0)

    def test_total_tokens_computed_when_missing(self):
        # 部分 provider 只回 prompt/completion 不回 total
        resp = {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        pt, ct, tt = parse_usage(resp)
        assert (pt, ct, tt) == (7, 3, 10)

    def test_input_output_tokens_alias(self):
        # 部分 provider 用 input_tokens / output_tokens
        resp = {"usage": {"input_tokens": 5, "output_tokens": 8,
                          "total_tokens": 13}}
        pt, ct, tt = parse_usage(resp)
        assert (pt, ct, tt) == (5, 8, 13)

    def test_zero_values(self):
        resp = {"usage": {"prompt_tokens": 0,
                          "completion_tokens": 0,
                          "total_tokens": 0}}
        assert parse_usage(resp) == (0, 0, 0)


# ----------------------------------------------------------------------
# get_store / reset_store 单例
# ----------------------------------------------------------------------
class TestSingleton:
    def test_get_store_returns_same_instance(self):
        reset_store()
        s1 = get_store()
        s2 = get_store()
        assert s1 is s2

    def test_reset_store_creates_new_instance(self):
        reset_store()
        s1 = get_store()
        reset_store()
        s2 = get_store()
        assert s1 is not s2


# ----------------------------------------------------------------------
# ProviderRouter 接入：post_nvidia 成功路径落日志
# ----------------------------------------------------------------------
class TestRouterIntegration:
    """验证 post_nvidia 成功/失败路径都落 RequestLog。

    用 monkeypatch 替换 get_store 指向 tmp_path 隔离的 store，
    避免污染 config/request_logs.db。
    """
    def test_success_logs_request_with_usage(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        # 让 provider_router 内部 get_store 返回我们的隔离 store
        import src.core.request_log as rlmod
        monkeypatch.setattr(rlmod, "_store", store, raising=False)
        monkeypatch.setattr(rlmod, "get_store", lambda: store)

        from src.core.provider_router import ProviderKey, ProviderRouter
        k = ProviderKey(
            id="nv-1", name="nv-1", provider="nvidia",
            api_key="nvapi-X",
            base_url="https://integrate.api.nvidia.com/v1",
            priority=10, isActive=True)
        r = ProviderRouter([k], rate_limit_per_min=1000)

        # mock session.post 返回带 usage 的 200 响应
        class FakeResp:
            status_code = 200
            _body = json.dumps({
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 42,
                          "completion_tokens": 8,
                          "total_tokens": 50},
            })
            @property
            def text(self):
                return self._body
            def json(self):
                return json.loads(self._body)
            def close(self):
                pass

        from unittest.mock import patch
        with patch.object(r._session, "post", return_value=FakeResp()):
            out = r.post_nvidia({"model": "m", "messages": []})
        assert out["choices"][0]["message"]["content"] == "hi"

        # 验证落了一条日志
        items = store.list_logs(limit=10)
        assert len(items) == 1
        row = items[0]
        assert row["provider"] == "nvidia"
        assert row["status_code"] == 200
        assert row["prompt_tokens"] == 42
        assert row["completion_tokens"] == 8
        assert row["total_tokens"] == 50
        assert row["key_id"] == "nv-1"
        assert row["model"] == "m"
        assert row["latency_ms"] > 0
        assert row["error"] == ""
        # request_preview 含 model
        assert "m" in row["request_preview"]

    def test_fatal_401_logs_error_and_raises(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        import src.core.request_log as rlmod
        monkeypatch.setattr(rlmod, "_store", store, raising=False)
        monkeypatch.setattr(rlmod, "get_store", lambda: store)

        from src.core.provider_router import ProviderKey, ProviderRouter
        k = ProviderKey(
            id="nv-1", name="nv-1", provider="nvidia",
            api_key="nvapi-X",
            base_url="https://integrate.api.nvidia.com/v1",
            priority=10, isActive=True)
        r = ProviderRouter([k], rate_limit_per_min=1000)

        class FakeResp:
            status_code = 401
            _body = "Unauthorized"
            @property
            def text(self):
                return self._body
            def json(self):
                return json.loads(self._body) if self._body.startswith("{") else {}
            def close(self):
                pass

        from unittest.mock import patch
        import pytest as _pytest
        with patch.object(r._session, "post", return_value=FakeResp()):
            with _pytest.raises(RuntimeError, match="放弃"):
                r.post_nvidia({"model": "m", "messages": []})

        items = store.list_logs(limit=10)
        assert len(items) == 1
        row = items[0]
        assert row["status_code"] == 401
        assert "HTTP 401" in row["error"]
        assert row["prompt_tokens"] == 0  # 失败无 usage

    def test_log_failure_does_not_raise(self, tmp_path, monkeypatch):
        """log_request 内部异常不应影响 post_nvidia 主流程。"""
        store = _make_store(tmp_path)
        import src.core.request_log as rlmod
        # 让 get_store 抛异常
        def boom():
            raise RuntimeError("db locked")
        monkeypatch.setattr(rlmod, "get_store", boom)

        from src.core.provider_router import ProviderKey, ProviderRouter
        k = ProviderKey(
            id="nv-1", name="nv-1", provider="nvidia",
            api_key="nvapi-X",
            base_url="https://integrate.api.nvidia.com/v1",
            priority=10, isActive=True)
        r = ProviderRouter([k], rate_limit_per_min=1000)

        class FakeResp:
            status_code = 200
            _body = json.dumps({"choices": [], "usage": {}})
            @property
            def text(self):
                return self._body
            def json(self):
                return json.loads(self._body)
            def close(self):
                pass

        from unittest.mock import patch
        with patch.object(r._session, "post", return_value=FakeResp()):
            # 不应抛——日志失败只警告
            out = r.post_nvidia({"model": "m", "messages": []})
        assert out == {"choices": [], "usage": {}}


# ----------------------------------------------------------------------
# Web API 集成（FastAPI TestClient）
# ----------------------------------------------------------------------
class TestWebAPI:
    def test_get_requests_and_stats(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        store.log_request(_entry(provider="nvidia", prompt_tokens=100,
                                  completion_tokens=50, total_tokens=150))
        store.log_request(_entry(provider="kilo", prompt_tokens=10,
                                  completion_tokens=5, total_tokens=15))
        import src.core.request_log as rlmod
        monkeypatch.setattr(rlmod, "get_store", lambda: store)
        # 同时 patch requests router 已 import 的引用
        import src.web.routers.requests as wr
        monkeypatch.setattr(wr, "get_store", lambda: store)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        app.include_router(wr.router)
        c = TestClient(app)

        # GET /api/requests
        r = c.get("/api/requests?limit=10")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert any(x["provider"] == "nvidia" for x in body["requests"])

        # provider 过滤
        r = c.get("/api/requests?provider=nvidia")
        assert r.status_code == 200
        assert r.json()["count"] == 1

        # GET /api/requests/stats
        r = c.get("/api/requests/stats")
        assert r.status_code == 200
        stats = r.json()["providers"]
        assert stats["nvidia"]["total_tokens"] == 150
        assert stats["kilo"]["total_tokens"] == 15

    def test_delete_clears(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        store.log_request(_entry())
        store.log_request(_entry())
        import src.core.request_log as rlmod
        monkeypatch.setattr(rlmod, "get_store", lambda: store)
        import src.web.routers.requests as wr
        monkeypatch.setattr(wr, "get_store", lambda: store)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        app.include_router(wr.router)
        c = TestClient(app)

        r = c.delete("/api/requests")
        assert r.status_code == 200
        assert r.json()["deleted"] == 2
        assert store.list_logs(limit=100) == []
