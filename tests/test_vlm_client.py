"""VLM 客户端测试（9.1 视觉理解接入）。

全部 Mock，不真实连 Ollama / 云端 VLM，不真实付费 API。

策略：
  - 自建 fake httpx module 注入 sys.modules，拦截 AsyncClient.post，
    断言 payload 拼装正确（base64 / data URL / model / headers）。
  - httpx 缺失场景：用 monkeypatch 从 sys.modules 暂时移除 httpx，
    并设为 None 触发 _import_httpx 的 ImportError 分支。
  - 不依赖 pytest-asyncio：用 asyncio.new_event_loop().run_until_complete
    驱动 async describe（与 test_im_gateway / test_remote_tunnel 同模式）。
"""
from __future__ import annotations

import asyncio
import base64
import sys
import types
from typing import Any, Dict, Optional, Tuple

import pytest

from src.core.vlm_client import (
    CloudVLMClient,
    MockVLMClient,
    OllamaVLMClient,
    build_vlm_client,
)


def run(coro):
    """同步驱动 async 调用（不依赖 pytest-asyncio）。"""
    return asyncio.new_event_loop().run_until_complete(coro)


# ----------------------------------------------------------------------
# Fake httpx module（注入 sys.modules 拦截网络）
# ----------------------------------------------------------------------
class _FakeResponse:
    """模拟 httpx.Response：json() + raise_for_status()。"""

    def __init__(self, json_data: Dict[str, Any], status_code: int = 200) -> None:
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Dict[str, Any]:
        return self._json


class _FakeAsyncClient:
    """模拟 httpx.AsyncClient 上下文管理器。

    每次实例化记录自身（让测试断言只创建一个 client），
    post() 返回 _FakeResponse，并把 (url, json, headers) 记录到
    模块级 _LAST_CALL 供断言。
    """

    def __init__(self, timeout: Any = None) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(self, url: str, json: Any = None, headers: Any = None) -> _FakeResponse:
        _FAKE_HTTPX.last_call = (url, json, headers)
        return _FAKE_HTTPX.response


class _FakeHttpxModule(types.ModuleType):
    """假 httpx 模块：暴露 AsyncClient，记录最后一次 post 调用。"""

    def __init__(self) -> None:
        super().__init__("httpx")
        self.AsyncClient = _FakeAsyncClient
        self.last_call: Optional[Tuple[str, Any, Any]] = None
        self.response: _FakeResponse = _FakeResponse({"message": {"content": ""}})

    def reset(self, response_data: Dict[str, Any]) -> None:
        self.response = _FakeResponse(response_data)
        self.last_call = None


_FAKE_HTTPX = _FakeHttpxModule()


@pytest.fixture
def fake_httpx(monkeypatch: pytest.MonkeyPatch) -> _FakeHttpxModule:
    """注入 fake httpx 到 sys.modules，让 vlm_client._import_httpx 拿到它。"""
    monkeypatch.setitem(sys.modules, "httpx", _FAKE_HTTPX)
    # 清掉 vlm_client 内已 import 的 httpx 缓存（_import_httpx 是函数级 import，
    # 每次重查 sys.modules，无需清属性，但保险清一下模块属性缓存）
    return _FAKE_HTTPX


# ======================================================================
# 1. MockVLMClient 按序返回
# ======================================================================
class TestMockVLMClient:
    def test_returns_in_order(self) -> None:
        client = MockVLMClient(["第一帧", "第二帧", "第三帧"])
        assert run(client.describe(b"x", "p")) == "第一帧"
        assert run(client.describe(b"x", "p")) == "第二帧"
        assert run(client.describe(b"x", "p")) == "第三帧"

    def test_exhausted_raises(self) -> None:
        client = MockVLMClient(["only"])
        run(client.describe(b"x", "p"))
        with pytest.raises(IndexError):
            run(client.describe(b"x", "p"))


# ======================================================================
# 2. build_vlm_client 三 provider 正确实例类型
# ======================================================================
class TestBuildVLMClient:
    def test_ollama(self) -> None:
        c = build_vlm_client("ollama", base_url="http://x:11434", model="m")
        assert isinstance(c, OllamaVLMClient)

    def test_cloud(self) -> None:
        c = build_vlm_client(
            "cloud", base_url="https://api.x.io", api_key="sk-x", model="gpt-4o"
        )
        assert isinstance(c, CloudVLMClient)

    def test_mock(self) -> None:
        c = build_vlm_client("mock", responses=["a", "b"])
        assert isinstance(c, MockVLMClient)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="未知"):
            build_vlm_client("azure")

    def test_cloud_missing_apikey_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 让 httpx 可 import 但不发请求（缺 key 应在构造阶段就 fail）
        monkeypatch.setitem(sys.modules, "httpx", _FAKE_HTTPX)
        with pytest.raises(ValueError, match="api_key"):
            build_vlm_client("cloud", base_url="https://api.x.io")

    def test_cloud_missing_apikey_via_constructor(self) -> None:
        # 直接构造也应 raise（不依赖 httpx）
        with pytest.raises(ValueError, match="api_key"):
            CloudVLMClient(base_url="https://api.x.io", api_key="")


# ======================================================================
# 3. OllamaVLMClient.describe payload 拼装
# ======================================================================
class TestOllamaVLMClient:
    def test_payload_correct(self, fake_httpx: _FakeHttpxModule) -> None:
        fake_httpx.reset({"message": {"content": "  一只猫  "}})
        img = b"\xff\xd8\xff\xe0fakejpeg"
        client = OllamaVLMClient(base_url="http://localhost:11434", model="llama-vision")

        result = run(client.describe(img, "描述画面"))

        # 返回值是 message.content 且 strip
        assert result == "一只猫"
        url, payload, headers = fake_httpx.last_call  # type: ignore[misc]
        assert url == "http://localhost:11434/api/chat"
        # model 字段正确
        assert payload["model"] == "llama-vision"
        # stream:false
        assert payload["stream"] is False
        # messages 结构
        msg = payload["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == "描述画面"
        # images[0] 是 base64 字符串，能解码回原图
        b64 = msg["images"][0]
        assert isinstance(b64, str)
        assert base64.b64decode(b64) == img

    def test_missing_httpx_raises_importerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 暂时让 httpx 在 sys.modules 中消失（None 触发 import 失败）
        monkeypatch.setitem(sys.modules, "httpx", None)
        # 同时清掉已 import 的缓存属性（vlm_client._import_httpx 每次重 import）
        client = OllamaVLMClient()
        with pytest.raises(ImportError, match="httpx 未安装"):
            run(client.describe(b"img", "p"))


# ======================================================================
# 4. CloudVLMClient.describe OpenAI 多模态格式
# ======================================================================
class TestCloudVLMClient:
    def test_payload_openai_format(self, fake_httpx: _FakeHttpxModule) -> None:
        fake_httpx.reset(
            {"choices": [{"message": {"content": "一只狗在跑"}}]}
        )
        img = b"\xff\xd8\xff\xe0fakejpeg"
        client = CloudVLMClient(
            base_url="https://api.example.com",
            api_key="sk-test-key",
            model="gpt-4o-mini",
        )

        result = run(client.describe(img, "这帧有什么"))

        assert result == "一只狗在跑"
        url, payload, headers = fake_httpx.last_call  # type: ignore[misc]
        assert url == "https://api.example.com/v1/chat/completions"
        # Authorization header
        assert headers == {"Authorization": "Bearer sk-test-key"}
        # model
        assert payload["model"] == "gpt-4o-mini"
        # content[0] text
        content = payload["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "这帧有什么"}
        # content[1] image_url.url 是 data:image/jpeg;base64,...
        url_field = content[1]["image_url"]["url"]
        assert url_field.startswith("data:image/jpeg;base64,")
        # 解码 data URL 后半段应等于原图
        b64_part = url_field.split(",", 1)[1]
        assert base64.b64decode(b64_part) == img

    def test_parse_choices_message_content(self, fake_httpx: _FakeHttpxModule) -> None:
        # 验证响应解析：choices[0].message.content
        fake_httpx.reset(
            {"choices": [{"message": {"content": "  解析正确  "}}]}
        )
        client = CloudVLMClient(
            base_url="https://api.x.io", api_key="k", model="m"
        )
        assert run(client.describe(b"img", "p")) == "解析正确"

    def test_empty_choices_returns_empty(self, fake_httpx: _FakeHttpxModule) -> None:
        fake_httpx.reset({"choices": []})
        client = CloudVLMClient(
            base_url="https://api.x.io", api_key="k", model="m"
        )
        assert run(client.describe(b"img", "p")) == ""
