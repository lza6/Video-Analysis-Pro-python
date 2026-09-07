"""VLM（视觉语言模型）客户端 — 9.1 视觉理解接入。

为 Agent 框架提供"看图说话"能力：把一帧图片 + 提示词交给 VLM，拿回
文字描述。后续由主控接入到 ReactLoopAgent 工具集（adapter.py 桥接），
本模块只负责**调用层**，不改 logic.py / agent_tools.py。

三种实现：
  - OllamaVLMClient：本地 Ollama（llama-vision 等），走 `/api/chat`，
    body 的 messages[0].images[0] 是 base64 字符串。
  - CloudVLMClient：OpenAI 兼容云端（如 OpenAI / NVIDIA / 自建），
    走 `/v1/chat/completions`，content 是多模态数组（text + image_url）。
  - MockVLMClient：测试用，按序 pop 预置响应，绝不真实网络。

依赖策略（守付费 API 红线 + 条件依赖）：
  - httpx 是**条件依赖**（不在 core requirements），缺失时 describe raise
    ImportError，主控按需 `pip install httpx`。这样不污染最小启动集。
  - 真实付费 API 预算为 0：测试全部注入 Mock httpx，不真实连云端。
  - api_key 缺失时 CloudVLMClient 构造即 raise ValueError（fail fast），
    避免误发匿名请求。

参考：
  - 项目条件依赖模式见 `src/core/im_gateway/cipher.py`（cryptography 缺失降级）
  - Protocol 模式见 `src/core/agent/loop.py:LLMClient`
  - OpenAI 多模态格式见 `src/core/kilo_provider.py` / `llm_gateway.py`
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Protocol

logger = logging.getLogger(__name__)

# 默认值（.env / 构造参数可覆盖）
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_VLM_MODEL = "llama-vision"
DEFAULT_CLOUD_VLM_MODEL = "gpt-4o-mini"


def _import_httpx() -> Any:
    """条件导入 httpx（不在 core requirements）。

    缺失时抛 ImportError，调用方应捕获并提示 `pip install httpx`。
    """
    try:
        import httpx  # type: ignore[import-untyped]
        return httpx
    except ImportError as e:
        raise ImportError(
            "httpx 未安装：VLM 客户端需要 httpx（条件依赖）。"
            "请运行 `pip install httpx` 后重试。"
        ) from e


def _encode_image_b64(image_bytes: bytes) -> str:
    """把原始图片字节编码成 base64 字符串（不带 data: 前缀）。"""
    return base64.b64encode(image_bytes).decode("ascii")


def _build_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """构造 data URL（OpenAI image_url 格式）。"""
    b64 = _encode_image_b64(image_bytes)
    return f"data:{mime};base64,{b64}"


class VLMClient(Protocol):
    """VLM 客户端协议：把图片 + prompt 翻译成文字描述。"""

    async def describe(self, image_bytes: bytes, prompt: str) -> str:
        """对一张图片提问，返回 VLM 的文字回答。

        Args:
            image_bytes: 原始图片字节（JPEG/PNG 等，由调用方解码）。
            prompt: 提示词，如"描述这帧画面中的关键物体"。

        Returns:
            VLM 输出的纯文本描述。
        """
        ...


# ----------------------------------------------------------------------
# Ollama 本地 VLM
# ----------------------------------------------------------------------
class OllamaVLMClient:
    """Ollama 本地 VLM 客户端（`/api/chat` + images 字段）。

    测试时注入 fake httpx module（sys.modules），不真实连本地 Ollama。
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_OLLAMA_VLM_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def describe(self, image_bytes: bytes, prompt: str) -> str:
        """POST `{base_url}/api/chat`，body 含 images base64 列表。"""
        httpx = _import_httpx()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [_encode_image_b64(image_bytes)],
                }
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:  # type: ignore[attr-defined]
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        # Ollama /api/chat 返回 {"message": {"content": "..."}, ...}
        return str(data.get("message", {}).get("content", "")).strip()


# ----------------------------------------------------------------------
# 云端 OpenAI 兼容 VLM
# ----------------------------------------------------------------------
class CloudVLMClient:
    """OpenAI 兼容云端 VLM 客户端（`/v1/chat/completions` 多模态）。

    api_key 缺失即 raise ValueError（fail fast，不发匿名请求）。
    httpx 为条件依赖，缺失 raise ImportError。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_CLOUD_VLM_MODEL,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "CloudVLMClient 需要 api_key（云端 VLM 鉴权缺失，拒绝发请求）"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def describe(self, image_bytes: bytes, prompt: str) -> str:
        """POST `{base_url}/v1/chat/completions`，content 多模态数组。"""
        httpx = _import_httpx()
        data_url = _build_data_url(image_bytes)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:  # type: ignore[attr-defined]
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        # OpenAI 格式：choices[0].message.content
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()


# ----------------------------------------------------------------------
# Mock VLM（测试专用）
# ----------------------------------------------------------------------
class MockVLMClient:
    """测试用 VLM 客户端：按序 pop 预置响应，绝不真实网络。

    用法：
        client = MockVLMClient(["第一帧描述", "第二帧描述"])
        await client.describe(b"...", "描述")  # → "第一帧描述"
        await client.describe(b"...", "描述")  # → "第二帧描述"
    """

    def __init__(self, responses: List[str]) -> None:
        # 复制一份避免外部 mutate；倒序方便 pop
        self._responses: List[str] = list(responses)

    async def describe(self, image_bytes: bytes, prompt: str) -> str:
        if not self._responses:
            raise IndexError("MockVLMClient 响应列表已耗尽")
        return self._responses.pop(0)


# ----------------------------------------------------------------------
# 工厂
# ----------------------------------------------------------------------
def build_vlm_client(provider: str, **kwargs: Any) -> VLMClient:
    """按 provider 名构造 VLM 客户端。

    Args:
        provider: "ollama" / "cloud" / "mock"。
        **kwargs: 透传给对应客户端构造函数。

        - "ollama" → OllamaVLMClient（base_url / model / timeout）
        - "cloud"  → CloudVLMClient（base_url / api_key 必填 / model / timeout）
        - "mock"   → MockVLMClient（responses: list[str]）

    Returns:
        VLMClient 实例。

    Raises:
        ValueError: 未知 provider，或 cloud 缺 api_key。
    """
    provider = provider.lower().strip()
    if provider == "ollama":
        return OllamaVLMClient(**kwargs)  # type: ignore[arg-type]
    if provider == "cloud":
        if not kwargs.get("api_key"):
            raise ValueError("cloud provider 需要 api_key 参数")
        return CloudVLMClient(**kwargs)  # type: ignore[arg-type]
    if provider == "mock":
        return MockVLMClient(**kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"未知 VLM provider: {provider!r}（支持: ollama / cloud / mock）"
    )
