"""全协议 LLM 网关抽象 (v5.0)

参考 cc-switch 的供应商分类思想，把多协议（Anthropic Messages / OpenAI Chat Completions /
OpenAI Responses / Gemini generateContent）统一到一套抽象后端，Agent 与上层只面向
统一接口，按 provider 配置路由到具体协议实现。

设计:
  - ProtocolBackend (ABC): chat_stream / list_models / probe
  - AnthropicBackend : /v1/messages (兼容 glm/claude 系)
  - OpenAIChatBackend : /v1/chat/completions (兼容 deepseek/qwen/openai 系)
  - OpenAIResponsesBackend : /v1/responses (新版 OpenAI Responses API)
  - GeminiBackend : generateContent (Google Gemini 原生)
  - GatewayRouter : 按 provider 字段路由，支持故障转移队列
"""
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests

logger = logging.getLogger("VideoAnalyzerCore")


class ProtocolBackend:
    """所有协议后端的基类。"""

    name: str = "base"
    # 该后端是否支持原生图像输入（决定 search_by_image 等是否直传图）
    supports_vision: bool = True

    def __init__(self, api_key: str, base_url: str, model: str,
                 timeout: int = 600, max_tokens: int = 4096):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        # 默认 trust_env=True 走系统代理（外部 API 需要）；本机 localhost 后端在子类覆盖
        self._session = requests.Session()

    # ---- 公共工具 ----
    @staticmethod
    def _encode_image(image_path) -> Optional[str]:
        p = Path(image_path) if not isinstance(image_path, Path) else image_path
        if not p.exists():
            return None
        return base64.b64encode(p.read_bytes()).decode("utf-8")

    # ---- 子类实现 ----
    def chat_stream(self, messages: List[Dict[str, Any]],
                    image_paths: Optional[List[str]] = None,
                    temperature: float = 0.2,
                    system: Optional[str] = None) -> Iterator[str]:
        raise NotImplementedError

    def list_models(self) -> List[str]:
        return []

    def probe(self) -> bool:
        """健康探测，默认实现可被子类覆盖。"""
        return bool(self.api_key and self.base_url)


class AnthropicBackend(ProtocolBackend):
    """Anthropic Messages 协议 (兼容 glm/claude 系中转)。

    端点: POST {base}/messages
    认证: x-api-key + anthropic-version
    流: SSE (event: content_block_delta, data: {...})
    """
    name = "anthropic"
    supports_vision = True

    def chat_stream(self, messages, image_paths=None, temperature=0.2, system=None):
        # 把 image_paths 注入到最后一条 user 消息
        msgs = [dict(m) for m in messages]
        if image_paths and msgs and msgs[-1]["role"] == "user":
            content = msgs[-1]["content"]
            if isinstance(content, str):
                parts = [{"type": "text", "text": content}]
            else:
                parts = list(content)
            for img in image_paths:
                b64 = self._encode_image(img)
                if b64:
                    parts.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    })
            msgs[-1]["content"] = parts

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": msgs,
            "temperature": temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            with self._session.post(f"{self.base_url}/messages",
                                    headers=headers, json=payload, stream=True,
                                    timeout=self.timeout) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data: "):
                        continue
                    data = raw[6:]
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    et = ev.get("type", "")
                    if et == "content_block_delta":
                        d = ev.get("delta", {})
                        if d.get("type") == "text_delta":
                            yield d.get("text", "")
                        elif d.get("type") == "thinking_delta":
                            yield f"<think>{d.get('thinking', '')}</think>"
        except Exception as e:
            yield f"[Anthropic error: {e}]"

    def list_models(self):
        try:
            r = self._session.get(f"{self.base_url}/models",
                                   headers={"x-api-key": self.api_key,
                                            "anthropic-version": "2023-06-01"},
                                   timeout=10)
            if r.status_code == 200:
                data = r.json().get("data", [])
                return [m.get("id") for m in data if m.get("id")]
        except Exception:
            pass
        return []


class OpenAIChatBackend(ProtocolBackend):
    """OpenAI Chat Completions 协议 (最广泛兼容)。"""
    name = "openai_chat"
    supports_vision = True

    def chat_stream(self, messages, image_paths=None, temperature=0.2, system=None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(dict(m) for m in messages)
        if image_paths and msgs and msgs[-1]["role"] == "user":
            parts = [{"type": "text", "text": msgs[-1]["content"]}]
            for img in image_paths:
                b64 = self._encode_image(img)
                if b64:
                    parts.append({"type": "image_url",
                                  "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            msgs[-1]["content"] = parts

        payload = {"model": self.model, "messages": msgs,
                   "temperature": temperature, "stream": True,
                   "max_tokens": self.max_tokens}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        try:
            with self._session.post(f"{self.base_url}/chat/completions",
                                    headers=headers, json=payload, stream=True,
                                    timeout=self.timeout) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data: "):
                        continue
                    data = raw[6:]
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = ev.get("choices", [{}])[0].get("delta", {})
                    rc = delta.get("reasoning_content")
                    if rc:
                        yield f"<think>{rc}</think>"
                    c = delta.get("content")
                    if c:
                        yield c
        except Exception as e:
            yield f"[OpenAI error: {e}]"

    def list_models(self):
        try:
            r = self._session.get(f"{self.base_url}/models",
                                  headers={"Authorization": f"Bearer {self.api_key}"},
                                  timeout=10)
            if r.status_code == 200:
                return [m["id"] for m in r.json().get("data", []) if m.get("id")]
        except Exception:
            pass
        return []


class OpenAIResponsesBackend(ProtocolBackend):
    """OpenAI Responses API (新版 stateful)。"""
    name = "openai_responses"
    supports_vision = True

    def chat_stream(self, messages, image_paths=None, temperature=0.2, system=None):
        # Responses API 用 input 数组
        inp = []
        if system:
            inp.append({"role": "system", "content": system})
        for m in messages:
            inp.append({"role": m["role"], "content": m["content"]})
        if image_paths and inp and inp[-1]["role"] == "user":
            parts = [{"type": "input_text", "text": inp[-1]["content"]}]
            for img in image_paths:
                b64 = self._encode_image(img)
                if b64:
                    parts.append({"type": "input_image",
                                   "image_url": f"data:image/jpeg;base64,{b64}"})
            inp[-1]["content"] = parts
        payload = {"model": self.model, "input": inp, "stream": True,
                   "max_output_tokens": self.max_tokens}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        try:
            with self._session.post(f"{self.base_url}/responses",
                                    headers=headers, json=payload, stream=True,
                                    timeout=self.timeout) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data: "):
                        continue
                    data = raw[6:]
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    t = ev.get("type", "")
                    if t in ("response.output_text.delta", "response.text.delta"):
                        yield ev.get("delta", "")
                    elif t == "response.reasoning.delta":
                        yield f"<think>{ev.get('delta', '')}</think>"
        except Exception as e:
            yield f"[Responses error: {e}]"


class GeminiBackend(ProtocolBackend):
    """Google Gemini generateContent 原生协议。"""
    name = "gemini"
    supports_vision = True

    def chat_stream(self, messages, image_paths=None, temperature=0.2, system=None):
        # Gemini 用 contents 数组，role: user/model
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            parts = []
            content = m["content"]
            if isinstance(content, str):
                parts.append({"text": content})
            else:
                parts.extend(content)
            if m["role"] == "user" and image_paths:
                for img in image_paths:
                    b64 = self._encode_image(img)
                    if b64:
                        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
            contents.append({"role": role, "parts": parts})
        payload = {"contents": contents,
                   "generationConfig": {"temperature": temperature,
                                        "maxOutputTokens": self.max_tokens}}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = (f"{self.base_url}/v1beta/models/{self.model}:streamGenerateContent"
               f"?key={self.api_key}&alt=sse")
        try:
            with self._session.post(url, json=payload, stream=True,
                                    timeout=self.timeout) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data: "):
                        continue
                    try:
                        ev = json.loads(raw[6:])
                    except json.JSONDecodeError:
                        continue
                    for cand in ev.get("candidates", []):
                        for p in cand.get("content", {}).get("parts", []):
                            if p.get("thought"):
                                yield f"<think>{p.get('text', '')}</think>"
                            elif p.get("text"):
                                yield p["text"]
        except Exception as e:
            yield f"[Gemini error: {e}]"


# ---- 协议路由 ----
_PROTOCOL_MAP = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIChatBackend,
    "openai_chat": OpenAIChatBackend,
    "openai_responses": OpenAIResponsesBackend,
    "gemini": GeminiBackend,
}


def build_backend(protocol: str, api_key: str, base_url: str, model: str,
                  **kw) -> ProtocolBackend:
    cls = _PROTOCOL_MAP.get(protocol.lower())
    if cls is None:
        # 默认回退到最兼容的 openai chat
        logger.warning(f"未知协议 {protocol!r}，回退到 openai_chat")
        cls = OpenAIChatBackend
    return cls(api_key=api_key, base_url=base_url, model=model, **kw)


def detect_protocol(base_url: str, model: str) -> str:
    """启发式协议探测（供自动适配）。"""
    u = base_url.lower()
    m = model.lower()
    if "gemini" in m or "gemini" in u or "generatecontent" in u:
        return "gemini"
    if "/responses" in u or "responses" in m:
        return "openai_responses"
    if "anthropic" in u or "claude" in m or "glm" in m or "/messages" in u:
        return "anthropic"
    return "openai_chat"


class GatewayRouter:
    """多供应商路由 + 故障转移队列 (cc-switch 思想)。

    providers: [{"id","protocol","base_url","api_key","model","category"}]
    current: 当前激活的 provider id
    """

    def __init__(self, providers: List[Dict[str, Any]], current: Optional[str] = None):
        self.providers = {p["id"]: p for p in providers}
        self.current = current or (providers[0]["id"] if providers else None)
        self._backend_cache: Dict[str, ProtocolBackend] = {}

    def _get_backend(self, provider_id: Optional[str] = None) -> Optional[ProtocolBackend]:
        pid = provider_id or self.current
        if not pid or pid not in self.providers:
            return None
        if pid not in self._backend_cache:
            p = self.providers[pid]
            proto = p.get("protocol") or detect_protocol(p["base_url"], p["model"])
            self._backend_cache[pid] = build_backend(
                proto, p["api_key"], p["base_url"], p["model"],
                max_tokens=p.get("max_tokens", 8192))
        return self._backend_cache[pid]

    def chat_stream(self, messages, image_paths=None, temperature=0.2,
                    system=None, provider_id: Optional[str] = None) -> Iterator[str]:
        backend = self._get_backend(provider_id)
        if backend is None:
            yield "[Gateway: 无可用 provider]"
            return
        yield from backend.chat_stream(messages, image_paths, temperature, system)

    def switch(self, provider_id: str):
        if provider_id in self.providers:
            self.current = provider_id
            return True
        return False

    def list_providers(self) -> List[Dict[str, Any]]:
        return [{"id": p["id"], "name": p.get("name", p["id"]),
                 "protocol": p.get("protocol", "auto"),
                 "model": p["model"], "category": p.get("category", "custom")}
                for p in self.providers.values()]
