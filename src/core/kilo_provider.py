"""Kilo provider 集成 (v5.5 T6)

Kilo 被 Anaconda 收购后提供 OpenAI 兼容协议（base_url https://kilocode.ai/v1），
免费模型如 nvidia/nemotron-3-ultra-550b-a55b:free、openai/gpt-oss-120b:free，
1M context，无视频能力，用于 agent ReAct / 编码 / 知识库问答。

多 key 轮换：VAP_KILO_API_KEYS 支持逗号/分号/空白分隔，遇 401/403/429 自动切下一 key，
全部失败 yield 错误文本（不抛异常，保持流式契约）。

接缝约定（与其他代理协作）:
  - provider_router.RateLimiter 由其他代理实现（T7），此处 try/except 兜底 import，
    缺失时降级为无限流（Kilo 免费层暂无限流）。
  - llm_gateway.OpenAIChatBackend 已实现单 key OpenAI 兼容协议，本类职责是多 key 池
    轮换 + embedding 端点，不继承 OpenAIChatBackend（避免 key 轮换与重试逻辑耦合）。
  - 真实付费 API 不调用（Kilo 虽免费但要 key），全部走 mock 测试。
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import requests

logger = logging.getLogger("VideoAnalyzerCore")

# 默认配置（.env 可覆盖）
DEFAULT_KILO_BASE_URL = "https://kilocode.ai/v1"
DEFAULT_KILO_MODEL = "openai/gpt-oss-120b:free"
_EMBED_MODEL_DEFAULT = "nvidia/nemotron-3-embed-1b"

# 触发 key 轮换的 HTTP 状态码（认证失败 / 限流）
_KEY_ROTATE_STATUS = {401, 403, 429}


@dataclass
class KiloConfig:
    """Kilo provider 配置（不可变值对象，__post_init__ 规范化）。"""
    base_url: str = DEFAULT_KILO_BASE_URL
    api_keys: List[str] = field(default_factory=list)
    default_model: str = DEFAULT_KILO_MODEL
    timeout: int = 600
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        # 过滤空 key（"k1,,k2" / 纯空白）
        self.api_keys = [k.strip() for k in self.api_keys if k and k.strip()]


class KiloClient:
    """OpenAI 兼容协议封装 + 多 key 轮换。

    与 llm_gateway.OpenAIChatBackend 互补：后者单 key 单后端，本类负责多 key 池
    轮换 + provider_router.RateLimiter 接缝（可选）。流式 chat 产出纯文本 delta
    （不泄漏 JSON 碎片），契约与 OllamaClient 一致。
    """

    def __init__(self, config: KiloConfig) -> None:
        self.config = config
        self._key_index = 0
        self._lock = threading.Lock()
        self._session = requests.Session()
        # provider_router 由其他代理实现（T7），try/except 兜底避免循环依赖
        self._rate_limiter: Any = None
        try:
            from src.core.provider_router import RateLimiter  # type: ignore
            self._rate_limiter = RateLimiter()
            logger.debug("[kilo] provider_router.RateLimiter 已接入")
        except Exception:
            logger.debug("[kilo] provider_router 不可用，降级无限流")

    # ---- key 轮换 ----
    def _next_key(self) -> str:
        """线程安全轮换到下一个 key。"""
        with self._lock:
            keys = self.config.api_keys
            if not keys:
                raise RuntimeError("Kilo: 无可用 API key（检查 VAP_KILO_API_KEYS）")
            key = keys[self._key_index % len(keys)]
            self._key_index += 1
            return key

    def _all_keys(self) -> List[str]:
        return list(self.config.api_keys)

    # ---- 流式 chat ----
    def chat_stream(self, messages: List[Dict[str, Any]],
                    model: Optional[str] = None,
                    temperature: float = 0.2,
                    system: Optional[str] = None,
                    max_tokens: Optional[int] = None) -> Iterator[str]:
        """流式 chat，多 key 轮换。yield 纯文本 delta。"""
        model = model or self.config.default_model
        msgs: List[Dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(dict(m) for m in messages)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens

        url = f"{self.config.base_url}/chat/completions"
        yield from self._stream_with_rotation(url, payload)

    def _stream_with_rotation(self, url: str,
                              payload: Dict[str, Any]) -> Iterator[str]:
        keys = self._all_keys()
        if not keys:
            yield "[Kilo: 无可用 key]"
            return
        tried: set = set()
        last_err = ""
        for _ in range(len(keys)):
            key = self._next_key()
            if key in tried:
                break
            tried.add(key)
            if self._rate_limiter is not None:
                try:
                    self._rate_limiter.acquire()  # type: ignore[attr-defined]
                except Exception:
                    pass  # 限流器故障不阻塞主链路
            headers = {"Authorization": f"Bearer {key}",
                       "Content-Type": "application/json"}
            try:
                resp = self._session.post(url, headers=headers, json=payload,
                                          stream=True, timeout=self.config.timeout)
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning(f"[kilo] 网络错误，切下一 key: {last_err}")
                continue
            if resp.status_code in _KEY_ROTATE_STATUS:
                last_err = f"HTTP {resp.status_code}"
                logger.warning(f"[kilo] {resp.status_code}，轮换 key "
                               f"（已试 {len(tried)}/{len(keys)}）")
                try:
                    resp.close()
                except Exception:
                    pass
                continue
            try:
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
                    delta = (ev.get("choices") or [{}])[0].get("delta", {})
                    c = delta.get("content")
                    if c:
                        yield c
                return
            except requests.HTTPError as e:
                last_err = str(e)
                logger.warning(f"[kilo] HTTPError，切下一 key: {last_err}")
            finally:
                try:
                    resp.close()
                except Exception:
                    pass
        yield f"[Kilo: 全部 key 均失败（{last_err}）]"

    # ---- 非流式 chat（RAG 问答用）----
    def chat(self, messages: List[Dict[str, Any]],
             model: Optional[str] = None,
             temperature: float = 0.2,
             system: Optional[str] = None,
             max_tokens: Optional[int] = None) -> str:
        """非流式 chat，返回完整文本。"""
        return "".join(self.chat_stream(messages, model, temperature,
                                        system, max_tokens))

    # ---- embedding（NVIDIA nemotron-3-embed-1b via OpenAI 兼容 /embeddings）----
    def embed(self, texts: List[str],
              model: Optional[str] = None) -> Optional[List[List[float]]]:
        """文本 embedding（用于 RAG 索引/查询）。

        返回 None 表示 Kilo embed 不可用，调用方应回退本地 sentence-transformers
        （kb_indexer.get_embedder，CLIP 512 维，与现有 kb_frames collection 一致）。
        本次不真实调用（需 key），mock 测试覆盖 payload + 轮换。
        """
        model = model or os.environ.get("VAP_NV_EMBED_MODEL", _EMBED_MODEL_DEFAULT)
        url = f"{self.config.base_url}/embeddings"
        keys = self._all_keys()
        for key in keys:
            headers = {"Authorization": f"Bearer {key}",
                       "Content-Type": "application/json"}
            payload = {"model": model, "input": texts}
            try:
                resp = self._session.post(url, headers=headers, json=payload,
                                          timeout=self.config.timeout)
                if resp.status_code in _KEY_ROTATE_STATUS:
                    logger.warning(f"[kilo] embed {resp.status_code}，切下一 key")
                    continue
                resp.raise_for_status()
                data = resp.json().get("data", [])
                return [item["embedding"] for item in data if "embedding" in item]
            except (requests.RequestException, ValueError) as e:
                logger.warning(f"[kilo] embed 错误，切下一 key: {e}")
                continue
        logger.warning("[kilo] embed 全 key 失败，调用方应回退本地 embedder")
        return None


def _parse_keys(raw: str) -> List[str]:
    """VAP_KILO_API_KEYS 支持逗号 / 分号 / 空白分隔多 key。"""
    if not raw:
        return []
    parts = re.split(r"[,;\s]+", raw.strip())
    return [p.strip() for p in parts if p.strip()]


def build_kilo_config(config_manager: Any = None) -> KiloConfig:
    """从 .env / config_manager 构建 KiloConfig。

    优先级：环境变量 > config_manager（如有）> 默认。
    不读密钥环（Kilo key 是 JWT，多 key 批量管理更适合 .env）。
    """
    base_url = os.environ.get("VAP_KILO_BASE_URL", DEFAULT_KILO_BASE_URL)
    keys = _parse_keys(os.environ.get("VAP_KILO_API_KEYS", ""))
    model = os.environ.get("VAP_KILO_DEFAULT_MODEL", DEFAULT_KILO_MODEL)
    # config_manager 兜底（未来 UI 可写入 ini [Kilo] 段）
    if not keys and config_manager is not None:
        try:
            ini_keys = config_manager.config.get("Kilo", "api_keys", fallback="")
            keys = _parse_keys(ini_keys)
        except Exception:
            pass
    return KiloConfig(base_url=base_url, api_keys=keys, default_model=model)


def build_kilo_client(config_manager: Any = None) -> Optional[KiloClient]:
    """工厂：构建 KiloClient，无 key 返回 None（调用方优雅降级）。"""
    cfg = build_kilo_config(config_manager)
    if not cfg.api_keys:
        logger.info("[kilo] VAP_KILO_API_KEYS 未配置，Kilo provider 禁用")
        return None
    return KiloClient(cfg)
