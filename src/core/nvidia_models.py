"""NVIDIA Integrate 模型能力矩阵 + Kilo 集成配置 (T5)

本模块登记 NVIDIA integrate.api.nvidia.com 上的全量模型（按用途分类），
并提供构造 raw REST payload 的工具函数。**关键坑点**：NVIDIA 的 OpenAI 兼容
端点走 raw REST（requests.post）时，**不接受** OpenAI Python SDK 的
`extra_body` 参数——必须把 `chat_template_kwargs` 与 `reasoning_budget`
**直接拍平到请求体顶层**，否则会被服务器忽略或报 400。

Kilo（已被 Anaconda 收购）走 OpenRouter 兼容协议，模型 id 带 `:free` 后缀，
1M context，**无视频能力**，用于 agent/编码任务。

本模块只读不改 llm_gateway.py / provider_router.py（其他代理在改）。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("VideoAnalyzerCore")

# ---- 常量 ----
NVIDIA_INTEGRATE_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_CHAT_ENDPOINT = f"{NVIDIA_INTEGRATE_BASE_URL}/chat/completions"

# Kilo 走 OpenRouter 兼容（:free 后缀），端点形如 https://api.kilo.ai/v1
KILO_DEFAULT_BASE_URL = "https://api.kilo.ai/v1"


# ---- 数据模型 ----
@dataclass(frozen=True)
class NvidiaModel:
    """NVIDIA Integrate 上一个模型的注册项。

    frozen=True 保证注册表不可变（不可变性原则），避免运行时误改能力矩阵。
    """
    id: str
    name: str
    category: str  # video / llm / embed / rerank / ocr / asr / tts / safety
    supports_video: bool = False
    supports_thinking: bool = False
    max_context: Optional[int] = None
    max_output: Optional[int] = None
    input_modalities: List[str] = field(default_factory=lambda: ["text"])
    # 视频分片配置（per-model，按官方文档；video/visual 模型显式填，其余默认）
    # 默认值对齐 NVIDIA Nemotron Omni 官方上限：mp4≤2min / 720p / 2fps / 256帧
    max_segment_sec: int = 120
    max_video_mb: int = 200
    max_frames: int = 256
    target_height: int = 720
    target_fps: int = 2
    notes: str = ""

    def __post_init__(self) -> None:
        # 输入校验：category 必须是已知枚举
        valid = {"video", "llm", "embed", "rerank", "ocr", "asr", "tts", "safety"}
        if self.category not in valid:
            raise ValueError(
                f"未知 category {self.category!r}，合法值: {sorted(valid)}"
            )


# ---- 全量注册表 ----
# 来源：NVIDIA integrate.api.nvidia.com 官方模型清单（用户提供的完整列表）。
# max_context / max_output 用官方文档标注值；未公开的留 None，不臆造。
NVIDIA_MODELS: List[NvidiaModel] = [
    # === 视频理解（原生 video_url）===
    NvidiaModel(
        id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        name="Nemotron-3-Nano-Omni-30B-A3B (Reasoning)",
        category="video",
        supports_video=True,
        supports_thinking=True,
        max_context=128000,
        max_output=65536,
        input_modalities=["text", "video_url"],
        # omni 官方视频上限：≤2min，1080p@1fps/128帧 或 720p@2fps/256帧
        # 选 720p/2fps/256帧（帧数更多，监控场景命中更稳）
        max_segment_sec=120,
        max_video_mb=200,
        max_frames=256,
        target_height=720,
        target_fps=2,
        notes="30B MoE (3B 激活)。视频≤2min：1080p@1fps/128帧 或 720p@2fps/256帧。"
              "仅英语输出。支持思考链。",
    ),
    NvidiaModel(
        id="nvidia/cosmos3-nano-reasoner",
        name="Cosmos3-Nano-Reasoner",
        category="video",
        supports_video=True,
        supports_thinking=True,
        max_context=128000,
        max_output=32768,
        input_modalities=["text", "video_url", "image_url"],
        # cosmos3 官方未公开视频上限，按 omni 同档保守对齐
        max_segment_sec=120,
        max_video_mb=200,
        max_frames=256,
        target_height=720,
        target_fps=2,
        notes="视频物理推理模型（世界模型方向）。",
    ),

    # === LLM / Agent ===
    NvidiaModel(
        id="nvidia/nemotron-3-ultra-550b-a55b",
        name="Nemotron-3-Ultra-550B-A55B",
        category="llm",
        supports_thinking=True,
        max_context=1000000,
        max_output=65536,
        input_modalities=["text"],
        notes="550B MoE (55B 激活)。1M context，旗舰 agent/长文档模型。",
    ),
    NvidiaModel(
        id="nvidia/nemotron-3.5-lightning-30b-a3b",
        name="Nemotron-3.5-Lightning-30B-A3B",
        category="llm",
        supports_thinking=True,
        max_context=128000,
        max_output=32768,
        input_modalities=["text"],
        notes="最快的 30B 模型，低延迟 agent/编码首选。",
    ),
    NvidiaModel(
        id="nvidia/nemotron-3-super-120b-a12b",
        name="Nemotron-3-Super-120B-A12B",
        category="llm",
        supports_thinking=True,
        max_context=128000,
        max_output=32768,
        input_modalities=["text"],
        notes="120B MoE (12B 激活)。平衡质量与速度。",
    ),

    # === 多模态视觉 LLM（非原生视频，但可逐帧分析）===
    NvidiaModel(
        id="nvidia/nemotron-nano-12b-v2-vl",
        name="Nemotron-Nano-12B-v2-VL",
        category="llm",
        supports_thinking=False,
        max_context=128000,
        max_output=32768,
        input_modalities=["text", "image_url"],
        notes="12B 视觉语言模型。可逐帧分析视频截图。",
    ),

    # === Embedding（RAG）===
    NvidiaModel(
        id="nvidia/nemotron-3-embed-1b",
        name="Nemotron-3-Embed-1B",
        category="embed",
        max_context=32768,
        max_output=2048,  # embedding 维度
        input_modalities=["text"],
        notes="2048 维向量，34 语言含中文。用于跨视频知识库 RAG 检索。",
    ),

    # === Rerank ===
    NvidiaModel(
        id="nvidia/llama-nemotron-rerank-vl-1b-v2",
        name="Llama-Nemotron-Rerank-VL-1B-v2",
        category="rerank",
        max_context=32768,
        input_modalities=["text", "image_url"],
        notes="视觉+文本重排序，RAG 二阶段精排。",
    ),

    # === OCR / 文档解析 ===
    NvidiaModel(
        id="nvidia/nemotron-ocr-v2",
        name="Nemotron-OCR-v2",
        category="ocr",
        max_context=32768,
        input_modalities=["image_url"],
        notes="通用 OCR（含表格/公式/手写）。",
    ),
    NvidiaModel(
        id="nvidia/nemotron-parse",
        name="Nemotron-Parse",
        category="ocr",
        max_context=32768,
        input_modalities=["image_url", "file_url"],
        notes="复杂文档结构化解析（PDF/发票/合同）。",
    ),

    # === ASR ===
    NvidiaModel(
        id="nvidia/parakeet-ctc-0.6b-asr",
        name="Parakeet-CTC-0.6B-ASR (English)",
        category="asr",
        input_modalities=["audio_url"],
        notes="英语语音识别。",
    ),
    NvidiaModel(
        id="nvidia/parakeet-ctc-0.6b-zh-cn",
        name="Parakeet-CTC-0.6B-ASR (zh-CN)",
        category="asr",
        input_modalities=["audio_url"],
        notes="普通话语音识别（本项目 Whisper 的 NVIDIA 替代）。",
    ),

    # === TTS ===
    NvidiaModel(
        id="nvidia/magpie-tts-zeroshot",
        name="Magpie-TTS-ZeroShot",
        category="tts",
        input_modalities=["text", "audio_url"],
        notes="零样本语音合成。",
    ),

    # === 内容安全 ===
    NvidiaModel(
        id="nvidia/nemotron-3.5-content-safety",
        name="Nemotron-3.5-Content-Safety",
        category="safety",
        max_context=32768,
        input_modalities=["text", "image_url"],
        notes="内容安全分类（文本+图像）。",
    ),
]


# ---- Kilo 注册表（OpenRouter 兼容，:free 后缀，无视频）----
@dataclass(frozen=True)
class KiloModel:
    """Kilo（OpenRouter 兼容）上的免费模型注册项。"""
    id: str  # 带 :free 后缀
    name: str
    category: str  # llm / embed / rerank
    max_context: Optional[int] = None
    supports_thinking: bool = False
    notes: str = ""


KILO_MODELS: List[KiloModel] = [
    KiloModel(
        id="nvidia/nemotron-3-ultra-550b-a55b:free",
        name="Nemotron-3-Ultra-550B (Kilo Free)",
        category="llm",
        max_context=1000000,
        supports_thinking=True,
        notes="1M context，agent/长文档。无视频。",
    ),
    KiloModel(
        id="nvidia/nemotron-3.5-lightning-30b-a3b:free",
        name="Nemotron-3.5-Lightning-30B (Kilo Free)",
        category="llm",
        max_context=128000,
        supports_thinking=True,
        notes="最快 30B，低延迟编码。",
    ),
    KiloModel(
        id="nvidia/nemotron-3-super-120b-a12b:free",
        name="Nemotron-3-Super-120B (Kilo Free)",
        category="llm",
        max_context=128000,
        supports_thinking=True,
        notes="120B MoE，平衡质量速度。",
    ),
]


# ---- 便捷查询 ----
def list_by_category(category: str) -> List[NvidiaModel]:
    """按 category 过滤 NVIDIA 模型。"""
    return [m for m in NVIDIA_MODELS if m.category == category]


def get_video_model() -> Optional[NvidiaModel]:
    """返回首选原生视频理解模型（omni reasoning）。无则 None。"""
    video_models = list_by_category("video")
    # 优先 omni（原生视频+思考），其次 cosmos3
    for m in video_models:
        if "omni" in m.id:
            return m
    return video_models[0] if video_models else None


def get_embed_model() -> Optional[NvidiaModel]:
    """返回 RAG 检索用的 embedding 模型。"""
    embeds = list_by_category("embed")
    return embeds[0] if embeds else None


def get_agent_model() -> Optional[NvidiaModel]:
    """返回 agent/编码首选 LLM（ultra 旗舰，1M context）。"""
    for m in list_by_category("llm"):
        if "ultra" in m.id and "vl" not in m.id:
            return m
    llms = list_by_category("llm")
    return llms[0] if llms else None


def get_model_by_id(model_id: str) -> Optional[NvidiaModel]:
    """按 id 精确查 NVIDIA 模型。"""
    for m in NVIDIA_MODELS:
        if m.id == model_id:
            return m
    return None


# 视频分片默认配置（未知模型兜底，对齐 Nemotron Omni 官方上限）
_VIDEO_CONFIG_DEFAULT: Dict[str, Any] = {
    "max_segment_sec": 120,
    "max_video_mb": 200,
    "max_frames": 256,
    "target_height": 720,
    "target_fps": 2,
}


def get_video_config(model_id: str) -> Dict[str, Any]:
    """返回某模型的视频分片配置。

    按 per-model 注册的 max_segment_sec / target_height / target_fps /
    max_frames / max_video_mb 动态返回；未知模型回退到默认（对齐 Nemotron
    Omni 官方上限：mp4≤2min / 720p / 2fps / 256帧）。batch_runner 不再写死
    30s/720p，而用此函数按模型能力动态切分。

    Returns:
        dict 含 max_segment_sec / target_height / target_fps / max_frames /
        max_video_mb（int）
    """
    m = get_model_by_id(model_id)
    if m is None:
        logger.debug(
            f"[nvidia] 未知模型 {model_id!r}，视频分片配置回退默认 "
            f"{_VIDEO_CONFIG_DEFAULT}"
        )
        return dict(_VIDEO_CONFIG_DEFAULT)
    return {
        "max_segment_sec": m.max_segment_sec,
        "max_video_mb": m.max_video_mb,
        "max_frames": m.max_frames,
        "target_height": m.target_height,
        "target_fps": m.target_fps,
    }


# ---- Payload 构造 ----
def build_nvidia_payload(
    model_id: str,
    messages: List[Dict[str, Any]],
    *,
    enable_thinking: bool = True,
    reasoning_budget: int = 8192,
    max_tokens: int = 65536,
    temperature: float = 0.2,
    stream: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造 NVIDIA integrate.api.nvidia.com 的 raw REST payload。

    **关键坑点**：raw REST（requests.post 直传 JSON）不接受 OpenAI Python SDK
    的 `extra_body` 参数。`chat_template_kwargs` 与 `reasoning_budget` 必须
    **直接放在请求体顶层**，不能嵌套进 `extra_body`。本函数已处理。

    Args:
        model_id: 模型 id，如 "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        messages: OpenAI 格式消息列表
        enable_thinking: 是否开启思考链（仅对 supports_thinking 模型生效）
        reasoning_budget: 思考 token 预算（思考链长度上限）
        max_tokens: 输出 token 上限
        temperature: 采样温度
        stream: 是否流式
        extra: 额外顶层字段（如 top_p），会被合并进 payload 顶层

    Returns:
        可直接传给 requests.post(json=...) 的 dict。**无 extra_body 字段**。
    """
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    # 思考链相关字段：查模型是否支持思考；未知模型默认按调用者意图放（服务器兜底）
    model = get_model_by_id(model_id)
    supports_thinking = model.supports_thinking if model else True

    if enable_thinking and supports_thinking:
        # 拍平到顶层：chat_template_kwargs 是 NVIDIA 约定的思考开关
        payload["chat_template_kwargs"] = {"thinking": True}
        # reasoning_budget 直接放顶层（不是 extra_body.reasoning_budget）
        payload["reasoning_budget"] = reasoning_budget
    elif enable_thinking and not supports_thinking:
        logger.debug(
            f"[nvidia] 模型 {model_id!r} 不支持思考链，跳过 thinking 字段"
        )

    if extra:
        # 合并额外顶层字段，但不允许覆盖思考链字段（防误关）
        for k, v in extra.items():
            if k in ("chat_template_kwargs", "reasoning_budget"):
                logger.warning(f"[nvidia] extra 试图覆盖受保护字段 {k!r}，已忽略")
                continue
            payload[k] = v

    return payload


# ---- Provider 探测 ----
def detect_provider(base_url: str) -> str:
    """根据 base_url 判断 provider。

    Returns:
        "nvidia" / "kilo" / "openrouter" / "unknown"
    """
    if not base_url:
        return "unknown"
    u = base_url.lower()
    if "integrate.api.nvidia.com" in u or "integrate.api.nvidia" in u:
        return "nvidia"
    if "kilo.ai" in u or "kilo" in u:
        return "kilo"
    if "openrouter.ai" in u:
        return "openrouter"
    return "unknown"


def get_nvidia_api_key() -> Optional[str]:
    """从环境变量读取 NVIDIA API Key（不日志、不打印）。"""
    return os.environ.get("VAP_NV_API_KEY") or os.environ.get("NVIDIA_API_KEY")
