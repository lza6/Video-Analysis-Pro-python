"""NVIDIA 模型能力矩阵 + Kilo 集成配置 测试。

覆盖：
- 全量注册表完整性（每个模型 id 唯一、category 合法、关键字段非空）
- build_nvidia_payload 输出结构（无 extra_body、thinking 字段拍平顶层）
- list_by_category / get_video_model / get_embed_model / get_agent_model 查询
- detect_provider 分类
- KILO_MODELS 注册表

真实 PONG 调用用 mark.real_nvidia 标注，CI 默认 skip（无 key/无网络时不阻塞）。
"""
import json
import os

import pytest

from src.core import nvidia_models as nm
from src.core.nvidia_models import (
    KILO_MODELS,
    NVIDIA_MODELS,
    NvidiaModel,
    build_nvidia_payload,
    detect_provider,
    get_agent_model,
    get_embed_model,
    get_model_by_id,
    get_video_model,
    list_by_category,
)


# ---- 注册表完整性 ----
class TestRegistryIntegrity:
    def test_all_models_have_unique_ids(self):
        ids = [m.id for m in NVIDIA_MODELS]
        assert len(ids) == len(set(ids)), f"模型 id 重复: {[i for i in ids if ids.count(i)>1]}"

    def test_all_categories_valid(self):
        valid = {"video", "llm", "embed", "rerank", "ocr", "asr", "tts", "safety"}
        for m in NVIDIA_MODELS:
            assert m.category in valid, f"{m.id} category={m.category} 不合法"

    def test_required_models_present(self):
        """用户给的完整清单里的关键模型必须全部登记。"""
        required = {
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "nvidia/nemotron-3-ultra-550b-a55b",
            "nvidia/nemotron-3.5-lightning-30b-a3b",
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3-embed-1b",
            "nvidia/llama-nemotron-rerank-vl-1b-v2",
            "nvidia/nemotron-ocr-v2",
            "nvidia/nemotron-parse",
            "nvidia/parakeet-ctc-0.6b-asr",
            "nvidia/parakeet-ctc-0.6b-zh-cn",
            "nvidia/magpie-tts-zeroshot",
            "nvidia/nemotron-3.5-content-safety",
            "nvidia/nemotron-nano-12b-v2-vl",
            "nvidia/cosmos3-nano-reasoner",
        }
        present = {m.id for m in NVIDIA_MODELS}
        missing = required - present
        assert not missing, f"缺失模型: {sorted(missing)}"

    def test_video_models_support_video_flag(self):
        for m in list_by_category("video"):
            assert m.supports_video, f"{m.id} category=video 但 supports_video=False"

    def test_no_extra_body_or_thinking_nested_keys_in_payload(self):
        """关键：payload 不能出现 OpenAI SDK 风格的 extra_body。"""
        p = build_nvidia_payload(
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            [{"role": "user", "content": "hi"}],
        )
        assert "extra_body" not in p, "raw REST payload 不应含 extra_body 字段"

    def test_frozen_dataclass_immutable(self):
        """NvidiaModel frozen=True，运行时改字段应抛 FrozenInstanceError。"""
        m = NVIDIA_MODELS[0]
        with pytest.raises(Exception):
            m.id = "hacked"  # type: ignore[misc]

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError):
            NvidiaModel(id="x", name="x", category="bogus")


# ---- build_nvidia_payload 结构 ----
class TestBuildPayload:
    def test_thinking_flattened_to_top_when_enabled(self):
        p = build_nvidia_payload(
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            [{"role": "user", "content": "PONG"}],
            enable_thinking=True,
            reasoning_budget=4096,
        )
        # chat_template_kwargs 与 reasoning_budget 必须在顶层
        assert "chat_template_kwargs" in p
        assert p["chat_template_kwargs"] == {"thinking": True}
        assert p["reasoning_budget"] == 4096
        # 不能嵌套在 extra_body 里
        assert "extra_body" not in p

    def test_thinking_omitted_when_disabled(self):
        p = build_nvidia_payload(
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            [{"role": "user", "content": "hi"}],
            enable_thinking=False,
        )
        assert "chat_template_kwargs" not in p
        assert "reasoning_budget" not in p

    def test_thinking_skipped_for_non_thinking_model(self):
        """nemotron-ocr-v2 不支持思考，enable_thinking=True 也不该加字段。"""
        p = build_nvidia_payload(
            "nvidia/nemotron-ocr-v2",
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,x"}}]}],
            enable_thinking=True,
        )
        assert "chat_template_kwargs" not in p
        assert "reasoning_budget" not in p

    def test_basic_fields_present(self):
        p = build_nvidia_payload(
            "nvidia/nemotron-3-ultra-550b-a55b",
            [{"role": "user", "content": "hi"}],
            max_tokens=1024,
            temperature=0.5,
            stream=False,
        )
        assert p["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
        assert p["messages"] == [{"role": "user", "content": "hi"}]
        assert p["max_tokens"] == 1024
        assert p["temperature"] == 0.5
        assert p["stream"] is False

    def test_extra_merged_to_top(self):
        p = build_nvidia_payload(
            "nvidia/nemotron-3-ultra-550b-a55b",
            [{"role": "user", "content": "hi"}],
            extra={"top_p": 0.9, "frequency_penalty": 0.1},
        )
        assert p["top_p"] == 0.9
        assert p["frequency_penalty"] == 0.1

    def test_extra_cannot_override_thinking_fields(self):
        """extra 试图覆盖受保护的思考链字段应被忽略并告警。"""
        p = build_nvidia_payload(
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            [{"role": "user", "content": "hi"}],
            enable_thinking=True,
            reasoning_budget=8192,
            extra={"chat_template_kwargs": {"thinking": False}, "reasoning_budget": 1},
        )
        # 原值不被覆盖
        assert p["chat_template_kwargs"] == {"thinking": True}
        assert p["reasoning_budget"] == 8192

    def test_unknown_model_defaults_to_thinking_on(self):
        """未知模型 id（未在注册表），仍按调用者意图放思考字段。"""
        p = build_nvidia_payload(
            "nvidia/some-future-model-not-registered",
            [{"role": "user", "content": "hi"}],
            enable_thinking=True,
        )
        assert "chat_template_kwargs" in p


# ---- 便捷查询 ----
class TestQueries:
    def test_list_by_category_video(self):
        vids = list_by_category("video")
        assert len(vids) >= 2
        assert all(m.supports_video for m in vids)

    def test_get_video_model_prefers_omni(self):
        m = get_video_model()
        assert m is not None
        assert "omni" in m.id
        assert m.supports_video is True
        assert m.supports_thinking is True

    def test_get_embed_model(self):
        m = get_embed_model()
        assert m is not None
        assert m.category == "embed"
        assert m.max_output == 2048

    def test_get_agent_model_prefers_ultra(self):
        m = get_agent_model()
        assert m is not None
        assert "ultra" in m.id
        assert m.max_context == 1000000

    def test_get_model_by_id_hit_and_miss(self):
        assert get_model_by_id("nvidia/nemotron-3-embed-1b") is not None
        assert get_model_by_id("nvidia/nope") is None


# ---- Kilo ----
class TestKilo:
    def test_kilo_models_have_free_suffix(self):
        for m in KILO_MODELS:
            assert m.id.endswith(":free"), f"{m.id} 缺 :free 后缀"

    def test_kilo_no_video_models(self):
        """Kilo 无视频能力，category 不应为 video。"""
        for m in KILO_MODELS:
            assert m.category != "video"

    def test_kilo_ultra_has_1m_context(self):
        ultra = next(m for m in KILO_MODELS if "ultra" in m.id)
        assert ultra.max_context == 1000000


# ---- detect_provider ----
class TestDetectProvider:
    @pytest.mark.parametrize("url,expected", [
        ("https://integrate.api.nvidia.com/v1", "nvidia"),
        ("https://integrate.api.nvidia.com/v1/chat/completions", "nvidia"),
        ("https://api.kilo.ai/v1", "kilo"),
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://api.example.com/v1", "unknown"),
        ("", "unknown"),
    ])
    def test_classify(self, url, expected):
        assert detect_provider(url) == expected


# ---- 真实 PONG 调用（可选，需 key+网络；CI 默认 skip）----
@pytest.mark.real_nvidia
class TestRealNvidiaPong:
    """用 1 次真实调用验证 payload 结构正确。仅在 VAP_NV_API_KEY 存在时跑。"""

    @pytest.fixture(autouse=True)
    def _skip_if_no_key(self):
        if not (os.environ.get("VAP_NV_API_KEY") or os.environ.get("NVIDIA_API_KEY")):
            pytest.skip("无 VAP_NV_API_KEY，跳过真实 NVIDIA PONG 调用")

    def test_pong_returns_non_empty(self, caplog):
        import requests as req
        key = os.environ.get("VAP_NV_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        p = build_nvidia_payload(
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            [{"role": "user", "content": "Reply with exactly: PONG"}],
            enable_thinking=True,
            reasoning_budget=2048,
            max_tokens=64,
            stream=False,
        )
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                   "Accept": "application/json"}
        resp = req.post(nm.NVIDIA_CHAT_ENDPOINT, headers=headers, json=p, timeout=60)
        # 允许 429/限流（仍算 payload 结构通过：服务器识别了请求）
        if resp.status_code == 429:
            pytest.skip(f"NVIDIA 限流 429（payload 已被服务器接受，结构正确）: {resp.text[:200]}")
        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"
        data = resp.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        assert content, f"空响应: {json.dumps(data)[:500]}"
