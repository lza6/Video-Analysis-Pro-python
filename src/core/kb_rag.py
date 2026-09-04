"""跨视频知识库 RAG 接入 (v5.5 T6)

把 HistoryManager 的 ChromaDB 全局知识库（kb_frames collection，CLIP 视觉 embedding）
接到 Kilo LLM 做 RAG 问答。Agent 通道：agent_tools.search_kb 返回原始片段，
KnowledgeBaseRAG 进一步用 Kilo 把片段组装成自然语言回答。

Embedding 策略:
  - 默认复用 kb_indexer.get_embedder()（CLIP clip-ViT-B-32，512 维，与现有 KB 一致）
  - 可选注入 Kilo embed（nemotron-3-embed-1b）做纯文本 RAG，但维度不同需独立
    collection — 本版暂不启用，留接口（embed_model 参数）。

禁区（只读接缝）:
  - 不改 history_manager / kb_indexer / agent_tools / llm_gateway
  - index_video 复用 history_manager.add_frame_to_kb（与 kb_indexer.index_frames 同一接口）
  - query 复用 history_manager.search_kb（与 agent_tools.search_kb 同一接口）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger("VideoAnalyzerCore")

# RAG prompt 模板（组装检索片段 → LLM 问答）
_RAG_SYSTEM = (
    "你是视频分析知识库助手。根据下方检索到的视频片段信息回答用户问题。"
    "若片段不足以回答，直接说明知识库中无相关信息，不要编造。"
)
_RAG_TEMPLATE = (
    "【检索到的视频片段】\n{context}\n\n【用户问题】\n{question}\n\n"
    "请基于片段信息作答，引用片段时标注视频名和时间戳。"
)


@dataclass
class FrameRef:
    """轻量帧引用（避免依赖 logic.py 的 FrameInfo，便于测试）。"""
    path: str
    timestamp: float
    vision_content: str = ""
    ocr_text: str = ""
    video_name: Optional[str] = None
    video_path: Optional[str] = None


class KnowledgeBaseRAG:
    """跨视频知识库 RAG。

    用法:
      rag = KnowledgeBaseRAG(history_manager, kilo_client)
      rag.index_video("run-xyz", frames, transcript="...")
      answer = rag.query("哪些视频出现过红色跑车？")
    """

    def __init__(self, history_manager: Any,
                 kilo_client: Any = None,
                 embedder: Any = None,
                 embed_model: Optional[str] = None) -> None:
        self.hm = history_manager
        self.kilo = kilo_client
        self._embedder = embedder  # 可注入（测试 mock）
        self._embed_model = embed_model  # Kilo embed 模型名（可选，本版未启用）

    # ---- embedder 解析 ----
    def _get_embedder(self) -> Any:
        """优先注入的 embedder，其次复用 kb_indexer.get_embedder()。"""
        if self._embedder is not None:
            return self._embedder
        try:
            from src.core.kb_indexer import get_embedder
            return get_embedder()
        except Exception as e:
            logger.warning(f"[kb_rag] 本地 embedder 不可用: {e}")
            return None

    # ---- 索引 ----
    def index_video(self, run_id: str, frames: List[Any],
                    transcript: str = "") -> int:
        """把视频关键帧写入全局 KB（复用 history_manager.add_frame_to_kb）。

        frames: FrameInfo / FrameRef 列表，需有 path/timestamp/vision_content/ocr_text。
        transcript: 整段转录文本（存入 session summary，不单独 embed）。
        返回成功写入条数。
        """
        if not frames:
            return 0
        embedder = self._get_embedder()
        if embedder is None:
            logger.warning("[kb_rag] embedder 不可用，跳过索引")
            return 0

        # 推断 video_name / video_path（首帧属性兜底）
        first = frames[0]
        video_name = getattr(first, "video_name", None) or run_id
        video_path = getattr(first, "video_path", None) or ""

        # 尝试 Kilo embed（若启用且 embedder 是 KiloClient）—— 本版默认走本地 CLIP
        # 保持与现有 kb_frames collection 维度一致

        indexed = 0
        for f in frames:
            try:
                from PIL import Image
                img = Image.open(str(f.path)).convert("RGB")
                emb = embedder.encode(img, convert_to_tensor=False)
            except Exception as e:
                logger.warning(f"[kb_rag] 跳过帧 {getattr(f, 'path', '?')}: {e}")
                continue
            emb_arr = np.asarray(emb, dtype=np.float32)
            ok = self.hm.add_frame_to_kb(
                session_id=run_id,
                video_name=str(video_name),
                video_path=str(video_path),
                timestamp=float(getattr(f, "timestamp", 0.0)),
                content=getattr(f, "vision_content", "") or getattr(f, "ocr_text", ""),
                embedding=emb_arr,
                ocr_text=getattr(f, "ocr_text", ""),
            )
            if ok:
                indexed += 1

        # transcript 存入 session summary（hm 支持 update_session_summary）
        if transcript:
            try:
                self.hm.update_session_summary(run_id, transcript[:1000])
            except Exception as e:
                logger.debug(f"[kb_rag] transcript 存档失败（非致命）: {e}")

        logger.info(f"[kb_rag] run={run_id}: {indexed}/{len(frames)} 帧已索引")
        return indexed

    # ---- 查询 ----
    def query(self, question: str, run_id: Optional[str] = None,
              top_k: int = 8) -> str:
        """检索相关片段 + Kilo LLM 组装回答。

        run_id: 限定单视频范围（None=跨视频全局检索）。
        返回自然语言回答；Kilo 不可用时返回原始片段列表。
        """
        embedder = self._get_embedder()
        if embedder is None:
            return "[kb_rag] embedder 不可用，无法检索"
        try:
            q_emb = embedder.encode(question, convert_to_tensor=False)
        except Exception as e:
            return f"[kb_rag] query embed 失败: {e}"
        q_arr = np.asarray(q_emb, dtype=np.float32)

        hits = self.hm.search_kb(q_arr, top_k=top_k)
        if run_id:
            hits = [h for h in hits if h.get("session_id") == run_id]
        if not hits:
            return "知识库中没有匹配结果。可能尚未索引任何视频，或描述差异过大。"

        # 组装 context（与 agent_tools.search_kb 相同的展示格式，便于复用）
        ctx_lines = []
        for i, h in enumerate(hits, 1):
            ctx_lines.append(
                f"{i}. {h.get('video_name', '?')} @ "
                f"{float(h.get('timestamp', 0.0)):.2f}s "
                f"(匹配度 {float(h.get('score', 0.0)):.2f})\n"
                f"   内容: {h.get('content', '')[:200]}"
            )
        context = "\n".join(ctx_lines)

        # 无 LLM → 返回原始片段（与 search_kb 工具行为一致）
        if self.kilo is None:
            return context

        prompt = _RAG_TEMPLATE.format(context=context, question=question)
        try:
            answer = self.kilo.chat(
                messages=[{"role": "user", "content": prompt}],
                system=_RAG_SYSTEM,
            )
            return answer
        except Exception as e:
            logger.warning(f"[kb_rag] Kilo 问答失败，回退原始片段: {e}")
            return context
