"""跨视频向量知识库索引器 (v4.5)

Phase 1/2 完成后把关键帧写入 ChromaDB 全局 collection（kb_frames），
供 Agent 的 search_kb 工具与主窗口搜索框进行跨视频语义搜索。
复用 logic.py 已加载的 SentenceTransformer('clip-ViT-B-32') 实例，
避免重复加载模型。
"""
import logging
from typing import Optional

import numpy as np

from src.core.logic import CLIP_AVAILABLE

if CLIP_AVAILABLE:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger("VideoAnalyzerCore")

_shared_embedder: Optional["SentenceTransformer"] = None


def get_embedder() -> Optional["SentenceTransformer"]:
    """进程内共享一个 CLIP embedding 模型（首次调用时加载）。"""
    global _shared_embedder
    if not CLIP_AVAILABLE:
        return None
    if _shared_embedder is None:
        try:
            logger.info("KB indexer: loading clip-ViT-B-32 embedder...")
            _shared_embedder = SentenceTransformer('clip-ViT-B-32')
        except Exception as e:
            logger.error(f"KB indexer: embedder load failed: {e}")
            return None
    return _shared_embedder


def index_frames(history_manager, session_id: str, video_name: str,
                 video_path: str, frames: list, batch_size: int = 64) -> int:
    """把关键帧批量写入全局知识库。返回成功写入的条数。

    帧的视觉内容 embedding 用图像本身；文本占位用 OCR/已识别内容。
    在 QThread 中调用，不要在主线程批量索引大量帧。
    """
    if not frames:
        return 0
    embedder = get_embedder()
    if embedder is None:
        logger.warning("KB indexer: CLIP 不可用，跳过知识库索引。")
        return 0

    indexed = 0
    # 分批 encode 以控制内存峰值
    for start in range(0, len(frames), batch_size):
        chunk = frames[start:start + batch_size]
        try:
            # SentenceTransformer 只接受 str / PIL.Image / np.ndarray；
            # Frame.path 是 Path，必须转成 str，否则 encode 抛
            # "Unsupported input type: WindowsPath"。
            embeddings = embedder.encode(
                [str(f.path) for f in chunk],
                convert_to_tensor=False,
            )
        except Exception as e:
            logger.error(f"KB indexer: batch encode failed: {e}")
            continue

        for frame, emb in zip(chunk, embeddings):
            ok = history_manager.add_frame_to_kb(
                session_id=session_id,
                video_name=video_name,
                video_path=str(video_path),
                timestamp=frame.timestamp,
                content=frame.vision_content or frame.ocr_text or "",
                embedding=np.asarray(emb),
                ocr_text=frame.ocr_text or "",
            )
            if ok:
                indexed += 1

    logger.info(f"KB indexer: {indexed}/{len(frames)} 帧已入知识库。")
    return indexed
