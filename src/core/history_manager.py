import sqlite3
import json
import shutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

class HistoryManager:
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.config_dir / "history.db"
        self._init_db()
        self.chroma_client = None
        if CHROMA_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(path=str(self.config_dir / "chroma_db"))
            except Exception as e:
                logging.error(f"ChromaDB initialization failed: {e}")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # History table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    video_path TEXT,
                    video_name TEXT,
                    output_dir TEXT,
                    summary TEXT,
                    status TEXT DEFAULT 'completed'
                )
            ''')
            # Checkpoints table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    session_id TEXT,
                    last_processed_second REAL,
                    data TEXT,
                    PRIMARY KEY(session_id)
                )
            ''')
            conn.commit()

    def save_checkpoint(self, session_id: str, second: float, data: dict):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO checkpoints (session_id, last_processed_second, data) VALUES (?, ?, ?)",
                (session_id, second, json.dumps(data, ensure_ascii=False))
            )
            conn.commit()

    def get_checkpoint(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_processed_second, data FROM checkpoints WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                return {"second": row[0], "data": json.loads(row[1])}
        return None

    def add_frame_to_memory(self, session_id: str, timestamp: float, content: str, embedding: np.ndarray):
        """将帧信息存入 ChromaDB 以后续进行毫秒级语义搜索。"""
        if not self.chroma_client: return

        try:
            collection = self.chroma_client.get_or_create_collection(name=f"session_{session_id}")
            collection.add(
                ids=[f"frame_{timestamp}"],
                embeddings=[embedding.tolist()],
                metadatas=[{"timestamp": timestamp, "content": content}],
                documents=[content]
            )
        except Exception as e:
            logging.error(f"Failed to add to ChromaDB: {e}")

    # ------------------------------------------------------------------
    # v4.5 跨视频知识库 (Global Knowledge Base)
    # 单一会话级 collection 之上新增一个全局 collection，使语义搜索可以
    # 跨越所有已分析的视频（"帮我找找过去一年所有视频里出现过的红色跑车"）。
    # 复用同一个 PersistentClient，不引入新的向量库。
    # ------------------------------------------------------------------
    KB_COLLECTION_NAME = "kb_frames"

    def add_frame_to_kb(self, session_id: str, video_name: str, video_path: str,
                        timestamp: float, content: str, embedding: np.ndarray,
                        ocr_text: str = "") -> bool:
        """把一帧写入全局知识库，metadata 携带会话与视频信息用于跳转与清理。"""
        if not self.chroma_client:
            return False
        try:
            collection = self.chroma_client.get_or_create_collection(name=self.KB_COLLECTION_NAME)
            # 同一会话可能分析多个视频，相同时间戳（0.0s/1.0s…）会互相
            # upsert 覆盖 → frame_id 必须含视频名（消毒后）
            import re as _re
            safe_vname = _re.sub(r"[^\w]", "_", video_name)[:40]
            frame_id = f"{session_id}_{safe_vname}_{timestamp:.6f}"
            collection.upsert(
                ids=[frame_id],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "session_id": session_id,
                    "video_name": video_name,
                    "video_path": video_path,
                    "timestamp": float(timestamp),
                    "content": (content or "")[:500],
                    "ocr_text": (ocr_text or "")[:500],
                }],
                documents=[(content or "")[:1000]],
            )
            return True
        except Exception as e:
            logging.error(f"Failed to add frame to KB: {e}")
            return False

    def search_kb(self, query_embedding: np.ndarray, top_k: int = 8,
                  min_score: float = 0.25) -> list:
        """跨视频语义搜索。返回 [{video_name, video_path, timestamp, score, content}]"""
        if not self.chroma_client:
            return []
        try:
            collection = self.chroma_client.get_collection(name=self.KB_COLLECTION_NAME)
        except Exception:
            return []  # 知识库为空（尚未建立）属正常场景
        try:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            logging.error(f"KB search failed: {e}")
            return []

        hits = []
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for meta, dist in zip(metadatas, distances):
            # ChromaDB 默认返回 L2 距离；转换为 0-1 相似度供 UI 展示
            score = max(0.0, 1.0 - float(dist) / 2.0)
            if score < min_score:
                continue
            hits.append({
                "video_name": meta.get("video_name", "Unknown"),
                "video_path": meta.get("video_path", ""),
                "session_id": meta.get("session_id", ""),
                "timestamp": float(meta.get("timestamp", 0.0)),
                "score": round(score, 3),
                "content": meta.get("content", ""),
            })
        return hits

    def kb_count(self) -> int:
        """知识库当前条目数（UI 展示用）。"""
        if not self.chroma_client:
            return 0
        try:
            collection = self.chroma_client.get_collection(name=self.KB_COLLECTION_NAME)
            return collection.count()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # P2-8: 跨视频用户偏好记忆 (user_preferences collection)
    # 记录用户的查询历史/关注点，让 Agent 跨会话更懂用户
    # ------------------------------------------------------------------
    PREFS_COLLECTION_NAME = "user_preferences"

    def remember_preference(self, kind: str, content: str) -> bool:
        """记录一条用户偏好（kind: query/interest/feedback）。"""
        if not self.chroma_client:
            return False
        try:
            from src.core.kb_indexer import get_embedder
            embedder = get_embedder()
            if embedder is None:
                return False
            collection = self.chroma_client.get_or_create_collection(
                name=self.PREFS_COLLECTION_NAME)
            emb = embedder.encode([content], convert_to_tensor=False)[0]
            import time as _t, uuid as _u
            collection.upsert(
                ids=[f"pref_{int(_t.time() * 1000)}_{_u.uuid4().hex[:8]}"],
                embeddings=[emb.tolist()],
                metadatas=[{"kind": kind, "content": content[:300],
                            "created": _t.strftime("%Y-%m-%d %H:%M:%S")}],
                documents=[content[:1000]],
            )
            return True
        except Exception as e:
            logging.error(f"Failed to remember preference: {e}")
            return False

    def recall_preferences(self, query: str, top_k: int = 3) -> list:
        """按语义召回相关偏好，供 Agent 个性化回答。"""
        if not self.chroma_client:
            return []
        try:
            from src.core.kb_indexer import get_embedder
            embedder = get_embedder()
            if embedder is None:
                return []
            collection = self.chroma_client.get_collection(name=self.PREFS_COLLECTION_NAME)
            emb = embedder.encode([query], convert_to_tensor=False)[0]
            res = collection.query(query_embeddings=[emb.tolist()], n_results=top_k)
            return [{"content": m.get("content", ""), "kind": m.get("kind", "")}
                    for m in res.get("metadatas", [[]])[0]]
        except Exception:
            return []

    def semantic_search_frames(self, session_id: str, query_embedding: np.ndarray, top_k: int = 5):
        if not self.chroma_client: return []
        try:
            collection = self.chroma_client.get_collection(name=f"session_{session_id}")
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            return results
        except Exception:
            return []

    def add_session(self, video_path: str, output_dir: str, summary: str = "", status: str = 'completed'):
        # Historical bug: seconds-resolution timestamps collided when two
        # sessions were created within the same second (PRIMARY KEY conflict).
        # A uuid has no such constraint.
        import uuid
        session_id = uuid.uuid4().hex
        timestamp = datetime.utcnow().isoformat()
        video_name = Path(video_path).name
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (id, timestamp, video_path, video_name, output_dir, summary, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, timestamp, str(video_path), video_name, str(output_dir), summary, status)
            )
            conn.commit()
        return session_id

    def update_session_summary(self, session_id: str, summary: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id))
            conn.commit()

    def get_history(self) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY timestamp DESC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_session(self, session_id: str):
        # Historical bug: the KB cleanup sat inside `if row:` — a session that
        # had KB entries but no row in `sessions` (or an already-deleted row)
        # short-circuited and left orphan vectors behind. Chroma cleanup now
        # runs unconditionally, and the function returns True if anything was
        # actually removed.
        deleted_something = False
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT output_dir FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                deleted_something = True
                out_dir = Path(row[0])
                if out_dir.exists() and out_dir.is_dir():
                    try:
                        shutil.rmtree(out_dir)
                    except Exception as e:
                        logging.warning(f"删除会话目录失败 (文件可能被占用): {e}")

                cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                cursor.execute("DELETE FROM checkpoints WHERE session_id = ?", (session_id,))
                conn.commit()

            # Cleanup Chroma: per-session collection + KB entries belonging
            # to this session (so cross-video search never shows ghosts).
            if self.chroma_client:
                try: self.chroma_client.delete_collection(name=f"session_{session_id}")
                except Exception: pass
                try:
                    kb = self.chroma_client.get_collection(name=self.KB_COLLECTION_NAME)
                    kb.delete(where={"session_id": session_id})
                except Exception:
                    pass

        return deleted_something

    def clear_all_history(self):
        history = self.get_history()
        for session in history:
            self.delete_session(session['id'])
        logging.info("All history cleared from SQLite.")

    def cleanup_old_sessions(self, retention_days=7):
        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM sessions WHERE timestamp < ?", (cutoff,))
            old_ids = [row[0] for row in cursor.fetchall()]
            for sid in old_ids:
                self.delete_session(sid)
        logging.info(f"Cleanup complete. Removed {len(old_ids)} old sessions.")
