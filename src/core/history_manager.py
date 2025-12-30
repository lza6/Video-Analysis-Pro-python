import sqlite3
import json
import os
import shutil
import time
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

    def semantic_search_frames(self, session_id: str, query_embedding: np.ndarray, top_k: int = 5):
        if not self.chroma_client: return []
        try:
            collection = self.chroma_client.get_collection(name=f"session_{session_id}")
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            return results
        except:
            return []

    def add_session(self, video_path: str, output_dir: str, summary: str = "", status: str = 'completed'):
        session_id = str(int(time.time()))
        timestamp = datetime.now().isoformat()
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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT output_dir FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                out_dir = Path(row[0])
                if out_dir.exists() and out_dir.is_dir():
                    try:
                        shutil.rmtree(out_dir)
                    except: pass
                
                # Cleanup Chroma
                if self.chroma_client:
                    try: self.chroma_client.delete_collection(name=f"session_{session_id}")
                    except: pass
                
                cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                cursor.execute("DELETE FROM checkpoints WHERE session_id = ?", (session_id,))
                conn.commit()
                return True
        return False

    def clear_all_history(self):
        history = self.get_history()
        for session in history:
            self.delete_session(session['id'])
        logging.info("All history cleared from SQLite.")

    def cleanup_old_sessions(self, retention_days=7):
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM sessions WHERE timestamp < ?", (cutoff,))
            old_ids = [row[0] for row in cursor.fetchall()]
            for sid in old_ids:
                self.delete_session(sid)
        logging.info(f"Cleanup complete. Removed {len(old_ids)} old sessions.")
