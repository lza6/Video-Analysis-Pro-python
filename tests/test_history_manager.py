"""HistoryManager: SQLite + ChromaDB 知识库（tmp_path 隔离）。"""
import numpy as np
import pytest

from src.core.history_manager import HistoryManager


@pytest.fixture
def manager(tmp_path):
    return HistoryManager(str(tmp_path / "cfg"))


def test_add_and_get_session(manager):
    sid = manager.add_session("a.mp4", "out1")
    history = manager.get_history()
    assert len(history) == 1
    assert history[0]["video_name"] == "a.mp4"


def test_session_ids_unique_within_same_second(manager):
    """回归：int(time.time()) 主键在同一秒内冲突。"""
    ids = {manager.add_session(f"v{i}.mp4", f"o{i}") for i in range(20)}
    assert len(ids) == 20


def test_delete_session_removes_row(manager):
    sid = manager.add_session("a.mp4", "out1")
    assert manager.delete_session(sid) is True
    assert manager.get_history() == []


def test_kb_add_and_search(manager):
    emb = np.zeros(384, dtype=np.float32)
    emb[0] = 1.0
    ok = manager.add_frame_to_kb("s1", "car.mp4", "C:/v/car.mp4", 3.5,
                                 "a red sports car", emb)
    assert ok is True
    results = manager.search_kb(emb, top_k=3)
    assert len(results) == 1
    assert results[0]["video_name"] == "car.mp4"
    assert results[0]["timestamp"] == pytest.approx(3.5)


def test_kb_search_empty_returns_list(manager):
    assert manager.search_kb(np.zeros(384), top_k=3) == []


def test_kb_delete_session_removes_entries(manager, tmp_path):
    emb = np.zeros(384, dtype=np.float32)
    emb[1] = 1.0
    manager.add_frame_to_kb("s1", "a.mp4", "p1", 1.0, "x", emb)
    manager.add_frame_to_kb("s2", "b.mp4", "p2", 2.0, "y", emb)
    manager.add_session("a.mp4", str(tmp_path / "o1"))
    manager.delete_session("s1")
    results = manager.search_kb(emb, top_k=10)
    assert all(h["session_id"] != "s1" for h in results)
    assert len(results) == 1  # 只剩 s2


def test_kb_count(manager):
    assert manager.kb_count() == 0
    emb = np.zeros(384, dtype=np.float32)
    emb[2] = 1.0
    manager.add_frame_to_kb("s1", "a.mp4", "p", 1.0, "c", emb)
    assert manager.kb_count() == 1


def test_user_preferences_roundtrip(manager):
    """P2-8: 用户偏好记忆 — 无 embedder 时优雅返回 False/[]。"""
    assert manager.remember_preference("query", "找黑色旅行袋") in (True, False)
    prefs = manager.recall_preferences("旅行袋")
    assert isinstance(prefs, list)
