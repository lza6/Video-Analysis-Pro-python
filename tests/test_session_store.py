"""SessionStore 验收测试：list_sessions / search_sessions / save / delete。

覆盖验收标准：
  - save 后 list 能查到
  - delete 后查不到
  - search 命中 / 不命中
  - 空库返回 []
"""
import time

import pytest

from src.core.agent.session import Session, SessionStore, SessionEvent


def _make_session(session_id: str, content: str = "默认内容") -> Session:
    """造一个带 user 事件的 Session。"""
    s = Session(session_id)
    s.append(SessionEvent("user", time.time(), {"content": content}))
    return s


@pytest.fixture()
def store(tmp_path):
    """临时 db 的 SessionStore。"""
    return SessionStore(str(tmp_path / "test_sessions.db"))


def test_empty_db_list_returns_empty(store):
    # Arrange
    # 空库

    # Act
    result = store.list_sessions()

    # Assert
    assert result == []


def test_save_then_list_contains_session(store):
    # Arrange
    s = _make_session("sess-1", "关于停车场的分析")
    store.save(s)

    # Act
    result = store.list_sessions()

    # Assert
    assert len(result) == 1
    assert result[0]["session_id"] == "sess-1"
    assert result[0]["size"] > 0


def test_delete_then_list_excludes_session(store):
    # Arrange
    store.save(_make_session("sess-a", "内容A"))
    store.save(_make_session("sess-b", "内容B"))

    # Act
    store.delete("sess-a")
    result = store.list_sessions()

    # Assert
    assert len(result) == 1
    assert result[0]["session_id"] == "sess-b"
    # 删除后 load 也应返回 None
    assert store.load("sess-a") is None


def test_search_hit(store):
    # Arrange
    store.save(_make_session("sess-parking", "发现停车场里有车辆"))

    # Act
    result = store.search_sessions("停车场")

    # Assert
    assert len(result) == 1
    assert result[0]["session_id"] == "sess-parking"
    assert "停车场" in result[0]["snippet"]


def test_search_miss(store):
    # Arrange
    store.save(_make_session("sess-1", "无关内容"))

    # Act
    result = store.search_sessions("不存在的关键词")

    # Assert
    assert result == []


def test_search_empty_db_returns_empty(store):
    # Act
    result = store.search_sessions("关键词")

    # Assert
    assert result == []
