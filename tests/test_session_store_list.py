"""SessionStore.list_sessions / search_sessions 单元测试。"""
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


def test_list_sessions_empty(store):
    # Arrange
    # 空库

    # Act
    result = store.list_sessions()

    # Assert
    assert result == []


def test_list_sessions_returns_saved(store):
    # Arrange
    s1 = _make_session("sess-1", "内容一")
    s2 = _make_session("sess-2", "内容二")
    store.save(s1)
    time.sleep(1.1)  # updated_at 精确到秒，需跨秒确保倒序
    store.save(s2)

    # Act
    result = store.list_sessions()

    # Assert
    assert len(result) == 2
    assert result[0]["session_id"] == "sess-2"
    assert result[1]["session_id"] == "sess-1"


def test_list_sessions_limit(store):
    # Arrange
    for i in range(5):
        store.save(_make_session(f"sess-{i}", f"内容{i}"))
        time.sleep(1.1)  # 跨秒确保 updated_at 倒序可分

    # Act
    result = store.list_sessions(limit=3)

    # Assert
    assert len(result) == 3


def test_search_sessions_hit(store):
    # Arrange
    s = _make_session("sess-parking", "发现停车场里有车辆")
    store.save(s)

    # Act
    result = store.search_sessions("停车场")

    # Assert
    assert len(result) == 1
    assert result[0]["session_id"] == "sess-parking"
    assert "停车场" in result[0]["snippet"]


def test_search_sessions_miss(store):
    # Arrange
    s = _make_session("sess-1", "无关内容")
    store.save(s)

    # Act
    result = store.search_sessions("不存在的关键词")

    # Assert
    assert result == []


def test_list_sessions_size_field(store):
    # Arrange
    store.save(_make_session("sess-size", "有大小的内容"))

    # Act
    result = store.list_sessions()

    # Assert
    assert len(result) == 1
    assert "size" in result[0]
    assert isinstance(result[0]["size"], int)
    assert result[0]["size"] > 0
