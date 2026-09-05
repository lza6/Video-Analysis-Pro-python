"""AgentOrchestrator 跨会话记忆层（load_session_memory / format_memory_text）。

断点 B4 / 改进项 I5.8-agent-4：agent 重启时读 run_store 历史，知道"之前跑到
哪 / 命中过什么"。用真实 RunStore + tmp_path sqlite 隔离，不污染 config/runs.db。
"""
from src.core.agent_orchestrator import (AgentOrchestrator,
                                         format_memory_text)
from src.core.run_store import RunStore


# ----------------------------------------------------------------------
# fixture：每个测试独立 tmp_path，互不污染
# ----------------------------------------------------------------------
def _make_store(tmp_path):
    return RunStore(str(tmp_path / "cfg"))


def _seed_done_run_with_hit(store, video_path="C:/videos/_388.mp4"):
    """构造 1 个 done run + 1 个命中 segment + 1 个 clip。"""
    run_id = store.create_run(
        video_path, duration_sec=600.0,
        model="glm-5.3-flash", provider="openai", mode="surveillance",
    )
    store.add_segment(run_id, {
        "seg_idx": 0, "start_sec": 536.0, "dur_sec": 60.0,
        "status": "ok", "match": 1,
        "confidence": 0.95, "reason": "匹配到目标物品",
        "abs_timestamp": "2026-09-04T10:08:56",
    })
    store.add_hit(run_id, {
        "hit_idx": 0,
        "abs_timestamp": "2026-09-04T10:08:56",
        "clip_path": "C:/clips/_388_hit0.mp4",
    })
    store.update_run(
        run_id,
        segments_total=1, segments_ok=1,
        status="done", finished_at="2026-09-04T10:10:00",
    )
    return run_id


def _seed_running_run(store, video_path="C:/videos/cam02.mp4"):
    """构造 1 个 running run（未完成）。"""
    run_id = store.create_run(
        video_path, duration_sec=1800.0,
        model="glm-5.3-flash", provider="openai", mode="surveillance",
        status="running",
    )
    return run_id


# ----------------------------------------------------------------------
# load_session_memory：done + running + hit 的混合场景
# ----------------------------------------------------------------------
def test_load_session_memory_mixed_done_and_running(tmp_path):
    # Arrange：1 done（含 1 hit）+ 1 running
    store = _make_store(tmp_path)
    _seed_done_run_with_hit(store, "C:/videos/_388.mp4")
    _seed_running_run(store, "C:/videos/cam02.mp4")
    orchestrator = AgentOrchestrator()

    # Act
    memory = orchestrator.load_session_memory(store, limit_runs=5,
                                              limit_hits=10)

    # Assert：未完成 1 个，视频名在列表里
    assert memory["unfinished_count"] == 1
    assert "cam02.mp4" in memory["unfinished_videos"]
    # 最近命中非空，且字段齐全
    assert len(memory["recent_hits"]) >= 1
    hit = memory["recent_hits"][0]
    assert hit["video_name"] == "_388.mp4"
    assert hit["timestamp"] == "2026-09-04T10:08:56"
    assert hit["confidence"] == 0.95
    assert hit["reason"] == "匹配到目标物品"
    # 总数
    assert memory["total_runs"] == 2
    assert memory["total_hits"] == 1


# ----------------------------------------------------------------------
# 空库不崩：全 0 / 空列表
# ----------------------------------------------------------------------
def test_load_session_memory_empty_store(tmp_path):
    store = _make_store(tmp_path)
    orchestrator = AgentOrchestrator()

    memory = orchestrator.load_session_memory(store)

    assert memory == {
        "unfinished_count": 0,
        "unfinished_videos": [],
        "recent_hits": [],
        "total_runs": 0,
        "total_hits": 0,
    }


# ----------------------------------------------------------------------
# run_store=None：返回空记忆，不崩
# ----------------------------------------------------------------------
def test_load_session_memory_none_store():
    orchestrator = AgentOrchestrator()
    memory = orchestrator.load_session_memory(None)
    assert memory["total_runs"] == 0
    assert memory["recent_hits"] == []


# ----------------------------------------------------------------------
# 异常 run_store：返回空记忆，不崩（agent 启动期不能因记忆层报错）
# ----------------------------------------------------------------------
def test_load_session_memory_broken_store():
    class BrokenStore:
        def list_runs(self, *a, **kw):
            raise RuntimeError("db locked")

        def get_run(self, *a, **kw):
            raise RuntimeError("db locked")

    orchestrator = AgentOrchestrator()
    memory = orchestrator.load_session_memory(BrokenStore())
    assert memory["total_runs"] == 0
    assert memory["recent_hits"] == []


# ----------------------------------------------------------------------
# limit_hits 截断：3 个 hit 只取 2 个
# ----------------------------------------------------------------------
def test_load_session_memory_limit_hits_truncates(tmp_path):
    store = _make_store(tmp_path)
    # 造 1 个 run 含 3 个 hit
    run_id = store.create_run("C:/videos/multi.mp4",
                              model="m", provider="p", mode="surveillance")
    for i in range(3):
        store.add_segment(run_id, {
            "seg_idx": i, "start_sec": float(i * 60),
            "dur_sec": 60.0, "status": "ok", "match": 1,
            "confidence": 0.7 + i * 0.1,
            "reason": f"匹配 {i}",
            "abs_timestamp": f"2026-09-04T10:0{i}:00",
        })
        store.add_hit(run_id, {
            "hit_idx": i,
            "abs_timestamp": f"2026-09-04T10:0{i}:00",
            "clip_path": f"C:/clips/hit{i}.mp4",
        })
    store.update_run(run_id, segments_total=3, segments_ok=3,
                     hits_count=3, status="done",
                     finished_at="2026-09-04T10:30:00")

    orchestrator = AgentOrchestrator()
    memory = orchestrator.load_session_memory(store, limit_runs=5,
                                              limit_hits=2)

    assert len(memory["recent_hits"]) == 2
    assert memory["total_hits"] == 3


# ----------------------------------------------------------------------
# format_memory_text：有历史时输出可读文本
# ----------------------------------------------------------------------
def test_format_memory_text_with_history(tmp_path):
    store = _make_store(tmp_path)
    _seed_done_run_with_hit(store, "C:/videos/_388.mp4")
    _seed_running_run(store, "C:/videos/cam02.mp4")
    orchestrator = AgentOrchestrator()
    memory = orchestrator.load_session_memory(store)

    text = format_memory_text(memory)

    assert text.startswith("📌 上次会话记忆：")
    assert "1 个视频未跑完" in text
    assert "_388.mp4" in text
    assert "0.95" in text
    assert "共跑过 2 个视频" in text
    assert "命中 1 次" in text
    # 长度受控
    assert len(text) <= 500


# ----------------------------------------------------------------------
# format_memory_text：空记忆返回首次使用提示
# ----------------------------------------------------------------------
def test_format_memory_text_empty():
    text = format_memory_text({})
    assert text == "📌 首次使用，无历史记忆。"


# ----------------------------------------------------------------------
# format_memory_text：全 done（无未完成）也能渲染
# ----------------------------------------------------------------------
def test_format_memory_text_all_done(tmp_path):
    store = _make_store(tmp_path)
    _seed_done_run_with_hit(store, "C:/videos/_388.mp4")
    orchestrator = AgentOrchestrator()
    memory = orchestrator.load_session_memory(store)

    text = format_memory_text(memory)

    assert "已全部跑完" in text
    assert "_388.mp4" in text
    assert "共跑过 1 个视频" in text
