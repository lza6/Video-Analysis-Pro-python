"""RunStore checkpoint/restore：长程断点续跑能力（tmp_path 隔离，不污染 config/runs.db）。

AAA 模式：Arrange 准备 fixture / Act 调用方法 / Assert 验证 phase/payload/ts 还原。
"""
import time

from src.core.run_store import RunStore


# ----------------------------------------------------------------------
# fixture：每个测试独立 tmp_path，互不污染
# ----------------------------------------------------------------------
def _make_store(tmp_path):
    return RunStore(str(tmp_path / "cfg"))


def _seed_run(store, video_path="C:/videos/cam01_20260904.mp4"):
    """构造一条最小 run（started 状态），返回 run_id。"""
    return store.create_run(
        video_path,
        duration_sec=3600.0,
        model="qwen3-vl-flash",
        provider="openai",
        mode="surveillance",
    )


# ----------------------------------------------------------------------
# checkpoint + restore：写入后能读回 phase
# ----------------------------------------------------------------------
def test_checkpoint_save_and_restore(tmp_path):
    # Arrange
    store = _make_store(tmp_path)
    run_id = _seed_run(store)

    # Act
    store.checkpoint(run_id, "plan", {"steps": 3})
    restored = store.restore(run_id)

    # Assert：phase 原样读回
    assert restored is not None
    assert restored["phase"] == "plan"
    assert restored["payload"] == {"steps": 3}
    assert "ts" in restored and restored["ts"]


# ----------------------------------------------------------------------
# restore 不存在的 run：返回 None
# ----------------------------------------------------------------------
def test_restore_missing_run_returns_none(tmp_path):
    store = _make_store(tmp_path)

    assert store.restore("不存在的-run-id") is None


# ----------------------------------------------------------------------
# list_checkpoints：按 ts 倒序
# ----------------------------------------------------------------------
def test_list_checkpoints_order(tmp_path):
    # Arrange
    store = _make_store(tmp_path)
    run_id = _seed_run(store)

    # Act：写 3 个 checkpoint，秒精度时间戳要 sleep 保证倒序
    store.checkpoint(run_id, "phase_a", {"i": 1})
    time.sleep(1.1)
    store.checkpoint(run_id, "phase_b", {"i": 2})
    time.sleep(1.1)
    store.checkpoint(run_id, "phase_c", {"i": 3})
    cps = store.list_checkpoints(run_id)

    # Assert：3 条全在，按 ts 倒序（phase_c 最近在前）
    assert len(cps) == 3
    assert [c["phase"] for c in cps] == ["phase_c", "phase_b", "phase_a"]
    # 每条结构完整
    for c in cps:
        assert set(c.keys()) == {"phase", "payload", "ts"}


# ----------------------------------------------------------------------
# clear_checkpoints：清空后 list 返回空
# ----------------------------------------------------------------------
def test_clear_checkpoints(tmp_path):
    # Arrange
    store = _make_store(tmp_path)
    run_id = _seed_run(store)
    store.checkpoint(run_id, "plan", {"steps": 3})
    store.checkpoint(run_id, "exec", {"done": 1})
    assert len(store.list_checkpoints(run_id)) == 2

    # Act
    store.clear_checkpoints(run_id)

    # Assert：list 空，restore 也 None
    assert store.list_checkpoints(run_id) == []
    assert store.restore(run_id) is None


# ----------------------------------------------------------------------
# 同 run 同 phase 写 2 次：restore 返回最新（追加不覆盖语义）
# ----------------------------------------------------------------------
def test_checkpoint_overwrite_latest(tmp_path):
    # Arrange
    store = _make_store(tmp_path)
    run_id = _seed_run(store)
    store.checkpoint(run_id, "plan", {"version": 1})
    time.sleep(1.1)
    store.checkpoint(run_id, "plan", {"version": 2})

    # Act
    restored = store.restore(run_id)

    # Assert：返回最新的一条（version=2）
    assert restored is not None
    assert restored["phase"] == "plan"
    assert restored["payload"] == {"version": 2}


# ----------------------------------------------------------------------
# payload 含嵌套 dict：restore 能还原
# ----------------------------------------------------------------------
def test_checkpoint_payload_dict(tmp_path):
    # Arrange
    store = _make_store(tmp_path)
    run_id = _seed_run(store)
    nested_payload = {
        "outer": {
            "inner": {
                "list": [1, 2, 3],
                "flag": True,
                "null_field": None,
            },
            "name": "监控分析",
        },
        "count": 5,
    }

    # Act
    store.checkpoint(run_id, "extract", nested_payload)
    restored = store.restore(run_id)

    # Assert：嵌套 dict 完整还原（含中文/None/bool/list）
    assert restored is not None
    assert restored["payload"] == nested_payload
    assert restored["payload"]["outer"]["inner"]["list"] == [1, 2, 3]
    assert restored["payload"]["outer"]["inner"]["null_field"] is None
    assert restored["payload"]["outer"]["inner"]["flag"] is True
    assert restored["payload"]["outer"]["name"] == "监控分析"
