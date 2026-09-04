"""RunStore: 监控视频分析运行记录数据库（tmp_path 隔离，不污染 config/runs.db）。

AAA 模式：Arrange 准备 fixture / Act 调用方法 / Assert 验证字段、级联、清空。
"""
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
# create_run + get_run：字段完整
# ----------------------------------------------------------------------
def test_create_and_get_run_fields_complete(tmp_path):
    # Arrange
    store = _make_store(tmp_path)

    # Act
    run_id = store.create_run(
        "C:/videos/cam01.mp4",
        duration_sec=120.0,
        model="glm-5.3-flash",
        provider="openai",
        mode="surveillance",
    )
    run = store.get_run(run_id)

    # Assert：所有 create_run 写入的字段都原样读回
    assert run is not None
    assert run["run_id"] == run_id
    assert run["video_path"] == "C:/videos/cam01.mp4"
    assert run["video_name"] == "cam01.mp4"
    assert run["duration_sec"] == 120.0
    assert run["status"] == "started"
    assert run["model"] == "glm-5.3-flash"
    assert run["provider"] == "openai"
    assert run["mode"] == "surveillance"
    # 计数字段初始为 0
    assert run["hits_count"] == 0
    assert run["segments_total"] == 0
    assert run["segments_ok"] == 0
    assert run["segments_failed"] == 0
    # 嵌套字段存在且空
    assert run["segments"] == []
    assert run["clips"] == []


# ----------------------------------------------------------------------
# add_segment：3 个 segment 写入并按 seg_idx 排序读回
# ----------------------------------------------------------------------
def test_add_segments_query_back(tmp_path):
    # Arrange
    store = _make_store(tmp_path)
    run_id = _seed_run(store)

    # Act：插入 3 个 segment
    seg_ids = []
    for i in range(3):
        sid = store.add_segment(run_id, {
            "seg_idx": i,
            "start_sec": i * 60.0,
            "dur_sec": 60.0,
            "status": "ok" if i < 2 else "failed",
            "match": 1 if i == 0 else 0,
            "confidence": 0.88 if i == 0 else None,
            "reason": "匹配到目标物品" if i == 0 else "无匹配",
            "abs_timestamp": f"2026-09-04T10:0{i}:00",
            "attempts": 1,
            "elapsed_sec": 2.5,
            "usage_json": '{"prompt_tokens": 100, "completion_tokens": 50}',
            "error": None if i < 2 else "timeout",
        })
        seg_ids.append(sid)

    run = store.get_run(run_id)

    # Assert：3 条 segment 全部写入，按 seg_idx 升序
    segs = run["segments"]
    assert len(segs) == 3
    assert [s["seg_idx"] for s in segs] == [0, 1, 2]
    # 字段完整
    s0 = segs[0]
    assert s0["seg_id"] == seg_ids[0]
    assert s0["run_id"] == run_id
    assert s0["start_sec"] == 0.0
    assert s0["dur_sec"] == 60.0
    assert s0["status"] == "ok"
    assert s0["match"] == 1
    assert s0["confidence"] == 0.88
    assert s0["reason"] == "匹配到目标物品"
    assert s0["attempts"] == 1
    assert s0["elapsed_sec"] == 2.5
    assert s0["usage_json"] == '{"prompt_tokens": 100, "completion_tokens": 50}'
    # failed 段
    assert segs[2]["status"] == "failed"
    assert segs[2]["error"] == "timeout"


# ----------------------------------------------------------------------
# add_hit + add_clip：hits_count 自增、clip 字段完整
# ----------------------------------------------------------------------
def test_add_hit_and_clip(tmp_path):
    # Arrange
    store = _make_store(tmp_path)
    run_id = _seed_run(store)

    # Act：1 hit + 1 单独 clip
    clip_id = store.add_hit(run_id, {
        "hit_idx": 0,
        "abs_timestamp": "2026-09-04T10:00:00",
        "clip_path": "C:/clips/cam01_hit0.mp4",
    })
    extra_clip_id = store.add_clip(
        run_id,
        "C:/clips/cam01_hit0_extra.mp4",
        hit_idx=0,
        abs_timestamp="2026-09-04T10:00:05",
    )

    run = store.get_run(run_id)

    # Assert：hits_count 被 add_hit 递增到 1，add_clip 不递增
    assert run["hits_count"] == 1
    # clips 表有 2 条
    clips = run["clips"]
    assert len(clips) == 2
    # add_hit 写入的 clip 字段完整
    c0 = [c for c in clips if c["clip_id"] == clip_id][0]
    assert c0["run_id"] == run_id
    assert c0["hit_idx"] == 0
    assert c0["abs_timestamp"] == "2026-09-04T10:00:00"
    assert c0["clip_path"] == "C:/clips/cam01_hit0.mp4"
    # add_clip 单独追加
    c1 = [c for c in clips if c["clip_id"] == extra_clip_id][0]
    assert c1["clip_path"] == "C:/clips/cam01_hit0_extra.mp4"


# ----------------------------------------------------------------------
# 完整链路：1 run + 3 segments + 1 hit + 1 clip，get_run 聚合正确
# ----------------------------------------------------------------------
def test_full_roundtrip_run_with_segments_hits_clips(tmp_path):
    # Arrange
    store = _make_store(tmp_path)

    # Act
    run_id = _seed_run(store)
    for i in range(3):
        store.add_segment(run_id, {
            "seg_idx": i,
            "start_sec": i * 30.0,
            "dur_sec": 30.0,
            "status": "ok" if i < 2 else "failed",
            "match": 1 if i == 1 else 0,
        })
    store.add_hit(run_id, {
        "hit_idx": 0,
        "abs_timestamp": "2026-09-04T10:00:30",
        "clip_path": "C:/clips/hit0.mp4",
    })
    # 调用方负责同步 run 计数字段（store 不隐式聚合，防并发竞争）
    store.update_run(
        run_id,
        segments_total=3,
        segments_ok=2,
        segments_failed=1,
        status="done",
        finished_at="2026-09-04T10:05:00",
        vlm_elapsed_sec=120.0,
        total_elapsed_sec=180.0,
    )

    run = store.get_run(run_id)
    progress = store.get_progress(run_id)

    # Assert：聚合后所有信息齐全
    assert run["status"] == "done"
    assert run["segments_total"] == 3
    assert run["segments_ok"] == 2
    assert run["segments_failed"] == 1
    assert run["finished_at"] == "2026-09-04T10:05:00"
    assert run["vlm_elapsed_sec"] == 120.0
    assert run["total_elapsed_sec"] == 180.0
    assert len(run["segments"]) == 3
    assert len(run["clips"]) == 1
    assert run["clips"][0]["clip_path"] == "C:/clips/hit0.mp4"
    # 进度
    assert progress == {"total": 3, "done": 3, "hits": 1, "failed": 1}


# ----------------------------------------------------------------------
# delete_run：级联清理 segments + clips
# ----------------------------------------------------------------------
def test_delete_run_cascades_segments_and_clips(tmp_path):
    # Arrange
    store = _make_store(tmp_path)
    run_id = _seed_run(store)
    store.add_segment(run_id, {"seg_idx": 0, "start_sec": 0.0, "dur_sec": 60.0})
    store.add_segment(run_id, {"seg_idx": 1, "start_sec": 60.0, "dur_sec": 60.0})
    store.add_hit(run_id, {
        "hit_idx": 0,
        "abs_timestamp": "2026-09-04T10:00:00",
        "clip_path": "C:/clips/hit0.mp4",
    })

    # Act
    deleted = store.delete_run(run_id)
    run_after = store.get_run(run_id)

    # Assert：run 删了，segments/clips 级联删了
    assert deleted is True
    assert run_after is None


def test_delete_run_returns_false_for_missing(tmp_path):
    store = _make_store(tmp_path)
    assert store.delete_run("nonexistent-run-id") is False


# ----------------------------------------------------------------------
# delete_run purge_files=True：删磁盘 clip 文件
# ----------------------------------------------------------------------
def test_delete_run_purge_files_removes_clip_on_disk(tmp_path):
    # Arrange
    store = _make_store(tmp_path)
    run_id = _seed_run(store)
    # 真实创建一个 clip 文件
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_file = clip_dir / "hit0.mp4"
    clip_file.write_text("fake clip content")
    assert clip_file.exists()
    store.add_hit(run_id, {
        "hit_idx": 0,
        "abs_timestamp": "2026-09-04T10:00:00",
        "clip_path": str(clip_file),
    })

    # Act
    deleted = store.delete_run(run_id, purge_files=True)

    # Assert：DB 行删了 + 磁盘文件删了
    assert deleted is True
    assert not clip_file.exists()


# ----------------------------------------------------------------------
# clear_all：清空所有 run
# ----------------------------------------------------------------------
def test_clear_all_empties_database(tmp_path):
    # Arrange
    store = _make_store(tmp_path)
    r1 = _seed_run(store, "cam01.mp4")
    r2 = _seed_run(store, "cam02.mp4")
    r3 = _seed_run(store, "cam03.mp4")
    for rid in (r1, r2, r3):
        store.add_segment(rid, {"seg_idx": 0, "start_sec": 0.0, "dur_sec": 60.0})
        store.add_hit(rid, {
            "hit_idx": 0,
            "abs_timestamp": "2026-09-04T10:00:00",
            "clip_path": f"C:/clips/{rid}.mp4",
        })

    # Act
    count = store.clear_all()

    # Assert：3 条 run 全部删，list_runs 空，segments/clips 也级联空
    assert count == 3
    assert store.list_runs() == []
    assert store.get_run(r1) is None
    assert store.get_run(r2) is None
    assert store.get_run(r3) is None


# ----------------------------------------------------------------------
# list_runs：按 started_at 倒序 + status 过滤
# ----------------------------------------------------------------------
def test_list_runs_orders_by_started_at_desc_and_filters_status(tmp_path):
    # Arrange
    store = _make_store(tmp_path)
    import time
    r1 = store.create_run("a.mp4")
    time.sleep(1.1)  # 秒精度时间戳，确保倒序
    r2 = store.create_run("b.mp4")
    time.sleep(1.1)
    r3 = store.create_run("c.mp4")
    store.update_run(r1, status="done", finished_at="2026-09-04T10:00:00")
    store.update_run(r2, status="failed", finished_at="2026-09-04T10:01:00")

    # Act
    all_runs = store.list_runs()
    done_runs = store.list_runs(status="done")
    failed_runs = store.list_runs(status="failed")

    # Assert：全量倒序（c 最晚在前，a 最后）
    assert [r["run_id"] for r in all_runs] == [r3, r2, r1]
    # status 过滤
    assert len(done_runs) == 1
    assert done_runs[0]["run_id"] == r1
    assert len(failed_runs) == 1
    assert failed_runs[0]["run_id"] == r2


# ----------------------------------------------------------------------
# update_run：白名单过滤（未知字段不报错但也不写入）
# ----------------------------------------------------------------------
def test_update_run_ignores_unknown_fields(tmp_path):
    store = _make_store(tmp_path)
    run_id = _seed_run(store)

    # Act：传入未知字段，应被白名单过滤，不报错
    ok = store.update_run(
        run_id,
        status="running",
        bogus_field="should_be_ignored",
    )

    # Assert：合法字段写入，非法字段忽略
    assert ok is True
    run = store.get_run(run_id)
    assert run["status"] == "running"
    assert "bogus_field" not in run


def test_update_run_invalid_status_raises(tmp_path):
    store = _make_store(tmp_path)
    run_id = _seed_run(store)
    try:
        store.update_run(run_id, status="bogus")
        assert False, "应抛 ValueError"
    except ValueError as e:
        assert "status" in str(e)


# ----------------------------------------------------------------------
# run_id 全局唯一（同秒批量创建不冲突）
# ----------------------------------------------------------------------
def test_run_ids_unique_within_same_second(tmp_path):
    """回归：uuid4 hex 主键，同秒批量创建 N 条不冲突。"""
    store = _make_store(tmp_path)
    ids = {store.create_run(f"v{i}.mp4") for i in range(20)}
    assert len(ids) == 20


# ----------------------------------------------------------------------
# get_progress：run 不存在返回 None
# ----------------------------------------------------------------------
def test_get_progress_missing_run_returns_none(tmp_path):
    store = _make_store(tmp_path)
    assert store.get_progress("nonexistent") is None


# ----------------------------------------------------------------------
# M3: first_token_ms 字段（首字耗时）
# ----------------------------------------------------------------------
def test_add_segment_persists_first_token_ms(tmp_path):
    """add_segment 写入 first_token_ms，get_run 原样读回。"""
    store = _make_store(tmp_path)
    run_id = _seed_run(store)
    store.add_segment(run_id, {
        "seg_idx": 0,
        "start_sec": 0.0,
        "dur_sec": 60.0,
        "status": "ok",
        "match": 1,
        "first_token_ms": 320,
        "elapsed_sec": 2.5,
    })
    run = store.get_run(run_id)
    segs = run["segments"]
    assert len(segs) == 1
    assert segs[0]["first_token_ms"] == 320


def test_add_segment_first_token_ms_optional(tmp_path):
    """不传 first_token_ms 时列存在但值为 None（老调用方兼容）。"""
    store = _make_store(tmp_path)
    run_id = _seed_run(store)
    store.add_segment(run_id, {
        "seg_idx": 0,
        "start_sec": 0.0,
        "dur_sec": 60.0,
        "status": "ok",
    })
    run = store.get_run(run_id)
    assert run["segments"][0]["first_token_ms"] is None


def test_legacy_db_auto_migrates_first_token_ms_column(tmp_path):
    """旧库（segments 表缺 first_token_ms 列）被 RunStore 打开时自动补列。

    模拟 M2 时代的库结构：segments 表无 first_token_ms 列。RunStore.__init__
    调 _ensure_column 做 ALTER TABLE ADD COLUMN，迁移后能正常写入新字段。
    """
    import sqlite3
    db_path = tmp_path / "runs.db"
    # 手建一份旧 schema（runs / segments / clips 三表，segments 缺 first_token_ms）
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, video_path TEXT, video_name TEXT,
                duration_sec REAL, status TEXT DEFAULT 'started',
                started_at TEXT, finished_at TEXT,
                hits_count INTEGER DEFAULT 0, segments_total INTEGER DEFAULT 0,
                segments_ok INTEGER DEFAULT 0, segments_failed INTEGER DEFAULT 0,
                vlm_elapsed_sec REAL, total_elapsed_sec REAL,
                model TEXT, provider TEXT, mode TEXT
            );
            CREATE TABLE segments (
                seg_id TEXT PRIMARY KEY, run_id TEXT, seg_idx INTEGER,
                start_sec REAL, dur_sec REAL, status TEXT, match INTEGER,
                confidence REAL, reason TEXT, abs_timestamp TEXT,
                attempts INTEGER, elapsed_sec REAL, usage_json TEXT, error TEXT
            );
            CREATE TABLE clips (
                clip_id TEXT PRIMARY KEY, run_id TEXT,
                hit_idx INTEGER, abs_timestamp TEXT, clip_path TEXT
            );
        """)
        conn.commit()

    # 用 RunStore 打开旧库 → _ensure_column 应补 first_token_ms 列
    store = RunStore(str(tmp_path))
    with sqlite3.connect(db_path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(segments)").fetchall()]
    assert "first_token_ms" in cols

    # 迁移后能正常写入 + 读回新字段
    run_id = store.create_run("cam01.mp4", model="m", provider="p", mode="surveillance")
    store.add_segment(run_id, {
        "seg_idx": 0, "start_sec": 0.0, "dur_sec": 60.0,
        "status": "ok", "match": 1, "first_token_ms": 280,
    })
    run = store.get_run(run_id)
    assert run["segments"][0]["first_token_ms"] == 280

    # 幂等：再开一次 RunStore 不应崩溃（列已存在，_ensure_column 返回 False）
    store2 = RunStore(str(tmp_path))
    run2 = store2.get_run(run_id)
    assert run2 is not None
    assert run2["segments"][0]["first_token_ms"] == 280
