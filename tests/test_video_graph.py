"""VideoGraph 跨视频时序推理图单测（v6.1 视频知识图谱）。

覆盖：
  - add_hit 单节点
  - trace_item 无命中返回空
  - 时序边：两同摄像头命中时间差 < 30min 连通
  - 时序边：时间差 > 30min 不连通
  - 空间边：同 camera_id 连通
  - 物品边：reason 含相同关键词连通
  - trace_item 返回按时间排序的链路
  - to_dict/from_dict 往返
"""
from datetime import datetime, timedelta

from src.core.video_graph import HitNode, VideoGraph


def _node(
    *, run_id="r1", video_name="36#2单元入口.mp4", video_path="D:/36.mp4",
    ts=100.0, conf=0.9, reason="匹配到黑色旅行袋", abs_time=None,
    camera_id=None, clip_path="D:/clips/h0.mp4", node_id=None,
) -> HitNode:
    return HitNode(
        run_id=run_id,
        video_name=video_name,
        video_path=video_path,
        timestamp_sec=ts,
        confidence=conf,
        reason=reason,
        camera_id=camera_id if camera_id is not None
        else HitNode._extract_camera_id(video_name),
        abs_time=abs_time,
        clip_path=clip_path,
        node_id=node_id or "",
    )


# ----------------------------------------------------------------------
# add_hit 单节点
# ----------------------------------------------------------------------
def test_add_hit_single_node():
    g = VideoGraph()
    nid = g.add_hit(_node(node_id="n1"))
    assert nid == "n1"
    # 幂等：重复 node_id 不新增
    g.add_hit(_node(node_id="n1"))
    chains = g.trace_item("黑色旅行袋")
    assert len(chains) == 1
    assert len(chains[0]) == 1
    assert chains[0][0].node_id == "n1"


# ----------------------------------------------------------------------
# trace_item 无命中返回空
# ----------------------------------------------------------------------
def test_trace_item_no_match_returns_empty():
    g = VideoGraph()
    g.add_hit(_node(reason="匹配到红色书包", node_id="n1"))
    assert g.trace_item("黑色旅行袋") == []
    # 空关键词也返回空
    assert g.trace_item("") == []


# ----------------------------------------------------------------------
# 时序边：两同摄像头命中时间差 < 30min 连通
# ----------------------------------------------------------------------
def test_temporal_edge_within_threshold_connects():
    g = VideoGraph()
    t0 = datetime(2026, 9, 4, 10, 0, 0)
    g.add_hit(_node(abs_time=t0, reason="黑色旅行袋出现", node_id="a"))
    g.add_hit(_node(abs_time=t0 + timedelta(minutes=10),
                    reason="黑色旅行袋消失", node_id="b"))
    assert "temporal" in g.edge_types("a", "b")
    chains = g.trace_item("黑色旅行袋")
    assert len(chains) == 1
    assert len(chains[0]) == 2


# ----------------------------------------------------------------------
# 时序边：时间差 > 30min 不连通（但同 camera_id 仍连空间边，故用不同摄像头）
# ----------------------------------------------------------------------
def test_temporal_edge_beyond_threshold_no_temporal_edge():
    g = VideoGraph()
    t0 = datetime(2026, 9, 4, 10, 0, 0)
    # 不同摄像头 + 不同 reason 关键词，仅靠时序边连通
    g.add_hit(_node(abs_time=t0, video_name="36#A.mp4",
                    reason="黑色旅行袋", node_id="a", camera_id="36"))
    g.add_hit(_node(abs_time=t0 + timedelta(minutes=45),
                    video_name="99#B.mp4",
                    reason="黑色旅行袋", node_id="b", camera_id="99"))
    assert "temporal" not in g.edge_types("a", "b")
    # 但因 item 边（同 reason）仍连通，所以 trace 返回 1 条含 2 节点链路
    chains = g.trace_item("黑色旅行袋")
    assert len(chains) == 1
    assert len(chains[0]) == 2


# ----------------------------------------------------------------------
# 空间边：同 camera_id 连通
# ----------------------------------------------------------------------
def test_spatial_edge_same_camera_connects():
    g = VideoGraph()
    t0 = datetime(2026, 9, 4, 10, 0, 0)
    # 时间差 > 30min（无时序边）+ 同摄像头（空间边）+ 同 reason（物品边）
    g.add_hit(_node(abs_time=t0, video_name="36#A.mp4",
                    reason="黑色旅行袋", node_id="a", camera_id="36"))
    g.add_hit(_node(abs_time=t0 + timedelta(hours=2),
                    video_name="36#B.mp4",
                    reason="黑色旅行袋", node_id="b", camera_id="36"))
    assert "spatial" in g.edge_types("a", "b")
    # 时间差 2h > 30min，无时序边
    assert "temporal" not in g.edge_types("a", "b")


# ----------------------------------------------------------------------
# 物品边：reason 含相同关键词连通
# ----------------------------------------------------------------------
def test_item_edge_shared_keyword_connects():
    g = VideoGraph()
    t0 = datetime(2026, 9, 4, 10, 0, 0)
    # 不同摄像头 + 时间差大 + 共享 "黑色旅行袋" 关键词
    g.add_hit(_node(abs_time=t0, video_name="36#A.mp4",
                    reason="发现黑色旅行袋", node_id="a", camera_id="36"))
    g.add_hit(_node(abs_time=t0 + timedelta(hours=3),
                    video_name="99#B.mp4",
                    reason="黑色旅行袋被拿走", node_id="b", camera_id="99"))
    assert "item" in g.edge_types("a", "b")
    assert "spatial" not in g.edge_types("a", "b")
    assert "temporal" not in g.edge_types("a", "b")


# ----------------------------------------------------------------------
# trace_item 返回按时间排序的链路
# ----------------------------------------------------------------------
def test_trace_item_returns_sorted_chain():
    g = VideoGraph()
    t0 = datetime(2026, 9, 4, 10, 0, 0)
    # 三节点同摄像头同物品，时间递增，应连成一条按时间排序的链路
    g.add_hit(_node(abs_time=t0 + timedelta(minutes=20),
                    reason="黑色旅行袋", node_id="c"))
    g.add_hit(_node(abs_time=t0, reason="黑色旅行袋", node_id="a"))
    g.add_hit(_node(abs_time=t0 + timedelta(minutes=10),
                    reason="黑色旅行袋", node_id="b"))
    chains = g.trace_item("黑色旅行袋")
    assert len(chains) == 1
    chain = chains[0]
    assert [n.node_id for n in chain] == ["a", "b", "c"]
    # 按时间升序验证
    times = [n.abs_time for n in chain]
    assert times == sorted(times)


# ----------------------------------------------------------------------
# 非种子节点经时序边被拉入链路
# ----------------------------------------------------------------------
def test_trace_item_pulls_non_seed_via_temporal_edge():
    """命中节点 a 含关键词，b 不含但时间相近 + 同摄像头 → 经时序边拉入同一链路。

    验证 BFS 不只走 item 边：非种子节点也可通过 temporal/spatial 边进入链路。
    """
    g = VideoGraph()
    t0 = datetime(2026, 9, 4, 10, 0, 0)
    # a 含 "黑色旅行袋"，b 不含但 5 分钟后 + 同摄像头
    g.add_hit(_node(abs_time=t0, video_name="36#A.mp4",
                    reason="发现黑色旅行袋", node_id="a", camera_id="36"))
    g.add_hit(_node(abs_time=t0 + timedelta(minutes=5),
                    video_name="36#B.mp4",
                    reason="可疑人员经过", node_id="b", camera_id="36"))
    chains = g.trace_item("黑色旅行袋")
    # a 是唯一种子，b 经 temporal+spatial 边拉入，1 条链路含 2 节点
    assert len(chains) == 1
    assert len(chains[0]) == 2
    assert [n.node_id for n in chains[0]] == ["a", "b"]



# ----------------------------------------------------------------------
# to_dict / from_dict 往返
# ----------------------------------------------------------------------
def test_to_dict_from_dict_roundtrip():
    g = VideoGraph()
    t0 = datetime(2026, 9, 4, 10, 0, 0)
    g.add_hit(_node(abs_time=t0, reason="黑色旅行袋", node_id="a"))
    g.add_hit(_node(abs_time=t0 + timedelta(minutes=10),
                    reason="黑色旅行袋", node_id="b"))
    data = g.to_dict()
    assert "nodes" in data and len(data["nodes"]) == 2
    assert data["nodes"][0]["camera_id"] == "36"

    g2 = VideoGraph.from_dict(data)
    # node_id 保留
    assert {n.node_id for n in g2._nodes.values()} == {"a", "b"}
    # 边重建后仍连通
    assert "temporal" in g2.edge_types("a", "b")
    # trace 结果一致
    chains = g2.trace_item("黑色旅行袋")
    assert len(chains) == 1
    assert len(chains[0]) == 2
    assert [n.node_id for n in chains[0]] == ["a", "b"]


# ----------------------------------------------------------------------
# HitNode.from_segment：从 RunStore 三表行聚合
# ----------------------------------------------------------------------
def test_hit_node_from_segment_aggregates_run_seg_clip():
    run = {
        "run_id": "r1", "video_name": "36#2单元.mp4",
        "video_path": "D:/36.mp4", "started_at": "2026-09-04T10:00:00",
    }
    seg = {
        "seg_idx": 0, "start_sec": 536.0, "match": 1,
        "confidence": 0.95, "reason": "黑色旅行袋",
        "abs_timestamp": "2026-09-04T10:08:56",
    }
    clip = {"hit_idx": 0, "clip_path": "D:/clips/h0.mp4"}
    n = HitNode.from_segment(run, seg, clip)
    assert n.run_id == "r1"
    assert n.video_name == "36#2单元.mp4"
    assert n.camera_id == "36"
    assert n.timestamp_sec == 536.0
    assert n.confidence == 0.95
    assert n.reason == "黑色旅行袋"
    assert n.clip_path == "D:/clips/h0.mp4"
    # abs_timestamp 是 ISO 字符串，优先解析为绝对时间
    assert n.abs_time == datetime(2026, 9, 4, 10, 8, 56)


def test_hit_node_from_segment_falls_back_to_started_at_plus_sec():
    """abs_timestamp 缺失时，用 started_at + start_sec 偏移。"""
    run = {
        "run_id": "r2", "video_name": "cam02.mp4",
        "video_path": "D:/cam02.mp4", "started_at": "2026-09-04T10:00:00",
    }
    seg = {"seg_idx": 1, "start_sec": 120.0, "match": 1,
           "confidence": 0.8, "reason": "红色书包"}
    n = HitNode.from_segment(run, seg, None)
    assert n.camera_id == "cam02"
    assert n.abs_time == datetime(2026, 9, 4, 10, 2, 0)
    assert n.clip_path == ""


# ----------------------------------------------------------------------
# build_from_run_store：集成 RunStore 真实表
# ----------------------------------------------------------------------
def test_build_from_run_store_collects_hits(tmp_path):
    from src.core.run_store import RunStore
    store = RunStore(str(tmp_path / "cfg"))
    run_id = store.create_run("D:/36#A.mp4", duration_sec=600.0)
    store.add_segment(run_id, {
        "seg_idx": 0, "start_sec": 100.0, "dur_sec": 60.0,
        "status": "ok", "match": 1, "confidence": 0.9,
        "reason": "黑色旅行袋", "abs_timestamp": "2026-09-04T10:01:40",
    })
    store.add_hit(run_id, {
        "hit_idx": 0, "abs_timestamp": "2026-09-04T10:01:40",
        "clip_path": "D:/clips/h0.mp4",
    })
    store.update_run(run_id, segments_total=1, segments_ok=1, status="done")

    g = VideoGraph.build_from_run_store(store)
    chains = g.trace_item("黑色旅行袋")
    assert len(chains) == 1
    assert len(chains[0]) == 1
    n = chains[0][0]
    assert n.run_id == run_id
    assert n.reason == "黑色旅行袋"
    assert n.clip_path == "D:/clips/h0.mp4"
