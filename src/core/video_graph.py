"""跨视频时序推理图（v6.1 视频知识图谱）。

纯 Python dict 实现的时序图，**不依赖 networkx**（避免改 requirements.txt 触发
launcher import 验证 + 新增重依赖）。节点 = 一次命中（HitNode），边 = 三种关系：

  - temporal  : 两节点绝对时间差 < 30 分钟（同摄像头内时序链 / 跨视频时序链）
  - spatial   : 同 camera_id（同摄像头不同时段）
  - item      : reason 含相同物品关键词

``trace_item(keyword)`` 从 reason 含关键词的节点出发，BFS 遍历图，返回按时间
排序的轨迹链路（跨视频追踪物品移动轨迹）。

数据来源：RunStore 的 runs + segments(match=1) + clips 三表。segments 表存
reason/confidence/abs_timestamp，clips 表存 clip_path，runs 表存 video_path/
video_name/started_at。``HitNode.from_segment`` 把三表行聚合成一个命中节点。
"""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# 时序边阈值：两节点绝对时间差小于此值则连时序边（默认 30 分钟）
TEMPORAL_THRESHOLD_SEC = 30 * 60

# 物品关键词提取：中文连续字符段（用于滑窗）+ 英文单词（2 字以上）
_ITEM_RE = re.compile(r"[一-鿿]+|[A-Za-z]{2,}")


@dataclass
class HitNode:
    """一次命中（match=1 的 segment）的图节点表示。"""
    run_id: str
    video_name: str
    video_path: str
    timestamp_sec: float              # 视频内秒数（seg.start_sec 或 clip abs_timestamp 数值）
    confidence: float = 0.0
    reason: str = ""
    camera_id: str = ""
    abs_time: Optional[datetime] = None  # 绝对时间（跨视频时序边用）
    clip_path: str = ""
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @staticmethod
    def _extract_camera_id(video_name: str) -> str:
        """从 video_name 提取摄像头 ID（同摄像头不同视频同前缀）。

        "36#2单元入口.mp4" → "36"；"cam02.mp4" → "cam02"；"_388.mp4" → "388"。
        规则：先按 ``#`` 切取首段；否则取前导字母数字串（跳过前导 ``_``/``-``）。
        """
        stem = Path(video_name).stem if video_name else ""
        if not stem:
            return ""
        if "#" in stem:
            return stem.split("#", 1)[0]
        m = re.match(r"^[_-]*([A-Za-z0-9]+)", stem)
        return m.group(1) if m else stem

    @staticmethod
    def _parse_time(value: Any) -> Optional[datetime]:
        """灵活解析时间值：ISO 字符串 / datetime / None。

        数值（秒数）不被当作绝对时间——秒数交由 abs_time 由调用方组合
        （run.started_at + start_sec）。
        """
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return None
        s = str(value).strip()
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_sec(value: Any) -> float:
        """解析秒数：数值字符串 / float。失败返回 0.0。"""
        if value is None or value == "":
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def from_segment(
        cls,
        run: Dict[str, Any],
        seg: Dict[str, Any],
        clip: Optional[Dict[str, Any]] = None,
    ) -> "HitNode":
        """从 RunStore 的 run + segment(+ clip) 行聚合一个命中节点。

        - ``timestamp_sec``: 优先 ``seg.start_sec``，次选 ``seg.abs_timestamp``
          解析为秒（batch_runner 写的是 ``f"{seg_start:.1f}"``）
        - ``abs_time``: 优先 ``seg.abs_timestamp`` 解析为 ISO；否则
          ``run.started_at + start_sec`` 偏移
        - ``camera_id``: 从 ``run.video_name`` 提取
        - ``clip_path``: 从 clip 行取（按 hit_idx=seg_idx 关联）
        """
        start_sec = cls._parse_sec(seg.get("start_sec"))
        ts_sec = start_sec or cls._parse_sec(seg.get("abs_timestamp"))
        abs_ts = cls._parse_time(seg.get("abs_timestamp"))
        if abs_ts is None:
            started = cls._parse_time(run.get("started_at"))
            if started is not None:
                abs_ts = started + timedelta(seconds=start_sec)
        video_name = run.get("video_name") or ""
        return cls(
            run_id=run.get("run_id") or "",
            video_name=video_name,
            video_path=run.get("video_path") or "",
            timestamp_sec=ts_sec,
            confidence=float(seg.get("confidence") or 0.0),
            reason=seg.get("reason") or "",
            camera_id=cls._extract_camera_id(video_name),
            abs_time=abs_ts,
            clip_path=(clip or {}).get("clip_path") or "",
        )


def _item_keywords(reason: str) -> Set[str]:
    """从 reason 提取物品关键词集合。

    中文采用 2-4 字滑窗（覆盖"黑色旅行袋"这类多字名词；单字如"的/了/是"
    噪声不收），英文按单词（2 字以上，小写化）。
    """
    if not reason:
        return set()
    kws: Set[str] = set()
    for seg in _ITEM_RE.findall(reason):
        # 英文单词段
        if seg[0].isascii():
            kws.add(seg.lower())
            continue
        # 中文段：2-4 字滑窗
        n = len(seg)
        for size in (2, 3, 4):
            if n < size:
                break
            for i in range(n - size + 1):
                kws.add(seg[i:i + size])
    return kws


class VideoGraph:
    """跨视频时序推理图。纯 dict 邻接表，无 networkx 依赖。"""

    def __init__(self, temporal_threshold_sec: float = TEMPORAL_THRESHOLD_SEC):
        self._nodes: Dict[str, HitNode] = {}
        self._adj: Dict[str, Dict[str, Set[str]]] = {}
        self._temporal_threshold = temporal_threshold_sec
        self._dirty = False  # 增量加节点后标记，下次查询前重建边

    # ------------------------------------------------------------------
    # 节点管理
    # ------------------------------------------------------------------
    def add_hit(self, node: HitNode) -> str:
        """加一个命中节点。重复 node_id 忽略（幂等），返回 node_id。"""
        if not node.node_id:
            node.node_id = uuid.uuid4().hex
        if node.node_id in self._nodes:
            return node.node_id
        self._nodes[node.node_id] = node
        self._dirty = True
        return node.node_id

    def _ensure_edges(self) -> None:
        """脏标记触发的全量边重建。O(n^2) 但命中数通常百级以下，可接受。"""
        if not self._dirty:
            return
        self._adj = {}
        ids = list(self._nodes.keys())
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                types = self._edge_types(self._nodes[a], self._nodes[b])
                if types:
                    self._adj.setdefault(a, {}).setdefault(b, set()).update(types)
                    self._adj.setdefault(b, {}).setdefault(a, set()).update(types)
        self._dirty = False

    def _edge_types(self, a: HitNode, b: HitNode) -> Set[str]:
        """计算两节点间的边类型集合。"""
        types: Set[str] = set()
        # 时序边：绝对时间差 < 阈值
        if a.abs_time and b.abs_time:
            delta = abs((a.abs_time - b.abs_time).total_seconds())
            if delta < self._temporal_threshold:
                types.add("temporal")
        # 空间边：同摄像头
        if a.camera_id and a.camera_id == b.camera_id:
            types.add("spatial")
        # 物品边：reason 共享关键词
        if _item_keywords(a.reason) & _item_keywords(b.reason):
            types.add("item")
        return types

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def edge_types(self, a_id: str, b_id: str) -> Set[str]:
        """查询两节点间的边类型集合（测试 / 调试用）。"""
        self._ensure_edges()
        return set(self._adj.get(a_id, {}).get(b_id, set()))

    def trace_item(self, item_keyword: str) -> List[List[HitNode]]:
        """追踪物品轨迹：找 reason 含关键词的节点，BFS 遍历图，返回按时间排序的链路。

        返回 ``List[链路]``，每条链路是按时间升序的 ``HitNode`` 列表。多个连通
        分量各成一条链路。无命中返回空列表。遍历走全部边类型（temporal + spatial
        + item），以支持跨摄像头追踪同一物品的移动轨迹。
        """
        self._ensure_edges()
        kw = (item_keyword or "").strip().lower()
        if not kw:
            return []
        seeds = [nid for nid, n in self._nodes.items() if kw in n.reason.lower()]
        if not seeds:
            return []
        visited: Set[str] = set()
        chains: List[List[HitNode]] = []
        for seed in seeds:
            if seed in visited:
                continue
            comp = self._bfs(seed)
            visited |= comp
            chain = sorted(
                (self._nodes[n] for n in comp),
                key=lambda n: (n.abs_time or datetime.min, n.timestamp_sec),
            )
            chains.append(chain)
        chains.sort(
            key=lambda c: (c[0].abs_time or datetime.min, c[0].timestamp_sec)
        )
        return chains

    def _bfs(self, seed: str) -> Set[str]:
        """从 seed 出发 BFS 遍历所有边类型，返回连通节点集合。"""
        comp: Set[str] = set()
        queue = [seed]
        while queue:
            cur = queue.pop()
            if cur in comp:
                continue
            comp.add(cur)
            for nb in self._adj.get(cur, {}):
                if nb not in comp:
                    queue.append(nb)
        return comp

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 友好的 dict（存 run_store 或文件）。"""
        self._ensure_edges()
        return {
            "temporal_threshold_sec": self._temporal_threshold,
            "nodes": [
                {
                    **asdict(n),
                    "abs_time": n.abs_time.isoformat() if n.abs_time else None,
                }
                for n in self._nodes.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoGraph":
        """从 ``to_dict`` 输出重建图（node_id 保留，边按需重建）。"""
        g = cls(temporal_threshold_sec=data.get(
            "temporal_threshold_sec", TEMPORAL_THRESHOLD_SEC))
        for nd in data.get("nodes", []):
            abs_raw = nd.get("abs_time")
            abs_dt = None
            if abs_raw:
                try:
                    abs_dt = datetime.fromisoformat(abs_raw)
                except (ValueError, TypeError):
                    abs_dt = None
            g.add_hit(HitNode(
                run_id=nd.get("run_id", ""),
                video_name=nd.get("video_name", ""),
                video_path=nd.get("video_path", ""),
                timestamp_sec=float(nd.get("timestamp_sec", 0.0)),
                confidence=float(nd.get("confidence", 0.0)),
                reason=nd.get("reason", ""),
                camera_id=nd.get("camera_id", ""),
                abs_time=abs_dt,
                clip_path=nd.get("clip_path", ""),
                node_id=nd.get("node_id") or uuid.uuid4().hex,
            ))
        g._dirty = True
        return g

    # ------------------------------------------------------------------
    # 从 RunStore 构建整图（trace_item 工具调用）
    # ------------------------------------------------------------------
    @classmethod
    def build_from_run_store(cls, run_store: Any, limit: int = 200) -> "VideoGraph":
        """从 RunStore 读取所有 run 的命中 segment，构建时序图。

        只取 ``segments.match=1`` 的行作为命中节点。每个命中节点 join 对应 clip
        （按 ``hit_idx=seg_idx``）取 ``clip_path``。``limit`` 控制扫描的 run 数
        （默认 200，防全表扫超时）。RunStore 不可用或空返回空图，不抛异常。
        """
        g = cls()
        try:
            runs = run_store.list_runs(limit=limit)
        except Exception:
            return g
        for r in runs:
            run_id = r.get("run_id") or ""
            if not run_id:
                continue
            try:
                full = run_store.get_run(run_id)
            except Exception:
                full = None
            if not full:
                continue
            clips_by_idx = {c.get("hit_idx"): c for c in (full.get("clips") or [])}
            for seg in (full.get("segments") or []):
                if not seg.get("match"):
                    continue
                hit_idx = seg.get("seg_idx")
                clip = clips_by_idx.get(hit_idx)
                g.add_hit(HitNode.from_segment(full, seg, clip))
        return g
