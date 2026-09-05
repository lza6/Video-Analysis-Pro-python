"""自检闭环逻辑（指南 4.6 自检闭环 v7.0）。

agent 跑完批量后，审查决策日志与 run_store，找灰色地带（confidence 0.6-0.7
的可能误判）与未二次验证的命中，生成人类可读报告供 agent 决定是否 deep_dive。

设计要点（付费 API 红线）
  - 本模块纯逻辑，不发起任何 LLM/AI 真实调用；只读 run_store 与 decision_log。
  - 二次验证本身（batch_runner._verify_segment）走真实 API，本模块只负责
    "找出可疑分片"，不重复调用付费接口。
  - 触发判定 should_trigger_self_check 只看决策日志的 status/risk 字段计数。

接缝说明
  - 不依赖 batch_runner / agent_orchestrator / agent_tools / main_window，
    纯函数 + 只读 run_store.get_run / decision_log.to_list。主控后续接线。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class GrayZone:
    """灰色地带分片：confidence 落在 [low, high] 区间，最可能误判。

    不可变（frozen=True），与项目不可变数据约定一致（学 decision_log 铁律），
    防 agent 跨步骤引用时被误改。
    """
    run_id: str
    seg_idx: int
    confidence: float
    reason: str
    video_name: str


def find_gray_zones(run_store: Any, run_id: str,
                    low: float = 0.6, high: float = 0.7) -> List[GrayZone]:
    """从 run_store 找 confidence 在 [low, high] 的灰色地带分片。

    这些既没明确命中（confidence<0.7 二次验证阈值）又非明确未命中（>0.6），
    是最可能误判的区间。返回列表供 agent 决定是否 deep_dive。

    边界闭区间：0.6 和 0.7 都算灰色地带（实测边界值常是阈值临界点）。
    confidence 为 None 的分片（异常/未跑完）跳过，不误报。
    run_id 不存在返回空列表（不 raise，agent 可安全调用）。
    """
    run = run_store.get_run(run_id)
    if run is None:
        return []
    video_name = str(run.get("video_name", "") or "")
    zones: List[GrayZone] = []
    for seg in run.get("segments", []):
        conf = seg.get("confidence")
        if conf is None:
            continue
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            continue
        if low <= conf_f <= high:
            zones.append(GrayZone(
                run_id=run_id,
                seg_idx=int(seg.get("seg_idx", -1)),
                confidence=conf_f,
                reason=str(seg.get("reason", "") or ""),
                video_name=video_name,
            ))
    zones.sort(key=lambda z: z.seg_idx)
    return zones


def find_unverified_hits(run_store: Any, run_id: str,
                         verify_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """找 match=true 但未二次验证的命中。

    判定：match=1 且 confidence < verify_threshold（默认 0.7）。

    原理：
      - enable_verification=True 时，confidence>=0.7 的命中会走二次验证，
        match=1 且 confidence<0.7 说明没达阈值、未触发二次验证 → 最可疑。
      - enable_verification=False 时，所有命中都没二次验证；confidence<0.7
        的更可疑。confidence>=0.7 但 enable_verification=False 的情况本方法
        无法从 DB 区分（segments 表无 verified 列），保守只报 < 阈值的。

    返回 [{run_id, seg_idx, confidence, reason, video_name}]，按 seg_idx 升序。
    run_id 不存在返回空列表。
    """
    run = run_store.get_run(run_id)
    if run is None:
        return []
    video_name = str(run.get("video_name", "") or "")
    hits: List[Dict[str, Any]] = []
    for seg in run.get("segments", []):
        if not seg.get("match"):
            continue
        conf = seg.get("confidence")
        try:
            conf_f = float(conf) if conf is not None else 0.0
        except (TypeError, ValueError):
            conf_f = 0.0
        if conf_f < verify_threshold:
            hits.append({
                "run_id": run_id,
                "seg_idx": int(seg.get("seg_idx", -1)),
                "confidence": conf_f,
                "reason": str(seg.get("reason", "") or ""),
                "video_name": video_name,
            })
    hits.sort(key=lambda h: h["seg_idx"])
    return hits


def build_self_check_report(gray_zones: List[GrayZone],
                            unverified: List[Dict[str, Any]]) -> str:
    """生成人类可读的自检报告（紧凑，≤500 字典型场景）。

    格式：
      🔍 自检报告：
      - 灰色地带 N 个（confidence 0.6-0.7），建议 deep_dive：
        1. video_name seg{idx} conf=0.65 reason
        ...
      - 未验证命中 M 个，建议补二次验证：
        ...
      （任一类为 0 时报"无误判风险"/"均已二次验证"，仍可见全貌）
    """
    lines: List[str] = ["🔍 自检报告："]
    if gray_zones:
        lines.append(
            f"- 灰色地带 {len(gray_zones)} 个（confidence 0.6-0.7），"
            f"建议 deep_dive：")
        for i, z in enumerate(gray_zones, 1):
            lines.append(
                f"  {i}. {z.video_name} seg{z.seg_idx} "
                f"conf={z.confidence:.2f} {z.reason}")
    else:
        lines.append("- 灰色地带 0 个，无误判风险")
    if unverified:
        lines.append(
            f"- 未验证命中 {len(unverified)} 个，建议补二次验证：")
        for i, h in enumerate(unverified, 1):
            lines.append(
                f"  {i}. {h['video_name']} seg{h['seg_idx']} "
                f"conf={h['confidence']:.2f} {h['reason']}")
    else:
        lines.append("- 未验证命中 0 个，均已二次验证")
    return "\n".join(lines)


def should_trigger_self_check(decision_log: Any) -> bool:
    """决策日志含 ≥3 个 error 或 ≥5 个 high risk 步骤时触发自检。

    返回 bool，agent 据此决定是否调 find_gray_zones / find_unverified_hits。

    decision_log 兼容三种形态：
      - DecisionLog 实例（有 to_list）→ 取 entries
      - 带 entries 属性的对象（如 dataclass）→ 取 entries
      - 可迭代的 entry 列表 → 直接遍历
    每个 entry 可是 DecisionEntry（属性访问）或 dict（键访问），都兼容。
    None 返回 False（agent 未传日志时安全不触发）。
    """
    if decision_log is None:
        return False
    if hasattr(decision_log, "to_list"):
        entries = decision_log.to_list()
    elif hasattr(decision_log, "entries"):
        entries = decision_log.entries
    else:
        entries = decision_log
    error_count = 0
    high_risk_count = 0
    for e in entries:
        if isinstance(e, dict):
            status = e.get("status")
            risk = e.get("risk")
        else:
            status = getattr(e, "status", None)
            risk = getattr(e, "risk", None)
        if status == "error":
            error_count += 1
        if risk == "high":
            high_risk_count += 1
    return error_count >= 3 or high_risk_count >= 5
