"""决策日志 —— 黑匣子透明化核心数据模型。

设计依据（scout-transparency 报告，已验证参考项目）：
  - video-use edl.json：reason 必须一句大白话（"最干净的 delivery，在 38.46
    口误前停住"），不许只记参数。
  - edit-mind DB：耗时/帧数/输出路径当一等公民字段，不藏在日志正文里。
  - OpenHands Event：id/cause_id/status 链路，便于跨步骤因果追溯。

不可变追加（append 返回新实例，原实例不变），防并发竞态；原子落盘
（.tmp + os.replace），防中途崩溃留下半截 JSON。
"""
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# 枚举值集中声明，make_entry 校验，避免拼写漂移。
_STATUS_VALUES: Tuple[str, ...] = ("ok", "error", "blocked")
_RISK_VALUES: Tuple[str, ...] = ("low", "medium", "high")


@dataclass(frozen=True)
class DecisionEntry:
    """单条决策记录。

    字段语义：
      id            本条 8 位短 id（uuid4 hex 前 8，便于人在 UI 里引用）。
      timestamp     ISO8601 时间戳（秒精度，人读友好）。
      step_name     步骤名（如"抽帧""视觉搜索"）。
      action_type   工具/动作类型（如 search_visual / run_ocr）。
      decision      决策结论（做了什么，如"停在 38.46s"）。
      reason        一句大白话原因（学 edl.json，必填非空）。
      cause_id      触发本步骤的上游条目 id（None 表示根步骤）。
      output_path   本步骤产物路径（None 表示无落盘产物）。
      duration_ms   本步骤耗时（毫秒，一等公民字段）。
      status        ok / error / blocked。
      risk          low / medium / high（危险操作打标，供 ask 阻塞）。
    """
    id: str
    timestamp: str
    step_name: str
    action_type: str
    decision: str
    reason: str
    cause_id: Optional[str] = None
    output_path: Optional[str] = None
    duration_ms: float = 0.0
    status: str = "ok"
    risk: str = "low"


@dataclass(frozen=True)
class DecisionLog:
    """不可变决策日志容器。

    append 返回新实例（原实例 entries 不变），调用方拿到新引用。这样 worker
    线程读旧引用不会看到半截追加，避免竞态。entries 是 tuple（不可变），
    默认值 () 安全（dataclass 不报可变默认值错）。
    """
    entries: Tuple[DecisionEntry, ...] = ()

    def append(self, entry: DecisionEntry) -> "DecisionLog":
        """返回包含新条目的新实例，self 不变（不可变语义）。"""
        return DecisionLog(self.entries + (entry,))

    def to_list(self) -> Tuple[DecisionEntry, ...]:
        return self.entries

    def to_json(self) -> str:
        """pretty JSON，ensure_ascii=False 保留中文可读。"""
        return json.dumps([asdict(e) for e in self.entries],
                          ensure_ascii=False, indent=2)

    def save(self, path: Path) -> None:
        """原子写：先写 .tmp，再 os.replace 覆盖目标（Windows 亦原子）。

        中途崩溃只会留下 .tmp，不会破坏既有目标文件。
        """
        p = Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(p)  # os.replace 语义：跨卷报错，同卷原子

    @classmethod
    def from_json(cls, s: str) -> "DecisionLog":
        """从 JSON 重建（保留原 id/timestamp，便于跨会话续写日志）。"""
        data = json.loads(s)
        return cls(tuple(DecisionEntry(**d) for d in data))


def make_entry(
    step_name: str,
    action_type: str,
    decision: str,
    reason: str,
    *,
    cause_id: Optional[str] = None,
    output_path: Optional[str] = None,
    duration_ms: float = 0.0,
    status: str = "ok",
    risk: str = "low",
) -> DecisionEntry:
    """工厂构造 DecisionEntry。

    reason 必填非空（一句大白话，学 edl.json 铁律）——空串/纯空格 raise
    ValueError，防止退化成"只记参数"的废日志。status/risk 拼写校验同上。
    id 与 timestamp 由工厂生成（uuid4 hex 8 位 / ISO8601 秒精度）。
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason 必须是一句非空大白话说明（不许只记参数）")
    if status not in _STATUS_VALUES:
        raise ValueError(f"status 必须是 {_STATUS_VALUES} 之一，得到 {status!r}")
    if risk not in _RISK_VALUES:
        raise ValueError(f"risk 必须是 {_RISK_VALUES} 之一，得到 {risk!r}")
    return DecisionEntry(
        id=uuid.uuid4().hex[:8],
        timestamp=datetime.now().isoformat(timespec="seconds"),
        step_name=step_name,
        action_type=action_type,
        decision=decision,
        reason=reason,
        cause_id=cause_id,
        output_path=output_path,
        duration_ms=duration_ms,
        status=status,
        risk=risk,
    )
