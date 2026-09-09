"""Skill 评分棘轮 — 只留改进、回滚退步（评分卡 + 落盘棘轮）。

评分卡（0-100，规则化，无 LLM）：
- 结构完整度 0-30：frontmatter 必填项 + 正文区块存在性
- 触发匹配 0-20：triggers 数量/字符质量 + description 含关键词数
- 步骤可执行性 0-30：正文含可执行步骤（列表/代码块）与参数
- 安全分 0-20：spectre 扫描干净则满分；每条 finding 按严重级扣分

棘轮语义（与指南 P1-2 一致）：
- 新 skill 首次创建：直接落盘（无历史分可比较）
- 重跑同 skill：只有新分 >= 旧分才更新落盘，否则标记 rejected（回滚退步）
- 全部原子写 skills_scores.json（tmp + os.replace）
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from src.utils.constants import CONFIG_DIR

try:
    from src.skills.spectre import ScanReport, scan_skill
except Exception:  # noqa: BLE001
    scan_skill = None  # type: ignore[assignment]
    ScanReport = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)

SCORES_FILENAME = "skills_scores.json"

# 评分卡分档
_MAX_STRUCTURE = 30
_MAX_TRIGGER = 20
_MAX_EXECUTABLE = 30
_MAX_SECURITY = 20

_FRONTMATTER_REQUIRED = ("name", "description")
_BODY_SECTIONS = ("## 适用场景", "## 算法", "## 参数", "## 何时用", "## 何时不该用", "## 降级行为")


@dataclass(frozen=True)
class ScoreCard:
    """单次评分结果。"""

    name: str
    total: float
    structure: float
    trigger: float
    executable: float
    security: float
    raw: Dict[str, Any] = None  # type: ignore[assignment]  # 内部明细


def _scores_path() -> Path:
    return Path(CONFIG_DIR) / SCORES_FILENAME


def _atomic_write(data: Dict[str, Any]) -> None:
    p = _scores_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, p)


def load_scores() -> Dict[str, Any]:
    """读 skills_scores.json，结构不合法返回空 dict。"""
    p = _scores_path()
    if not p.exists():
        return {}
    try:
        with p.open(encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("读取 skills_scores.json 失败，按空处理：%s", exc)
    return {}


def get_score(name: str) -> Optional[float]:
    """取某 skill 的历史总分，无记录返回 None。"""
    data = load_scores()
    entry = data.get(name)
    if isinstance(entry, dict):
        val = entry.get("score")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def score_skill(skill_dir: Path, *, existing_scores: Optional[Dict[str, Any]] = None
                ) -> ScoreCard:
    """对 <skill_dir>/SKILL.md 评分。

    不依赖外部状态（existing_scores 仅用于棘轮比较，评分本身只看文件内容）。
    """
    root = Path(skill_dir)
    name = root.name
    content = ""
    if (root / "SKILL.md").exists():
        content = (root / "SKILL.md").read_text(encoding="utf-8-sig")

    # -- 结构完整度 ---------------------------------------------------------
    structure = 0.0
    fm = _extract_frontmatter(content)
    for req in _FRONTMATTER_REQUIRED:
        if req in fm and str(fm[req]).strip():
            structure += 8
    for sect in _BODY_SECTIONS:
        if sect in content:
            structure += 2.0
    structure = min(structure, _MAX_STRUCTURE)

    # -- 触发匹配 -----------------------------------------------------------
    trigger = 0.0
    triggers = _extract_triggers(fm)
    trigger += min(len(triggers) * 4, 12)  # 最多 3 个算满
    desc = str(fm.get("description", ""))
    trigger += min(len(desc) / 5, 4)
    for kw in ("视频", "分析", "监控", "识别"):
        if kw in desc:
            trigger += 1.0
    trigger = min(trigger, _MAX_TRIGGER)

    # -- 步骤可执行性 -------------------------------------------------------
    executable = 0.0
    lines = content.splitlines()
    code_blocks = sum(1 for ln in lines if ln.strip().startswith("```"))
    if code_blocks >= 2:
        executable += 10
    elif code_blocks >= 1:
        executable += 6
    bullets = sum(1 for ln in lines if ln.strip().startswith(("-", "*")))
    numbered = sum(1 for ln in lines if ln.strip()[:2] in ("1.", "2.", "3.") or
                   ln.strip()[:1].isdigit() and "." in ln.strip()[:3])
    if bullets + numbered >= 4:
        executable += 8
    elif bullets + numbered >= 2:
        executable += 4
    if ("sample_fps" in content or "threshold" in content or "=" in content):
        executable += 6
    executable = min(executable, _MAX_EXECUTABLE)

    # -- 安全分 -------------------------------------------------------------
    security = _MAX_SECURITY
    if scan_skill is not None:
        try:
            report = scan_skill(root)
            if isinstance(report, ScanReport):
                if not report.clean:
                    for f in report.findings:
                        security -= {
                            "critical": 12, "high": 8, "medium": 4, "low": 0,
                        }.get(f.severity, 0)
                    security = max(security, 0.0)
        except Exception:  # noqa: BLE001
            security = 0.0

    total = structure + trigger + executable + security
    return ScoreCard(
        name=name,
        total=round(total, 2),
        structure=round(structure, 2),
        trigger=round(trigger, 2),
        executable=round(executable, 2),
        security=round(security, 2),
        raw={
            "frontmatter_present": bool(fm),
            "code_blocks": code_blocks,
            "sections_present": [s for s in _BODY_SECTIONS if s in content],
        },
    )


def apply_ratchet(skill_dir: Path) -> Dict[str, Any]:
    """棘轮落盘：新 skill 首次直接写；重跑同 skill 只留 >= 旧分的版本。

    返回 {"skill": name, "score": float, "ratchet": "new"|"improved"|"rejected"}。
    """
    root = Path(skill_dir)
    card = score_skill(root)
    name = card.name
    scores = load_scores()
    old = get_score(name)
    if old is None:
        action = "new"
        scores[name] = {"score": card.total, "status": "ok"}
    elif card.total >= old:
        action = "improved"
        scores[name] = {"score": card.total, "status": "ok"}
    else:
        action = "rejected"
        # 回滚退步：保留旧分，标记 rejected（不覆盖旧分）
        entry = dict(scores.get(name, {}))
        entry["status"] = "rejected"
        entry["rejected_score"] = card.total
        entry["last_good_score"] = old
        scores[name] = entry
    _atomic_write(scores)
    return {
        "skill": name,
        "score": card.total,
        "ratchet": action,
    }


def _extract_frontmatter(content: str) -> Dict[str, str]:
    """提取 frontmatter 键值（yaml 优先，降级手写 key:value）。"""
    import re
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL)
    if not m:
        return {}
    fm_text = m.group(1)
    try:
        import yaml  # type: ignore[import-untyped]
        data = yaml.safe_load(fm_text)
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items()}
    except Exception:  # noqa: BLE001
        pass
    result: Dict[str, str] = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip().strip("\"'")
    return result


def _extract_triggers(fm: Dict[str, str]) -> Sequence[str]:
    raw = fm.get("triggers", "")
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    return []


# -- 测试/工具便捷 ---------------------------------------------------------

def normalize_scores_path(path: Path) -> None:
    """测试隔离：把 skills_scores.json 重定向到 tmp。"""
    global _scores_path
    _scores_path = lambda p=path: Path(p)  # noqa: E731