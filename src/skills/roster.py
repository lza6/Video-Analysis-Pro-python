"""按需分配（roster）— 渐进披露：按用户意图实时加载相关 skill 子集。

核心参考 skills-best-practices 的 progressive disclosure（不注入全量 skills）。
resolve_skills_for_intent(intent, skills_dir) → 相关 skill 列表：

1. 显式 triggers 命中（关键词子串，最高优先级）
2. description 关键词命中（fuzzy token 交集）
3. Experience 历史命中（从 ExperienceStore 读到该 intent 的历史 tool_chain，
   按工具名倒查哪些 skill 的 algorithm 引用它 → 更精准）

默认只返回 enabled 的 skill；断/禁用（enabled=False）一律不返回。

v10.2 B-P2-2：新增 4 个领域 skill（ecommerce-merchant / ecommerce-shopper /
ppt-deck / marketing-publish）。对「做PPT」「公众号」「小红书」「上架」「比价」
等口语意图，triggers 子串匹配常命不中（如 intent「帮我做个路演PPT」含「PPT」
大写在 triggers 中不存在）。为此增加领域关键词语义路由（re.IGNORECASE 子串
匹配 + 口语别名表），保证这些领域 skill 能被按需召回；不影响既有 8 个内置
skill 的 triggers/description 命中（优先级低于显式 triggers，且仅对未命中的
领域关键词生效）。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from src.skills.loader import load_skills
from src.skills.schema import Skill

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v10.2 B-P2-2 领域关键词语义路由（intent → 命中 skill 名）
# ---------------------------------------------------------------------------
# 优先级：显式 triggers（match_triggers）> 领域关键词（本表）> description token。
# 领域关键词用正则子串匹配（忽略大小写），意图含「PPT」「ppt」「小红书」等
# 均能命中。仅当 triggers 未命中时才查本表，避免覆盖既有 skill 的 triggers 命中。
_DOMAIN_KEYWORD_MAP: Dict[str, tuple[str, ...]] = {
    "ecommerce-merchant": (
        r"上架", r"改价", r"调价", r"库存", r"补货", r"商品管理", r"电商运营",
        r"卖家", r"发布商品", r"定价",
    ),
    "ecommerce-shopper": (
        r"比价", r"选品", r"性价比", r"哪个划算", r"购物清单", r"买哪个",
        r"买家", r"买什么",
    ),
    "ppt-deck": (
        r"ppt", r"幻灯片", r"演示文稿", r"路演", r"讲稿", r"提纲", r"大纲",
    ),
    "marketing-publish": (
        r"公众号", r"小红书", r"微博", r"抖音", r"营销", r"推广", r"文案",
        r"广告语", r"宣传", r"种草",
    ),
}


def _tokenize(text: str) -> Set[str]:
    """把文本切成 token；对中文附加双字 bigram 以支持子串匹配。"""
    if not text:
        return set()
    toks: Set[str] = set()
    for w in re.split(r"[\W_]+", text):
        w = w.strip()
        if not w:
            continue
        if len(w) >= 2:
            toks.add(w.lower())
        # 中文/无空格串：附加 bigram（"监控分析" → 监控/控分/分析）
        if len(w) >= 2:
            for i in range(len(w) - 1):
                big = w[i:i + 2]
                if re.match(r"^[一-鿿]{2}$", big):
                    toks.add(big)
    return toks


def _extract_tool_names(algorithm: str) -> Set[str]:
    """从 algorithm 中抽取工具名（蛇形命名的函数/调用名）。"""
    return set(re.findall(r"[a-z_][a-z0-9_]{2,}", algorithm or ""))


def match_triggers(intent: str, skills: Sequence[Skill]) -> List[Skill]:
    """显式 triggers 关键词命中（子串匹配，全启用 skill）。"""
    out: List[Skill] = []
    for s in skills:
        for t in s.triggers:
            if t and t in intent:
                out.append(s)
                break
    return out


def match_description(intent: str, skills: Sequence[Skill]) -> List[Skill]:
    """description 关键词模糊命中（token 交集 >=1）。"""
    itoks = _tokenize(intent)
    if not itoks:
        return []
    out: List[Skill] = []
    for s in skills:
        dtoks = _tokenize(s.description)
        if itoks & dtoks:
            out.append(s)
    return out


def match_domain_keywords(intent: str, skills: Sequence[Skill]) -> List[Skill]:
    """v10.2 B-P2-2 领域关键词语义路由。

    intent 含领域关键词（正则子串，忽略大小写）→ 返回对应 skill。
    仅命中 enabled 且存在于 skills 集合中的 skill。
    """
    if not intent:
        return []
    out: List[Skill] = []
    for skill_name, patterns in _DOMAIN_KEYWORD_MAP.items():
        if any(re.search(p, intent, re.IGNORECASE) for p in patterns):
            for s in skills:
                if s.enabled and s.name == skill_name:
                    out.append(s)
                    break
    return out


class RosterResolver:
    """默认 resolver：resolve_skills_for_intent 的实现。"""

    def _existing_skills(self, skills_dir: Path) -> Sequence[Skill]:
        return load_skills(skills_dir)

    def resolve(
        self, intent: str, skills_dir: Optional[Path] = None
    ) -> List[Skill]:
        available = [s for s in self._existing_skills(skills_dir or Path("config/skills"))
                     if s.enabled]
        if not intent:
            return []
        trig_hits = match_triggers(intent, available)
        # 领域关键词语义路由：仅当显式 triggers 未命中时才查（避免覆盖既有
        # triggers 命中；如「走廊」「剪辑」等仍由原 triggers 先行）
        if trig_hits:
            domain_hits: List[Skill] = []
        else:
            domain_hits = match_domain_keywords(intent, available)
        desc_hits = match_description(intent, available)

        # 合并不重复（triggers 命中优先排序在前）
        seen: Set[str] = set()
        merged: List[Skill] = []
        for s in trig_hits + domain_hits + desc_hits:
            if s.name in seen:
                continue
            seen.add(s.name)
            merged.append(s)
        return merged

    def with_experience(
        self, intent: str, skills_dir: Path, experiences: Sequence[object]
    ) -> List[Skill]:
        """结合 Experience 历史命中：intent 历史 tool_chain 引用的技能列入。"""
        base = self.resolve(intent, skills_dir)
        chain_steps: Set[str] = set()
        for e in experiences:
            if not _intent_match(intent, getattr(e, "intent", "")):
                continue
            for t in (getattr(e, "tool_chain", ()) or ()):
                chain_steps.add(str(t))
        if not chain_steps:
            return base
        extra: List[Skill] = []
        for s in base:
            if s.name in {x.name for x in extra}:
                continue
        all_skills = [s for s in self._existing_skills(skills_dir) if s.enabled]
        for s in all_skills:
            st = _extract_tool_names(s.description)
            if st & chain_steps:
                if s.name not in [x.name for x in base]:
                    extra.append(s)
        return base + extra


def _intent_match(a: str, b: str) -> bool:
    """轻量意图匹配：一方是另一方的子串或 token 有交集。"""
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return bool(_tokenize(a) & _tokenize(b))


def resolve_skills_for_intent(
    intent: str, skills_dir: Optional[Path] = None
) -> List[Skill]:
    """主入口：按意图返回相关 skill 子集（渐进披露）。

    - 未匹配返回空列表（不返回全量）
    - 显式 triggers 命中优先；其次 description token 命中
    - 禁用 skill 一律不返回
    """
    return RosterResolver().resolve(intent, skills_dir)