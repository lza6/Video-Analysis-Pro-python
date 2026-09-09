"""MemoryLayeredConnector — run_turn 的主控接入（热层注入 + 经验/三元组记录）。

feature flag `VAP_MEMORY_LAYERED`（默认 1=开）由 `from_env` 读取装配。
store 全部可选（None=该层禁用）。

- turn 开始：热层注入（WorkingMemory.inject_system_messages，在
  derive_messages 之后作为附加 system 段）
- turn 结束：Experience 记录 + 同步热层 + 三元组抽取
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from src.core.agent.experience import ExperienceStore, ExperienceExtractor
from src.core.memory.triplestore import TripleStore
from src.core.memory.working import WorkingMemory

log = logging.getLogger("core.agent.memory_layered")

DEFAULT_WORKING_DB = "config/agent_working_memory.db"
DEFAULT_TRIPLES_DB = "config/agent_triples.db"
DEFAULT_EXPERIENCE_DB = "config/agent_experiences.db"


class MemoryLayeredConnector:
    """run_turn 分层记忆接入点（feature_flag=False 时全层 no-op）。

    组件：WorkingMemory（热层）+ ExperienceStore（语义层）+ TripleStore（时序层）。
    """

    def __init__(
        self,
        feature_flag: bool,
        working: Optional[WorkingMemory] = None,
        experience: Optional[ExperienceStore] = None,
        triples: Optional[TripleStore] = None,
    ) -> None:
        self.enabled = bool(feature_flag)
        self.working = working if (self.enabled and working is not None) else None
        self.experience = experience if (self.enabled and experience is not None) else None
        self.triples = triples if (self.enabled and triples is not None) else None

    @classmethod
    def from_env(cls, feature_flag: Optional[bool] = None,
                 working: Optional[WorkingMemory] = None,
                 experience: Optional[ExperienceStore] = None,
                 triples: Optional[TripleStore] = None) -> "MemoryLayeredConnector":
        """按 feature flag 装配。

        显式传 feature_flag 优先；否则读 `VAP_MEMORY_LAYERED`（默认 1=开）。
        store 传 None 时用默认 db 路径自建（惰性，不构造不落库）。
        """
        if feature_flag is None:
            flag = os.environ.get("VAP_MEMORY_LAYERED", "1").strip().lower()
            feature_flag = flag in ("1", "true", "yes", "on")
        wm = working
        exp = experience
        tr = triples
        if feature_flag:
            if wm is None:
                wm = WorkingMemory(DEFAULT_WORKING_DB)
            if exp is None:
                exp = ExperienceStore(DEFAULT_EXPERIENCE_DB)
            if tr is None:
                tr = TripleStore(DEFAULT_TRIPLES_DB)
        return cls(feature_flag=feature_flag, working=wm, experience=exp, triples=tr)

    # -- 热层注入 -----------------------------------------------------------

    def inject(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """turn 开始：把热层事实注入为附加 system 段（不改 system_prompt）。"""
        if not self.enabled or self.working is None:
            return messages
        try:
            return self.working.inject_system_messages(list(messages))
        except Exception:  # noqa: BLE001
            log.warning("working memory 注入失败，跳过", exc_info=True)
            return messages

    # -- turn 结束记录 ------------------------------------------------------

    def record(self, session: Any, exp: Optional[Any] = None) -> None:
        """turn 结束：Experience 记录 + 热层同步 + 三元组抽取。

        exp 为 None 时用 ExperienceExtractor.from_session 现场提取（空 session
        返回 None → 全部 no-op）。
        """
        if not self.enabled:
            return
        if exp is None:
            try:
                exp = ExperienceExtractor.from_session(session)
            except Exception:  # noqa: BLE001
                log.warning("experience 提取失败，跳过", exc_info=True)
                exp = None
        if exp is None:
            return
        try:
            if self.experience is not None:
                self.experience.record(exp)
            if self.working is not None:
                self.working.record_intent(
                    exp.session_id, exp.intent, list(exp.tool_chain),
                    ts=float(exp.timestamp))
            if self.triples is not None:
                self.triples.record_experience(
                    exp.intent, list(exp.tool_chain), exp.success,
                    ts=float(exp.timestamp))
        except Exception:  # noqa: BLE001
            log.warning("分层记忆写入失败，跳过", exc_info=True)