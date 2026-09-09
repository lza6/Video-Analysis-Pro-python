"""三层记忆子系统 — 热层(WorkingMemory) + 语义层(ExperienceStore) + 时序图层(TripleStore)。

《改进指南》P1-1：扁平记忆升为三层，全部 SQLite 本地、纯 stdlib、
feature flag `VAP_MEMORY_LAYERED` 可独立开关（默认 1=开）。

注：MemoryLayeredConnector 不在此 re-export（避免与 src.core.agent 的循环导入），
主控侧直接用 `from src.core.memory.connector import MemoryLayeredConnector`。
"""
from src.core.memory.triplestore import Triple, TripleStore
from src.core.memory.working import WorkingFact, WorkingMemory

__all__ = [
    "Triple",
    "TripleStore",
    "WorkingFact",
    "WorkingMemory",
]
