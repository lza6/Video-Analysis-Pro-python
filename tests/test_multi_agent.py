"""v7.0 多 agent 协作测试（指南 4.2）。

MultiAgentOrchestrator：Planner 拆任务 → Executor 执行 → Critic 审查 → Reporter 汇总。
纯逻辑（无 LLM 真实调用，付费 API 红线）。
"""
from __future__ import annotations

import sys

import pytest


def test_plan_complex_task_surveillance() -> None:
    """含监控/找包 → 拆成 Executor+Critic+Reporter 三角色。"""
    from src.core.agent_orchestrator import (
        MultiAgentOrchestrator, AgentRole)
    ma = MultiAgentOrchestrator()
    tasks = ma.plan_complex_task("分析监控找包")
    roles = [t.role for t in tasks]
    assert AgentRole.EXECUTOR in roles
    assert AgentRole.CRITIC in roles
    assert AgentRole.REPORTER in roles
    # Executor 跑 batch_analyze
    exec_tasks = [t for t in tasks if t.role == AgentRole.EXECUTOR]
    assert any(t.tool_name == "batch_analyze" for t in exec_tasks)


def test_plan_complex_task_clip() -> None:
    """含剪辑 → Executor 跑 create_highlights。"""
    from src.core.agent_orchestrator import (
        MultiAgentOrchestrator, AgentRole)
    ma = MultiAgentOrchestrator()
    tasks = ma.plan_complex_task("剪辑精彩集锦")
    exec_tasks = [t for t in tasks if t.role == AgentRole.EXECUTOR]
    assert any(t.tool_name == "create_highlights" for t in exec_tasks)


def test_plan_complex_task_report() -> None:
    """含报告/汇总 → Reporter 生成。"""
    from src.core.agent_orchestrator import (
        MultiAgentOrchestrator, AgentRole)
    ma = MultiAgentOrchestrator()
    tasks = ma.plan_complex_task("生成报告")
    rep_tasks = [t for t in tasks if t.role == AgentRole.REPORTER]
    assert len(rep_tasks) >= 1


def test_plan_complex_task_empty() -> None:
    """无关键词 → Planner 单任务。"""
    from src.core.agent_orchestrator import (
        MultiAgentOrchestrator, AgentRole)
    ma = MultiAgentOrchestrator()
    tasks = ma.plan_complex_task("你好")
    assert len(tasks) == 1
    assert tasks[0].role == AgentRole.PLANNER


def test_run_next_executes_tasks() -> None:
    """run_next 逐步执行任务，is_done 推进。"""
    from src.core.agent_orchestrator import MultiAgentOrchestrator

    class FakeRegistry:
        def execute_tool_call(self, name, args):
            return f"executed {name}"

    ma = MultiAgentOrchestrator(tool_registry=FakeRegistry())
    ma.plan_complex_task("分析监控找包")
    n = len(ma.get_tasks())
    done = 0
    while not ma.is_done():
        t = ma.run_next()
        if t and t.status == "done":
            done += 1
    assert done == n  # 全部 done


def test_run_next_no_registry_skips() -> None:
    """无 tool_registry → 任务 skipped。"""
    from src.core.agent_orchestrator import MultiAgentOrchestrator
    ma = MultiAgentOrchestrator()  # 无 registry
    ma.plan_complex_task("分析监控找包")
    t = ma.run_next()
    assert t.status == "skipped"


def test_run_next_done_returns_none() -> None:
    """全部跑完 run_next 返回 None。"""
    from src.core.agent_orchestrator import MultiAgentOrchestrator
    ma = MultiAgentOrchestrator()
    ma.plan_complex_task("你好")
    ma.run_next()  # 跑唯一任务
    assert ma.run_next() is None


def test_critic_review_rule_based() -> None:
    """无 LLM 时 Critic 规则审查。"""
    from src.core.agent_orchestrator import MultiAgentOrchestrator
    ma = MultiAgentOrchestrator()
    # 含"命中" → 建议深挖
    r = ma.critic_review("发现命中 conf=0.65")
    assert "Critic" in r
    assert "deep" in r or "二次验证" in r
    # 含 error → 建议重试
    r2 = ma.critic_review("执行 error")
    assert "Critic" in r2
    # 无异常
    r3 = ma.critic_review("一切正常")
    assert "Critic" in r3


def test_critic_review_with_llm() -> None:
    """有 LLM 时 Critic 调 LLM 审查。"""
    from src.core.agent_orchestrator import MultiAgentOrchestrator

    def fake_llm(prompt, images):
        return "LLM 审查意见"

    ma = MultiAgentOrchestrator(llm_callback=fake_llm)
    r = ma.critic_review("命中")
    assert "LLM" in r


def test_agent_role_enum() -> None:
    """AgentRole 四角色枚举。"""
    from src.core.agent_orchestrator import AgentRole
    assert AgentRole.PLANNER == "planner"
    assert AgentRole.EXECUTOR == "executor"
    assert AgentRole.CRITIC == "critic"
    assert AgentRole.REPORTER == "reporter"


def test_agent_task_dataclass() -> None:
    """AgentTask 数据类字段。"""
    from src.core.agent_orchestrator import AgentTask, AgentRole
    t = AgentTask(task_id="t1", role=AgentRole.EXECUTOR,
                  description="test", tool_name="x", args={})
    assert t.status == "pending"
    assert t.result is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
