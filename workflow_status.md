# Workflow Status — v10.3.1 交付闭环（Phase B 完成）

> 依据 `参考的结果计划指南.md` v3（Critic A/B/C 独立审查后）落地。用户全权授权所有阶段（含提交/推送/发行版）。
> 本文件是任务单一状态源。

## Task Contract
- **原始目标**：指南 v3 P0 全部批次真实落地 + 真实 E2E + 独立审查 + 版本五同步 + 提交推送 + 发行版。
- **当前阶段**：DONE（等 Critic 终审后 commit/push/release）。
- **成功标准**：默认用户（零 env 配置）能用到监督层/写审批/记忆/roster/skills 正文/32 工具；审批链路真实 E2E 全通；全量测试绿。

## Task Graph / Batches
| ID | 批 | 状态 | 证据 |
|----|----|------|------|
| P0-1 | 默认路径接通（backend=react / supervisor=on / roster=on / 工具面 16→32） | **VERIFIED** | `_get_backend` 默认 react（agent.py:194）；32 工具实测注册；3 个默认值测试翻转全绿 |
| P0-2 | 审批 SSE 契约修复（/chat 只出契约，执行移 /run_stream + text 重放） | **VERIFIED** | 真实后端 E2E：chat→auto_run=True→run_stream→approval-request(SSE)→decide→done 全通 |
| P0-3 | 工具分类修正（make_* → write；cdp_evaluate/eval_write → dangerous_write）+ Ask.priority 分级 + 审批弹窗中文/后果/配色 | **VERIFIED** | 新分类断言 4 个新测试全绿；SSE 实测 priority=dangerous_write |
| P0-4 | set_error_policy + set_lock_resolver 接线（install_tool_guard 默认装配） | **VERIFIED** | 2 个新测试断言 registry._error_policy/_lock_resolver 非 None |
| P0-5 | WindowsJobSandbox 移除 KILL_ON_JOB_CLOSE（防自杀） | **VERIFIED** | sandbox 测试 16 passed |
| P0-6 | 压缩消费修复（loop 用 cond.messages 替代全量投影） | **VERIFIED** | 新端到端测试 test_loop_uses_condensed_messages：压缩后消息数 < 全量 |
| P0-7 | electron-builder 补 **/*.md（安装包 0 skill→12）+ 版本五同步 10.3.1 + CHANGELOG 诚实性 | **VERIFIED** | filter 模拟匹配 SKILL.md ✅；app.py/constants/package.json×2/bat/CLAUDE.md 全 10.3.1 |
| P0-8 | skills 正文注入（_build_skills_block + VAP_SKILLS_BODY_BUDGET=2000） | **VERIFIED** | 烟雾测试正文块 694 字符含模板标题+截断标注；assembly 21 passed |
| A11 | parse_tool_call 兼容 `<function=NAME>` 变体（真实 LLM E2E 发现） | **VERIFIED** | 真实 LLM 输出解析 None → 修复后 ('delete_history', {})；70 passed |
| D4 | logs 页实时按钮死锁修复 | **VERIFIED** | disabled={live} 移除；tsc 0 错误 |
| E2E | Playwright 全套 | **23 passed (8.7s)** | 真实 chromium，后端 8002 |
| 全量 | pytest 标准子集 | **1336 passed + 2 skipped (605.57s)** | P0 全批次后实跑 |
| 覆盖率 | --cov=src | **TOTAL 80%** | 1336 passed 同跑 |

## 真实 E2E（curl/python 直打 127.0.0.1:8031，非 mock）
1. `/api/health` 200 + openapi version=10.3.1 ✅
2. `/chat` → `{"auto_run": true, "session_id": ..., "plan_steps": []}` ✅
3. `/run_stream?text=…&session_id=…` → 真实 LLM 回复 done ✅
4. **审批允许链路**：approval-request(SSE, priority=dangerous_write) → decide(True) → done ✅
5. **审批拒绝链路**：decide(False) → tool_result denied（工具未执行）✅
6. 发现并修复：真实 GLM 输出 `<function=NAME></function></tool_call>` 不被 legacy `<tool name=>` 解析器识别 → `_FUNC_RE` 兼容。

## Review Findings（独立 Critic）
- Critic-v10.3.1 审查进行中（六维度）。结果见下次更新。

## Next Gate
- Critic 终审 → commit → push → tag v10.3.1 → GitHub Release（附 Setup.exe 重传）。
