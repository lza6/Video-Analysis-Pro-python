# Workflow Status — v10.3.1 交付闭环（全部完成 ✅）

> 依据 `参考的结果计划指南.md` v3 落地。用户全权授权所有阶段。
> 本文件是任务单一状态源。**状态: DONE（2026-09-11）**

## Task Contract
- **原始目标**：指南 v3 P0 全部批次真实落地 + 真实 E2E + 独立 Critic 审查 + 版本五同步 + 提交推送 + 发行版。
- **成功标准**：默认用户（零 env 配置）用到监督层/写审批/记忆/roster/skills 正文/32 工具；审批链路真实 E2E 全通；全量测试绿；Release 资产可下载。

## Task Graph / Batches（全部 VERIFIED）
| ID | 批 | 状态 | 证据 |
|----|----|------|------|
| P0-1 | 默认路径接通（backend=react / supervisor=on / roster=on / 工具 16→32） | VERIFIED | `_get_backend` 默认 react；32 工具实测；3 个默认值测试翻转全绿 |
| P0-2 | 审批 SSE 契约（/chat 只出契约；/run_stream 执行+text 重放） | VERIFIED | 真实后端 E2E 全链路通（见下）；Critic B-1(前端不传 text)修复+回归测试 |
| P0-3 | 工具分类修正 + Ask.priority 分级 + 审批弹窗增强 | VERIFIED | 4 新分类测试；SSE 实测 priority=dangerous_write |
| P0-4 | set_error_policy + set_lock_resolver 接线 | VERIFIED | 2 新测试断言 registry 策略/锁非 None |
| P0-5 | WindowsJobSandbox 移除 KILL_ON_JOB_CLOSE | VERIFIED | sandbox 16 passed |
| P0-6 | 压缩消费修复（cond.messages 替代全量投影） | VERIFIED | test_loop_uses_condensed_messages：压缩后 < 全量 |
| P0-7 | 安装包补 skill + 版本五同步 10.3.1 + CHANGELOG 勘误 | VERIFIED | win-unpacked config/skills 实测 12 目录 |
| P0-8 | skills 正文注入（budget 2000 字符） | VERIFIED | 烟雾测试正文块+截断标注；assembly 21 passed |
| A11 | parse_tool_call 兼容 <function=NAME>（真实 LLM E2E 发现） | VERIFIED | 修复前 None→修复后 ('delete_history',{})；70 passed |
| D4 | logs 页死锁修复 + .gitignore /logs/ 收编该页 | VERIFIED | disabled={live} 移除；page.tsx 首次入库 |
| E2E | Playwright | **23 passed ×2 轮** | 打包前后各一轮 |
| 全量 | pytest 标准子集 | **1336 passed + 2 skipped**（终跑 669s） | 修复后实跑 |
| 覆盖率 | --cov=src | **TOTAL 80%** | 合跑实测 |
| 审查 | 独立 Critic（六维度） | CONDITIONAL PASS → **修复 B-1/M-2/Mi-1 后 PASS** | 10 项改动 7 REAL 3 PARTIAL → 复审无 BLOCKER |
| 发布 | commit c673be7 + push + tag v10.3.1 + Release | **资产 uploaded** | API 终验 state=uploaded, 697,507,972 bytes |

## 真实 E2E（127.0.0.1:8031 真实 LLM）
1. /api/health + openapi version=10.3.1 ✅
2. /chat → auto_run=True + session_id ✅
3. /run_stream?text=… → 真实 LLM 回复 done ✅
4. 审批允许: approval-request(priority=dangerous_write) → decide(True) → done ✅
5. 审批拒绝: decide(False) → tool_result denied（工具未执行）✅
6. E2E 过程发现并修复: GLM 输出 <function=NAME></function></tool_call> 不被解析 → _FUNC_RE 兼容

## 剩余风险 / 已知限制
- 审批 SSE 队列为进程级单例（多窗口互偷事件）—— 单用户桌面场景可接受，多窗口待 P1
- eli5 模板未覆盖新工具面（make_*/web_*/cdp_* 走兜底文案）
- M-1 降级为已知限制（Critic 同意）；P1 批次（记忆/采纳闭环/onboarding）留待下轮
