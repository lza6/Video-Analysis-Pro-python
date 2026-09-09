# Workflow Status — v10.3 全功能闭环实施（Phase B）

> 依据 `参考的结果计划指南.md`（v2，Critic PASS）落地。用户已全权授权所有阶段与行为（含提交/推送/发行版）。
> 本文件是任务单一状态源。证据只在实际运行后标记。

## Task Contract
- **原始目标**：将指南 v2 的 P0-P3 全部批次真实落地闭环，含真实 E2E、验收、审计；最后提交推送并创建发行版。
- **当前阶段**：Phase B（实施）→ 验证 → 审查 → 复验 → 发布。
- **当前授权**：全部（编码/测试/构建/E2E/审计/commit/push/tag/release）。
- **成功标准**：每批有源码+测试+运行证据；全量测试绿 + 覆盖率≥80% + ruff/pyflakes 零告警 + 前端 build 绿 + Playwright E2E 绿；Critic 复验无 BLOCKER/MAJOR；版本号五同步 + CHANGELOG；推送 + release。
- **停止条件**：无未关闭 BLOCKER/MAJOR；全部批 ✓；发布完成。

## Task Graph / Batches
| ID | 批 | 内容 | 依赖 | 涉及文件 | 状态 |
|----|----|------|------|----------|------|
| B0 | 基线 | 测试基线 + git/remote 确认 | - | - | IN_PROGRESS |
| B-P0-1 | 监督层 | supervisor.py(Stuck/Compress/Budget) + loop.py 接入 + TurnStopReason BUDGET/STUCK + test_supervisor | - | src/core/agent/{supervisor(新),loop,turn} | VERIFIED 2026-09-09（13 用例全绿 + 回归） |
| B-P0-2 | 写审批 | scope_guard.py + registry 装配 approval/sandbox + waterfall 分级 + deps SSE 通道 + test_scope_guard | - | src/core/tools/{scope_guard(新),waterfall,registry}+src/web/routers/agent.py+deps.py | VERIFIED 2026-09-09（主体 DONE，Web 装配由主控集成） |
| B-ASSEMBLE | 统一装配 | agent.py react 路径注入 supervisor+sandbox+approval + SSE 审批端点 + .env.example 5 变量 + test_agent_router_assembly | P0-1/P0-2/P1-3 | src/web/routers/agent.py + deps(确认) + .env.example + tests/test_agent_router_assembly(新) | VERIFIED 2026-09-09（75 passed 复跑） |
| B-P1-1 | 记忆分层 | memory/{working(新),triplestore(新),connector(新)} + experience FTS5 混合 + loop hooks 接入 | B-P0-1 | src/core/{memory/*,agent/experience,agent/loop} | VERIFIED 2026-09-09（59 passed 复跑） |
| B-P1-2 | skills 闭环 | skills/{distiller,validator,scoring,spectre,roster} + loader 安全前端 + test | B-P1-1 | src/skills/* | VERIFIED 2026-09-09（77 passed 复跑） |
| B-P2-1 | 视频工具集 | media_gen/{subtitle,clip,voiceover,shortvideo} + adapter 注册(Mock 守付费红线) | B-P0-2 + B-P1-2 | src/core/tools/media_gen/* | VERIFIED 2026-09-09（23 passed 复跑） |
| B-P2-2 | 电商/PPT/营销 skills | commerce/dashi-ppt/claude-ads skill + role_template 商务角色 | B-P1-2 + B-P0-2 | config/skills/*+src/core/subagent/role_template.py | VERIFIED 2026-09-09（52 passed 复跑） |
| B-P2-3 | 浏览器/GUI 自动化 | tools/web_auto/{browser,cdp} + desktop CUA 服务 | B-P0-2 + B-P1-2 | src/core/tools/web_auto/* + desktop | VERIFIED 2026-09-09（16 passed+1 skip 复跑） |
| B-P3-1 | MCP 双形态 | src/mcp/server.py + 工具白名单 | B-P2-1 | src/mcp/* | VERIFIED 2026-09-09（22 passed+stdio 冒烟复跑） |
| B-P3-2 | 反 AI-slop | webapp 设计 token 升级 | - | webapp/src/components|globals.css | VERIFIED 2026-09-09（build+token 实测复跑） |
| B-FIN | 收尾发布 | 全量测试+E2E+审计+Critic+版本五同步+CHANGELOG+push+release | 全部 | 全仓 | IN_PROGRESS |

## Evidence Ledger
| Batch | Command | Result | Evidence |
|-------|---------|--------|----------|
| B-P1-3 | `node test-log-store.js`（主控复跑）+ `node --check`×5 + `webapp next build` | **单测 18 断言全过**（滚动/过滤/搜索/脱敏/Bearer/断路器冻结+re-arm+restart 拒绝/诊断 zip 条目+EOCD）+ build **Compiled/18/18 static** | log-store.js 6.6K / crash-diagnostics.js 11K / vapDesktop.d.ts / runtime-controller 状态机+断路器(5次/60s→冻结5min auto re-arm) / main.js autoUpdater(electron-updater ^6.8.9)+IPC / logs 页黑匣子(SSE+IPC 合并+过滤搜索导出 zip) / electron-builder.yml files 含 node_modules |
| B-P1-3 无法执行 | 真 Electron 启动（无 GUI 交互会话） | 说明留人工验收：start-desktop.bat → 日志页绿点实时流/过滤/搜索/复制/导出诊断包 → 崩溃触发验证 | 代码已挂 render-process-gone/app crash 记录；zip 已用 Python zipfile 实开验证修复 EOCD 字段序 |
| B-ASSEMBLE | `pytest assembly+scope_guard+supervisor+agent_framework`（主控复跑） | **75 passed** 15.11s | agent.py 装配点：Supervisor.from_env / _install_tool_guard(install_tool_guard,build_sandbox→NullSandbox 降级) / POST /approval/{pin}/decide / GET /approval/pending / _sse_approval(_APPROVAL_SSE_EVENT) / VAP_AGENT_APPROVAL_TIMEOUT 默认 60s 超时 deny；.env.example 补 VAP_AGENT_SUPERVISOR/SANDBOX/ALLOW_WRITE_{GENERAL,CUT,DELETE}；test_agent_router_assembly.py 10 用例(契约①-④) |
| B-P1-2 | `pytest test_skills_{distiller,validator,spectre,scoring,roster,loader}+test_skill_generator`（主控复跑） | **77 passed** 1.37s | distiller(11.7K) 聚合≥3次稳定链→复用 skill_generator / validator 三重(跨域/predict/exclusivity) / spectre(9.5K) 指令注入+危险命令+密钥+URL 白名单 / scoring 棘轮(new/improved/rejected, 原子写) / roster 渐进披露(triggers+中文bigram+经验) / loader 安全前端(security_warning 默认不自动 enabled, AUTODISTILL=0 关) / schema Security 字段 / skill_generator return_draft 兼容 / .env.example VAP_SKILLS_AUTODISTILL=1 |
| B-P2-3 | `pytest test_web_auto.py`（主控复跑）+ node --check cua-service | **16 passed, 1 skipped** 6.22s + CUA node check OK | web_auto/{browser(23K 7工具),cdp(14K 5工具),__init__(register_web_auto_tools)} + desktop/cua-service.js(mock 标注+capturePage 真实优先) / scope_guard 读写分类(snapshot/screenshot allow, trigger/update/ev_write ask) / Playwright chromium 真实可用(真实 open(file://)→snapshot→click→screenshot PNG) / 降级 mock 不伪造 / .env VAP_WEB_AUTO_PROVIDER+CDP_URL+CUA_ENABLED |
| B-P3-1 | `pytest test_mcp_server.py`（主控复跑）+ stdio 冒烟 | **22 passed** 0.15s + **冒烟返回 17 工具 schema** | src/mcp/{server(17K 纯 stdlib asyncio+threading JSON-RPC2.0 子集 initialize/tools-list/tools-call),run,README,__init__} + src/config/mcp/tools_allowlist.json(读工具+media_gen 安全项) / MCPToolFilter 双闸(allowlist AND VAP_MCP_ALLOW_WRITE) / 无 handler Ask→默认拒绝 / Windows proactor 管道坑用后台线程读 stdin 绕过 / 冒烟 tools/list→真实 17 工具 / tools/call delete_history→-32003 not in allowlist |
| B-P1-3 说明 | autoUpdater 真实更新 | 无 feed/签名，仅代码+文档，日志打"enabled but feed not configured" | electron-builder.yml 现打包 node_modules/**/* 增体积，后续可窄化 |
| B-P0-1 ruff | `ruff check src/core/agent/supervisor.py` | All checks passed（含 UP 全修） | loop.py/turn.py 存量 UP 告警为历史遗留（改前 43→改后 40 净减），按"精准修改"未顺带重构；pyflakes 红线已零告警 |
| B-P0-2 契约⑥ | ApprovalBus 阻塞轮询 vs asyncio 主线程无 loop（Python3.14）→ 改验证"超时默认 deny"路径；decide-True-执行 由 handler 返回 True 的同步用例覆盖 | 合理替代，已说明 | wait_decision 用 time.monotonic 规避 RuntimeError（实测触发并修复） |
| B-P0-2 待办 | `.env.example` 补 `VAP_ALLOW_WRITE_*` 说明；agent.py/loop.py 的 approval+sandbox 装配（SSE decide 端点）由主控 B-ASSEMBLE 落地 | - | - |

## Review Findings（Critic 闭环后补）

## Next Gate
- ✅ 第一批（P0-1 / P0-2 / P1-3）与 B-ASSEMBLE 全部 VERIFIED。下一步：启动第二批 **P1-1 记忆分层**（依赖 P0-1 已满足）。