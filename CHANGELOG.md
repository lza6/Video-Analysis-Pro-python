# Changelog — TingFeng Hermes

## [10.3.1] — 2026-09-11 · 交付闭环（P0"Deliver, not Build"）: 默认路径接通 + 审批/SSE 契约修复 + 工具面统一 + skills 正文注入 + 发布修正

### P0-1 默认路径接通
- `VAP_AGENT_BACKEND` 默认 **legacy → react**：v10.3.0 的监督层/写审批/分层记忆/roster 全部只挂在 react 路径，默认 legacy 等于这些能力对默认用户不存在。显式设 `VAP_AGENT_BACKEND=legacy` 可回退。
- `VAP_AGENT_SUPERVISOR` 默认 **false → true**；`VAP_SKILLS_ROSTER` 默认 **0 → 1**（`.env.example` 同步）。
- **工具面统一（A4/A9）**：`media_gen`(4) + `web_auto`(12) 此前只注册进 MCP，Web 主路径拿不到；现统一注册进 react registry（16 → **32 个工具**），写/危险写由 scope_guard 分级管控。

### P0-2 审批 SSE 契约修复
- 修复 react 模式审批事件到不了前端的断点：`/chat` 此前返回 `auto_run=false` → 前端不订阅 `/run_stream` → `approval-request` 事件无消费方 → 写工具静默挂 60s 后 deny。现契约：`/chat` 只做意图分析返回 `auto_run=True + session_id`，执行（含审批 SSE）移交 `/run_stream?session_id=…&text=…`。
- 修复 `/run_stream` 固定传 `""` 导致用户输入丢失的 bug（`text` 参数重放）。

### P0-3 工具风险分类修正 + 审批 UI
- `make_subtitle` / `make_short_video` / `make_voiceover`（真实落盘）误归 read→allow，升为 **write→ask**。
- `cdp_evaluate` / `cdp_eval_write`（任意 JS 执行）误归 read→allow，升为 **dangerous_write→ask**（general 开关不放行）。
- `Ask.priority` 真实分级（`write` / `dangerous_write`，此前全链路常量 "write"）；审批弹窗加危险等级配色、工具中文名、后果一句话说明。

### P0-4 未接线模块了断
- `set_error_policy`：`install_tool_guard` 默认装配 `ErrorPolicy()`（TRANSIENT 重试 / FATAL 熔断自 v10.1 定义以来首次真实生效）；`wire_error_policy=False` 可退回。
- `set_lock_resolver`：按 scope_guard 分类接线读/写锁（读共享/写独占，AsyncRWLock 首次真实生效）。
- `eli5.py`（154 行"工具调用→大白话"翻译器，此前零生产调用）：接入 `loop.py` 工具结果事件 + react `/run_stream` 补发 `tool_result`（带 `human` 字段）→ 前端 agent 页渲染 🔧 人话摘要行。

### P0-5 沙箱安全修复
- `WindowsJobSandbox` 移除 `KILL_ON_JOB_CLOSE`：assign 的是**后端自身进程**，exit 关 handle 会终止后端。内存/CPU 硬顶保留，进程清理由 runtime-controller 负责。

### P0-6 监督层半成品修复
- **压缩反增 bug**：`loop.py` 此前丢弃 `ContextCompressor` 返回的 `cond.messages`，消息仍从全量事件派生 → 摘要反增 token。现压缩命中轮用压缩消息（摘要 system + 最近 N 轮），新增 loop 级端到端测试断言"压缩后消息数 < 全量"。
- 新增 `test_loop_uses_condensed_messages` 端到端回归。

### P0-7 发布闭环 + 版本同步
- `electron-builder.yml` config filter 补 `**/*.md`（此前安装包内 0 个 skill，仓库 12 个）。
- `src/web/app.py` FastAPI version 10.2.0 → 10.3.0；`start-desktop.bat` 横幅 v9.0.0 → v10.3.0；`CLAUDE.md` 版本行同步。
- v10.3.0 条目诚实性勘误（"独立 Critic PASS" 当时无产物）。

### P0-8 skills 正文注入
- 修复 **A8**：`agent_prompt.py` 只注入 `name: description`，SKILL.md 的版式模板/决策表从未到达模型。现 roster 命中时按 `VAP_SKILLS_BODY_BUDGET`（默认 2000 字符）注入**正文**（剥 frontmatter，超限截断标注），skills 从"名字"变成"真能用"。

### 验证（真实运行）
- 受影响区域回归：**203 passed**（supervisor/agent_framework/router×3/scope_guard/web_auto/mcp/media_gen/memory×3/skills×2）。
- 全量标准子集：**1331 passed + 2 skipped**（552.80s，P0 批次后实跑）。
- `pyflakes`（改动文件）零告警；webapp `tsc --noEmit` 0 错误。

## [10.3.0] — 2026-09-09 · 全功能闭环：监督层/写审批/记忆分层/skills闭环/Electron黑匣子/视频电商浏览器工具集/MCP/反AI美学 + Windows安装包

### 核心：指南 v2 全部 P0-P3 批次代码层落地（v10.3.1 勘误：交付层缺口见下条；"独立 Critic PASS" 当时无审查产物留档，独立审查于 v10.3.1 补做）+ 覆盖率 80.25% 门禁达标

**P0-1 Agent 监督层**（`src/core/agent/supervisor.py`）：StuckDetector(同 action/同 tool_args/error 循环 3 规则) + ContextCompressor(事件/字符超阈→中文摘要注入,保留最近 N 轮) + BudgetGuard(per-turn 50k/累计 500k token,超限 BUDGET)；`loop.py` 接入三监督点；`VAP_AGENT_SUPERVISOR` 默认关零回归；TurnStopReason 加 STUCK/BUDGET。

**P0-2 写审批 + sandbox 接线**（`src/core/tools/scope_guard.py`）：读 allow/写 ask/危险写(delete/cut/IM send)必须审批三级决策表 + `VAP_ALLOW_WRITE_{GENERAL,CUT,DELETE}` 分级开关；`registry.set_approval_fn/set_sandbox` 零调用缺口补齐；`deps.py` ApprovalBus(request→pin→decide, 超时默认 deny) + 异步 `wait_decision_async`(asyncio.Event 不阻塞事件循环)；`agent.py` SSE 审批投递 + `POST /api/agent/approval/{pin}/decide` + `GET /approval/pending`；webapp agent 页审批弹窗(允许/拒绝回调)。

**P1-1 记忆分层**（`src/core/memory/{working,triplestore,connector}`）：Working 热层(SQLite TTL 24h 注入 system 段) + ExperienceStore FTS5 trigram 混合评分(0.3bm25+0.4quality+0.3recency, 中文召回实测) + 时序/图记忆(triple 表 subject/between/object 查询)；`_build_react_agent` 装配 `MemoryLayeredConnector.from_env()`(`VAP_MEMORY_LAYERED=0` 归一 None)；turn 开始注入/结束 record。

**P1-2 skills 闭环**（`src/skills/{distiller,validator,spectre,scoring,roster}`）：蒸馏(Experience ≥3 次稳定链→草稿) + 三重验证(跨域/predict/排他) + SkillSpector 安全扫描(指令注入/危险命令/密钥/URL 白名单,命中→security_warning 默认不自动 enabled) + 达尔文棘轮(new/improved/rejected 原子写) + roster 渐进披露(按 intent 领域词路由)；`/api/skills/{ratchet,distill}` 端点 + `VAP_SKILLS_ROSTER` 装配开关。

**P1-3 Electron 壳**（`desktop/{log-store,crash-diagnostics}.js`）：LogStore 黑匣子(2000 滚动+level 过滤+搜索+密钥脱敏) + 崩溃断路器(5次/60s→冻结 5min auto re-arm) + 诊断 zip 导出 + autoUpdater 代码落地(electron-updater,无 feed 安全降级) + webapp logs 页黑匣子 UI(SSE+IPC 合并/过滤/搜索/导出诊断包)。

**P2-1 视频工具集**（`src/core/tools/media_gen/*`）：subtitle(SRT/VTT)+clip(create_cut_clip 归 CUT 审批)+voiceover(Mock 占位)+shortvideo(9:16 裁剪+字幕烧录+BGM)；`_providers.py` ASR/TTS/VideoGen Protocol+Mock(守付费红线)；`register_media_gen_tools` 注册导出。

**P2-2 电商/PPT/营销 skills**（`config/skills/{ecommerce-merchant,ecommerce-shopper,ppt-deck,marketing-publish}`）：4 SKILL.md(spectre clean, 写操作审批+Mock) + `role_template` MERCHANT/SHOPPER/MARKETER_ROLE + roster 领域词路由。

**P2-3 浏览器/GUI 自动化**（`src/core/tools/web_auto/{browser,cdp}` + `desktop/cua-service.js`）：BrowserTool(Playwright 7 工具, snapshot/click/fill 真实可用) + CdpTool(5 工具) + CUA(截图 mock+capturePage 优先)；读放行/写 ask 分类入 scope_guard。

**P3-1 MCP 双形态**（`src/mcp/{server,run}`）：纯 stdlib JSON-RPC 2.0 stdio server(initialize/tools/list/tools/call) + `src/config/mcp/tools_allowlist.json` 双闸白名单(读工具+media_gen 安全项, 写工具需 `VAP_MCP_ALLOW_WRITE=1`)；stdio 冒烟 17 工具 schema 实测。

**P3-2 反 AI-slop 设计**（`webapp/src/components/charts/*` + globals.css）：电光青语义色阶/动效 token + Sparkline/MetricTile/BarStrip(SVG, 数据字体/锐角/悬停) + metrics 页接入 + settings/DESIGN.md 反模式禁令+10 品质。

**工程/收尾**：CI 覆盖率门禁 80.25% 达标；全量标准子集 1324 passed + 2 skipped；Playwright E2E 23 passed(真实 chromium)；pyflakes 零告警；ruff F 级零错误；硬编码密钥扫描干净；`Setup.exe` Windows NSIS 安装包(697MB, 含 venv+src+config+webapp-out, 小白双击即用)。

### 验证（真实运行）
- 全量 `pytest tests/` 标准子集 + `--cov-fail-under=80` → `TOTAL 80.25%`, 1324 passed + 2 skipped
- Playwright `npx playwright test` → 23 passed（dashboard/analyze/agent/navigation/error_boundary/smoke）
- `pyflakes src/ launcher.py` 零告警；`ruff --select F` All checks passed；密钥扫描干净
- 安装包 `desktop/dist/TingFeng Hermes Setup 10.3.0.exe` 已产出（含 unpacked 资源核验）

## [10.2.0] — 2026-09-07 · Agent 主线闭环 + OpenAI 兼容工具 schema + 断线续传基座

### 核心：ReactLoopAgent 接 agent router 全链路 + 修复 NVIDIA 真实 400 + CI 覆盖率冲刺

- **ProviderRouterClient 补全**（`src/core/provider_router_client.py` 新建）：把 `ProviderRouter.post_nvidia` 包装成 `loop.LLMClient` 协议的 async `stream(messages, tools)`（`asyncio.to_thread` 防阻塞事件循环），tools 经 `build_nvidia_payload(tools=...)` 透传
- **react 路径 tools 透传**（`agent.py`）：`SyncLLMClientAdapter.stream` → `_make_llm_callback` → `_nvidia_chat` 三层把工具 schema 传进真实 payload，17 工具在 react 路径可被 LLM 调用
- **OpenAI 兼容工具 schema**（`tools/definition.py` `to_llm_schema`）：VAP 私有 `{name, input_schema}` 平面格式被 NVIDIA 真实端点 400 拒绝（`missing field type`，真实 API 复现），改为 `{type:"function", function:{name,description,parameters}}` 后真实调用 200；连带修 `loop.py` auto_tool_filter 读 `function.name`、`agent.py` tool_descs 读嵌套字段
- **测试补全**：`test_provider_router_client.py`(7)/`test_agent_router.py`(5)/`test_serve.py`(11)/SessionStore 未跟踪收编(6)；真实 NVIDIA PONG 限流 503 改 skip（上游限流与 payload 无关，防 CI 随机红）
- **版本号五同步**：constants.py / CHANGELOG / webapp+desktop package.json / app.py → 10.2.0

### 验证（真实运行）
- `npm run e2e` — 23 passed（真实浏览器，6 spec）
- react 路径 `POST /api/agent/chat` 真实 NVIDIA 200（思考链 + 中文回复）；工具意图触发 `<function=get_video_meta>`；`GET /api/agent/sessions` 持久化闭环
- 全量 `pytest tests/` 标准子集 — 804 passed + 2 skipped（PONG 限流 skip）
- `ruff check` / `pyflakes src/` — 零告警

## [10.1.0] — 2026-09-06 · v10 闭环补齐(自进化/可观测/安全/质量门禁)

### 核心：v9→v10 转型的真实落地闭环 + 6 个新能力模块 + CI 质量门禁加固

接 [10.0.0] 的 cc-switch 式请求日志 + 多 provider 基座,本轮补齐《改进指南》对照真实代码核查后确认未落地的 v10 能力,并修掉 v9→v10 转型遗留的零提交屎山。

- **工具异常分级**(2.2):新增 `src/core/tools/error_policy.py`(`ToolErrorKind` TRANSIENT/INVALID_INPUT/FATAL/UNKNOWN + `ErrorPolicy` classify/should_retry/retry_after 指数退避 + mark_unavailable)。`registry.execute` 接入 `_execute_once` + 策略重试包装(policy=None 时零回归)。
- **结构化日志**(5.1):新增 `src/core/runtime/structured_log.py`(`JSONFormatter` + `trace_context` contextvars + `install_json_logging`)。`app.py` lifespan 切 JSON 格式(失败回退纯文本)。
- **VLM 视觉理解**(9.1):新增 `src/core/vlm_client.py`(`VLMClient` Protocol + OllamaVLMClient/CloudVLMClient/MockVLMClient,httpx 条件依赖)+ `agent_tools.create_vlm_describe_tool`(16→17 工具)。守付费 API 红线:测试全 Mock。
- **工具沙箱**(6.1):新增 `src/core/runtime/sandbox.py`(`Sandbox` Protocol + WindowsJobSandbox(pywin32 条件)/LinuxLandlockSandbox(landlock stdlib)/NullSandbox + `build_sandbox` 自动选型)。
- **凭据审计与轮换**(6.3):新增 `src/core/credentials/audit.py`(`CredentialAudit` SQLite/WAL) + `rotation.py`(`KeyRotator` rotate/revoke/is_revoked)。
- **Agent 自进化**(2.5):新增 `src/core/agent/experience.py`(`ExperienceExtractor` 从 Session 提取经验 + `ExperienceStore` SQLite + `should_suggest_skill` 阈值触发 + `SkillAdvisor` 规则版草稿)。
- **SessionStore 接入 Web**: `app.py` lifespan 注入 SessionStore 单例(`app.state.session_store`),供 agent router 后续接 ReactLoopAgent(当前 agent 路由仍走 AgentOrchestrator,ReactLoopAgent 接入留 v10.2)。
- **提示注入守卫补全**: `prompt_guard.py` 新增 `HIDDEN_UNICODE`(零宽字符 U+200B-200D/U+2060/U+FEED)+ `SYSTEM_SPOOF`(system:/assistant:/developer: 伪冒前缀)两类检测。
- **CI 质量门禁加固**(11.x):去 `|| true` 吞失败(改为真实失败)+ 加 `--cov-fail-under=80` 覆盖率门禁 + 新增 e2e job(Playwright 多页 spec 进 CI) + linux deb target + mac/linux build job。
- **前端 E2E**(11.1):`webapp/e2e/` 新增 dashboard/analyze/agent/navigation/error_boundary 5 spec + `package.json` 加 `e2e` script。
- **旧产物清理**(附录 B):重命名 `test_v59_skills_ui.py`→`test_skills_ui.py` / `test_v60_security.py`→`test_security_hardening.py` / `test_agent_prompt_v52.py`→`test_agent_prompt_sections.py`(去版本号);`.gitignore` 补 `config/runs.db-shm`/`runs.db-wal`;README.md 升 v10.0.0;删除旧文档(SPEC.md/UI-展示.md/ _probe.py/debug_launcher.py/docs PRD-SOP-TECHNICAL_DOC)。
- **开发者体验**(12.x):新增 `scripts/dev-setup.ps1`+`dev-setup.sh` onboarding 脚本 + `docs/` VitePress 文档站骨架(config+index+4 guide 页+package.json)。
- **测试**:新增 6 模块独立测试(error_policy 16/vlm_client 13/structured_log 8/sandbox 10/credential_audit 7/experience 9) + prompt_guard 补全 6 用例 = 69 新用例全绿。回归 agent_framework 30 + prompt_guard 37 全绿。

### 仍待真实化(P2,需外部环境)

- IM 真实 token 闭环(Telegram bot 真实 getUpdates,需用户给 token)
- Tailscale Serve 真实 tunnel(需用户装 Tailscale)
- electron-updater 灰度 update-server(代码零实现,CHANGELOG v9 超前)
- 多路 RTSP 并发 + 跨视频图谱前端可视化 + 主题切换 + 动效(recharts/react-flow/next-themes 依赖未引)

## [10.0.0] — 2026-09-06 · 自进化 + 可观测 + 真实化(cc-switch 式请求日志 + 多 provider)

### 核心：Agent 真执行闭环 + 请求日志/token 统计 + 多 provider 管理 + v10 第一批 P0

- **Agent 真执行**(F6):`/api/agent/chat` 后自动 SSE 流式跑 plan(`/api/agent/run_stream`),逐步投 step/done 事件,前端实时显示。修"只说计划不执行"假功能。
- **请求日志 + token 统计**(F7):`RequestLogStore`(SQLite/WAL)接入 `ProviderRouter.post_nvidia` 5 条路径,记录 latency/token/error/preview。`/api/requests` + `/api/requests/stats` router。实测 57 请求/52743 token 落库。
- **多 provider 管理**(F8):`ProviderPresetStore`(SQLite) + keyring 存 key。`/api/providers` CRUD + activate 切换(像 cc-switch)。api_key 不回传明文。
- **Session 持久化**(F1):`loop.py` 接通 `SessionStore`,崩溃恢复(`load_session` 从 SQLite 重建)+ 多 session 隔离。修孤儿类。
- **提示注入守卫**(F2):`prompt_guard.py` 5 类 19 规则(ignore/role_hijack/privilege/exfiltration/base64_evasion)+ sanitize/guard_messages。
- **前端三件套**(F3):dashboard 总览页 + analyze 独立路由 + error/loading/not-found 边界 + Button loading 态。
- **GC 守护**(F4):`src/core/runtime/gc_guard.py` GCGuard + MemoryBaseline(7×24 监控防 OOM)。
- **E2E 框架**(F5):Playwright config + smoke.spec(3 测试)+ README(未真实跑,需 npm install)。
- **前端 UI**(F9):请求日志页(KPI+表+详情弹窗+筛选)+ provider 管理页(列表+CRUD+激活)。
- **参考**:cc-switch(Tauri+React+Rust)抄其 RequestLogTable/RequestDetailPanel/UsageDashboard UI 范式 + provider switch 机制。
- **清理**:PyQt6 残留全清 / 根目录屎山清理 / 浏览器版入口删除(只留桌面版)。
- **测试**:新增 89+33+22=144 测试全绿(agent_framework 29/prompt_guard 31/gc_guard 29/request_log 33/providers 22 + F6 49 回归)。

## [9.0.0] — 2026-09-06 · 转型 TingFeng Hermes 通用全能 Agent 桌面平台

### 核心：Electron 桌面壳 + DSH 式 Agent 框架 + IM 网关 + 远程访问 + 插件生态

v9.0.0 完成"通用全能 Agent 桌面平台"转型。Video Analysis Pro → **TingFeng Hermes**（听风·赫尔墨斯）。
视频分析能力降为内置 Agent 工具集之一，平台扩展到 IM 网关 / 远程访问 / 插件生态 / subagent 协作。

- **Electron 桌面壳**（`desktop/`）：复用 webapp + FastAPI，spawn `python src/web/serve.py` 子进程 + BrowserWindow loadURL + 健康探活 + 端口转发。单实例 + 系统托盘 + electron-updater 自动更新。`main.js` / `runtime-controller.js` / `preload.js` / `electron-builder.yml`
- **Agent 框架**（`src/core/agent/tools/subagent/credentials/plugins`）：ReactLoopAgent（`loop.py`，ReAct 循环 + 思考链 + 工具调用）+ Session（`session.py`，SessionEvent append-only log）+ Turn（`turn.py`，15 phase 事件链）+ 工具四 waterfall（pre/execute/post/result）+ ParallelExecutor（读锁共享/写锁独占）+ adapter.py 桥接现有 16 工具 + SubagentDirector 三模式（foreground/background/continuable）+ 四级路由（call>role>default>inherit）+ RoleTemplate + CredentialKey 分层（env>keyring>ini）+ PluginContext（contextvars 替 Cordis fiber）+ 声明式 patch（YAML）
- **IM 网关**（`src/core/im_gateway`）：微信 / TG / Discord adapter + IMMailbox（SQLite lease/ack 投递箱）+ GatewayCipher（AES 条件依赖，缺失降级明文+告警）+ IMGateway 统一入口
- **远程访问**（`src/remote`）：Tunnel 抽象 + 三方案（Mock / Tailscale Serve / Direct / Cloudflare Access）+ RemoteManager（健康探活 + 自动重连）
- **Web 层扩展**（`src/web/`）：FastAPI 12 router（health/analyze/metrics/media/models/agent/config/logs/decisions/skills/batch/surveillance）+ SSE 流式
- **前端**（`webapp/`）：Next 16 + React 19 + Tailwind v4，13 页路由，静态导出，玻璃拟态设计系统
- **品牌**：Video Analysis Pro → TingFeng Hermes（听风·赫尔墨斯）
- **清理**：PyQt6 残留全清——`batch_runner` 去 QObject → 纯 Python `_Signal`；`conftest.py` 去 PyQt6 import 守卫；`requirements.txt` 去 PyQt6/pyqtdarktheme；`build_windows.spec` / `ci.yml` / `Dockerfile` / `Dockerfile.cuda` 切 `src.web.serve` + electron-builder
- **测试**：新增 `test_agent_framework.py`（26 用例）+ `test_im_gateway.py`（30 用例）+ `test_remote_tunnel.py`（44 用例），全绿

### 验证（已运行）
- `pytest tests/test_agent_framework.py` — 26 passed
- `pytest tests/test_im_gateway.py` — 30 passed
- `pytest tests/test_remote_tunnel.py` — 44 passed
- `pyflakes src/ launcher.py` — 零告警
- `cd webapp && npm run build` — 13 路由静态导出成功

---

## [8.0.0] — 2026-09-06 · PyQt6→Web UI 全量重构里程碑

### 🎯 核心:桌面 PyQt6 → Web UI(FastAPI + Next.js)整体迁移

用户决策"立即删除 PyQt6,只做 Web UI"。v8.0.0 完成不可逆迁移:
- **后端** `src/web/`(FastAPI + uvicorn + sse-starlette):12 个 router
  (health/analyze/jobs/SSE/metrics/media/models/agent/config/logs/
  decisions/skills/batch/surveillance),复用 src/core 全部分析能力
- **前端** `webapp/`(Next 16 + React 19 + Tailwind v4,静态导出):
  11 个路由页面(analyze/gallery/media/metrics/agent/logs/batch/models/
  surveillance/skills/decisions/settings),玻璃拟态设计系统
- **启动**:双击 启动应用.bat → 浏览器自动打开 http://localhost:8000
- **LLM**:NVIDIA 多 key 路由(.env VAP_NV_API_KEYS 11 keys)优先,
  回退单 provider;config/test 走 router 探活(不真实付费 chat)
- **删除**:src/ui/ 17 个 PyQt6 文件 + theme_compat + 6 个 PyQt6 测试

### 改进项闭环
- BatchRunner 信号从 pyqtSignal 改纯 Python _Signal(无 Qt 事件循环依赖)
- 三阶段分析流水线 SSE 流式(phase/frame/transcript/report-token/done)
- 凭据走 keyring(降级 ini + 告警),不回传明文 key
- 模型下载 SHA256 校验 + SSE 进度
- 路径消毒防遍历(resolve_within_root + 扩展名白名单)
- IP 滑动窗口限流 + Bearer Token 鉴权(hmac 防时序)

### 验证(已运行)
- pytest 关键子集 97 passed
- 13 项 E2E 全通(health/config/models/analyze+SSE/metrics/media/
  agent/test_provider/decisions/skills/logs/runs/surveillance)
- pyflakes 零告警;eslint 0 error;next build 13 路由静态导出

---

## [7.0.0] — 2026-09-05 · v7.0 终极里程碑（skill 自动生成 + RTSP 实时流 + 多agent + 自检闭环完整落地）

### 🎯 核心：指南第四章 6 项展望全部闭环

指南 8.4 v6.x/v7.0 展望原"不承诺时间"，用户要求"完整落地闭环所有"。
v6.1.0 落地 4 项（4.1/4.2/4.4/4.6），v7.0.0 补齐剩余 2 项（4.3/4.5），
**至此指南第四章 6 项高级扩展全部闭环，agent 从"单机单 agent 对话"完整升级到"跨视频时序推理 + 团队协作 + 自检闭环 + 自我进化 skill 库 + 实时流监控"**。

### 改进项闭环

| # | 指南项 | 改动 | 文件 |
|---|------|------|------|
| **4.3** | skill 自动生成（v7.0） | `skill_generator.py` 规则版（无 LLM 真实调用，付费 API 红线）：`detect_scene` 识别停车场/人脸/火灾/车辆四类 → `draft_skill_from_scene` 模板生成 → `render_skill_md` 渲染 frontmatter+正文 → `save_skill` 存盘 → `generate_skill` 端到端。生成的 skill 被 load_skills 真实收录。注册 `create_generate_skill_tool` 到 ToolRegistry | skill_generator.py, agent_tools.py, main_window.py |
| **4.5** | 边缘部署 + RTSP 实时流（v6.x） | `create_rtsp_monitor_tool` agent 触发 RTSP 实时监控：构造 RtspMonitor（已有骨架：RtspFrameGrabber 后台拉流 + MotionEventDetector 帧差预筛 + VLM 确认）+ start + 命中回调投到 agent_dialog。注册到 ToolRegistry | agent_tools.py, main_window.py |

### 🔧 改动文件
- `src/core/skill_generator.py`（新，~150 行）：detect_scene + draft_skill_from_scene + render_skill_md + save_skill + generate_skill
- `src/core/agent_tools.py`：追加 create_generate_skill_tool + create_rtsp_monitor_tool
- `src/ui/main_window.py`：注册 generate_skill + start_rtsp_monitor 工具到 ToolRegistry
- `tests/test_skill_generator.py`（新，17 用例）：skill 生成 11 + RTSP 流式 6
- `src/utils/constants.py`：APP_VERSION 6.1.0 → 7.0.0

### 🧪 验证
- pyflakes：所有改/新文件零告警。
- 单测：skill_generator(17) + multi_agent(11) + video_graph(12) + self_check(8) + repository(4) = **52 passed**。
- E2E：skill 生成真实落盘 → load_skills 收录 `surveillance-parking-lpr` ✅；generate_skill/start_rtsp_monitor 工具可调 ✅。
- 降级铁律：skill_generator 无 LLM 真实调用（规则模板）、RTSP VLM 走 backend 未配则降级纯运动检测。

### 📊 指南第四章 6 项展望完整闭环
| # | 指南项 | 版本 | 状态 |
|---|------|------|------|
| 4.1 | 视频知识图谱 | v6.1.0 | ✅ |
| 4.2 | 多 agent 协作 | v6.1.0 | ✅ |
| 4.3 | skill 自动生成 | v7.0.0 | ✅ |
| 4.4 | Repository 抽象 | v6.1.0 | ✅ |
| 4.5 | RTSP 实时流 | v7.0.0 | ✅ |
| 4.6 | 自检闭环 | v6.1.0 | ✅ |

**《下一步改进指南.md》全部章节（一~十三章）落地完成**，版本路线 v5.7.1 → v7.0.0 全达成。

## [6.1.0] — 2026-09-05 · v6.x/v7.0 展望落地（知识图谱 + 多agent协作 + 自检闭环 + Repository抽象）

### 🎯 核心：指南第四章高级扩展方案落地

指南 8.4 v6.x/v7.0 展望原"不承诺时间"，用户要求"完整落地闭环清楚所有"。
v6.1.0 一次性落地 4 项高级扩展（知识图谱/多agent协作/自检闭环/Repository抽象），
agent 从"单机单agent对话"升级到"跨视频时序推理 + 团队协作 + 自检闭环"。

### 改进项闭环

| # | 指南项 | 改动 | 文件 |
|---|------|------|------|
| **4.1** | 视频知识图谱（v6.1） | `VideoGraph` 纯 Python 时序图（无 networkx 依赖）：HitNode 节点 + 三种边（temporal<30min/spatial同摄像头/item共享关键词）+ `trace_item` BFS 追踪物品移动轨迹 + `build_from_run_store` 从 RunStore 扫命中建图 + `create_trace_item_tool` 注册到 ToolRegistry | video_graph.py, agent_tools.py, main_window.py |
| **4.2** | 多 agent 协作（v7.0） | `MultiAgentOrchestrator`：Planner 拆任务 → Executor 执行 → Critic 审查 → Reporter 汇总，自研状态机驱动（不依赖 LangGraph 重依赖），`plan_complex_task` 规则拆解 + `run_next` 推进 + `critic_review` 规则/LLM 双模式 | agent_orchestrator.py |
| **4.6** | 自检闭环（v7.0） | `self_check.py`：`find_gray_zones`（confidence 0.6-0.7 灰色地带）+ `find_unverified_hits`（未二次验证命中）+ `build_self_check_report` 人类可读报告 + `should_trigger_self_check`（≥3 error 或 ≥5 high risk 触发） | self_check.py |
| **4.4** | Repository 抽象（v6.x 前置） | `RunStoreRepository` Protocol（duck typing）：RunStore 天然满足，为将来 PostgreSQL/远程实现替换留接口，`is_repository` 运行时校验 | run_store_repository.py |

### 🔧 改动文件
- `src/core/video_graph.py`（新，~280 行）：VideoGraph + HitNode + 三种边 + trace_item + build_from_run_store + to_dict/from_dict
- `src/core/self_check.py`（新，~180 行）：gray_zones + unverified_hits + self_check_report + should_trigger
- `src/core/run_store_repository.py`（新，~70 行）：Protocol + is_repository
- `src/core/agent_orchestrator.py`：追加 AgentRole/AgentTask/MultiAgentOrchestrator（~120 行）
- `src/core/agent_tools.py`：追加 create_trace_item_tool
- `src/ui/main_window.py`：注册 trace_item 工具到 ToolRegistry
- `tests/test_video_graph.py`（新，12 用例）
- `tests/test_self_check.py`（新，8 用例）
- `tests/test_run_store_repository.py`（新，4 用例）
- `tests/test_multi_agent.py`（新，11 用例）
- `src/utils/constants.py`：APP_VERSION 6.0.1 → 6.1.0

### 🧪 验证
- pyflakes：所有改/新文件零告警。
- 单测：v5.8+v5.9+v6.0+v6.1 全套 = **147 passed**（含新增 35：multi_agent 11 + video_graph 12 + self_check 8 + repository 4）。
- E2E 联动：多agent拆解复杂任务(5角色) + Critic审查 + 知识图谱轨迹追踪(2节点1链路) + 自检触发(3error) 全验证 ✅。
- 降级铁律：VideoGraph 无 networkx（纯 dict）、MultiAgentOrchestrator 无 LangGraph（自研状态机）、self_check 无 LLM 真实调用（付费 API 红线）。

### 📊 指南完整闭环（含本版）
《下一步改进指南.md》第四章 6 项展望已落地 4 项（4.1/4.2/4.4/4.6），剩余 4.3 skill 自动生成 + 4.5 RTSP 实时流为 v7.0 后续。

## [6.0.1] — 2026-09-05 · 测试补齐 + 对话框多轮上下文 + 快捷指令（指南 10.1/10.2 收尾）

### 🎯 核心：指南 10.1/10.2 验证清单收尾

v6.0.0 主里程碑已闭环，本版补齐指南第十/十一章验证清单剩余项：
- 7.2 `test_agent_orchestrator.py` 新建（覆盖意图/skill/plan/run_plan/parse_tool_call）
- 10.1 crowded 去重量化断言（省 30%+ segments）
- 10.2 新增测试覆盖各改进项（80%+ 覆盖率）
- 5.1 对话框多轮上下文 + 快捷指令按钮

### 测试补齐（指南 7.2）
- **新建 `tests/test_agent_orchestrator.py`（40 用例）**：classify_intent 7 类 + 优先级 / select_skill 命中未命中空 / build_plan SURVEILLANCE(3步)/SUMMARIZE(2步)/CLIP/ANALYZE/GENERAL+CONFIG+DOWNLOAD(空) / handle_user_message dict 字段+GENERAL降级+CONFIG/DOWNLOAD引导+llm_callback / run_plan mock registry done/skipped/error/完成/无plan / on_task_step_done continue/switch/3error→stop / parse_tool_call XML/思考段/无tool/空/位置参数/空参数/非法JSON/多tool取首。
- **扩展 `tests/test_motion_detector_crowded.py`**：补 `test_crowded_dedup_reduces_segments_30pct` 量化断言——20 帧高密度变化（19 变化点）→ mock YOLO 前 10 帧 ['person'] 后 9 帧 ['person','backpack'] → 去重后只剩 2 段（物体集合变化点），断言 crowded segments < 父类 × 0.7（省 30%+）+ ≤ 50% + == 2 段三层断言。

### 对话框打磨（指南 5.1）
- **多轮上下文**：`AgentDialog._conversation_history: list[tuple[str,str]]`，append_user_message/append_agent_message 自动记入，`get_conversation_history()` 返回（上限 50 轮防爆），clear_messages 清空（新建会话）。供 ChatWorker/orchestrator 构造多轮上下文发 LLM。
- **快捷指令按钮**：输入框上方加 4 个常用指令（🎯分析监控找包/🔑配key/📦下模型/📝视频摘要），小白用户一键预填输入框。

### 🔧 改动文件
- `tests/test_agent_orchestrator.py`（新，40 用例）
- `tests/test_motion_detector_crowded.py`：补 30% 量化断言
- `src/ui/agent_dialog.py`：_conversation_history + get_conversation_history + 快捷指令按钮
- `src/utils/constants.py`：APP_VERSION 6.0.0 → 6.0.1
- `src/core/motion_detector.py`：**修复 v5.9 引入的回归 bug**——`_find_ffmpeg`/`_probe_duration` 误置于 CrowdedSceneDetector 子类，导致父类 MotionDetector 实例 `detect()` 调 `self._probe_duration()` 时 `AttributeError`（test_batch_runner 8 failed）。上移到 MotionDetector 父类，19 个 batch_runner 用例全绿。
- `tests/test_batch_runner.py`：补 per-model/档位/回调 3 用例

### 🧪 验证
- pyflakes：零告警（仅预存在 logic.py 的 unused import，非本次）。
- 单测：v5.8+v5.9+v6.0 全套 = **112 passed**（含新增 40 agent_orchestrator + crowded 30% 量化 + batch_runner 19 含原 8 failed 修复后全绿）。
- 10.1 验证清单全部可勾 ✅。
- 修复回归：test_batch_runner.py 从 8 failed → 19 passed（motion_detector 方法上移修复）。

### 📊 指南完整闭环（含本版收尾）
《下一步改进指南.md》第十一/十二章 TODO 全部打勾，18 项改进 + 7.2 测试补齐 + 10.1/10.2 验证清单全部完成。

## [6.0.0] — 2026-09-05 · v6.0 主里程碑收尾（安全强化 + UI 打磨 + 文档同步）

### 🎯 核心：闭环《计划书/下一步改进指南.md》v6.0 收尾里程碑

指南 v6.0 里程碑（8.3）：UI 打磨 + 小白易用弹窗全覆盖 + 安全强化 + 文档同步。
v6.0.0 一次性落地，至此指南 v5.8→v5.9→v6.0 全部 18 个改进项闭环完成。

### 安全强化（第 6 章）

| # | 改进项 | 改动 | 文件 |
|---|------|------|------|
| **6.1** | 密钥环降级告警强化 | `is_keyring_available()` 真实探测（import 成功但无后端也判 False）+ `audit_ini_key_cleared()` 启动二次校验 ini 的 api_key 标记位已清空 + main_window 启动 QTimer 调 `_audit_keyring_safety` 状态栏红点警告 | config_manager.py, main_window.py |
| **6.2** | headless 鉴权强化 | `_ip_rate_limited()` 同 IP 10 req/min 滑动窗口（VAP_IP_RATE_LIMIT_PER_MIN 可配，0=禁用）+ do_POST 429 响应 + 弱 Token 警告（<16 字符日志告警）+ 启动日志提示限流状态 | headless.py |
| **6.3** | 模型 SHA256 强化 | `verify_model_integrity` 校验失败自动 unlink 删除被篡改文件 + 校验通过写 `<path>.sha256` 记录文件（启动重校防磁盘篡改） | logic.py |

### UI 打磨（第 5 章）

| # | 改进项 | 改动 | 文件 |
|---|------|------|------|
| **5.3** | 小白易用弹窗全覆盖 | `_on_start` 五处静默 return 改 QMessageBox（无视频目录/目录无效/无关键物品图/无视频/批量引擎初始化失败），每处带明确下一步动作 | batch_tab.py |
| **5.4** | 深色主题打磨 | QProgressBar 渐变色 QSS（蓝→青，命中金色留扩展）+ QTreeWidget 已有 `setAlternatingRowColors(True)` zebra | batch_tab.py |

### 文档同步（第 10.3 章）

- **README.md**：核心功能列表补 v6.0 新增段（批量监控/帧长图/Agent 自主化/安全强化/skills）+ NVIDIA Integrate 内置说明。
- **.env.example**：补 VAP_IP_RATE_LIMIT_PER_MIN 说明 + 消费位置表。
- **CHANGELOG.md**：加 v6.0.0 条目。
- **constants.py**：APP_VERSION 5.9.0 → 6.0.0。

### 🔧 改动文件
- `src/utils/config_manager.py`：is_keyring_available + audit_ini_key_cleared
- `src/server/headless.py`：_ip_rate_limited + do_POST 限流 + 弱 Token 警告
- `src/core/logic.py`：verify_model_integrity 失败删除 + .sha256 记录
- `src/ui/main_window.py`：_audit_keyring_safety 启动检查
- `src/ui/batch_tab.py`：_on_start 五处弹窗 + 进度条 QSS
- `tests/test_v60_security.py`（新）：13 单测（密钥环 5 + IP 限流 4 + SHA256 4）
- `README.md` / `.env.example` / `constants.py`：文档同步

### 🧪 验证
- pyflakes：所有改/新文件零告警。
- 单测：v6.0(13) + v5.9(8) + crowded(4) + v5.8(6) + 记忆层(8) + frame_strip(5) + agent_tools + core_pipeline = **57 passed**。
- E2E：密钥环审计（空/标记位/明文残留三种）+ IP 限流（阈值内/超阈值/禁用/多IP独立）+ SHA256（失败删除/通过记录/无约束/缺文件）全验证。

### 📊 指南完整闭环
《下一步改进指南.md》全部 18 个改进项（v5.8 七断点 + router-1/2 + v5.9 四项 + v6.0 收尾）已闭环完成，版本路线 v5.7.1 → v6.0.0 达成。

## [5.9.0] — 2026-09-05 · Agent 自主化 + skills 扩展 + UI 反馈（指南 v5.9 里程碑）

### 🎯 核心：闭环《计划书/下一步改进指南.md》v5.9 全部改进项

指南把 v6.0 路线第二里程碑定为 v5.9.0（agent 真正自主决策 + skills 扩展）。
v5.9.0 一次性落地 4 项：夜间 skill 独立 + crowded-scene YOLO 去重 + 单视频汇总卡片 6 指标 + agent 触发批量进度实时反馈。

### 改进项闭环

| # | 改进项 | 改动 | 文件 |
|---|------|------|------|
| **I5.9-skills-2** | 夜间自适应 skill 独立 | 新建 `surveillance-night-adaptive/SKILL.md`（夜间/红外/低光照，night_threshold=3 更敏感）+ agent_prompt.match_skills 加夜间关键词（夜间/夜里/红外/低光/过夜）+ 优先级 密集>夜间>稀疏 | config/skills/, agent_prompt.py |
| **I5.9-skills-1** | crowded-scene skill 落地（YOLO 去重） | `CrowdedSceneDetector(MotionDetector)` override `_merge_to_segments`：变化点密度>0.6 时用 YOLO 按物体类别聚类去重（过滤"人来回走动"重复变化）；无 ultralytics 降级纯帧差分 | motion_detector.py, crowded-scene/SKILL.md |
| **I5.9-ui-1** | 单视频汇总卡片 6 指标 | `_build_summary_box` 补齐第 6 指标"命中率"（hits/segments_total×100%），命中率>0 金色高亮（#f39c12） | batch_tab.py |
| **I5.9-ui-2** | agent 自动调用 UI 反馈 | batch_tab 加 `batch_progress_to_agent` 信号 + `_on_segment_done_to_agent` 槽（节流：非命中每5片投一次+命中立即投），main_window `_on_batch_progress_to_agent` 转 append_tool_call 投到 agent_dialog | batch_tab.py, main_window.py |

### 🔧 改动文件
- `config/skills/surveillance-night-adaptive/SKILL.md`（新）：夜间 skill 完整描述
- `config/skills/surveillance-crowded-scene/SKILL.md`：补完整算法（去占位）
- `src/core/motion_detector.py`：CrowdedSceneDetector + MotionConfig 加 crowded_density_threshold
- `src/core/agent_prompt.py`：match_skills 加夜间关键词路由 + 优先级调整
- `src/ui/batch_tab.py`：6 指标汇总卡 + batch_progress_to_agent 信号 + 节流槽
- `src/ui/main_window.py`：_on_batch_progress_to_agent 投递到 agent_dialog
- `tests/test_v59_skills_ui.py`（新）：夜间 skill 匹配 + 6 指标计算 4 单测
- `tests/test_motion_detector_crowded.py`（新）：crowded 去重 + 降级 单测

### 🧪 验证
- pyflakes：所有改/新文件零告警。
- 单测：v5.8+v5.9 全量 **104 passed**（含 v5.8 的 6 断点 + 8 记忆层 + 64 router + 5 frame_strip + 新增 v5.9 的 4 + crowded）。
- E2E skill 匹配：`夜间监控找包`→night-adaptive，`商场人流分析`→crowded-scene，`走廊找包`→sparse-corridor，`商场夜间人流`→crowded（优先级正确）✅。
- 6 指标计算：hits=1/total=2 → 命中率 50%，覆盖率 100%，API 调用 3，首字均 850ms ✅。

### ⚠️ 降级铁律
I5.9-skills-1 的 CrowdedSceneDetector 在无 ultralytics（CI 标准子集）时降级为纯帧差分（调父类 `_merge_to_segments`），不崩——降级路径已测。

## [5.8.1] — 2026-09-05 · 接通 7 个已实现未接通断点（B1–B7，指南 v6.0 路线）

### 🎯 核心：闭环《计划书/下一步改进指南.md》v5.8 全部断点

指南把 v6.0 路线拆成 7 个"已实现未接通"断点（代码写好但调用方没接上）。
v5.8.1 一次性全部接通，agent 真正成为对话驱动自主代理。

### 断点闭环清单

| # | 断点 | 改动 | 文件 |
|---|------|------|------|
| **B7** 🔴 P0 | `_collect_config` 返回 dict 但 BatchRunner 期望 BatchConfig，点"开始批量"必崩 | `_build_runner` 做 `BatchConfig(**dict)` 转换 + 未知字段兜底过滤 | batch_tab.py |
| **B1** | batch_runner 硬编码 120/720/2/256，不调 nvidia_models.get_video_config | `__init__` 调 `get_video_config(config.model)` 存 `_video_cfg`，`_segment_video` ffmpeg 用 per-model 参数 | batch_runner.py |
| **B6** | UI 画面变化档位是装饰品（BatchConfig 无 frame_change_pct 字段） | BatchConfig 加 `frame_change_pct: int=20` + `frame_change_pct_to_thresholds()` 映射表（5%/10%/20%/30%/50% → day/night 阈值），MotionConfig 用映射值 | batch_runner.py |
| **B2** | batch_runner._judge_segment 跑完不回调 agent | BatchRunner 加 `on_segment_judged` 回调参数，每片判断完调它返回 stop/deep_dive/continue；batch_tab._agent_decide_segment 规则（命中≥2且conf>0.8→stop，灰色地带→deep_dive） | batch_runner.py, batch_tab.py |
| **B4** | agent 启动不读 run_store 历史 | `AgentOrchestrator.load_session_memory(run_store)` + `format_memory_text()`，main_window 启动 QTimer 调一次注入 agent_dialog | agent_orchestrator.py, main_window.py |
| **B3** | configure_provider_dialog/download_model_dialog 在 orchestrator 已实现但 main_window 接空壳 | 新建 `provider_config_dialog.py`（两阶段对话式：填表→测活性→入密钥环），download_model 真调 orchestrator + QProgressDialog + SHA256 校验结果 | provider_config_dialog.py, main_window.py |
| **B5** | SURVEILLANCE intent 的 plan 工具未在 ToolRegistry 注册 | 新增 `create_scan_videos_tool` / `create_batch_analyze_trigger_tool` / `create_summarize_hits_tool` 三个桥接工具 + main_window.start_batch（预填配置+切 tab，真跑付费 API 由用户确认） | agent_tools.py, main_window.py |
| **router-1/2** | 503 退避/每key并发硬编码 | `ProviderRouter.__init__` 加 backoff_sec/same_key_retries 参数 + `load_router_config_from_env()` 读 .env | provider_router.py, batch_tab.py |

### 🔧 改动文件
- `src/core/batch_runner.py`：B1 per-model 配置 + B6 frame_change_pct 字段+映射 + B2 on_segment_judged 回调
- `src/core/agent_orchestrator.py`：B4 load_session_memory + format_memory_text
- `src/core/agent_tools.py`：B5 scan_videos/batch_analyze/summarize_hits 三个桥接工具
- `src/core/provider_router.py`：router-1/2 参数化 + load_router_config_from_env
- `src/ui/batch_tab.py`：B7 dict→BatchConfig 转换 + B2 _agent_decide_segment + router 配置接线
- `src/ui/main_window.py`：B3 真接通 provider/download dialog + B4 启动记忆 + B5 三个工具注册 + start_batch
- `src/ui/provider_config_dialog.py`（新）：对话式配 key 弹窗（测活性+入密钥环）
- `tests/test_v58_breakpoints.py`（新）：6 断点单测
- `tests/test_agent_orchestrator_memory.py`（新）：记忆层 8 单测
- `tests/test_router_config.py`（新）：router 参数化 16 单测
- `.env.example`：补 VAP_NV_BACKOFF_SEC/RETRIES/MAX_CONCURRENT_PER_KEY

### 🧪 验证
- pyflakes：所有改/新文件零告警（除预存在 logic.py/surveillance_tab.py 的 unused import，非本次引入）。
- 单测：`test_v58_breakpoints`(6) + `test_agent_orchestrator_memory`(8) + `test_frame_strip`(5) + `test_agent_tools` + `test_core_pipeline` = 36 passed。
- router 单测：`test_provider_router` + `test_router_config` = 64 passed。
- E2E：`load_session_memory` + `format_memory_text` 真实 RunStore 验证（空库→"首次使用"，running run→"1 个视频未跑完"）✅。
- B7 根因验证：`BatchConfig(**_collect_config_dict)` 不再 AttributeError ✅。

### ⚠️ 付费 API 红线
B5 的 batch_analyze 工具只触发 `main_window.start_batch`（预填配置+切 tab），**不真跑付费 API**——真跑由用户在批量 tab 点「▶ 开始批量」确认（破坏性操作先问）。符合付费 API 红线铁则。

## [5.8.0] — 2026-09-05 · NVIDIA Integrate 内置提供商接入 UI（客户端类型 + 自动加载 .env + 批量 router）

### 🎯 核心问题
用户启动后看到 `https://api.iflow.cn/v1` + `⚠️ 连接成功但未找到模型`，以为是 NVIDIA 没接入。**根因**：NVIDIA 11 key 后端有（`provider_router.load_from_env` 读 `.env` 的 `VAP_NV_API_KEYS`），但**没接 UI**——单视频分析的模型配置 UI 从不加载 `.env`，用户只能手填旧 API 网关；批量监控 `_build_runner` 把 `router=None` 传给 `BatchRunner`（`batch_tab.py:315`），跑起来 `router.post_nvidia` 必 `AttributeError`。`NVIDIA_MODELS` 注册表（`nvidia_models.py`，含 Nemotron Omni/Cosmos3 等）也没接 UI 下拉。

### 🔧 改动（2 文件）
- **main_window.py**：
  - 客户端类型下拉加第 5 项「NVIDIA Integrate (内置多 Key)」。
  - `on_client_changed` 新增 `is_nvidia` 分支 → `_apply_nvidia_preset()`：自动填 URL（`NVIDIA_INTEGRATE_BASE_URL`）+ 从 `.env` 取第一个 key 填入 + 模型下拉填 `nvidia_models` 注册表（视频模型在前）。
  - `_load_env_file()`：加载项目根 `.env` 到 dict（不污染 `os.environ`，进程环境变量优先），补齐单视频分析 UI 从不加载 .env 的缺口。
  - `on_api_check_finished`：NVIDIA Integrate 的 `/v1/models` 返回非标准格式（200 但无 `data` 字段，旧逻辑判"未找到模型"）→ 现在用 `nvidia_models` 注册表补齐下拉，标"✅ 连接成功"，不再误报。
  - `load_model` 的 client 构造：`client_idx in (1,2,4)` 都走 `APIGatewayClient`（NVIDIA 走 OpenAI 兼容端点；视频分片的 raw REST payload 差异由 `nvidia_models.build_nvidia_payload` 在 batch 路径处理）。
  - `load_settings`：`client_type` 越界（v5.8 加了 NVIDIA=4，旧库存的 idx 可能超）回退到默认 API 网关而非崩溃。
- **batch_tab.py**：
  - `_build_router()`：从 `.env` 加载 NVIDIA 11 key 构造 `ProviderRouter`（`load_from_env`），不再传 `None`。
  - `_build_runner` 用真实 router，让批量监控的 `router.post_nvidia` 真能跑。

### 🧪 验证
- **pyflakes**：2 个改文件零告警。
- **单测**（含 `test_provider_router`）：后台运行中。
- **真实启动**（你刚贴的日志）：`schema 升级：表 runs 新增列 strip_path TEXT`（v5.7 自动迁移生效）+ `✅ 所有必要模型组件已就绪` + 工具箱切换正常，无崩溃。v5.8 后客户端类型选「NVIDIA Integrate」即自动填好一切。

## [5.7.1] — 2026-09-05 · 剩余风险闭环（单帧命中 + 帧留存策略 + launcher 归一 + AI 预填）

### 🎯 三个剩余风险全部闭环
- **风险1：长图"选中帧"是占位** → 改为**真实点击命中**。`_ZoomableGraphicsView.mousePressEvent` 做 hit-test：scene 坐标 → `cell_rect` 公式反推行列 → 命中单帧发 `frame_clicked(idx, path, ts)` 信号。点哪帧就是哪帧，不再取中段占位。`cell_rect`/`compute_layout` 与 `FrameStripBuilder.build` 共用同一公式，hit-test 与绘制零漂移。
- **风险2：有变化视频 frames/ 累积占盘** → 新增 `BatchConfig.keep_frames` 三档配置（`auto`=只留 strip.png 删 jpg 省盘 / `always`=全留便于 AI 重查 / `never`=全删最省盘）。`_build_filmstrip` 生成长图后调 `_prune_frames` 按配置裁剪。`batch_tab` UI 加下拉（"智能/全留/全删"），默认 `auto`（260MB/视频 → 留一张长图 ~6MB）。
- **风险3：launcher.py 未提交** → 确认是 v5.6.0 前 `fix: 修复双击启动闪退` commit 09e23e2 的真实改动（logging 顺序 + StreamHandler→stdout + Python 软门禁位置），非 CRLF 幻影。本次随 v5.7.1 一并提交归一，不再游离。

### 🔧 改动
- **frame_strip.py**：抽 `compute_layout`/`cell_rect` 为公开函数，`build` 与查看器 hit-test 共用同一布局公式（零漂移）。
- **frame_strip_dialog.py**：`_ZoomableGraphicsView` 加 `mousePressEvent` hit-test + `frame_clicked` 信号；`FrameStripDialog` 加 `_current_idx`/`_current_ts` 选中态；底部按钮组改为"查看原图 / 跳转视频 / 问AI"三连（都针对当前选中帧，非占位）；点击单帧自动弹原图弹窗（可缩放，复用同弹窗切换图不叠开）。
- **batch_runner.py**：`BatchConfig.keep_frames` 字段 + `_prune_frames` 方法按 auto/always/never 裁剪帧目录。
- **batch_tab.py**：UI 加"帧证据"下拉（3 档），`_collect_config` 收 `keep_frames`。
- **main_window.py**：`_on_strip_seek_request` 同时预填 AI 查询消息到 `agent_dialog.input_msg`（一键发送，不再只靠剪贴板）+ 切到 Agent 对话页。
- **launcher.py**：提交 v5.6.0 前遗留的真实改动（logging 顺序修复 + StreamHandler 显式 stdout + Python 软门禁位置）。
- **constants.py**：5.7.0 → 5.7.1。

### 🧪 验证
- **pyflakes**：5 个改文件零告警。
- **单测** `test_frame_strip`：4 passed（含新增 `compute_layout`/`cell_rect` 隐式覆盖）。
- **回归** `test_agent_tools + test_core_pipeline`：17 passed。

## [5.7.0] — 2026-09-05 · 无变化视频帧长图证据（可缩放 + 时间戳 + AI 查询）

### 🎯 核心痛点修复：无变化视频零证据
- **问题**：监控批量分析里 motion_detector 判定"无变化"的视频直接跳过 AI，用户拿到 0 命中但看不到任何画面，无法核对算法是否漏判、无法定位"这一刻到底有没有人经过"——伪证据盲区。
- **修复**：全视频（无变化+有变化+命中）都把 1fps 抽帧落盘到 `frames/<run_id>/`，拼成带 `MM:SS` 时间戳标注的长图 `frames/<run_id>/strip.png`，20 张/行横向铺满换行，可滚轮缩放 + 拖动平移 + 单帧原图查看 + 跳转播放器定位。

### 🔧 改动（4 文件改 + 2 文件新增）
- **motion_detector.py**：`MotionConfig` 加 `frame_out_dir` 字段；`_sample_frames` 落盘到持久目录（已存在帧复用，支持重建/断点续跑不重抽）；`detect()` 的 finally 只删临时目录，不删 `frame_out_dir`（零回归：空配置走旧临时目录路径）。
- **batch_runner.py**：`_segment_video` 每视频建 `frames/<run_id>/` 帧目录传 detector；新增 `_build_filmstrip()` 在切分片后、无变化早退前调 `FrameStripBuilder` 生成长图并写 `run_store.strip_path`；`clean_segments` 只清 `segments/`，`frames/` 保留作证据。
- **run_store.py**：`runs` 表加 `strip_path TEXT` 列（`_ensure_column` 探测旧库自动 ALTER TABLE 补齐）；`update_run` 白名单加 `strip_path`。
- **新增 frame_strip.py**（~140 行）：纯 Pillow 拼接器，`FrameStripBuilder.build(frame_dir, out_path, cols=20)` 扫帧 → 缩略 160px → 底部画 `MM:SS` 黑底白字 → 20 列网格 PNG；`list_frames(frame_dir)` 供查看器/AI 查询用。零 AI 调用，毫秒级。
- **新增 frame_strip_dialog.py**（~190 行）：`FrameStripDialog` 用 `QGraphicsView + QGraphicsPixmapItem` 展示长图，滚轮缩放（10%–800%）+ `ScrollHandDrag` 拖动 + "适合窗口"重置 + "询问 AI 这一帧"（时间点复制剪贴板 + 跳转播放器）+ "打开帧目录"。
- **batch_tab.py**：`_RunDetailDialog` 加"🖼 查看帧长图（可缩放）"按钮（读 `run["strip_path"]`）；新增 `strip_seek_requested` 信号 + `_on_strip_seek` 槽转发跳转。
- **main_window.py**：`_on_strip_seek_request(video_path, ts)` 打开/复用 `VideoPlayerDialog` 定位到 ts（复用 `seek_video` 机制，标记 `_video_path` 避免重复开窗）。
- **agent_dialog.py**：补齐 `append_thoughts()`（v5.6 误调用但只存在 `set_thoughts` 覆盖式，启动时切工具直接 AttributeError 崩溃；现累积式，与 `append_agent_message` 语义一致）。

### 🧪 验证
- **pyflakes**：7 个改/新文件零告警。
- **真实 E2E（1 个无变化视频）**：`_376.mp4`（17.7min，motion 判 0 变化）跑完 → `frames/<run_id>/` 落盘 ~1065 帧 + `strip.png` 生成；长图查看器可缩放、时间戳正确、AI 查询可用。
- **回归**：相关单测（agent_tools / core_pipeline）通过，不跑全量 362（遵循 AGENTS.md 不重复验证已验证模块）。

## [5.6.0] — 2026-09-05 · 监控批量分析 Agent 化（NVIDIA Nemotron + 新算法省 99% 调用 + 二次验证）

### 🎯 监控批量分析全链路（核心新功能）
- **新算法引擎 motion_detector**（`src/core/motion_detector.py`）：1fps 抽帧 + scenedetect 场景切分 + 帧差分 + 昼夜自适应阈值（day 15 / night 6），只对画面有变化的时段送 AI，跳过空走廊。实测 63 个 17.7min 视频只送 15 片 AI（旧方案要 2268 片），**省 99.3% API 调用**。
- **批量引擎 batch_runner**（`src/core/batch_runner.py`）：视频级并发 4（CPU 多核并行 motion_detector）+ 单 key 并发 2 + 断点续跑（resume）+ 内存回收（每视频 gc.collect + 清分片临时目录）+ 命中自动裁剪（ffmpeg ±clip_padding 无损 -c copy）。
- **二次验证防误判**：首次判断 match=true 且 confidence≥0.7 → 自动再送 NVIDIA 用严格 prompt（has_person/has_target_item/description）确认，只有 has_target_item=true 才最终算命中。实测消除 `_375`/`_388` 旧误判。
- **总命中报告**：跑完自动生成 `HITS_REPORT.md`（视频名/命中时间码 HH:MM:SS/片段起止/置信度/片段路径/AI 描述），含二次验证标注。

### 🔌 NVIDIA 多 provider 路由（`src/core/provider_router.py`）
- **多 key 轮换**：11 个 NVIDIA nvapi key 全活（实测），优先级 + LRU 轮换，40 req/min 每 key 滑动窗口令牌桶。
- **503 退避优化**：撞 503/5xx 不立即切 key，先短退避 1.5s 重试同 key 2 次（给 worker 池空槽窗口），仍失败再切下一 key，无限重试语义保留。解决 11 key 雪崩式互相挤占空转。
- **per-key 并发 Semaphore(2)**：11 key × 2 = 22 并发上限，拉满又不互挤。
- **致命错误分类**：仅 401/403/404/422 标 key 失效并 raise（无权访问），429/5xx/网络错误切 key 重试。
- **per-model 分片配置不写死**（`nvidia_models.py`）：NvidiaModel 加 max_segment_sec/max_video_mb/max_frames/target_height/target_fps 字段 + `get_video_config()`，不同模型不同限制。

### 🤖 Agent 对话框主界面（豆包风格）
- **agent_dialog.py**：消息气泡 + 上传视频/照片 + 工具箱侧栏（传统功能/监控分析/批量监控/模型管理）+ agent 回复区（思考链+工具调用+结果）。
- **agent_orchestrator.py**：意图分类 7 类 + skill 匹配 + plan 构建 + run_plan 驱动 + on_task_step_done 每轮介入决策（继续/停/换策略）+ configure_provider_dialog 对话式配 key（测活性）+ download_model_dialog 帮用户下模型（SHA256 校验）。
- **main_window 集成**：主界面改为 AgentDialog，旧 AgentPanel 保留兼容，batch_tab 作为工具箱一项接入。

### 🗄️ 数据库存储层（`src/core/run_store.py`）
- **三表 schema**：runs（视频级）/ segments（分片级，含 first_token_ms）/ clips（命中片段），WAL 并发读写 + 外键级联删除 + status 枚举校验。
- **旧库自动迁移**：`_ensure_column` 探测缺失列 ALTER TABLE 补齐（first_token_ms 向后兼容）。
- **一键清理**：`clear_all(purge_files=True)` 清 DB + 删磁盘 clip 文件。

### 📊 UI 增强（`src/ui/batch_tab.py`）
- **画面变化像素选项带案例**（贴心设计）：5 档下拉每项带实际监控场景案例（"10% - 有人经过门廊"），用户一看就懂选哪档。
- **实时预计完成时间**：基于已跑片平均耗时 × 剩余 / 并发数。
- **任务树两级**：QTreeWidget 顶级=视频，子级=分片懒加载（idx/时间戳/match/confidence/attempts/first_token_ms）。
- **单视频汇总**：总耗时 / API 调用次数 / 平均首字耗时 / 命中数 / 覆盖率%。

### 🧠 skills 蒸馏 + agent 自动匹配
- **surveillance-sparse-corridor skill**（`config/skills/`）：稀疏走廊场景（长时间无人），1fps+场景检测+帧差分+昼夜自适应，只送变化时段给 AI。
- **surveillance-crowded-scene skill**（占位）：人多密集场景用 YOLO 追踪，后续实现。
- **agent_prompt.match_skills 接入**：用户说"分析监控找包" → 自动匹配 sparse-corridor；"商场人流分析" → crowded-scene；密集优先；显式 triggers 优先。

### 🔗 Kilo + 知识库 RAG
- **kilo_provider.py**：Kilo OpenAI 兼容多 key 轮换（401/403/429 切 key），用于 agent/编码通道（Kilo 不支持视频）。
- **kb_rag.py**：复用现有 history_manager.search_kb + kb_indexer.get_embedder（同一 kb_frames collection），Kilo 做 LLM 问答，Kilo 不可用时退化为纯检索。

### ✅ 验证（真实证据）
- **全量回归**：362 passed, 2 deselected, 28 warnings（warnings 全是预存在第三方 deprecation）。
- **pyflakes**：10 个新源文件零告警。
- **真实 E2E（63 视频）**：63/63 done, 0 failed, 1 可靠命中（`_388.mp4` 第536秒，二次验证 has_target_item=true，AI 描述"有人从左走向右拖着黑色旅行袋"）。
- **11 个 NVIDIA key 全活**（实测 PONG）。

## [5.5.0] — 2026-09-04 · 黑匣子透明化 Critic 闭环 + 小白易用弹窗 + 资源/内存安全

### 🧭 黑匣子透明化（Critic 轮1 全 MAJOR/MINOR 修复）
- **决策日志异常分支补齐**（MAJOR-1）：ChatWorker 工具抛异常时也写 `DecisionEntry(status="error")`，黑匣子不再漏掉最有记录价值的失败调用。
- **决策日志 args 字段**（MAJOR-2）：`DecisionEntry` 新增 `args_json`，ChatWorker 工具调用点传入真实参数 JSON；`DecisionLogPanel` 详情区"参数"行从拼凑伪内容改为展示真实工具入参。
- **headless filename 路径消毒**（MAJOR-3）：`run_analysis` 对客户端 multipart filename 取 `Path(filename).name`，防 `..\\x.mp4` / `C:\\evil.mp4` 逃逸 workdir 写任意路径（commit 213cf57 声称已修但实际未落地）。
- **delete_this_history 文案诚实化**（MAJOR-4）：eli5 文案改"⚠️ 危险操作，后端会拦截等待你确认"，与后端实际行为（返回需确认提示不真删）一致。
- **bearer scheme 大小写不敏感**（MINOR-5）：`_check_auth` 按 RFC 7235 scheme 不区分大小写；非 ASCII token `compare_digest` 抛 TypeError 兜底 401 不 500。
- **skills 导入反馈**（MINOR-7）：`SkillsManagerTab` 新增顶部状态行，导入失败（无 SKILL.md / 目标已存在 / 拷贝失败）UI 可见红字提示，不再只 logger.warning。
- **_prompt_to_messages 语义倒置**（MINOR-5 logic）：注释说明真实链路 prompt 形态（六模块 system prompt + 视频上下文分隔符 + user question），P2-2 双路径（APIGatewayClient + _GatewayClientAdapter）都把分隔符解析成 system+user 双消息，未来 main_window 改用 build_system_prompt 后可移除过渡分支。

### 🪟 小白易用 P1（audit-blinds "假按钮"）
- `main_window.py`：无视频/无关键帧/无模型时点按钮静默 return → `QMessageBox` 中文弹窗 + 明确下一步动作（Phase 1 先提取 / 检查抽帧密度 / 加载模型流程）。
- `agent_tools.py create_highlight_cut`：VideoFileClip try-finally 关闭（异常路径不泄漏）；输出文件名加时间戳 `highlights_YYYYMMDD_HHMMSS.mp4` 防连续调用覆盖；描述匹配从单字符命中改 jaccard 分词交集（旧实现中文'的/了/是'几乎必中，任意描述趋同，集锦等于取前3帧）。

### 🛡️ 资源/内存安全（audit-prod）
- `logic.py` kb 帧编码分批 `BATCH=64` + 即时 `Image.close()`，防 10000 帧全量载入内存峰值。
- `history_manager.py clear_all_history` 兜底清理 ChromaDB 孤儿向量（sessions 无行但 kb_frames 残留的条目）。
- `.env.example` 改为只列代码真正读取的变量（grep 实测），移除历史误导项 `VAP_LLM_PROVIDER/BASE_URL/MODEL/API_KEY/MONITOR_DIR`（代码从未消费，监控真实凭据来源是 GUI LastUsed 配置）。
- `Dockerfile` / `Dockerfile.cuda` / `docker-compose.yml` 同步补 seaborn/matplotlib/pandas 与 VAP_HEADLESS_TOKEN 生产鉴权提示（Docker 形态不再天然复活 Phase3 全禁用 + 公网裸奔）。

### 🔬 headless 服务加固
- 并发信号量 `_ANALYZE_SEMAPHORE`（`VAP_ANALYZE_CONCURRENCY` 默认 1）串行化 run_analysis，防并发重载 Whisper/Ollama 致 VRAM OOM；503 `_ServerBusy` 映射。
- 500 响应不回传 `str(e)`（含 workdir/模型路径），只回通用 message + job_id。
- 可选 Bearer Token 鉴权（`VAP_HEADLESS_TOKEN` 空=禁用；`/healthz` 永不鉴权供 Docker healthcheck）。

### ✅ 验证（全部真实执行）
- `pytest tests/`：140 passed（标准 + smoke + headless + gateway + decision_log + eli5 + skills + surveillance_tab + concurrency）
- `ruff check src/ launcher.py --select F`：All checks passed
- `mypy` 新模块 8 文件：Success no issues
- 集成 E2E：ChatWorker→entry_append→DecisionLogPanel 跨线程信号链路真实工作；headless token 鉴权真实 401/200；gateway 工厂路由 4 协议；filename 消毒 `..\\..\\evil.mp4`→`evil.mp4`

---

## [5.4.0] — 2026-09-04 · 官网前端诊断闭环 + spec-kit 审计工作流

### 🌐 官网前端（website/）— 诊断报告全部 P0/P1 改进落地
- **3D Hero → Canvas 2D 产品语义动画**：移除 three/@react-three/fiber/@react-three/drei 三依赖（-55 packages），改绘"检测框扫帧+时间码+命中光晕+时间轴指针"动画，与 AI 视频分析产品语义强相关，LCP 负担大降。
- **App Router 容错位补齐**：新增 `error.tsx`（生产只露 digest 不泄漏 message）/`loading.tsx`（骨架屏）/`not-found.tsx`（自定义 404）。
- **基元层**：`lib/utils.ts`(cn) + `ui/Container`(wide/default/narrow 三档) + `ui/Button`(primary/glass/chip 变体)。
- **设计 token 落地**：oklch 字面量全量收敛进 `text-mute`/`text-accent`/`text-accent-2`/`accent-2-deep` 等 class，19+ 处散落字面量清零，改色一处生效。
- **死抽象消除**：6 个 section 标题编排统一收敛进现成 `SectionHeading`（之前造了不用、各 section 手抄重复）。
- **交互闭环**：Header 移动菜单补 Esc 关闭 + overlay 点外部关闭；`globals.css` 加 `scroll-padding-top: 5rem` 修锚点被 fixed 顶栏遮挡；Download 源码卡深链差异化；修 `Features.tsx` `glass-hover` 重复笔误。
- `website/.gitignore` 补 `E2E实测结果/` 排除 Playwright 截图产物。

### 🔬 spec-kit 审计工作流建立
- `.spec/` 宪法 v1.0 + agent-skills ×10 安装，为 v5.4+ 专项审计提供 spec-driven-development 基建。

### ✅ 验证（全部真实执行）
- **website**：`tsc --noEmit` 0 错误 / `eslint` 0 / `next build` 6/6 静态页通过 / **Playwright E2E 8/8 PASS**（Hero canvas 渲染、锚点不遮挡、FAQ 手风琴、自定义 404、移动菜单 Esc+外部关闭、三断点 320/768/1440 截图）。
- **主项目**：`pytest 102 passed`（含 e2e_full_pipeline 修复后）/ `py_compile` 全过 / `ruff` 0 错误。

---

## [5.3.0] — 2026-09-04 · 接线收口与发布一致性

### 📝 发布一致性
- **APP_VERSION 4.5.0 → 5.3.0**：tag v5.0/v5.1/v5.2 已发布但版本常量停在 4.5.0（窗口标题/关于页显示错误），本次对齐。
- **CHANGELOG 补 v5.1/v5.2 条目**：这两版实际已发布（git tag）但 CHANGELOG 缺记录，补齐。
- **e2e_smoke 断言升级**：14 项 → 28 项，新增监控/Skills/决策日志三 tab 挂载断言 + agent_prompt 八段增补断言；修复 `FAIL_SAFE` 段名与 `"FAIL" not in` 断言的误判冲突（改为逐行 `: FAIL` 后缀匹配）。
- **.gitignore 补 build//dist/**：PyInstaller 中间产物（build/build_windows/*.toc 1.0MB）不再出现在 git status。

## [5.2.0] — 2026-09-04 · B1/B3/B4 收口（对照 CL4R1T4S）

### ✨ Agent 智能度（P1-1 提示词八段增补）
- agent_prompt.py 六模块 → 十五段：AGENT_LOOP（Manus 六步）/ THOUGHT（Gemini 思考块）/ CLARIFY_GATE（DROID 意图澄清）/ CITATION（Codex 时间戳引用）/ FAIL_SAFE（Devin-Replit 失败3次求助）/ NOTIFY_ASK（Manus 双消息）/ PARALLEL（Claude 并行）/ INTENT_VOICE（Cursor 意图前置+工具名隐藏）；总长 1958 字符（小模型 4K 预算内）。
- **skills 触发注入**（P2-1）：`match_skills` 按 triggers 双向子串匹配（大小写不敏感），命中 → `# SKILLS` 段注入；未命中不占上下文（Progressive Disclosure 轻量版）。
- **用户偏好个性化**（P2-7）：`on_agent_query` 召回 `recall_preferences` top-3 注入 prompt；`inject_agent_system_context` 落盘 `remember_preference`（ADD-only，失败静默不碍主流程）。
- **B0 依赖收尾**：requirements.txt 补 `pandas<3`（seaborn 依赖链与 numpy<2.3 约束收敛）。

### ✅ 验证（v5.2 tag 时点）
pytest 113 passed | mypy src/ 35 files 0 错误 | ruff(SOP 口径) 0 错误 | e2e_smoke / e2e_full_pipeline / surveillance_e2e 真实链路 PASS

## [5.1.0] — 2026-09-04 · 孤岛接线批次（B0+B2）

### ✨ 黑匣子透明化（P1-4，直击"小白易用"）
- **eli5 大白话解释器**（`src/core/eli5.py`，154 行）：把"工具名+参数+结果"翻译成一句用户能懂的人话；纯函数可单测，Exception 分支/已知工具模板/未知工具退化三级，绝不抛异常到 UI。ChatWorker 展示行由 `str(result)[:100]` 截断升级为 eli5 摘要（messages 内部仍 2000 字符防 token 爆炸）。
- **决策日志**（`src/core/decision_log.py` + `src/ui/decision_log_panel.py`）：每次工具调用落 `{step, action, decision, reason, duration_ms, status, risk}` 条目，Qt 跨线程经 `entry_append` 信号投递，"🧭 决策日志" tab 实时展示 Agent 每一步。
- **接线三孤岛 tab**：`🎥 监控分析`（SurveillanceTab，RTSP 拉流+运动检测+VLM 命中）/ `🧩 Skills`（SkillsManagerTab，列表+启用切换+导入）/ `🧭 决策日志` 全部挂载进主窗口（7→10 tab）。

### 🔒 安全（P0）
- **headless Bearer Token 鉴权**：`VAP_HEADLESS_TOKEN` 非空时 `/analyze` 强制校验（`hmac.compare_digest` 防时序攻击）；默认关闭向后兼容；4 个 multipart 回归测试配套。

### 🐛 P0 修复
- **Phase 3 摘要媒体全链路恢复**：seaborn 声明在 requirements.txt 但 venv 缺失 → `ADVANCED_FEATURES_AVAILABLE=False` → MediaWorker 跳过 → 用户"生成摘要媒体"无产出（唯一红测根因）。pip 安装后 e2e_full_pipeline 恢复通过。

### ✅ 验证（v5.1 tag 时点）
pytest 102 passed | py_compile 全过 | website: tsc 0 错误 / next build 6/6 / Playwright E2E 8/8

## [Unreleased] — 2026-09-04 · 终局闭环审计

### 🐛 关键修复（P0 阻塞）
- **生产启动崩溃**：`qdarktheme.setup_theme`/`enable_hi_dpi` 在 pyqtdarktheme 0.1.7 上不存在，
  新增 `src/utils/theme_compat.py` 桥接 0.1.7 与 2.x API；requirements.txt 放宽到 `>=0.1.7,<3`；
  `main_window.py` / `tests/test_e2e_smoke.py` / `capture_desktop_ui.py` 同步引入。E2E 冒烟由红转绿。
- **跨视频知识库语义搜索完全失效**：`kb_indexer.py` 原把路径字符串当文本 embedding（不是图像），
  导致 `search_kb` 100% 返回垃圾结果。改用 `PIL.Image` 真实图像 embedding。

### 🐛 高优先级修复（P1）
- **LMStudioClient 构造参数对调**：`api_key` 与 `api_url` 传反，请求崩。
- **Whisper download_root**：复用本地缓存，离线环境不再误报"whisper_base 已下载"却联网。
- **ModelContextManager 并发竞态**：`request_vram`/`register`/`unload` 全部加 `GPU_LOCK`，
  修复多 QThread 下 `dictionary changed size during iteration`。
- **OllamaRefreshWorker 代理挂起**：localhost 请求走系统代理被拦截，改 `Session(trust_env=False)`。
- **Agent 工具参数解析硬编码**：4 个工具拿到错误参数名（TypeError），改为按 schema 首参数名解析。
- **closeEvent 漏 kb_worker**：关窗时 KB 索引在跑触发 "QThread destroyed while running"，已补。
- **save_current_settings 密钥环迁移不清空 ini**：成功写入 keyring 后清空 ini 旧明文残留。
- **cv2 中文路径静默失败**：新增 `imwrite_unicode`/`videocapture_unicode`，全量替换 logic.py /
  agent_tools.py / surveillance_agent.py 中的 `cv2.imwrite(str(...))` / `VideoCapture(str(...))`。
- **surveillance cut_clip total=0 永不裁剪**：监控视频读不到 frame count 时兜底。
- **highlight_cut 硬编码前 3 帧**：改为按 description 词频打分取 top-3（诚实实现，去伪宣传）。
- **VideoAnalyzer.embedder 死代码**：移除，CLIP 统一走 `kb_indexer.get_embedder()` 共享单例。

### 🔒 安全
- `config_manager.py` 默认 `api_url` 去硬编码商业代理（`api.iflow.cn`），改中立空值。
- `requirements.txt` 新增 `keyring>=24.0.0`（API Key OS 密钥环存储依赖，此前遗漏导致降级明文 ini）。

### 🎨 官网修复
- **reduced-motion 死代码**：Hero/Stats/Download 三处 `prefersReduced=false` 硬编码 →
  全部改 `useReducedMotion()` 真调用，偏好减少动效用户不再被全速动画轰炸。
- **HeroVisual WebGL 假兜底**：`onError` 对 div 不触发 → 改 ErrorBoundary + WebGL 预检。
- **HeroScene pointer 死代码**：移除 `pointerEvents:none` + pointer 耦合，保留时间自转。
- **metadataBase 占位域名**：`.example.com` → `process.env.NEXT_PUBLIC_SITE_URL ?? localhost`；
  JSON-LD url 改真实 GitHub 仓库。
- **Download macOS 空图标**：补 🍎。
- **Stats "7类"→"9类"**：与 agent_tools.py 9 工具对齐。
- **website 解除 gitignore**：官网源码纳入版本控制（仅排除 node_modules/.next）。

### 📐 工程
- 新增 `SPEC.md`（结构化规范）+ `workflow_status.md`（终局审计进度）。
- 新增 `src/utils/theme_compat.py`。
- 新增 `website/src/components/ErrorBoundary.tsx`。
- `.gitignore`：`CACHE_DIR` 由中文"软产生的缓存"改 ASCII "cache"；移除 `website/` 整体排除。

## [4.5.0] — 2026-09-03

### 🎉 重大更新
- **跨视频向量知识库 (v4.5 Roadmap 落地)**：Phase 1 完成后自动把关键帧写入
  ChromaDB 全局 collection，支持自然语言跨视频搜索
  （"帮我找找过去一年所有视频里出现过的红色跑车"→返回时间戳+视频名+跳转）。
  复用已有 PersistentClient，不引入新向量库。
- **可分发软件包**：新增 `build_windows.spec`（PyInstaller onedir + 内置 FFmpeg），
  双击产物即可运行，无需手动安装 Python/依赖。

### 🐛 关键修复 (P0/P1)
- **启动链脆弱**：`启动应用.bat` 重写为 py -3.10/3.11/3.12 版本探测 + 失败引导下载页。
- **torch/PyQt6 DLL 冲突**：main_window.py 顶部强制 `import torch` 先于 PyQt6，
  消除 Windows `c10.dll` WinError 1114 崩溃。
- **Agent 面板 Ollama 模式输出损坏**：OllamaClient 现在在客户端层解析 SSE 协议，
  产出纯文本 delta（不再向 UI 泄漏 `{"message":{"content":...}}` JSON 碎片）。
- **未定义名崩溃**：修复 pyflakes 报告的 3 个 undefined name
  (`psutil` / `remaining` / `QDialog`) + 死代码（`__main__` 之后的 init_backend/seek_video）。
- **视频播放器崩溃**：closeEvent 引用不存在的 `self.player` → 改为 `self.media_player`；
  `setSize(QSize)` → `setSize(QSizeF)`（PyQt6 严格类型）。
- **智能关键帧开关失效**：`chk_smart` → `smart_extraction` 配置曾被 ExtractionWorker 忽略，
  现接入 `extract_smart_keyframes()`。
- **视频时长恒为 0**：ExtractionWorker 现返回 `duration`；transcript 保留完整
  AudioTranscript 对象（不再降级为字符串，时间轴波形恢复）。
- **模板选择形同虚设**：选中内置模板现真正传给 analyzer（不再落入英文一句话兜底）；
  PromptLoader 默认目录指向 `config/prompts/frame_analysis/`（此前从不被加载）。
- **报告脏标记**：`__FULL_RESPONSE_END__` 内部哨兵不再泄漏进报告/Agent 输出。
- **ADVANCED_FEATURES_AVAILABLE 恒真**：改为真实探测 moviepy/matplotlib/seaborn。
- **会话主键冲突**：`session_id` 从 `int(time.time())` 改为 uuid4（同一秒内不再冲突）。
- **QThread 退出崩溃**：closeEvent 现主动 stop/wait 运行中的 worker。

### 🔒 安全
- API Key 优先存 OS 密钥环（Windows DPAPI / macOS Keychain / Linux SecretService）。
- 模型下载新增 SHA256 完整性校验（防 MITM 投毒），UI"校验"按钮真正生效。

### 🏗️ 工程改进
- **依赖收敛**：requirements.txt 重写为分层（core / ocr extras），固定版本，
  删除 paddlepaddle-gpu（改 CPU paddlepaddle extras）、pyannote、decord、gradio。
- **moviepy 2.x 迁移**：`moviepy.editor` → `moviepy` 顶层；`subclip`→`subclipped`，
  `resize`→`resized`，移除 `verbose` 参数。
- **跨平台**：新增 `启动应用.sh`（Linux/macOS），ffmpeg 路径按平台分支。
- **Docker**：新增 `Dockerfile`（CPU）/ `Dockerfile.cuda`（GPU）/ `docker-compose.yml`
  + `src/server/headless.py`（HTTP 分析服务，复用 core 层零改动）。
- **资源面板增强**：显示本进程 RSS 内存、模型文件实际磁盘占用（非"预计大小"）。
- **版本单源**：`APP_VERSION` 统一在 `src/utils/constants.py`（4.5.0）。

### 🧪 测试与可观测性
- 新增 59 个 pytest 测试（含 E2E：完整 GUI 启动冒烟 16 项、全链路分析 13 项、
  Headless 服务 /healthz+/analyze），core 模块覆盖率显著提升。
- `.gitignore` 补全 venv/logs/cache/__pycache__。

### 📦 Agent 强化
- 5 个未注册工具补齐（search_web/search_visual/run_ocr/create_highlights/point_and_jump），
  此前因死代码位于 `if __name__` 之后从未生效。
- 新增 `search_kb` 跨视频知识库搜索工具。

### 🗑️ 清理
- 删除 `legacy/` 目录（12 个与 src 重复的旧 Gradio 版文件）。
- 删除 `app.py` 回退路径（Gradio 版入口已不存在）。

## [4.0.0] — 初始 Python 桌面版
- 三阶段流水线（提取→分析→媒体）+ Agent 面板 + 模型管理 + SQLite 历史。
