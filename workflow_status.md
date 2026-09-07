# v10.1.0 闭环补齐 — 任务状态

> 基于 5 个 explorer 子代理对真实代码的核查，对照《改进指南/下一步改进指南.md》。
> 工作树基线：v9→v10 转型代码零提交屎山（221 条）。
> 维护者：主控代理（Orchestrator） · 日期：2026-09-06 · 版本：10.1.0

## 已完成（本轮闭环）

### P0 收口
- ✅ .gitignore 补 `config/runs.db-shm`/`runs.db-wal`（line 60-62）
- ✅ README.md 升 v10.0.0（hero/tagline/简介/v10 新增能力/文档日期）
- ✅ 旧测试去版本号：test_v59_skills_ui→test_skills_ui / test_v60_security→test_security_hardening / test_agent_prompt_v52→test_agent_prompt_sections（git mv，回归 39 passed）
- ✅ error.tsx 上报 /api/logs（apiPostJson + 静默兜底）+ logs router 新增 POST /api/logs 端点
- ✅ Button 加 success/error 反馈态（feedback prop + shake keyframe）
- ✅ prompt_guard 补 HIDDEN_UNICODE（零宽 U+200B-200D/U+2060/U+FEFF）+ SYSTEM_SPOOF（system:/assistant:/developer: 伪冒）+ 6 新测试用例

### P1 新模块（6 个，Wave 1 worker 并行交付，主控接入）
- ✅ error_policy.py（ToolErrorKind + ErrorPolicy + 指数退避 + mark_unavailable）→ registry.execute 接入 `_execute_once` + 策略重试包装（policy=None 零回归）
- ✅ structured_log.py（JSONFormatter + trace_context + install_json_logging 幂等）→ app.py lifespan 切 JSON 日志（失败回退纯文本）
- ✅ vlm_client.py（VLMClient Protocol + Ollama/Cloud/Mock，httpx 条件依赖）+ agent_tools.create_vlm_describe_tool（16→17 工具）
- ✅ sandbox.py（Sandbox Protocol + WindowsJobSandbox(pywin32) + LinuxLandlockSandbox + NullSandbox + build_sandbox 自动选型）
- ✅ credentials/audit.py + rotation.py（CredentialAudit SQLite/WAL + KeyRotator rotate/revoke）
- ✅ experience.py（ExperienceExtractor + ExperienceStore + SkillAdvisor 规则版）

### P1 接入（主控串行）
- ✅ app.py lifespan 注入 SessionStore 单例（app.state.session_store）
- ✅ runtime/__init__.py 导出 structured_log（JSONFormatter/trace_context/install_json_logging/get_trace_id）
- ✅ registry.py 接入 ErrorPolicy（set_error_policy + execute 重试分级 + _execute_once 拆分）

### P2 前端/脚本/文档（Wave 2 worker 交付）
- ✅ Playwright E2E 5 spec（dashboard/analyze/agent/navigation/error_boundary）+ package.json e2e script + ci.yml e2e job（启后端→health wait→npm run e2e→artifact）
- ✅ scripts/dev-setup.ps1 + dev-setup.sh（onboarding）
- ✅ docs/ VitePress 骨架（config + index + 4 guide + package.json）

### CI 质量门禁
- ✅ ci.yml 去 `|| true`（测试真实失败阻断）
- ✅ 加 `--cov=src --cov-report=term-missing --cov-fail-under=80`
- ✅ build-windows 改 `needs: [test, e2e]`

## 验证日志（真实命令输出）

### Python 后端
```
$ pytest tests/ --ignore=test_headless_server --ignore=test_e2e_smoke --ignore=test_nvidia_models --cov=src --cov-fail-under=80
690 passed, 1 skipped, 998 warnings in 512.69s
TOTAL coverage: 66%  ← 低于 80% 门禁(预期:serve.py/router 未测,真实 CI 会 fail-under)
```
注:全量覆盖率 66% < 80% 门禁。根因:`src/web/serve.py`(115 行 0%)、各 router 的端点层大量未单测覆盖(走 TestClient 集成测试覆盖部分)。这是**真实 CI 会 fail-under**的状态——按铁则不伪造,如实记录。降门禁到 66% 可让 CI 绿但违背质量意图;保持 80% 让 CI 红倒逼补测试是正解。本轮保留 80% 门禁,标记为**待补测试**项。

```
$ pytest tests/test_agent_framework.py tests/test_error_policy.py tests/test_prompt_guard.py tests/test_vlm_client.py tests/test_structured_log.py tests/test_sandbox.py tests/test_credential_audit.py tests/test_experience.py
129 passed, 1 skipped  ← Wave1 6 新模块 + agent_framework 回归全绿
```

### pyflakes（CI 强制零告警）
```
$ pyflakes src/core/tools/registry.py src/core/tools/error_policy.py src/core/agent/prompt_guard.py
  src/core/runtime/__init__.py src/core/runtime/structured_log.py src/core/vlm_client.py
  src/core/runtime/sandbox.py src/core/credentials/audit.py src/core/credentials/rotation.py
  src/core/agent/experience.py src/web/app.py src/web/routers/logs.py src/core/agent_tools.py
exit 0  ← 零告警
```

### 前端
```
$ npx tsc --noEmit -p tsconfig.json  → 0 errors  ← Button/ error.tsx/ batch 修复后类型干净
$ npm run lint  → 0 errors, 5 warnings(react-hooks/set-state-in-effect,既有页非本轮引入)
```

### Playwright E2E（worker 报告）
```
$ npx playwright test  → 23 passed(含原 smoke 3 + 新 5 spec 20)
后端不可达时 20 skipped,0 失败(容错设计)
```

## 未完成 / 阻塞项（如实披露）

### P1 未接入（留 v10.2）
- ⚠️ ReactLoopAgent 未接 agent router：agent.py 仍走旧 AgentOrchestrator（/chat /run /run_stream），SessionStore 单例已注入 app.state 但 agent router 未消费。接入需重写 agent router 用 ReactLoopAgent + SessionStore.load/save，工作量 1-2d，本轮未做（避免破坏现有 /api/agent/chat 闭环）。
- ⚠️ SSE 断线重连（id/Last-Event-ID/useSSE.ts）：sse.py 仅 heartbeat，未加 event id 序号 + Last-Event-ID 续推 + 前端 useSSE hook。
- ⚠️ Turn 时间轴 API + 前端、run_store.checkpoint 长程断点、Subagent 多后端 backends/、凭据轮换前端、dashboard 时间轴+资源 mini 图、批量分片级进度前端、electron-updater 灰度、linux deb target + mac/linux build job。

### P2 需外部环境（标"待验证"，守付费 API 红线）
- IM 真实 token 闭环（Telegram bot 真实 getUpdates，需用户给 token）
- Tailscale Serve 真实 tunnel（需用户装 Tailscale）
- VLM 真实 key（Ollama/Cloud 真实 describe，需用户配 .env）
- electron 真实打包（需 windows-latest CI runner + tag push）
- 浏览器真实 E2E（Playwright 本地已验证，CI 待跑）

### 覆盖率门禁
- 全量覆盖率 66% < 80% 门禁。真实 CI push 会 fail-under。需补 serve.py/router 单测或下调门禁。本轮**保留 80%**（倒逼补测试），标记待补。

## 发版状态
- 版本号四同步：constants.py 10.1.0 / webapp+desktop package.json 10.1.0 / app.py version 10.1.0
- CHANGELOG 加 [10.1.0] 条目
- 待用户确认后：commit → push → tag v10.1.0 → gh release create
