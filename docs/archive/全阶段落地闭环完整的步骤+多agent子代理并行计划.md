# TingFeng Hermes v10.2 全阶段落地闭环步骤 + 多 Agent 并行计划

> **任务契约**：对照《计划书/下一步改进指南.md》（v10.2 版，19 项改进：P0×5 + P1×10 + 外部环境×4），
> 给出"全阶段落地闭环完整步骤 + 多 Agent 并行执行计划"，指导后续 implement 环节**逐个真实落地**（改代码 + 跑测试），
> 直到验收清单全勾、CI 全绿、发版闭环。
>
> 本文所有事实均已基于真实代码核验（2026-09-07）：
> - `src/web/routers/agent.py:201` 仍走旧 `AgentOrchestrator` + `_PLAN_CACHE`（`agent.py:142`，32 条上限）
> - `app.py:90-98` 已注入 `SessionStore` 单例但 agent router 未消费
> - `src/core/agent/loop.py:95` `ReactLoopAgent` 已有 `store` 参数 + `run_turn` 内 `_persist`（每 phase 写库）
> - `src/core/agent/session.py:125` `SessionStore`（SQLite/WAL）`save/load/delete` 完整，**缺 `list()`**
> - `src/core/agent/turn.py:29` 15 phase 枚举完整；`turn.py:117` `emit()` 会合并 extra 进 ctx
> - `src/web/sse.py:21-50` 仅 heartbeat comment 保活，**无 `id` 字段**
> - `src/web/routers/analyze.py:186-203` `job_stream` + `_guard_stream`，EventSourceResponse 包装
> - `src/core/run_store.py:42` `RunStore` 存在但**无 checkpoint/restore 方法**
> - `src/core/agent_tools.py:5-650` 共 **17 个** `create_*_tool`（Tool + ToolRegistry 在 `src/core/agent_tools.py:5-18`）
> - `webapp/src/lib/sse.ts:22` 有 fetch 版 `streamSSE`（**无自动重连、无 Last-Event-ID**）；`webapp/src/lib/api.ts:39-53` 有 apiGet/apiPostJson
> - `tests/` 已有 60 个测试文件（test_agent_framework.py 26 用例全绿）
>
> **发版状态**：v10.1.0 版本号已四同步但 **commit/push/tag 均未执行**（workflow_status L96，待用户确认）。
> **工作树**：HEAD 停在 v8.0.0，v9→v10 全部为未提交变更（发版动作本计划**不执行**，只规划）。
>
> **守付费红线**：IM 真实 token / Tailscale 真实 tunnel / VLM 真实 key / electron 真实上传 → 一律 Mock/本地验证，标"待验证"。

---

## 第一章 总策略：三阶段闭环 + 双层并行

### 1.1 阶段划分（每阶段有独立验收门，过门才进下一阶段）

```
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 0  基线锁定（0.5d）                                              │
│   冻结现状：跑 1 次全量基线测试 + 记录 coverage 66% 起点 + git stash   │
│   交付：baseline.json（测试数/覆盖率/失败清单）                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 1  P0 主线闭环（2-3d）—— 串行主线，内嵌并行                      │
│   S1.1 ReactLoopAgent 接 agent router（1-2d，唯一阻塞主线）          │
│   S1.2 覆盖率 66% → 80%（可全程并行，与 S1.1 文件不重叠）             │
│   S1.3 SessionStore 全链路持久化（随 S1.1，0.5d）                    │
│   S1.4 v10.1.0 补发版（需用户确认，独立可并行准备）                   │
│   门：test_agent_router.py 全绿 + coverage ≥80% + pyflakes 零告警     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 2  断线续传 + 可观测（2-3d）—— 依赖 S1.1，串行主线，内嵌并行     │
│   S2.1 SSE 断线重连（id/Last-Event-ID/useSSE 重连）                  │
│   S2.2 Turn 时间轴 API + 前端（依赖 S1.1 的 session 数据）            │
│   S2.3 长程断点续跑（run_store.checkpoint）                          │
│   并行池：Subagent in-process 后端 / 凭据轮换前端 / 沙箱接入 / guard  │
│   门：test_sse_reconnect.py + turns.spec.ts 全绿 + lint 0 error       │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 3  收尾 + 发版（1-2d）—— 并行池 + 串行收口                       │
│   S3.1 dashboard mini 图 / 批量分片级进度 / electron-updater 本地验证 │
│   S3.2 CI 全绿（真实 push 触发 test + e2e + build-windows）           │
│   S3.3 v10.2.0 发版（五同步 + CHANGELOG + tag + gh release）         │
│   门：验收清单 9.1/9.2/9.3 全勾 + CI 三 job 全绿                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 多 Agent 并行模型

```
                    ┌─────────────────────────┐
                    │  主控（Orchestrator）     │
                    │  串行主线推进 + 门禁把关   │
                    └───────────┬─────────────┘
          ┌─────────────────────┼──────────────────────┐
          ▼                     ▼                      ▼
   ┌─────────────┐      ┌──────────────┐      ┌──────────────┐
   │ Builder A   │      │ Builder B    │      │ Builder C    │
   │ (agent主线) │      │ (测试补全)    │      │ (前端/并行池) │
   │ 文件：       │      │ 文件：        │      │ 文件：        │
   │ agent.py    │      │ tests/*      │      │ sse.ts       │
   │ session.py  │      │ serve.py     │      │ pages/*      │
   │ loop.py(只读)│      │ 只碰 tests/   │      │ 只碰 webapp/ │
   └──────┬──────┘      └──────┬───────┘      └──────┬───────┘
          │                    │                      │
          └────────┬───────────┴──────────────────────┘
                   ▼
          ┌─────────────────────────┐
          │  Critic（独立审查）        │
          │  只读，不改代码            │
          │  每阶段结束审一次          │
          └─────────────────────────┘
```

**并行约束（铁则）**：
1. **文件不重叠**是并行的唯一前提。三条线文件域：A=`src/web/routers/agent.py`+`src/core/agent/`；B=`tests/`+`src/web/serve.py`；C=`webapp/`+`src/web/sse.py`。
2. **共享文件走主控串行**：`src/web/app.py`（版本号 + lifespan）只有主控改；`src/web/routers/agent.py` 只有 Builder A 改；`src/web/sse.py` 只有 Builder C 改。
3. **每 Builder 交付 = 代码 + 测试 + 真实运行证据**（pytest/npm run lint 输出截图），不允许"理论可行"。
4. **每阶段末 Critic 只读审查**：查需求完整性 / 边界 / 回归 / 是否破坏 v10 已落地能力。
5. **并行上限**：同时最多 3 个 Builder + 1 个 Critic，避免上下文碎片化。

---

## 第二章 阶段 0：基线锁定（0.5d，主控执行）

### 目标
锁定"改进前"的真实基线，防止落地过程把已有能力改坏而不自知。

### 步骤
1. `git status` 全量快照 → 确认工作树脏文件清单（v9→v10 未提交）
2. 跑基线测试（真实命令，输出存档 `logs/baseline.log`）：
   ```bash
   QT_QPA_PLATFORM=offscreen PYTHONIOENCODING=utf-8 \
     python -m pytest tests/ -q --ignore=tests/test_headless_server.py --ignore=tests/test_e2e_smoke.py \
     --cov=src --cov-report=term-missing
   ```
   → 预期 `690 passed, 1 skipped`，coverage **66%**（此数字是阶段 1 的门前基线）
3. 前端基线：`cd webapp && npx tsc --noEmit && npm run lint` → 0 errors（5 warnings 是既有页，容忍）
4. 产出 `logs/baseline.json`：`{"tests": 690, "skipped": 1, "coverage": 66, "tsc_errors": 0, "lint_warnings": 5}`

### 验收
- [ ] baseline.log 存在且命令真实跑过
- [ ] 确认无 P0 级既有失败（若有，先修再进阶段 1）

---

## 第三章 阶段 1：P0 主线闭环（2-3d）

### S1.1 ReactLoopAgent 接 agent router（P0，1-2d，Builder A 主改，主控串行验收）

**目标**：`/api/agent/chat` `/run` `/run_stream` 从旧 AgentOrchestrator 切换到 ReactLoopAgent + SessionStore，关窗不丢历史。

**落地方案（已核实到行号）**：

1. **`src/core/agent/session.py` 补 `list()` 方法**（现在只有 save/load/delete）：
   ```python
   def list(self, limit: int = 50) -> list[Session]:
       """列出全部会话（按 updated_at 倒序）。"""
       with sqlite3.connect(self.db_path) as conn:
           cur = conn.execute(
               "SELECT blob FROM sessions ORDER BY updated_at DESC LIMIT ?",
               (limit,))
           return [Session.deserialize(r[0]) for r in cur.fetchall()]
   ```

2. **`src/web/routers/agent.py` 重构**（保留旧函数名与端点签名，**只加不改**）：
   - 新增 `_build_react_agent(request, session_id)`：从 `request.app.state.session_store`（`app.py:94` 注入）load session（用 `ReactLoopAgent.load_session`，`loop.py:123`），构造 `ReactLoopAgent(config, llm_client, tool_registry, store=session_store)`。
   - LLM 客户端：新写 `ProviderRouterClient`（`src/core/agent/provider_client.py`，新文件）实现 `LLMClient` Protocol（`loop.py:49`）——包 `src/core/provider_router.py` 的 `ProviderRouter.post_nvidia` 流式 + 回退 `src/core/logic.py:build_llm_client`。**只读 import，不改旧文件**。
   - 工具：复用 `_build_orchestrator`（`agent.py:199`）里已注册的 5 工具 + `agent_tools.py` 全部 17 个 `create_*_tool` 走 `adapter.py` 桥接（17 工具全注册）。
   - `/chat`：`await agent.run_turn(session, text)`（async 化 handler）→ 返回 `{session_id, reply, phases}`。
   - `/run_stream`：SSE 推送 Turn 15 phase 事件（`turn.py:29-46`），结束 `session_store.save(session)`（`loop.py` 的 `_persist` 已自动做，确认即可）。
   - 新增 `GET /api/agent/sessions`：`session_store.list()` 返回历史会话列表。
   - 旧 `AgentOrchestrator` / `_PLAN_CACHE` **保留不删**（视频分析闭环在用），agent router 切走后加 `# deprecated（v10.2 后仅视频分析使用）` 注释。

3. **`src/web/app.py`**：`lifespan` 启动时若 `session_store` 非空则 log 恢复条数（`app.py:90-98` 已有初始化，只加一行 log）。

**验证（真实命令）**：
```bash
python -m pytest tests/test_agent_router.py -q          # 新测试文件
python -m pytest tests/test_agent_framework.py -q        # 26 用例回归（loop 未改，必须全绿）
python -m pytest tests/test_web_api.py -q -k agent        # 旧 agent 端点回归
python -m pyflakes src/web/routers/agent.py src/core/agent/session.py src/core/agent/provider_client.py
```

**新测试文件 `tests/test_agent_router.py`（Builder A 写）**：
- `test_chat_returns_session_id`：POST /api/agent/chat → 200，含 session_id
- `test_session_persist_across_restart`：POST /chat → 销毁 app → 重建 app → GET /sessions 仍有该 session（**关窗不丢历史**）
- `test_run_stream_emits_phases`：SSE 收到 turn/start → … → turn/end 15 phase 全序
- `test_turns_timeline`：GET /api/agent/sessions/{id}/turns → 按 turn 分组 phase 时间轴
- `test_prompt_injection_rejected`：POST 含 SYSTEM_SPOOF → 400
- Mock：`MockLLMClient`（`loop.py:69`）脚本回放，不真实调 LLM

### S1.2 覆盖率 66% → 80%（P0，1-2d，Builder B 主改，可与 S1.1 并行）

**目标**：不降门禁，补测试冲到 80%+（`--cov-fail-under=80` 真实通过）。

**落地方案（按性价比排序，已核实缺测点）**：

1. **`tests/test_serve.py`（新建）**：`unittest.mock.patch("uvicorn.run")` 测 `serve.py:128 run_server()`：
   - 参数解析（host/port 环境变量优先，`serve.py:157-158`）
   - 安全守卫：非 loopback + 无 `VAP_HEADLESS_TOKEN` → 拒绝启动返回 1（`serve.py:165-173`）
   - 端口自动跳：`VAP_PORT` 被占 → 下一个可用（`serve.py:178-187`）
   - `--no-browser` / `--reload` 参数（`serve.py:272-287`）
2. **router 端点补齐**（`tests/test_web_api.py` 扩展，已核实 16 router）：
   - `/api/providers/*`（providers.py 全部端点）
   - `/api/im/*`（im_gateway.py）
   - `/api/remote/*`（remote.py）
   - `/api/requests/stats`（requests.py）
   - `/api/skills/*`（skills.py）
   - `/api/surveillance/*`（surveillance.py）
3. **`coverage report --skip-covered`** 找 0% 模块逐个消灭，优先 `serve.py` / `deps.py` / `security.py` / `schemas.py` / `sse.py`。

**验证（真实命令）**：
```bash
QT_QPA_PLATFORM=offscreen PYTHONIOENCODING=utf-8 \
  python -m pytest tests/ -q --ignore=tests/test_headless_server.py --ignore=tests/test_e2e_smoke.py \
  --cov=src --cov-report=term-missing --cov-fail-under=80
```
→ 全绿 + `TOTAL coverage ≥ 80%`（**这是阶段 1 的主验收门**）

### S1.3 SessionStore 全链路持久化（P0，0.5d，随 S1.1，Builder A）

**目标**：Session 生命周期全链路落库，重启可恢复。
- 接 S1.1 后 `loop.py:run_turn` 的 `_persist` 已每 phase 写库（`loop.py:137-140, 159, 237, 245`），**不改 loop.py 内部逻辑**。
- 验证：`test_session_persist_across_restart`（S1.1 已覆盖）。
- 附加：`app.py` lifespan 恢复日志（S1.1 步骤 3）。

### S1.4 v10.1.0 补发版（P0，0.5d，需用户确认，主控准备不执行）

**现状**：版本号四同步已到 10.1.0，CHANGELOG 已加 [10.1.0]，但 commit/push/tag 未执行。
**动作**：**在阶段 1 门通过后、阶段 2 开始前**，向用户确认后执行：
```
git add -A && git commit -m "chore(v10.1.0): ..."
git push && git tag v10.1.0 && git push --tags
gh release create v10.1.0 --generate-notes
```
**注意**：本计划本身禁止 commit/push（任务约束），S1.4 是**规划**，执行时单独请求用户确认。

**阶段 1 验收门（全部真实跑过才算过）**：
- [ ] `tests/test_agent_router.py` 全绿（≥6 用例）
- [ ] `tests/test_agent_framework.py` 26 用例全绿（loop 未改回归）
- [ ] 全量 pytest + `--cov-fail-under=80` 通过，coverage ≥ 80%
- [ ] pyflakes `src/ launcher.py` 零告警
- [ ] `tests/test_serve.py` 全绿

---

## 第四章 阶段 2：断线续传 + 可观测（2-3d）

### S2.1 SSE 断线重连 + useSSE 重连（P1，1d，Builder C 主改）

**现状（已核实）**：`sse.py:21-50` 仅 heartbeat；`analyze.py:186` 用 EventSourceResponse；`webapp/src/lib/sse.ts:22` 是 fetch 版一次性流。

**落地方案**：
1. **`src/web/sse.py`**：
   - `JobRecord` 加 `seq: int = 0` 字段（`job_store.py:32` dataclass，向后兼容默认值）
   - `stream_job_events(rec, last_event_id=None)`：每条事件 `id=<seq>`，seq 自增并写回 `rec.seq`；`last_event_id` 传入时跳过 ≤ 该 seq 的事件（从 queue 尾部续推）
   - **关键设计**：JobRecord.queue 是无界 asyncio.Queue（`job_store.py:59`）——重连续推靠 **seq 序号**而非重放队列：客户端带 `Last-Event-ID` → 服务端**跳过序号 ≤ last 的事件**，后续新事件照常推（弱网断开期间的事件不重放，但进度不丢，因为数据源在 JobRecord 非流式字段）
2. **`src/web/routers/analyze.py:186-203`**：`job_stream` 读 `request.headers.get("last-event-id")` 传给 `stream_job_events`；`EventSourceResponse` 保持。
3. **`webapp/src/lib/sse.ts`**：
   - `streamSSE` 签名加 `lastEventId?: number` 参数（fetch 带 `Last-Event-ID` header）
   - 解析 `id:` 行，回调透传 event id
   - 新增 `useSSE(path, onEvent, {maxRetries=5, backoff=1s×2^n 上限 30s})` hook：断连自动重连，重连带上次收到的最新 id
4. **接入页面**：agent 页 + analyze 页改走 `useSSE`。

**验证**：
- 后端 `tests/test_sse_reconnect.py`（新建）：TestClient 发 2 事件 → 断 → 带 Last-Event-ID 重连 → 只收 seq>2 的增量
- 前端 `npm run lint` + `npx tsc --noEmit` 0 errors

### S2.2 Turn 时间轴 API + 前端（P1，1d，依赖 S1.1，主控串行）

**现状（已核实）**：`turn.py:29-46` 15 phase 枚举；`session.py:29` SessionEvent append-only。

**落地方案**：
1. 后端：`agent.py` 新增 `GET /api/agent/sessions/{session_id}/turns`：读 SessionStore，把 events 按 turn 分组（`user` 事件开启新 turn），返回 `[{turn_id, started_at, phases: [{phase, ts, duration_ms, tool?, content?}]}]`。**数据源已有，无需新存储**。
2. 前端：`webapp/src/components/agent/TurnTimeline.tsx`（新建）：竖排时间轴，每 phase 节点（plan=⚙️ / tool=🔧 / result=📄），hover 展开；agent 页 + dashboard 页接入。
3. **hook 方案**：复用 S2.1 的 `useSSE` 做实时增量（或 POST /chat 后轮询 GET turns）。

**验证**：`test_turns_timeline`（S1.1 已含）+ Playwright `turns.spec.ts`（新建：POST /chat → 断言 ≥3 个 phase 节点渲染）。

### S2.3 长程任务断点续跑（P1，1d，依赖 S2.1，主控串行）

**现状（已核实）**：`src/core/run_store.py:42` `RunStore` 存在但无 checkpoint/restore。

**落地方案**：
1. `run_store.py` 加：
   ```python
   def checkpoint(self, run_id: str, phase: str, payload: dict) -> None: ...
   def restore(self, run_id: str) -> dict | None: ...
   ```
   （SQLite 新表 `checkpoints`，WAL 同模式；`phase` 取 plan/tool/assistant 三档）
2. `agent.py /run_stream` 每 phase 结束时写 checkpoint；SSE 重连（S2.1）时读 checkpoint 从对应 phase 续推。

**验证**：`tests/test_run_store.py` 扩展：写 checkpoint → 模拟崩溃（新实例）→ restore 恢复到正确 phase。

### S2.4 并行池（P1，1-2d，Builder C 前端 + 安全线独立，可全程并行）

| 任务 | 文件域 | 落地方案（已核实） | 验证 |
|------|--------|-------------------|------|
| **Subagent in-process 后端**（2.3，P1） | `src/core/subagent/backends/inprocess.py`（新建）+ `director.py` | `InProcessBackend`：同进程 asyncio task 执行，支持 foreground/background；`director.py` 注册表加 `inprocess` 条目 | `tests/test_agent_framework.py` subagent 用例改走 InProcess 全绿 + 新增并发 2 子代理用例 |
| **凭据轮换前端**（4.1，P1） | `providers.py` + `webapp/src/app/(app)/providers/page.tsx` | 后端 `POST /api/providers/{name}/rotate` + `/revoke`（调 `rotation.py:KeyRotator`，**只回指纹不回明文**）；前端每行加轮换/吊销按钮 + 确认弹窗 | `tests/test_credential_audit.py` 扩展端点用例 + `npm run lint` |
| **工具沙箱 Web 接入**（5.1，P1） | `src/core/tools/registry.py` + `src/core/runtime/sandbox.py` | registry.execute 包 `with sandbox.guard()`，`VAP_SANDBOX_ENABLED` 开关（默认 False 防无 pywin32 回归） | `tests/test_sandbox.py` 全绿 + 新增 registry+sandbox 集成用例 |
| **prompt_guard 覆盖 Web 入口**（5.2，P1） | `agent.py`（与 A 共享！） | `/chat` 入口统一过 guard，命中高危（SYSTEM_SPOOF）→ 400 + 结构化错误 | `test_prompt_injection_rejected`（S1.1 已含）——**此任务并入 Builder A，不做并行** |

> **冲突规避**：S2.4 的 prompt_guard 任务修改 `agent.py`，与 S1.1 同文件——**并入 Builder A 串行做**，不另开 worker。其余三项文件域独立，可并行。

**阶段 2 验收门**：
- [ ] `tests/test_sse_reconnect.py` 全绿
- [ ] `tests/test_run_store.py` checkpoint/restore 用例全绿
- [ ] Playwright `turns.spec.ts` 通过（npm run e2e）
- [ ] `cd webapp && npm run lint && npx tsc --noEmit` 0 errors
- [ ] Subagent InProcess 用例全绿
- [ ] 全量回归 pytest 通过（覆盖率不掉回 80% 以下）

---

## 第五章 阶段 3：收尾 + 发版（1-2d）

### S3.1 收尾并行池（P1，0.5-1d，Builder C + 独立 worker）

| 任务 | 文件域 | 落地方案 | 验证 |
|------|--------|---------|------|
| **dashboard 资源 mini 图**（3.3，P1） | `webapp/src/components/dashboard/ResourceMiniChart.tsx`（新建） | 折线图（内存/CPU/磁盘），复用 `/api/metrics`，`useSSE` 或 5s 轮询，单文件 <100 行；dashboard 页 + `TurnTimeline` 组件复用 | Playwright `dashboard.spec.ts` 断言 mini 图数据非空 |
| **批量分片级进度前端**（4.2，P1） | `batch.py`（确认/补接口）+ `webapp/src/app/(app)/batch/page.tsx` | 后端确认 `GET /api/batch/{id}/tasks` 返回分片级数据（缺则补）；前端每任务行展开详情（分片 N/M + 每片耗时） | Playwright `batch.spec.ts` 断言展开后分片进度非空 |
| **electron-updater 本地验证**（4.3，P1，外部） | `desktop/package.json` + `desktop/electron-builder.yml` + `desktop/main.js` | 加 `electron-updater` 依赖 + `publish: [{provider: github, owner: lza6, repo: Video-Analysis-Pro-python}]` + main.js `autoUpdater` 静默检查；**真实上传需 GitHub token，标待验证** | `npm run dist` 产物含 `latest.yml` 结构正确（本地可验） |
| **linux deb + mac/linux build job**（10.2 遗留，P1） | `.github/workflows/ci.yml` | build-windows 外补 `build-linux`（deb target 已配）+ `build-macos` job | **需 CI runner，标待验证** |

### S3.2 CI 全绿（P0，0.5d，随 S1.4 发版）

**前置**：S1.4 v10.1.0 发版后 CI 首次真实跑：`test` job（cov 80%）+ `e2e` job（Playwright 23+ 用例）+ `build-windows`（needs: [test, e2e]）全绿。
**若 fail-under 仍红**：回查阶段 1 覆盖率的真实输出，补缺测点，重跑。
**注**：`e2e` job 在 CI 从未跑过（workflow_status L88），首次结果即真实验证——**这是 6.4 的唯一真实闭环路径**。

### S3.3 v10.2.0 发版（P0，0.5d，需用户确认，主控执行）

**五同步**：
1. `src/utils/constants.py:6` → `10.2.0`
2. `CHANGELOG.md` 顶部加 `[10.2.0]` 条目（列出全部落地项）
3. `webapp/package.json` version → 10.2.0
4. `desktop/package.json` version → 10.2.0
5. `src/web/app.py` version → 10.2.0

**文档同步**：
- `README.md` / `CLAUDE.md` 升 v10.2 能力与版本号
- `.env.example` 补 `VAP_SANDBOX_ENABLED`（如新增）
- `workflow_status.md` 更新：已完成项 ✅、外部环境项标"待验证"
- `计划书/下一步改进指南.md` 头部版本信息刷新

**发版动作（单独请求用户确认后执行，本计划不执行）**：
```
git add -A && git commit -m "chore(v10.2.0): Agent 主线闭环 + 断线续传 + 覆盖率80%"
git push && git tag v10.2.0 && git push --tags
gh release create v10.2.0 --generate-notes
```

---

## 第六章 外部环境项（待验证，守付费红线）

| 项 | 前置条件 | 本地可做 | 真实闭环 |
|----|---------|---------|---------|
| IM 真实 token 闭环（6.1） | 用户提供 Telegram bot token | Mock 闭环验证（现有 30 用例） | `VAP_IM_ADAPTERS=telegram` + 真实 getUpdates |
| Tailscale 真实 tunnel（6.2） | 用户装 Tailscale + token | Mock 验证（现有 44 用例） | `VAP_TUNNEL_PROVIDER=tailscale` + 真实 serve |
| VLM 真实 key（6.3） | 用户配 .env / Ollama | Mock 验证（test_vlm_client.py 全绿） | 真实 describe 一帧 |
| electron 真实打包（10.1） | windows-latest CI + tag | `npm run dist` 本地验证产物 | CI build-windows job 真实产出 |
| 浏览器 E2E CI（6.4） | 发版触发 CI | 本地 Playwright 全绿 | CI e2e job 首次真实跑 |

**铁则**：以上任一真实闭环前，workflow_status.md 保持"待验证"标记；**绝不**用真实凭据发起付费请求。

---

## 第七章 多 Agent 执行明细表（每阶段每 worker 的契约）

| 阶段 | Worker | 角色 | 输入 | 修改范围（禁区） | 交付物 | 验证 | 完成标准 |
|------|--------|------|------|-----------------|--------|------|---------|
| 0 | 主控 | Baseline | git 工作树 | 只读 | baseline.log/json | 真实命令 | 基线记录完成 |
| 1 | Builder A | agent 主线 | 指南 2.1/2.2/5.2 | `agent.py`+`session.py`+`provider_client.py`（新建） | 代码+test_agent_router.py | pytest 3 命令 | 6 用例全绿+回归全绿 |
| 1 | Builder B | 测试补全 | 指南 7.1 | `tests/*`+`serve.py`（只读） | test_serve.py+端点用例 | cov-fail-under=80 | coverage ≥80% |
| 1 | 主控 | 门禁 | A+B 产物 | `app.py`（唯一改权） | 版本 log | 全量回归 | 阶段 1 门全勾 |
| 2 | Builder C | SSE/前端 | 指南 3.1/3.3/4.2 | `sse.py`+`sse.ts`+`webapp/` | useSSE+重连+mini图 | test_sse_reconnect+lint | 全绿 |
| 2 | Worker D | Subagent | 指南 2.3 | `src/core/subagent/backends/`（新建） | inprocess.py+用例 | agent_framework 回归 | 全绿 |
| 2 | Worker E | 凭据轮换 | 指南 4.1 | `providers.py`+providers 页 | rotate/revoke 端点+UI | test_credential_audit 扩展 | 全绿 |
| 2 | Worker F | 沙箱接入 | 指南 5.1 | `registry.py`+`sandbox.py` | sandbox 开关 | test_sandbox 全绿 | 全绿 |
| 2 | Critic | 独立审查 | 阶段 2 全部产物 | 只读 | 发现清单（P0/P1/P2） | 代码核验 | 无未解 P0/P1 |
| 3 | 主控 | 收口 | 全量产物 | 版本五同步+文档 | v10.2.0 交付 | 验收清单 9.1/9.2/9.3 | 全勾 |

**Critic 审查点（每阶段末）**：
1. 需求完整性：指南 19 项是否全部覆盖（或明确标"外部/延后"）
2. 边界：SSE 重连死循环防护（maxRetries）、SessionStore 并发写、checkpoint 幂等
3. 回归：loop.py 未改断言、旧 AgentOrchestrator 未删断言、17 工具未改名断言
4. 证据：每个"完成"都有真实命令输出，无"理论可行"

---

## 第八章 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 覆盖率达到 80% 耗时超预期（serve.py/router 端点量大） | 中 | 阶段 1 阻塞 | 按性价比排序（test_serve.py → 高频端点 → 0% 模块），每 0.5d 检查进度，必要时与用户确认是否临时分阶段达标（不降门禁） |
| ReactLoopAgent 接入破坏既有 `/api/agent/chat` 前端闭环 | 中 | 回归 | 新旧端点并存：agent router 加 `?engine=react` 参数，默认仍走旧逻辑，前端逐步切换，S1.4 发版后再切默认 |
| SSE 重连在代理断开场景续推语义复杂 | 低 | 体验缺陷 | 简化设计：重连不重放断开期间事件（数据在 JobRecord 非流式字段可查），只保证"不断流、进度可查" |
| 多 worker 并行改同一文件 | 中 | 冲突 | 铁则 1：文件域划分 + 主控唯一改权 app.py；prompt_guard 任务并入 Builder A |
| 用户未及时确认 v10.1.0 发版 | 中 | S3.2 阻塞 | 发版与开发解耦：开发照跑，发版待确认；CI 全绿验证可用本地全量回归替代 |
| 外部环境项（IM/Tailscale/VLM）永远等不到凭据 | 低 | 验收缺口 | 按红线标记"待验证"不阻塞主线；v10.2 验收清单明确区分"已闭环"与"待验证" |

---

## 第九章 总验收清单（对应指南 9.1/9.2/9.3，全勾才算闭环）

### 9.1 功能验收
- [ ] `/api/agent/chat` 走 ReactLoopAgent，返回 `session_id`
- [ ] 关窗/重启后 `GET /api/agent/sessions` 能恢复历史会话
- [ ] SSE 断线后带 `Last-Event-ID` 重连，进度不断流
- [ ] agent 页展示 Turn 时间轴（≥3 个 phase 节点）
- [ ] dashboard 展示资源 mini 图（内存/CPU 曲线非空）
- [ ] providers 页可轮换/吊销 key（前端调 rotate 端点）
- [ ] batch 页展开任务显示分片级进度
- [ ] agent 页 `SYSTEM_SPOOF` 注入请求被 400 拒绝

### 9.2 质量验收
- [ ] 全量 pytest + `--cov-fail-under=80` 通过，coverage ≥ 80%
- [ ] `pyflakes src/ launcher.py` 零告警
- [ ] `cd webapp && npm run lint` 0 error；`npx tsc --noEmit` 0 errors
- [ ] Playwright 本地全绿（新增 turns.spec.ts / batch.spec.ts）
- [ ] CI 三个 job（test / e2e / build-windows）真实跑过且全绿

### 9.3 文档验收
- [ ] CHANGELOG.md 顶部加 [10.2.0] 条目
- [ ] README.md / CLAUDE.md 同步 v10.2 能力与版本号
- [ ] `.env.example` 补 `VAP_SANDBOX_ENABLED`
- [ ] workflow_status.md 更新（已完成 ✅ / 外部项"待验证"）
- [ ] 发版：先 v10.1.0（补发）再 v10.2.0，tag + gh release（用户确认后）

---

*计划撰写：2026-09-07* / *基于版本：v10.1.0* / *目标版本：v10.2.0* / *维护者：lza6（听风公司）*
