# v10.2.0 Sprint1+2 工作流状态（终局记录）

> 分支 `feat/v10.2-sprint1` · 22 提交 · 12 个新增 router/agent 测试文件 · +2300 行测试
> 更新：2026-09-08

## 已完成并验证（P0/P1 核心闭环）

### Sprint 1：Agent 主线闭环（8 提交）
- [x] `ProviderRouterClient` 补全（`src/core/provider_router_client.py`，7 用例全绿）
- [x] react 路径 tools 透传（`SyncLLMClientAdapter`+`_make_llm_callback`+`_nvidia_chat`）
- [x] `to_llm_schema` 改 OpenAI 兼容（真实 NVIDIA 400→200，`missing field type` 根因修复）
- [x] ReactLoopAgent + SessionStore 全链路（legacy 零回归）
- [x] 版本号五同步 v10.2.0 + CHANGELOG

### Sprint 2：router 覆盖率冲刺（14 提交）
| router | 覆盖率变化 | 用例数 | 测试文件 |
|---|---|---|---|
| media | 28%→100% | 13 | test_media_router.py |
| metrics | 20%→98% | 14 | test_metrics_router.py |
| models | 23%→**修复 422/latest-job/stream_closed** | 18 | test_models_router.py |
| remote | 41%→100% | 13 | test_remote_router.py |
| skills | 43%→94% | 14 | test_skills_router.py |
| requests | 74%→100% | 14 | test_requests_router.py |
| logs | 51%→89% | 8+4 | test_logs_router.py + stream |
| agent | — | 5+11 | test_agent_router.py + react |

### 修复的真实 bug（非测试迁移）
- **models.py download_stream 422**：`request` 缺类型标注被 FastAPI 当 query 参数
- **models.py 下载流订阅旧 job**：多次 POST 同一模型后 stream 死等旧线程 → 改取 latest
- **models.py stream_closed 顺序**：先置 closed 再推 `__close__`，避免多等 15s 心跳（对齐 analyzer_service）
- **合跑 429 限流**：IP 限流器进程级单例跨文件叠加 → 测试 fixture 显式 `init_ip_limiter(0)`
- **NVIDIA 400 missing field type**：VAP 私有 schema → 真实 API 复现 → OpenAI 兼容修复

## 验证证据（真实运行）
- 全量标准子集：`pytest tests/`（ignore headless/e2e_smoke）＝ **902 passed, 3 skipped**（679s）
- 11 个 router/agent 测试文件合跑：**121 passed**（63s）
- `pyflakes src/ launcher.py` ＋ `ruff check src/ launcher.py`：**零告警**
- Playwright E2E 23/23（Sprint1 时验证）
- react 路径真实 NVIDIA chat 200 + `GET /api/agent/sessions` 持久化闭环

## 遗留（诚实披露）
| 项 | 状态 | 说明 |
|---|---|---|
| surveillance router 测试 | ❌ 未落地 | 覆盖率 25%→约 40%；子代理被中断，`src/web/routers/surveillance.py` 需补 |
| im_gateway router 测试 | ❌ 未落地 | 覆盖率 38%；`tests/test_im_gateway.py` 是 core 层 30 用例，web router 未覆盖 |
| 全量覆盖率 ≥80% 门禁 | ⚠️ 未达成 | 约 68%→现约 70%+，surveillance/im_gateway 补齐后接近；analyzer_service 69% 仍低 |
| 推送/发行版 | ⚠️ 待用户确认 | 分支 22 提交未 push、未 tag、未建 release（gh 未安装） |
| models/yolo11n.pt | ✅ 已清理 | 已 gitignore，从跟踪移除 |

## 下一步（收尾清单）
1. 补 `tests/test_surveillance_router.py` + `tests/test_im_gateway_router.py`（覆盖率模型同上）
2. 全量 `--cov-fail-under=80` 终验
3. `git push origin feat/v10.2-sprint1`（需用户确认）
4. tag `v10.2.0` + GitHub Release（需用户确认 + gh CLI）