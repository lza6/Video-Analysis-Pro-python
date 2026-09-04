# Workflow Status — v5.4 终局闭环总审计（spec-kit 宪法 + agent-skills 驱动）

## Capability Map

| Module id | Responsibility | Depends on | 状态 |
|---|---|---|---|
| agent-prompt | CL4R1T4S 十五段系统提示词（v5.2 八段增补） | — | 已闭环(22 tests) |
| skills | SKILL.md 加载/触发注入/管理 UI（4 内置 skill） | agent-prompt | 已闭环 |
| decision-log | eli5 大白话 + 决策日志面板（黑匣子透明化） | desktop-ui | 已闭环 |
| llm-gateway | 4 协议 LLM 抽象+路由+重试退避+build_llm_client 接线 | — | 已闭环(9+7 tests) |
| surveillance | 监控两阶段搜索+RTSP 实时流+监控 Tab | llm-gateway | 已闭环(e2e 2 tests) |
| headless-server | HTTP 分析服务 + Bearer Token 鉴权 | core 全部 | 已闭环(10 tests) |
| kb-memory | 跨视频知识库+用户偏好（recall 注入已接线） | kb-indexer | 已闭环 |
| desktop-ui | PyQt6 主窗体 10 tab + Agent 面板 | core 全部 | 已闭环(e2e 28 断言) |
| release-consistency | APP_VERSION/CHANGELOG/README 版本对齐 | — | v5.3 收口 |

## v5.4 审计任务队列

| ID | Owner | Goal | Depends | Status |
|----|-------|------|---------|--------|
| T0 | Orchestrator | spec-kit init + agent-skills×10 安装 + 宪法 v1.0 | - | DONE |
| T1 | audit-blinds | 六角色盲点扫描（21 项清单：小白路径/headless 调用方/部署/数据一致/代码质量/文档一致性） | - | IN_PROGRESS |
| T2 | audit-prod | 生产架构审计（headless 并发/资源释放/安全/性能/可观测） | - | IN_PROGRESS |
| T3 | Orchestrator | 需求追踪矩阵 + spec.md | T1,T2 | PENDING |
| T4 | Builder | P0 修复 | T3 | PENDING |
| T5 | Builder | P1 修复 | T4 | PENDING |
| T6 | Critic | 独立审查（需求完整/逻辑/边界/质量/测试/运行 六方面） | T5 | PENDING |
| T7 | Builder | 修复 Critic 发现 → Critic 复验（≤3 轮） | T6 | PENDING |
| T8 | Orchestrator | 文档同步（README/SOP/CHANGELOG） | T7 | PENDING |
| T9 | Orchestrator | HTML 变更报告 + 底部测验 | T8 | PENDING |
| T10 | Orchestrator | 项目工作流+内部 skills 沉淀（新 API/功能接入 SOP） | T8 | PENDING |
| T11 | Orchestrator | 审计记录持久化（防重复盲扫） | T10 | PENDING |
| T12 | agent-manager | 联网挑高价值项（高并发/安全/UI）自主补强 | T9 | PENDING |

## 历史任务队列（v5.0→v5.3，全部完成）

- [x] T1 [P0] llm_gateway: 429/网络错误自动重试+退避 ✅
- [x] T2 [P0] rtsp_stream: RTSP URL 密码脱敏 ✅
- [x] T3 [P1] kb_indexer: _shared_embedder 双重检查锁 ✅
- [x] T4 [P1] surveillance_agent: 抽帧临时目录 rmtree ✅
- [x] T5 [P1] headless: Content-Length 先校验 + Bearer 鉴权 ✅
- [x] T6 [P0] 孤岛接线（v5.1）：surveillance_tab/skills_manager_tab/decision_log_panel 挂载 7→10 tab ✅
- [x] T7 [P1] agent_prompt 八段增补 + skills 触发注入 + 偏好个性化（v5.2）✅
- [x] T8 [P1] 发布一致性（v5.3）：APP_VERSION 4.5.0→5.3.0 对齐 tag / CHANGELOG 补 v5.1+5.2 / README 版本同步 / e2e_smoke 断言 14→28 / .gitignore 补 build/dist ✅
- [x] T9 [P2] 全链路回归 ✅（见验证日志）

## 验证日志
- 2026-09-04 (v5.0): pytest 74 passed | ruff 0 | mypy 10 files | 合成 E2E 149s PASS
- 2026-09-04 (v5.1): pytest 102 passed | py_compile 全过 | website tsc 0 / next build 6/6 / Playwright 8/8
- 2026-09-04 (v5.2): pytest 113 passed | mypy 35 files 0 错误 | ruff(SOP 口径) 0
- 2026-09-04 (v5.3): pytest tests/ -q → **135 passed**（269s 全量 + smoke 增强后复跑）
  - e2e_smoke 28/28（含监控/Skills/决策日志 tab 挂载 + prompt 八段断言）
  - e2e_full_pipeline PASS（Phase3 真实链路 33.6s）
  - test_headless_server 10 passed（含 VAP_HEADLESS_TOKEN 401/403/放行 3 场景）
  - skills 触发注入闭环实测：match_skills('请帮我总结这个视频') → 命中 builtin-video-summary → prompt 含 # SKILLS 段
  - 4 内置 skill 就位：builtin-video-summary / epic-infographics / funclip-clip / luxtts-voiceover
  - ruff(SOP 口径 --select F --ignore F403,F401) → All checks passed
  - mypy src/ 35 files → Success
- 2026-09-04 (v5.4 审计基线复核): pytest tests/ -q → **135 passed**（265.47s，Orchestrator 亲测，确认并行会话宣称属实）
- tag：v5.0.0 / v5.1.0 / v5.2.0 已推送远程并发布 Release；v5.3.0 待发布

## 独立审查记录
- 轮次1 (v5.0): 发现 T1-T5 五项缺陷 → 已全部修复 → 复验通过
- 轮次2 (audit-round2): UC1-UC7 + R5/R6 七项阻塞级 → 已全部修复 → 复验通过
- 轮次3 (v5.3 发布审计): 发现 6 项发布不一致（版本常量/CHANGELOG/README/workflow_status/smoke 覆盖/build 忽略）→ 全部修复 → 135 passed 复验
- 轮次4 (v5.4 终局审计): 进行中——audit-blinds（盲点扫描）+ audit-prod（生产架构）双线程并行，发现回填 §审计发现

