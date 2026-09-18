# Workflow Status — v10.5.0 交付闭环（2026-09-18）

> 本文件是任务单一状态源。上一版(v10.3.1 终态)已被本版取代。

## 已交付（真实证据，非宣称）

### v10.4.0（2026-09-17，主题提交 + tag + Release）
- 10 个主题 commit（`fc1625a`..`04c9a49`）：启动性能急救(29.18s→1.42s 基线)/
  eli5 工具面全覆盖/validator 准入闸门/SkillAdvisor/插件框架真相化/cua-service 接线/
  CI 三道门/batch 失败落库/版本五同步
- tag `v10.4.0` 推送 + GitHub Release 已创建
- 工作区曾未提交(26 改 + 15 新) —— 本批已全部入库

### v10.5.0（2026-09-17~18，对话体验闭环）
- P1-7 react 实时事件流：TurnHooks→SSE(assistant_delta/tool_start/tool_done/supervise)，
  run_stream 每请求独立队列+sentinel 直读(移除 50ms 忙轮询)
- P1-8 审批体验：实时倒计时/Enter-Esc 快捷键/pending?session_id 过滤/APPROVAL 映射守护测试
- P1-9 会话管理 UI：新建/切换/删除历史会话(关窗重开可续接)
- P1-10 单步软超时：VAP_AGENT_STEP_TIMEOUT 接线(LLM 流整体超时/工具超时落 Error 继续轮次)
- P0-B 启动脚本加固：cleanup-stale.js 不再全机杀 Electron(按命令行精确匹配+--dry-run)；
  start-desktop.bat 版本去硬编码
- tag `v10.5.0` 推送 + GitHub Release 已创建

### CI 修复（历史全红 → 绿）
- 根因1: test/e2e/build 三 job 缺 requirements-web.txt → ModuleNotFoundError: fastapi
- 根因2: pytest-cov 不识别 --skip-empty(coverage.py CLI 参数)
- 根因3: 矩阵只装核心子集却跑 tests/ 全目录 → 改全量依赖 + ffmpeg(apt/brew/choco)
- 根因4: torch(CPU index)与 torchvision(PyPI)分源 → torchvision::nms 注册失败 →
  transformers 惰性导入链报 'Could not import module PreTrainedModel' → 同源安装修复
- 真实安全修复: sanitize 用 Path().name 在 Linux 不剥反斜杠路径(可 C:\\evil.mp4 逃逸) →
  新增 secure_basename(双分隔符)并应用到 security + headless
- 接口顺序修复: batch 空目录校验移到装配 runner 之前(无 key 时应 400 而非 500)
- 测试隔离: torch 惰性代理用例复位 _mod; tkinter 用例 Linux 无 DISPLAY 跳过;
  step_timeout 3.10 asyncio.wait_for+queue.get 组合 bug 重写回退
- 最新一轮: macOS 3.10/3.11 全绿(项目 CI 历史首批通过)；ubuntu/windows 的
  KB/CLIP 11 项已由 torch/torchvision 同源 + kb_indexer 惰性导入修复，最终结果以
  最新 CI run 为准(见 Actions)

## 未完成（下一迭代主战场，见 计划书/下一步改进指南.md）
- P1-1 术语气泡 / P1-2 Onboarding / P1-3 用户画像 / P1-4 skill 采纳闭环 /
  P1-5 记忆读侧闭环 / P1-6 全局交互层
- P2-1~P2-9(压缩硬化/工具结果预算/卡死恢复/模型分档/子智能体/看板/tape/评测/双路径统一)
- P3-1~P3-7(pptx/图像生成/电商/插件市场/IM 真实 adapter/MCP 扩展/技能 12→18+)
- 第六章扩展清单(桌面自动更新/SQLite VACUUM/PWA/无障碍/成本看板等)

## 验收证据索引
- 计划书/下一步改进指南.md —— 批次方案 + 状态核销表 + TDD 清单
- CHANGELOG.md —— [10.4.0] / [10.5.0]
- 本会话本地实测: step_timeout 5 / realtime 5 / react 路由 45 / desktop 15 /
  eli5+skills+plugin 契约 65+43 / media+e2e pipeline 24 / KB 假 embedder 3 全绿
- Playwright E2E: 既有 24 passed + agent-realtime 3 passed；前端 tsc 0 错 + next build 17 路由
