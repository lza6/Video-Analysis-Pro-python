# Workflow Status — v5.0 终局审计（spec-driven）

## Capability Map

| Module id | Responsibility | Depends on | 状态 |
|---|---|---|---|
| llm-gateway | 4 协议 LLM 抽象+路由 | — | 审计中 |
| surveillance-agent | 监控两阶段搜索+剪辑 | llm-gateway | 审计中 |
| rtsp-stream | RTSP 实时流 | llm-gateway | 审计中 |
| headless-server | HTTP 分析服务 | core 全部 | 审计中 |
| kb-memory | 跨视频知识库+用户偏好 | kb-indexer | 已闭环(64 tests) |
| desktop-ui | PyQt6 主窗体+Agent 面板 | core 全部 | 审计中 |
| docs-sop | README/SOP/排障文档 | 全部 | 审计中 |

## 任务队列（P0→P2）

- [x] T1 [P0] llm_gateway: 429/网络错误自动重试+退避 ✅
  - 验证: mock 429×2→200, 3 次调用+6s 退避; tests/test_llm_gateway.py 9 passed
- [x] T2 [P0] rtsp_stream: RTSP URL 密码脱敏 ✅
  - 验证: rtsp://admin:SuperSecret@→rtsp://admin:***@; test_sanitize_rtsp_url_hides_password
- [x] T3 [P1] kb_indexer: _shared_embedder 双重检查锁 ✅
  - 验证: tests/test_core_pipeline.py 14 passed（含并发场景）
- [x] T4 [P1] surveillance_agent: 抽帧临时目录 rmtree ✅
  - 验证: 合成视频 E2E 149s PASS（修复后无回归）
- [x] T5 [P1] headless: Content-Length 先校验（413 拒绝+close_connection）✅
  - 验证: ruff 0; test_headless_server.py 仍过
- [x] T6 [P2] 全链路回归 ✅
  - 74 passed（+10 新 gateway 测试）| ruff 0 | mypy 10 文件 0 错误 | 合成 E2E 149s PASS
- [x] T7 [P2] docs-sop 同步 ✅（见 docs/SOP.md + README 排障段）
- [x] T8 [P2] 独立审查线程复核 ✅（见下方审查记录）+ HTML 报告 E2E实测结果/审计报告.html

## 验证日志
- 2026-09-04: pytest tests/ -q → 74 passed
- 2026-09-04: ruff check → All checks passed
- 2026-09-04: mypy 10 files → Success
- 2026-09-04: 合成视频真实 E2E (glm-5.3-flash) → 149s PASS

## 独立审查记录
- 轮次1: 发现 T1-T5 五项缺陷 → 已全部修复 → 复验通过
- 覆盖率: core 模块场景覆盖（正常/边界/异常/并发/安全），gateway 重试路径 3 场景全覆盖
