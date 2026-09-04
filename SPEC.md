# SPEC.md — Video Analysis Pro

> 结构化规范 · 真实可交付边界 · 2026-09-04
> 基于 spec-driven-development skill 生成。代码改动前先读此文件，过时即更新。

---

## 0. 能力地图 (Capability Map)

本项目是**单用户本地桌面应用 + 静态官网**，非 SaaS。所有"高并发/分布式"基础设施不适用。

| Module id | Responsibility | Depends on | 适用边界 |
|-----------|----------------|------------|---------|
| desktop-ui | PyQt6 主窗口、7 Tab、Agent 面板、模型管理 | core-logic, agent-tools | 单进程单用户，本地 GUI |
| core-logic | VideoProcessor / AudioProcessor / VideoAnalyzer / ModelContextManager | kb-indexer, utils | 单机，可选 GPU |
| agent-tools | 9 个 Agent 工具（截图/搜索/OCR/集锦/跳转/元数据/删除/KB搜索/视觉搜索） | core-logic | ReAct 循环，本地调用 |
| kb-indexer | 跨视频向量知识库（ChromaDB 嵌入） | core-logic, history | 本地 sqlite+chromadb，单文件 |
| headless-server | 可选 HTTP 分析服务（Docker 部署） | core-logic | 仅本地/受信网络，可选 Bearer Token 鉴权（`VAP_HEADLESS_TOKEN` 环境变量，空=禁用） |
| website | Next.js 16 官网落地页（玻璃拟态 2.0） | — | 纯静态，无后端 API |
| docs | README / UI-展示 / SPEC / TECHNICAL_DOC / PRD | 全部 | 文档与真实状态一致 |

**Build order**: core-logic → agent-tools → kb-indexer → desktop-ui → headless-server → website → docs

---

## 1. Objective

**真实目标**：交付一个本地运行、隐私至上、链路完整的 AI 视频分析桌面工具，配以精致官网落地页。已在生产使用，需终局闭环审计。

**不是什么**：不是 SaaS、不是云服务、不需要 Kafka/Redis/Sharding/Load Balancer。这些一律标注"不适用"，不假装做。

---

## 2. Commands

| 场景 | 命令 |
|------|------|
| 桌面应用启动 | `./venv/Scripts/python launcher.py`（或双击 `启动应用.bat`） |
| 官网预览 | `cd website && npm run dev` → http://localhost:3000 |
| 官网构建 | `cd website && npm run build` |
| 桌面 UI 截图 | `./venv/Scripts/python capture_desktop_ui.py` |
| 官网截图 | `cd website && node scripts/capture.mjs` |
| 测试 | `./venv/Scripts/python -m pytest tests/ -p no:cacheprovider` |
| 语法检查 | `./venv/Scripts/python -m py_compile <file>` |

---

## 3. Project Structure

```
Video-Analysis-Pro-python/
├── src/
│   ├── core/          logic.py, agent_tools.py, kb_indexer.py, history_manager.py,
│   │                  llm_gateway.py(v5.0 未接入), surveillance_agent.py(v5.0 未接入)
│   ├── ui/            main_window.py, agent_panel.py, model_manager_tab.py, ...
│   ├── utils/         config_manager.py, constants.py, theme_compat.py(新)
│   └── server/        headless.py
├── tests/             test_e2e_smoke.py, test_agent_tools.py, ...
├── website/           Next.js 16 官网（已纳入版本控制）
├── docs/              PRD.md, TECHNICAL_DOC.md, screenshots/, 计划书/
├── launcher.py        启动器（venv 探测+依赖校验）
├── capture_desktop_ui.py  桌面 UI 截图脚本
├── requirements.txt   核心依赖（已放宽 pyqtdarktheme 版本）
├── Dockerfile / docker-compose.yml  可选 headless 部署
├── README.md          含 UI 预览章
├── UI-展示.md         完整截图集
├── SPEC.md            本文件
└── workflow_status.md 终局审计进度
```

---

## 4. Code Style

- **Python**: PEP 8 + 类型注解 + `logging`（禁 print）+ 参数化 SQL + subprocess 列表形式
- **TypeScript**: strict + `===` + 类型显式 + 无未处理 promise
- **React**: 函数组件 + 命名 props interface + `useReducedMotion()` 真调用（禁硬编码 false）
- **CSS**: oklch token + 仅动 transform/opacity/clip-path/filter
- **提交**: Conventional Commits（feat/fix/refactor/docs/test/chore）

---

## 5. Testing Strategy

| 层级 | 框架 | 范围 |
|------|------|------|
| 单元 | pytest | agent_tools, api_clients, history_manager, model_manager, headless_server |
| E2E 冒烟 | pytest 子进程 + offscreen Qt | DesktopApp 14 项 UI/后端状态 |
| 真实浏览器 | Playwright | 官网 3 断点截图 + 渲染校验 |
| 静态 | py_compile + tsc | 全部源码编译通过 |
| 安全 | grep 密钥 + npm audit | 仓库无硬编码、依赖无已知漏洞 |

**不真实跑**：付费 API（DeepSeek/GPT-4o 调用）、GB 级模型权重下载（YOLO/Whisper 真实推理）。这些用 mock/fixture 验证参数拼装与响应解析。

---

## 6. Boundaries

**始终做**:
- 改代码前读 SPEC.md + workflow_status.md，过时先更新
- 修完跑 py_compile + 相关 pytest
- 截图同步到 `docs/screenshots/`（gitignore 已不排除）
- 诚实标注"已验证/静态确认/合理推断/待验证"

**先问再做**:
- 删除真实数据/迁移
- 引入核心新依赖（torch/cv2 升级）
- 生产部署/发布

**绝不做**:
- 真实付费 API 调用（预算默认 0）
- 用 mock/占位冒充已实现
- 未跑验证就宣称"完成"
- 硬编码密钥

---

## 7. 已知限制与外部边界

| 项 | 状态 | 说明 |
|----|------|------|
| GitHub 仓库 lza6/Video-Analysis-Pro-python | 待用户确认 | 官网 Download 链接指向，HTTP 200 已验但归属需用户确认 |
| 付费 LLM API 真实调用 | 不跑 | 用 mock 验证参数/响应/流式解析 |
| YOLO/Whisper 真实推理 | 不跑 | GB 级权重，属烧钱重资源，e2e_smoke 仅验证 import+实例化 |
| 真实 GPU WebGL | 截图验 | HeroVisual 有 ErrorBoundary + WebGL 预检降级 |
| SaaS 基础设施（Kafka/Redis 等） | 不适用 | 单机桌面应用，已在 A.4 裁定 |
