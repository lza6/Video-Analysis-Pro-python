# 📋 Video Analysis Pro — 产品需求文档（PRD）

> **版本**：v4.5.0（已交付现状）+ Phase B 增量需求（待审批）
> **维护方**：听风公司（Tingfeng）
> **文档日期**：2026-09-03
> **单一状态源**：本文件描述产品规格；技术实现详见 `docs/TECHNICAL_DOC.md`；改造路线与证据链详见 `参考的结果计划指南.md`。

---

## 0. 文档分层说明

本 PRD 分为两层：

| 层 | 范围 | 状态 | 章节 |
|----|------|------|------|
| L1 | v4.5 已交付功能的产品规格 | ✅ 已发布 | §2–§7 |
| L2 | Phase B 增量需求（Agent 智能度 + 黑匣子透明化 + skills 扩展面） | ⏳ 待审批 | §8 |

L2 不得在用户批准前实施。每条 L2 需求标注前置依赖与回滚方式。

---

## 1. 引言（产品概述）

**Video Analysis Pro** 是一款基于 Python 的本地化 AI 视频深度分析桌面应用，融合计算机视觉（CV）、自动语音识别（ASR）与大语言模型（LLM），将长视频浓缩为结构化报告、可视化报表、集锦片段，并提供可调用工具的 ReAct 智能体面板。

- **产品代号**：`Video Analysis Pro`（`src/utils/constants.py:5` `APP_NAME`）
- **版本**：`4.5.0`（`src/utils/constants.py:6` `APP_VERSION`）
- **协议**：GPL-3.0
- **形态**：桌面 GUI（PyQt6）+ 可选 Headless HTTP 服务（Docker）+ 可分发安装包（PyInstaller onedir）

### 1.1 核心价值

| 价值 | 说明 |
|------|------|
| 隐私至上 | 支持完全离线运行（Ollama + 本地模型），视频不出本地 |
| 零成本 | 依托开源模型，无强制 API 付费 |
| 可扩展 | 代码分层清晰（core/ui/server/utils），便于添加功能 |
| 跨视频记忆 | v4.5 落地 ChromaDB 全局知识库，支持自然语言跨视频语义搜索 |

---

## 2. 问题陈述

在信息过载时代，用户面对长视频（会议录屏、课程、监控、自媒体素材）面临三个痛点：

1. **检索慢**：1 小时视频人工定位关键片段需回看整段。
2. **理解难**：画面 + 语音 + 语义三层信息，人工综合成本高。
3. **隐私风险**：商业云端分析服务要求上传隐私视频，并按月收费。

现有方案（Gradio/Streamlit Web 界面、纯 CLI、商业 SaaS）在响应速度、交互丰富度、隐私保障之间难以兼得。Video Analysis Pro 以 PyQt6 桌面形态 + 本地推理 + 可选云端 API，提供兼顾三者的一体化方案。

---

## 3. 目标与对象（Objectives）

### 3.1 业务目标

| 编号 | 目标 | 衡量指标 |
|------|------|---------|
| O1 | 三阶段全自动分析（提取→AI 总结→媒体生成） | 1 小时视频 → 3 分钟内完成提取 + 报告 + 集锦 |
| O2 | 本地优先，离线可用 | 无网络时仍可完成 Phase 1/2/3（本地模型） |
| O3 | 跨视频可检索 | 跨 N 个已分析视频的语义搜索返回时间戳 + 跳转 |
| O4 | 智能体可调用工具完成多轮任务 | Agent 面板支持 ≥8 工具，ReAct 多轮 |
| O5 | 可分发 | 双击安装包即可运行，无需手动装 Python/依赖 |

### 3.2 非业务目标（明确排除）

- 不做云端托管 SaaS（核心形态是本地桌面应用）
- 不做实时流处理（RTSP，留待 v5.x+）
- 不做模型训练（仅推理 + 索引，训练留待 v5.x+）
- 不做插件市场分发（v4.5 范围内不引入）

---

## 4. 范围（Scope）

### 4.1 包含（In Scope）

- 三阶段流水线：数据提取、AI 分析、媒体生成
- Agent 面板（ReAct + 9 工具）
- 跨视频向量知识库（ChromaDB `kb_frames` 全局 collection）
- 模型管理（下载/校验/显存互斥/CPU 回退）
- 多 LLM 后端（Ollama / OpenAI 格式 API / LMStudio / 本地 GGUF）
- 历史会话管理（SQLite + ChromaDB）
- Headless HTTP 服务（Docker CPU/CUDA）
- 可分发软件包（PyInstaller onedir + 内置 FFmpeg）
- 跨平台启动入口（Win/Linux/macOS）
- API Key 密钥环存储（OS keyring）

### 4.2 排除（Out of Scope）

- 实时流（RTSP/摄像头）分析
- 模型微调/训练中心
- Web 版前端（FastAPI + React 重构）
- 插件市场/社区分发平台
- 多用户/多租户（单机单用户形态）

---

## 5. 功能需求（Functional Requirements）

### 5.1 Phase 1 — 数据提取

| 编号 | 需求 | 验收标准 | 代码位置 |
|------|------|---------|---------|
| FR-1.1 | 拖拽视频到提取区，支持常见格式 | mp4/mov/mkv/avi 可加载 | `main_window.py` 拖拽区 |
| FR-1.2 | 设置提取密度（0.1–1.0） | 密度越高抽帧越多，可调 | `VideoProcessor.extract_keyframes(density)` `logic.py:146` |
| FR-1.3 | OpenCV 智能抽帧（按密度） | 默认按 density 均匀抽帧，返回 `List[Frame]` | `logic.py:146` |
| FR-1.4 | 智能关键帧提取（场景切分） | 勾选后用 PySceneDetect 切场景 + 模糊过滤 | `extract_smart_keyframes` `logic.py:227`；接线 `main_window.py:106-110` |
| FR-1.5 | 音频提取 + Whisper 转录 | 提取音频 → faster-whisper 转录 → `AudioTranscript` | `AudioProcessor.extract_audio` `logic.py:487`；`transcribe` `logic.py:504` |
| FR-1.6 | YOLO 目标检测 | 识别帧内物体，结果写入 `frame.vision_content` | `VideoAnalyzer.detect_objects_in_frame` `logic.py:1026` |
| FR-1.7 | 返回真实视频时长 | `ExtractionWorker` 返回 `duration`（修复恒为 0 bug） | `main_window.py:144` |
| FR-1.8 | 写入跨视频知识库 | Phase 1 完成后关键帧入 `kb_frames` | `KBIndexWorker` `main_window.py:306`；`kb_indexer.index_frames` `kb_indexer.py:38` |

### 5.2 Phase 2 — AI 分析

| 编号 | 需求 | 验收标准 | 代码位置 |
|------|------|---------|---------|
| FR-2.1 | 选择 LLM 后端 | Ollama / OpenAI 格式 API / LMStudio / 本地 GGUF 四选一 | `OllamaClient` `logic.py:643`；`APIGatewayClient` `logic.py:742`；`LMStudioClient` `logic.py:928`；`LocalModelClient` `logic.py:933` |
| FR-2.2 | 选择提示词模板 | 内置 3 模板（内容总结/技术分析/情感识别）+ 自定义 | `PromptLoader` `logic.py:614`；默认目录 `config/prompts/frame_analysis/` |
| FR-2.3 | 逐帧构建提示词 + LLM 推理 | 帧信息 + 字幕拼入 prompt，流式返回 Markdown 报告 | `VideoAnalyzer.analyze_video` `logic.py:1068` |
| FR-2.4 | 流式输出（不卡顿） | SSE/流式 token 经 QThread 信号回 UI | `AnalysisWorker` `main_window.py:154`；`_process_stream` `logic.py:1035` |
| FR-2.5 | Ollama SSE 客户端层解析 | 产出纯文本 delta，不泄漏 JSON 碎片 | `OllamaClient.chat_stream` `logic.py:652`（v4.5 修复） |
| FR-2.6 | 内部哨兵不泄漏 | `__FULL_RESPONSE_END__` 不出现在报告 | `logic.py`（v4.5 修复） |
| FR-2.7 | 可视化报表 | 生成亮度/清晰度/饱和度趋势图 | `plot_metrics` `main_window.py:2165`（注：v4.5 为基础折线，见 NFR-3） |

### 5.3 Phase 3 — 媒体生成

| 编号 | 需求 | 验收标准 | 代码位置 |
|------|------|---------|---------|
| FR-3.1 | 自动生成集锦视频 | 按关键帧拼接近距离片段 → mp4 | `MediaWorker` `main_window.py:263`；`agent_tools.create_highlight_cut_tool` `agent_tools.py:213` |
| FR-3.2 | 生成 GIF 摘要 | 集锦片段转 GIF | `MediaWorker`（依赖 moviepy 2.x） |
| FR-3.3 | moviepy 2.x 兼容 | `subclipped`/`resized`，无 `verbose` | `agent_tools.py:220`（v4.5 迁移） |

### 5.4 Agent 面板（ReAct 智能体）

| 编号 | 需求 | 验收标准 | 代码位置 |
|------|------|---------|---------|
| FR-4.1 | 思维链可视化 | 类 DeepSeek R1 思考过程展示 | `ThinkingWidget` `agent_panel.py:7` |
| FR-4.2 | 多轮对话气泡 | 用户/AI 气泡区分 | `ChatBubble` `agent_panel.py:73` |
| FR-4.3 | 9 个工具全注册 | `search_web/search_visual/run_ocr/create_highlights/point_and_jump/search_kb/get_video_meta/get_frame_details/delete_history` 全接线 | `init_backend` `main_window.py:1655-1730`（v4.5 修复死代码） |
| FR-4.4 | 跨视频知识库搜索 | `search_kb` 工具返回带时间戳结果 + 可跳转 | `agent_tools.py:273-310`；`history_manager.search_kb` `history_manager.py:123` |
| FR-4.5 | 视频跳转 | Agent 调 `point_at_object` 后 `seek_video` 跳转 | `seek_video` 定义在 `DesktopApp` `main_window.py:1735`（v4.5 修复） |
| FR-4.6 | ReAct 多轮循环 | max_turns 上限，工具结果回灌 | `ChatWorker` `main_window.py:175`（注：当前实现简陋，见 §8 P1-2） |

### 5.5 模型管理

| 编号 | 需求 | 验收标准 | 代码位置 |
|------|------|---------|---------|
| FR-5.1 | 本地模型下载 | 支持下载 .pt/.gguf 到 `models/` | `ModelManager.download_model` `logic.py:420` |
| FR-5.2 | SHA256 完整性校验 | 防篡改/防 MITM 投毒 | `verify_model_integrity` `logic.py:399` |
| FR-5.3 | 显存互斥 | 多模型不抢占 VRAM | `ModelContextManager.request_vram` `logic.py:115` |
| FR-5.4 | CUDA 健康检查 + CPU 回退 | GPU 不可用时自动回退 | `logic.py:42-52` |
| FR-5.5 | 模型类型探测 | .pt→YOLO，.gguf→llama.cpp | `detect_model_type` `logic.py:376` |

### 5.6 历史与知识库

| 编号 | 需求 | 验收标准 | 代码位置 |
|------|------|---------|---------|
| FR-6.1 | 会话持久化 | SQLite `sessions` 表，uuid4 主键 | `add_session` `history_manager.py:182` |
| FR-6.2 | 断点续传检查点 | `checkpoints` 表 | `save_checkpoint` `history_manager.py:54` |
| FR-6.3 | 会话级帧记忆 | ChromaDB `session_{id}` collection | `add_frame_to_memory` `history_manager.py:72` |
| FR-6.4 | 跨视频全局 KB | ChromaDB `kb_frames` collection，upsert 防重 | `add_frame_to_kb` `history_manager.py:95`；`KB_COLLECTION_NAME` `history_manager.py:93` |
| FR-6.5 | 跨视频语义搜索 | 返回 video_name/timestamp/score/content | `search_kb` `history_manager.py:123` |
| FR-6.6 | KB 条目计数 | UI 展示当前条目数 | `kb_count` `history_manager.py:160` |
| FR-6.7 | 删除会话清理 KB | 删 session + 删 KB 中该 session 条目（无孤儿） | `delete_session` `history_manager.py:213`（v4.5 修复孤儿 bug） |
| FR-6.8 | 过期清理 | 按 retention_days 清理 | `cleanup_old_sessions` `history_manager.py:256` |

### 5.7 部署与分发

| 编号 | 需求 | 验收标准 | 代码位置 |
|------|------|---------|---------|
| FR-7.1 | Windows 一键启动 | `启动应用.bat` 版本探测 + 失败引导 | `启动应用.bat` |
| FR-7.2 | Linux/macOS 启动 | `启动应用.sh` | `启动应用.sh` |
| FR-7.3 | launcher 版本门禁 + venv | 自动建 venv + 装依赖 | `launcher.py` |
| FR-7.4 | PyInstaller onedir 打包 | 内置 FFmpeg，双击运行 | `build_windows.spec` |
| FR-7.5 | Docker CPU 镜像 | `Dockerfile` | `Dockerfile` |
| FR-7.6 | Docker CUDA 镜像 | `Dockerfile.cuda` + docker-compose GPU profile | `Dockerfile.cuda`；`docker-compose.yml` |
| FR-7.7 | Headless HTTP 服务 | `GET /healthz` + `POST /analyze` | `src/server/headless.py` |

---

## 6. 非功能需求（NFR）

| 编号 | 维度 | 需求 | 验收/现状 |
|------|------|------|----------|
| NFR-1 | 性能-提取 | 1 小时视频 Phase 1 在 5 分钟内完成（GPU） | 依赖硬件；CPU 较慢 |
| NFR-2 | 性能-搜索 | 跨视频 KB 搜索 < 500ms（已有 KB 索引） | `search_kb` 已用 ChromaDB；但 `search_visual` 单会话搜索无缓存（见 §8 P2-1） |
| NFR-3 | 报表专业度 | "高级可视化报表"应超越基础折线 | ⚠️ 当前 `plot_metrics` 为基础 matplotlib 折线，与 README 宣称不符（见 §8 P2-5） |
| NFR-4 | 兼容性 | Python 3.10+，Win/Linux/macOS | `launcher.py` 版本门禁；三平台启动脚本 |
| NFR-5 | 安全-凭据 | API Key 用 OS 密钥环 | `_secure_set` `config_manager.py:19`（DPAPI/Keychain/SecretService） |
| NFR-6 | 安全-模型 | 下载模型 SHA256 校验 | `verify_model_integrity` `logic.py:399` |
| NFR-7 | 稳定性-QThread | 退出时 stop/wait worker，无崩溃 | `closeEvent` 主动 stop/wait（v4.5 修复） |
| NFR-8 | 可观测性 | 资源面板显示 RSS/VRAM/模型 MB | `status_console.py` |
| NFR-9 | 测试覆盖 | 59 个 pytest（含 E2E 冒烟 + 全链路 + Headless） | `tests/` 11 文件 |
| NFR-10 | 中立性 | 默认 API 端点应中立空，不预置商业推荐 | ⚠️ 当前硬编码 `https://api.iflow.cn/v1`（见 §8 P2-4） |
| NFR-11 | 上下文窗口 | Ollama num_ctx 应可配置 | ⚠️ 当前硬编码 4096（见 §8 P2-3） |
| NFR-12 | 协议稳健 | APIGatewayClient 不应用字符串分隔符解析 system 消息 | ⚠️ 当前用 `--- System Context ---` 分隔（见 §8 P2-2） |

---

## 7. 用户故事（User Stories）

### 7.1 自媒体创作者

> 作为一个自媒体创作者，我希望把 1 小时的素材视频拖进软件，3 分钟后得到一份内容总结报告 + 高光集锦 GIF，这样我能快速提炼爆款点并分享到社交媒体。

**覆盖**：FR-1.1–1.7、FR-2.3、FR-3.1–3.2

### 7.2 会议记录员

> 作为一个会议记录员，我希望软件自动转录 Zoom/腾讯会议录屏的语音并生成纪要，这样我不用逐句手打。

**覆盖**：FR-1.5、FR-2.3

### 7.3 安防监控员

> 作为一个安防监控员，我希望输入"红色跑车"就能跨过去一年的所有监控视频定位出现时刻并跳转，这样我能快速取证。

**覆盖**：FR-6.4–6.5、FR-4.4、FR-4.5

### 7.4 学生

> 作为一个学生，我希望几分钟后就能了解一节 2 小时网课的核心内容，这样我能高效复习。

**覆盖**：FR-2.3、FR-3.1

### 7.5 开发者/极客

> 作为一个开发者，我希望 Agent 能听我指令调用工具（截图、OCR、网络搜索、剪辑），这样我能把它当可编程的视频助手。

**覆盖**：FR-4.3–4.6

---

## 8. Phase B 增量需求（待审批）

> 本节需求来自 `参考的结果计划指南.md` §6/§10 的真实瓶颈，均带 file:line 证据。
> **当前未授权实施。** 每条标注前置依赖、风险、验证方式、回滚。

### 8.1 B1 — Agent 系统提示词模块化重写（P1-1）

| 项 | 内容 |
|----|------|
| **需求** | 重写 `inject_agent_system_context`，按 identity/capabilities/agent_loop/todo_rules/tool_use_rules/finalize 守卫分段 |
| **现状证据** | `main_window.py:2257-2286` 仅拼 `"--- System Context ---"` + 视频元数据，无模块化分段 |
| **参考** | CL4R1T4S（Manus 6 模块 / Devin 三模式 / OpenHands system_prompt.j2 分段） |
| **前置依赖** | 无 |
| **风险** | L1 低（纯文本，可回滚） |
| **验证** | 本地 qwen2.5-vl 跑 3 个典型问题对比前后 |
| **回滚** | 还原旧 `context_str` |
| **数据迁移** | 无 |
| **兼容性** | 不影响现有功能 |

### 8.2 B2 — ChatWorker ReAct 结构化改造（P1-2）

| 项 | 内容 |
|----|------|
| **需求** | ① 用 `messages` 列表替代 `current_prompt` 累积拼接；② 工具参数用 JSON schema 严格解析（替代猜 key 名）；③ 工具结果完整展示（不截断 100 字符） |
| **现状证据** | `main_window.py:191-235` `ChatWorker.run`：`current_prompt` 累积重发；`args = {"query": args_str} if "search" in tool_name else {"seconds": args_str}` 靠猜；工具结果截断 100 字符展示 |
| **前置依赖** | B1（提示词引用结构化工具说明） |
| **风险** | L2 中（影响主交互链路） |
| **验证** | 单元测试覆盖 3 轮工具调用 + args 解析 |
| **回滚** | 还原旧 `run` |
| **兼容性** | 不影响数据层 |

### 8.3 B3 — 黑匣子透明化（P1-4）+ 集锦语义化（P1-3）+ search_visual 缓存（P2-1）

| 项 | 内容 |
|----|------|
| **P1-4 黑匣子** | 工具调用气泡展示完整参数 + 返回值 + eli5 plain-language 解释；新增可展开"Agent 决策日志"面板 |
| **P1-3 集锦语义化** | `create_highlight_cut_tool` 按 `description` 语义选段（复用 `search_visual` 找 top-N 时刻±2s），替代硬编码前 3 帧 |
| **P2-1 缓存** | `search_visual` 复用 `kb_indexer.get_embedder` + 帧缓存，替代每次重载 `SentenceTransformer` |
| **现状证据** | P1-3：`agent_tools.py:211-247` 取 `[f for f in app.frames if f.vision_content][:3]` 忽略 description；P2-1：`agent_tools.py:167-174` 每次新建 `SentenceTransformer('clip-ViT-B-32')`，与 `kb_indexer.py:18-40` 共享 embedder 不复用 |
| **前置依赖** | B2 |
| **风险** | L2 中 |
| **验证** | 1 个工具试点 eli5；真实视频 3 种 description 测集锦；对比 `search_visual` 调用耗时 |
| **回滚** | 隐藏面板；还原旧逻辑 |

### 8.4 B4 — skills 包机制 v0（P1-5，需单独审批）

| 项 | 内容 |
|----|------|
| **需求** | 定义 `SKILL.md` 规范（frontmatter `name`/`description`/`triggers`）+ 加载器 + 管理 UI 雏形（`src/ui/skills_manager_tab.py`） |
| **现状证据** | 无 skills 沉淀机制（现有 `prompts.json` 仅存模板文本，非工作流） |
| **参考** | hermes-agent optional-skills 分类 / skills-manager Tauri GUI |
| **前置依赖** | B1（提示词可引用已加载 skills） |
| **风险** | L2 中（新模块 + UI 入口） |
| **验证** | 加载 1 个内置 skill 成功 |
| **回滚** | 删除新模块 |
| **需单独审批** | 是（新增 UI 入口 + 新模块） |

### 8.5 B5 — 工程质量收尾（P2-2/2-3/2-4/2-5）

| 编号 | 需求 | 现状证据 |
|------|------|---------|
| P2-2 | `APIGatewayClient` 用结构化 `messages` 替代字符串分隔符 | `logic.py:781-789` 用 `"--- System Context ---"` 解析 |
| P2-3 | `OllamaClient` `num_ctx` 可配置 | `logic.py:664` 硬编码 `4096` |
| P2-4 | 默认 API 端点改中立空 | `config_manager.py:75` 硬编码 `https://api.iflow.cn/v1`；`api_intro_page.py:21-56` |
| P2-5 | `plot_metrics` 升级 seaborn/交互图 | `main_window.py:2165` 基础 matplotlib 折线 |

| 项 | 内容 |
|----|------|
| **前置依赖** | 无（可与 B1–B4 并行穿插） |
| **风险** | L1 低 |
| **验证** | 现有 59 测试保持全绿 |

### 8.6 B6 — 扩展增强（P2-6/2-7/2-8，建议放 v5.x）

| 编号 | 需求 | 现状证据 |
|------|------|---------|
| P2-6 | 图→帧跨模态搜索（`search_by_image` 工具） | `search_visual` 仅文→帧 |
| P2-7 | `main_window.py` 拆分（Worker 抽到 `src/workers/`） | God Object 2355 行 + ~74 方法 + 10 Worker |
| P2-8 | 跨视频用户偏好记忆（ChromaDB `user_preferences` collection） | 无用户画像 |

| 项 | 内容 |
|----|------|
| **前置依赖** | B3/B4 |
| **风险** | L2/L3（main_window 拆分影响面大，需独立 Critic 审查） |
| **建议** | P2-7 拆分放 v5.x；P2-6/P2-8 可在 v4.6 跟进 |

### 8.7 Phase B 实施批次与依赖顺序

| 批次 | 内容 | 风险 | 依赖 | 需单独审批 |
|------|------|------|------|-----------|
| B1 | P1-1 Agent 提示词重写 | L1 | 无 | 否 |
| B2 | P1-2 ChatWorker ReAct 改造 | L2 | B1 | 否 |
| B3 | P1-3 + P1-4 + P2-1 | L2 | B2 | 否 |
| B4 | P1-5 skills 包机制 v0 | L2 | B1 | **是** |
| B5 | P2-2/2-3/2-4/2-5 工程收尾 | L1 | 无 | 否 |
| B6 | P2-6/2-7/2-8 扩展 | L2/L3 | B3/B4 | 是（拆分影响面大） |

**建议优先批准**：B1 + B2（纯核心增强，低风险高收益，直击"agent 更完美"）；其次 B3（直击"黑匣子全开 + 小白易用"）；B4 需单独审批但是"扩展面 + 用户沉淀"核心，建议批准；B5 可穿插；B6 建议放 v5.x。

---

## 9. 待补充信息（需用户决策）

| # | 问题 | 影响 | 建议默认 |
|---|------|------|---------|
| Q1 | 扩展面（图片/视频生成、电商、PPT）是"可选 skills 包"还是"内置功能"？ | 决定 P1-5 skills 机制形态 | 可选 skills 包 |
| Q2 | 是否引入跨会话用户偏好记忆（P2-8）？用 ChromaDB 还是更重方案？ | 决定记忆层复杂度 | ChromaDB `user_preferences` collection（不引 mem0） |
| Q3 | 用户工作区未提交的 lint 清理 diff（bare except + 类型注解 + mypy.ini）是否纳入 Phase B 一并提交？ | 避免与用户改动冲突 | 保留用户 diff，Phase B 在其基础上增量 |
| Q4 | 本地测试模型是 qwen2.5-vl 7B 还是更大？ | P1-1 提示词升级效果依赖模型能力 | qwen2.5-vl 7B |
| Q5 | 是否批准 P1-5 skills 包机制（新增 UI 入口 + 新模块）？ | 扩展面 + 用户沉淀核心 | 建议批准 |

---

## 10. 约束与依赖

### 10.1 技术约束

- **Python 3.10+**：`launcher.py` 版本门禁
- **FFmpeg 必需**：视频/音频处理核心；Windows 可由打包内置，其他平台需系统安装
- **GPU 推荐**：本地 YOLO + LLM 推荐 NVIDIA GPU；纯 CPU 可运行但较慢
- **torch/PyQt6 DLL 冲突**：`main_window.py` 顶部强制 `import torch` 先于 PyQt6（Windows c10.dll WinError 1114）

### 10.2 依赖清单

分层依赖见 `requirements.txt`（core 分层）+ `requirements-ocr.txt`（OCR extras）。核心依赖：

- GUI：`PyQt6`、`pyqtdarktheme`
- CV/媒体：`opencv-python-headless`、`moviepy`、`imageio-ffmpeg`、`scenedetect`
- AI：`ultralytics`（YOLO11）、`faster-whisper`、`sentence-transformers`（CLIP）、`torch`
- 向量库：`chromadb`
- 配置/凭据：`keyring`
- 工具：`numpy`、`requests`、`psutil`、`markdown2`
- 可选 OCR：`paddleocr`（CPU extras）

### 10.3 外部服务依赖（可选）

- **Ollama**：本地 LLM 运行时（默认 `http://localhost:11434`）
- **OpenAI 格式 API**：DeepSeek/GPT-4o/Claude 等（用户自配 URL + Key）

---

## 11. 术语表

| 术语 | 含义 |
|------|------|
| Phase 1/2/3 | 数据提取 / AI 分析 / 媒体生成 三阶段 |
| ReAct | Reason + Act 循环，Agent 多轮工具调用模式 |
| KB | Knowledge Base，跨视频向量知识库（ChromaDB `kb_frames`） |
| 显存互斥 | `ModelContextManager` 保证多模型不抢占 VRAM |
| Headless | 无 GUI 的 HTTP 服务模式（Docker 部署） |
| eli5 | "Explain Like I'm 5"，plain-language 寄存器（Phase B P1-4 参考） |
| skills 包 | 可加载的用户/内置工作流单元（Phase B P1-5） |

---

*本 PRD 为单一产品规格源。技术实现细节见 `docs/TECHNICAL_DOC.md`；改造路线与证据链见 `参考的结果计划指南.md`。Phase B 获批后在此追加 Implementation Log。*
