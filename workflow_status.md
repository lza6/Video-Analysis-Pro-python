# Workflow Status — PyQt6 → Web UI 全量重构

> 复杂任务台账(全局 CLAUDE.md 要求)。只记事实与证据,不记私有推理。
> 最后更新:2026-09-05

## 一、任务契约

**目标**:把 Video Analysis Pro 的桌面 PyQt6 GUI 整体重构为 Web UI(FastAPI 后端 + Next.js 前端),删除 PyQt6,保留全部现有功能,本地双击 .bat → 浏览器打开 → 全功能可用。

**用户决策(已确认)**:
- 部署形态:本地 Web 服务(Python 后端 + Web 前端,浏览器访问 localhost)
- PyQt6 去留:立即删除,只做 Web UI(不双轨维护)
- 节奏:全部完整落地闭环,分轮交付,每轮真实可跑

**硬约束**:
- 保留"隐私不出本机"核心卖点(本地运行)
- 复用 `src/core/` 全部分析能力,不重写算法
- SSE/WebSocket 流式推送分析进度 + Agent 思考链(替代 QThread 信号)
- 凭据仍走 keyring,不落 ini 明文
- 模型下载保留 SHA256 校验
- Windows 优先(.bat 启动),macOS/Linux 兼容

## 二、架构定稿

### 后端 — `src/web/`(新建,FastAPI)

```
src/web/
├── __init__.py
├── app.py              # FastAPI 实例 + lifespan(探测能力、初始化配置) + 路由挂载 + 静态资源挂载
├── config.py           # pydantic-settings: VAP_PORT / VAP_HEADLESS_TOKEN / VAP_MAX_UPLOAD_MB / 并发 / 限流
├── deps.py             # 依赖注入: ConfigurationManager / 能力矩阵 / 作业仓库
├── schemas.py          # pydantic 请求/响应模型
├── security.py         # Bearer Token 鉴权(复用 headless 的 hmac.compare_digest 逻辑)+ 路径消毒
├── ratelimit.py        # IP 滑动窗口限流(从 headless.py 迁移)
├── run_store.py        # 作业状态仓库(内存 + 可选 SQLite 持久,复用 src/core/run_store.py)
├── routers/
│   ├── health.py       # GET /api/health                → 能力矩阵 + 磁盘余量 + keyring 状态
│   ├── analyze.py      # POST /api/analyze              → 创建作业,返回 job_id
│   │                   # GET  /api/jobs/{id}/stream     → SSE: phase/progress/frame/transcript/report-token
│   ├── jobs.py         # GET /api/jobs / GET /api/jobs/{id} / DELETE
│   ├── gallery.py      # GET /api/jobs/{id}/frames / GET /api/frames/{sanitized} (FileResponse + 路径消毒)
│   ├── media.py        # POST /api/jobs/{id}/highlights → 生成 GIF/Clips(SSE 进度)/ GET /api/media/{path}
│   ├── metrics.py      # GET /api/jobs/{id}/metrics     → 亮度/清晰度/饱和度趋势数据
│   ├── config.py       # GET/PUT /api/config            → provider/model/api_url(api_key 走 keyring)/prompts/presets
│   ├── models.py       # GET /api/models / POST /api/models/download (SSE + SHA256) / DELETE
│   ├── agent.py        # POST /api/agent/chat           → SSE: thought/action/observation/answer(ReAct 循环)
│   ├── kb.py           # POST /api/kb/index / GET /api/kb/search
│   ├── batch.py        # POST /api/batch / GET /api/batch/{id}/stream(SSE 逐任务进度)
│   ├── surveillance.py # POST /api/surveillance/start / GET /api/surveillance/stream
│   ├── skills.py       # GET /api/skills / POST /api/skills/generate
│   └── decisions.py    # GET /api/decisions
├── services/           # 后台任务适配层,把 src/core 的同步能力包成 BackgroundTask + SSE 发射
│   ├── analyzer_service.py   # 包 VideoProcessor/AudioProcessor/VideoAnalyzer,发 SSE 事件
│   ├── agent_service.py      # 包 AgentOrchestrator + ToolRegistry,流式 ReAct
│   ├── model_service.py      # 包模型下载 + SHA256 校验
│   ├── batch_service.py      # 包 batch_runner
│   ├── surveillance_service.py
│   └── kb_service.py
└── static_assets.py    # 安全的静态资源挂载(frames/gifs/clips,严格消毒防路径遍历)
```

### 前端 — `webapp/`(新建,独立 Next.js 项目,与 `website/` 营销站分离)

```
webapp/
├── package.json        # next 16 / react 19 / tailwind v4 / motion / @microsoft/fetch-event-source(SSE)
├── next.config.ts      # rewrites: /api/** → http://localhost:8000 (dev 代理)
├── src/
│   ├── app/
│   │   ├── layout.tsx          # 根布局:字体 + 全局样式 + Providers(设计 token 与 website/ 对齐)
│   │   ├── (app)/
│   │   │   ├── layout.tsx      # 应用 shell:左侧侧边栏(9 模块导航)+ 右侧 Agent 面板(可折叠)
│   │   │   ├── page.tsx        # 默认 → /analyze
│   │   │   ├── analyze/page.tsx       # ① AI 摘要报告(核心闭环:选视频→三阶段→报告)
│   │   │   ├── gallery/page.tsx       # ② 关键帧画廊
│   │   │   ├── media/page.tsx         # ③ 摘要媒体 GIF/Clips
│   │   │   ├── metrics/page.tsx       # ④ 元数据与画质图表
│   │   │   ├── logs/page.tsx          # ⑤ 系统日志(实时流)
│   │   │   ├── models/page.tsx        # ⑥ 模型管理(下载+校验)
│   │   │   ├── api-help/page.tsx      # ⑦ 获取 API 引导
│   │   │   ├── surveillance/page.tsx  # ⑧ 监控分析(RTSP)
│   │   │   ├── skills/page.tsx        # ⑨ Skills 管理
│   │   │   ├── decisions/page.tsx     # ⑩ 决策日志
│   │   │   └── batch/page.tsx         # 批量处理(独立路由)
│   │   ├── error.tsx / loading.tsx / not-found.tsx
│   │   └── globals.css         # 复用 website/ 的玻璃拟态 token(单一设计系统)
│   ├── components/
│   │   ├── shell/              # AppSidebar / AgentPanel / StatusBar / SkipLink
│   │   ├── ui/                 # Button / Card / Container / Badge / Dialog / Progress / Tabs(从 website/ 抽公共)
│   │   ├── analyze/            # VideoPicker / PhaseIndicator / FrameStrip / TranscriptView / ReportView
│   │   ├── gallery/ ...
│   │   └── agent/              # ThoughtChain / ToolCallCard / AgentInput
│   ├── lib/
│   │   ├── api.ts              # fetch 封装(含 SSE 流式消费)
│   │   ├── sse.ts              # SSE 事件流解析 hook
│   │   ├── types.ts            # 与后端 schemas 对齐的 TS 类型
│   │   └── utils.ts            # cn() 从 website/ 复用
│   └── hooks/                  # useAnalyzeJob / useAgentStream / useBatchStream ...
```

### 启动方式(本地优先)
- `启动应用.bat` → 检测 venv → 装依赖 → 启动 `uvicorn src.web.app:app`(生产:先 `cd webapp && next build`,FastAPI 挂载 standalone 静态)→ 打开浏览器 `http://localhost:8000`
- 开发态:FastAPI :8000 + Next.js dev :3000(rewrite 代理 /api)

## 三、UI 功能面 → Web 路由映射(待两个 Explore agent 回来后补全细节)

| PyQt6 界面 | 行数 | Web 路由 | 后端 API | 状态 |
|-----------|------|---------|---------|------|
| main_window.py 主框架+9 tab | 3154 | (app)/layout.tsx shell + 10 路由 | — | ⏳ |
| batch_tab.py | 1254 | /batch | /api/batch + SSE | ⏳ |
| agent_panel.py + agent_dialog.py | 911 | shell 内 AgentPanel + /agent | /api/agent/chat SSE | ⏳ |
| frame_strip_dialog.py | 377 | /analyze 内 FrameStrip 组件 | /api/jobs/{id}/frames | ⏳ |
| surveillance_tab.py | 307 | /surveillance | /api/surveillance/* | ⏳ |
| skills_manager_tab.py | 236 | /skills | /api/skills/* | ⏳ |
| decision_log_panel.py | 222 | /decisions | /api/decisions | ⏳ |
| provider_config_dialog.py | 200 | /settings(或 config dialog) | /api/config | ⏳ |
| model_manager_tab.py | 159 | /models | /api/models/* SSE | ⏳ |
| status_console.py | 144 | shell StatusBar | /api/health 轮询 | ⏳ |
| video_player_dialog.py | 123 | /analyze 内播放器 | /api/media/{path} | ⏳ |
| timeline_widget.py | 119 | /analyze 内 Timeline | /api/jobs/{id}/frames | ⏳ |
| carousel_widget.py | 90 | /gallery 内 Carousel | 同上 | ⏳ |
| api_intro_page.py | 90 | /api-help | 静态 | ⏳ |
| help_dialog.py | 91 | /help(或并入 docs) | 静态 | ⏳ |

## 四、任务图(依赖顺序)

```
阶段 0  地基(本轮)
  [0.1] workflow_status.md + 架构定稿                         ✅ 本文件
  [0.2] src/web/ FastAPI 骨架 + config + health + 鉴权 + 限流   ⏳
  [0.3] src/web/run_store.py 作业仓库                          ⏳
  [0.4] src/web/services/analyzer_service.py + SSE             ⏳
  [0.5] src/web/routers/analyze.py (POST + SSE stream)         ⏳
  [0.6] webapp/ Next.js 项目脚手架 + 设计 token               ⏳
  [0.7] webapp shell(sidebar + agent panel 骨架)              ⏳
  [0.8] webapp /analyze 页(核心闭环端到端打通)                 ⏳
  [0.9] 启动应用.bat 改造:启 FastAPI + 开浏览器               ⏳
  [0.10] 烟雾验证:启动→选视频→看到报告 SSE 流               ⏳

阶段 1  核心 tab 闭环(逐个)
  [1.1] /gallery 关键帧画廊 + Carousel
  [1.2] /media GIF/Clips 生成 + 播放器
  [1.3] /metrics 画质图表
  [1.4] /logs 实时日志流
  [1.5] AgentPanel ReAct 流式(替代 agent_panel.py)

阶段 2  管理/配置 tab
  [2.1] /models 下载 + SHA256 + 进度
  [2.2] /settings provider 配置(keyring)
  [2.3] /api-help 引导页
  [2.4] /skills 管理 + 生成
  [2.5] /decisions 日志

阶段 3  高级
  [3.1] /batch 批量处理 SSE
  [3.2] /surveillance RTSP 监控流

阶段 4  收尾
  [4.1] 删除 src/ui/ 全部 PyQt6
  [4.2] requirements.txt 移除 PyQt6/pyqtdarktheme
  [4.3] launcher.py 改为启 Web 服务(删 tkinter 向导里的 PyQt6 引用)
  [4.4] constants.py REQUIRED_PACKAGES 去 PyQt6,加 fastapi/uvicorn
  [4.5] tests/ 全量补 Web 后端测试 + 前端 e2e
  [4.6] README / CLAUDE.md 同步
  [4.7] 完整烟雾 + 交付报告
```

## 五、验收标准

1. 双击 `启动应用.bat` → 浏览器自动打开 `http://localhost:8000`,看到完整应用(非营销站)
2. 9 个 tab + 批量 + Agent 面板全部可用,功能对齐原 PyQt6
3. 选视频 → 三阶段分析 → SSE 流式看到进度/帧/转录/报告 token
4. Agent 对话 → ReAct 思考链流式可见,工具调用可触发
5. 模型下载有 SHA256 校验 + 进度条
6. 凭据走 keyring,ini 无明文
7. `src/ui/` 目录删除,`requirements.txt` 无 PyQt6
8. `python -m pytest tests/ -q` 标准子集绿
9. `python -m pyflakes src/ launcher.py` 零告警

## 六、阻塞项 / 待决策

- ~~UI 侦察~~:已完成,见下方"UI→Web 映射定稿"
- core 侦察:agent 仍在跑(未回),核心 API 已从 headless.py/logic.py/agent_tools.py 掌握,不阻塞地基
- 作业持久化:阶段0 用内存 JobStore(单机单用户够用);阶段3 batch 时桥接 src/core/run_store.py(SQLite 断点续跑)

## 七、UI→Web 映射定稿(来自 UI 侦察 agent)

**QThread → 后端 async job + SSE/WebSocket**:每个 QThread(ExtractionWorker/AnalysisWorker/ChatWorker/MediaWorker/SurveillanceWorker/ModelDownloadWorker/KBIndexWorker)对应一个后端 job,通过 SSE 推送 pyqtSignal 等价事件。Qt"信号跨线程回主线程"→ Web 端 SSE 消息在 JS 事件循环处理,天然安全。

**AgentDialog 工具箱路由 = Next.js (app)/layout.tsx + 子路由**:add_tool_page 每个工具页对应一个子路由(/analyze /gallery /media /metrics /models /surveillance /skills /decisions /batch),list_tools.currentRowChanged = navigate()。

**ChatWorker ReAct 循环 = 后端 SSE 流**:10 轮工具调用循环必须放后端(浏览器不能跑 Python 工具),前端 SSE 接 chunk/tool_call/entry_append 三类事件,分别渲染气泡/工具调用卡片/决策日志表。

**核心后端能力(已从 logic.py 确认签名)**:
- `VideoProcessor(video_path, output_dir).extract_keyframes(density, max_frames) -> List[Frame]` + `extract_smart_keyframes(min_scene_len) -> List[Frame]`
- `AudioProcessor().extract_audio(video_path, output_dir) -> Optional[Path]` + `transcribe(audio_path, diarize) -> Optional[AudioTranscript]`
- `VideoAnalyzer(client, model, prompt_loader, use_yolo, use_ocr).analyze_video(frames, transcript, custom_template) -> Iterator[str]`(流式)
- `OllamaClient` / `PromptLoader` / `Frame{path, timestamp, metrics, vision_content, ocr_text}`
- `ToolRegistry` + 12 个 create_*_tool 工厂(需 app_context_getter,Web 下改为 service 注入)
- `AgentOrchestrator`(意图解析 + ReAct)
- `ModelManager`(下载 + SHA256)
- `RtspMonitor`(监控)+ `surveillance_agent`
- `BatchRunner` + `RunStore`(SQLite)
- `kb_indexer.index_frames` + `HistoryManager`(ChromaDB)

**10 个 QTabWidget tab → 10 个 Web 路由**(见 workflow_status 顶部映射表,已与 agent 核对一致)

## 七、验证日志

| 时间 | 验证项 | 命令 | 结果 |
|------|--------|------|------|
| (待填) | | | |
