# TingFeng Hermes — 项目级指南

> 本文件是本项目（通用全能 AI Agent 桌面平台）的工程指南。通用行为准则、安全、测试、
> Git 工作流、复杂度分级遵循用户全局 `~/.claude/CLAUDE.md` 与 `~/.claude/rules/`；
> 此处只记录**本项目特有**的事实、命令、约定与已知坑。冲突时以本文件为准。

- **身份**：TingFeng Hermes（听风·赫尔墨斯），由 听风公司 (Tingfeng) 出品，维护者 `lza6`
- **仓库**：https://github.com/lza6/Video-Analysis-Pro-python
- **版本**：`v10.3.1`（见 `src/utils/constants.py:APP_VERSION`，改版本要同步 `CHANGELOG.md`）
- **License**：GPL-3.0（传染性开源，修改必开源）
- **Python**：3.10+（CI 矩阵测 3.10 / 3.11，本地 venv 子目录名 `venv`）
- **Node**：18+（webapp/ Next 16 + desktop/ Electron）
- **本项目不使用 OpenWolf**（`.wolf/` 不存在），全局 CLAUDE.md 的 OpenWolf 协议段不适用。

---

## 一、技术栈与架构

v9.0.0 完成"通用全能 Agent 桌面平台"转型：Electron 桌面壳 + Python FastAPI 后端 + Next.js 前端 + DSH 式 Agent 框架 + IM 网关 + 远程访问 + 插件生态。视频分析能力降为内置 Agent 工具集之一。

### 分层架构

| 层 | 目录 | 职责 |
|----|------|------|
| **Electron 桌面壳** | `desktop/` | `main.js` 主进程 + `runtime-controller.js`（spawn `python src/web/serve.py` 子进程 + 健康探活 + 端口转发）+ `preload.js`（IPC 桥）+ `electron-builder.yml`（Windows 安装包 + 单实例 + 托盘 + electron-updater） |
| **FastAPI 后端** | `src/web/` | `app.py` 装配 12 router（health / analyze / metrics / media / models / agent / config / logs / decisions / skills / batch / surveillance），`serve.py` 入口跑 uvicorn，SSE 流式 |
| **Next.js 前端** | `webapp/` | Next 16 + React 19 + Tailwind v4，静态导出（`webapp/out/`），13 页路由（analyze / gallery / media / metrics / agent / logs / batch / models / surveillance / skills / decisions / settings / dashboard），玻璃拟态设计系统 |
| **Agent 框架** | `src/core/agent/` | `loop.py` ReactLoopAgent（ReAct 循环 + 工具调用 + 思考链）+ `session.py` Session（SessionEvent append-only log）+ `turn.py` Turn（15 phase 事件链：user_msg → plan → tool_pre → tool_exec → tool_post → tool_result → … → assistant_msg） |
| **工具系统** | `src/core/tools/` | `definition.py` ToolDefinition + 四 waterfall（pre / execute / post / result）+ `parallel.py` ParallelExecutor（读锁共享 / 写锁独占）+ `adapter.py` 桥接现有 16 个视频分析工具 |
| **Subagent Director** | `src/core/subagent/` | 三模式（foreground / background / continuable）+ 四级路由（call > role > default > inherit）+ `role_template.py` RoleTemplate（planner / critic / researcher / executor） |
| **凭据管理** | `src/core/credentials/` | `credential_key.py` 分层（env > keyring > ini）+ 运行时合并 + 写回优先级 |
| **插件框架** | `src/core/plugins/` | `context.py` PluginContext（contextvars 替 Cordis fiber）+ `loader.py`（声明式 patch YAML 加载）+ `patch.py`（运行时挂载/卸载） |
| **IM 网关** | `src/core/im_gateway/` | `mailbox.py` IMMailbox（SQLite lease/ack 投递箱）+ `cipher.py` GatewayCipher（AES 条件依赖）+ 三 adapter（Mock / 微信 / TG / Discord）+ `gateway.py` IMGateway 统一入口 |
| **远程访问** | `src/remote/` | `tunnel.py` Tunnel 抽象 + 三方案（Mock / Tailscale Serve / Direct / Cloudflare Access）+ `manager.py` RemoteManager（健康探活 + 自动重连） |
| **视频分析核心** | `src/core/logic.py` | 三阶段流水线（Phase 1 抽帧/转录/检测 → Phase 2 LLM 推理 → Phase 3 媒体生成），降为 Agent 工具集之一，能力标志 `CLIP_AVAILABLE` / `NVIDIA_GPU_AVAILABLE` / `ADVANCED_FEATURES_AVAILABLE` / `FFMPEG_AVAILABLE` |
| **Utils** | `src/utils/` | `config_manager.py`（配置 + 密钥环）、`constants.py`（版本/路径/常量/REQUIRED_PACKAGES）、`ui_components.py`（tkinter 安装向导，遗留） |

### 关键依赖分层

- `requirements.txt` = core（应用启动最小集，**分层标注**，固定版本区间）
- `requirements-ocr.txt` = 可选 PaddleOCR（~1GB，缺失时自动跳过 OCR）
- `desktop/package.json` = Electron 壳依赖（electron / electron-builder / electron-updater）
- `webapp/package.json` = 前端依赖（next / react / tailwind / swr / radix）

核心库：`fastapi` / `uvicorn` / `sse-starlette` / `opencv-python-headless` / `ultralytics`(YOLOv11) / `scenedetect` / `faster-whisper` / `sentence-transformers` / `chromadb>=1.5.9` / `moviepy>=2.0` / `torch>=2.2` / `imageio-ffmpeg` / `nvidia-ml-py` / `electron` / `next` / `react` / `tailwindcss`。

### 三阶段处理流程（视频分析内置能力）

1. **Phase 1 数据提取**：OpenCV 智能抽帧 + Whisper 音频转录 + YOLO 物体检测 → 结构化缓存 + ChromaDB 全局 collection
2. **Phase 2 AI 分析**：选模型（Ollama 本地 / OpenAI 格式 API 云端 / NVIDIA Integrate 多 key）+ 提示词模板 → LLM 推理 → Markdown 报告
3. **Phase 3 媒体生成**：MoviePy 智能剪辑高光片段 + GIF 摘要 + 可视化数据图表（亮度/清晰度/饱和度）

---

## 二、运行 / 构建 / 测试命令

### 启动

```bash
# Windows 桌面版（唯一入口）：双击 → Electron 壳启动 + 自动 spawn FastAPI 后端 + BrowserWindow loadURL
start-desktop.bat

# macOS / Linux / 开发模式
cd desktop && npm start           # Electron 壳（自动 spawn 后端）
python -m src.web.serve           # 单独后端（开发用）
cd webapp && npm run dev          # 前端独立开发

# Headless 服务（Docker / 无 GUI）
python -m src.web.serve --port 8000

# 前端独立开发模式
cd webapp && npm run dev

# Electron 独立开发模式（需后端已启动）
cd desktop && npm run dev
```

`launcher.py` 含**版本门禁 + venv 自动创建 + import 验证脚本**；`desktop/runtime-controller.js` 含**Python 子进程托管 + 健康探活 + 端口转发**。

### 测试

```bash
# 标准子集（CI 跑这个，不依赖 ultralytics/faster-whisper 等重依赖）
QT_QPA_PLATFORM=offscreen PYTHONIOENCODING=utf-8 \
  python -m pytest tests/ -q \
  --ignore=tests/test_headless_server.py \
  --ignore=tests/test_e2e_smoke.py

# 全量套件（需先 pip install -r requirements.txt 全量）
QT_QPA_PLATFORM=offscreen PYTHONIOENCODING=utf-8 \
  python -m pytest tests/ -q --ignore=tests/test_headless_server.py
```

- 测试框架：`pytest`（`conftest.py` 把项目根插入 `sys.path`）
- v9 新增测试文件：
  - `tests/test_agent_framework.py`（26 用例）— ReactLoopAgent + Session + Turn 15 phase + 工具四 waterfall + ParallelExecutor 读/写锁
  - `tests/test_im_gateway.py`（30 用例）— IMMailbox lease/ack + GatewayCipher AES + 三 adapter（Mock/微信/TG/Discord）+ IMGateway 路由
  - `tests/test_remote_tunnel.py`（44 用例）— Tunnel 抽象 + 三方案（Mock/Tailscale/Direct/Cloudflare）+ RemoteManager 健康探活 + 自动重连
- 现有测试：`test_agent_tools` / `test_api_clients` / `test_api_gateway_stream` / `test_core_pipeline` / `test_e2e_full_pipeline` / `test_e2e_smoke` / `test_headless_server` / `test_history_manager` / `test_model_manager` / `test_web_api`

### 静态检查

```bash
python -m pyflakes src/ launcher.py     # CI 强制，零告警才放行
mypy src/                                 # mypy.ini: python 3.10, ignore_missing_imports
cd webapp && npm run lint                 # eslint
cd desktop && npm run lint                # eslint
```

### 打包发布

```bash
# Electron 桌面安装包（推荐分发形态）
cd desktop && npm run dist                # electron-builder → Windows NSIS installer

# Web 版（Docker / 无 GUI）
docker build -t tingfeng-hermes .           # CPU 镜像
docker build -f Dockerfile.cuda -t tingfeng-hermes:cuda .   # GPU 镜像
docker compose up                          # 编排（含 GPU profile）

# 前端静态导出
cd webapp && npm run build                # → webapp/out/ 静态文件
```

CI 在打 `v*` tag 时触发 `build-windows` job 跑全量测试 + electron-builder + 上传 artifact。

---

## 三、关键约定与已知坑（改代码前必读）

### 1. Agent 框架 asyncio 约定

`src/core/agent/loop.py` 的 `ReactLoopAgent` 是 async 协程驱动：
- 所有工具调用走 `await tool.execute()`，**不要**混入同步阻塞 IO
- `Session` 的 `SessionEvent` 是 append-only log，**不要**回写或删除已有事件
- `Turn` 的 15 phase 是有序事件链，**不要**跳过 phase 或乱序触发
- 工具四 waterfall（pre → execute → post → result）的 `pre`/`post` 可短路（return SkipResult），`execute` 必跑
- `ParallelExecutor` 读锁共享 / 写锁独占：纯读工具可并行，写工具串行
- 新建工具注册到 `ToolRegistry` 后，`adapter.py` 自动桥接到 Agent 框架

### 2. 凭据存储：分层优先级（安全）

`src/core/credentials/credential_key.py` 的分层优先级：**env > keyring > ini**。
- `VAP_NV_API_KEYS`（11 key）走 env，运行时读不入库
- 单 provider API Key 走 keyring（Windows DPAPI / macOS Keychain / Linux SecretService），不可用降级 ini + 告警
- **禁止**把真实 Key 写进 ini 或硬编码，`.env` 已 gitignore

### 3. IM 网关 Mock 红线

`src/core/im_gateway/` 的三 adapter：
- **Mock adapter**：本地测试用，**绝不**真实连接外部 IM 服务
- **微信 / TG / Discord adapter**：需用户提供 bot token，**测试时用 Mock**，付费 API 红线
- `GatewayCipher` 的 AES 加密是**条件依赖**（cryptography 不在 core requirements），缺失时降级明文 + 告警，**不要**强制把 cryptography 加进 core
- `IMMailbox` 的 SQLite lease/ack 机制保证消息不丢：lease 后必须 ack，超时自动重投

### 4. 插件框架 fiber → contextvars

`src/core/plugins/context.py` 用 **contextvars**（Python 标准库）替代 Cordis 框架的 fiber：
- `PluginContext` 是 contextvars.ContextVar，**不要**引入 Cordis 重依赖
- 插件 patch 是声明式 YAML（`plugin.yaml`），**不要**让插件直接写 Python 代码 hot-patch
- 插件挂载/卸载通过 `loader.py` + `patch.py`，运行时可热插拔

### 5. 模型下载需 SHA256 校验

`model_manager_tab.py`（PyQt6 遗留，将随 v10 清理）的模型下载流程含完整性校验（防 MITM 投毒）。新增下载源必须保留校验逻辑。

### 6. 依赖双写同步

`src/utils/constants.py` 的 `REQUIRED_PACKAGES` 列表与 `requirements.txt` **手动同步**（launcher 用它做 import 验证）。改 `requirements.txt` 时必须同步改 `constants.py`，否则 venv 验证漏装。

### 7. moviepy 2.x API

代码已迁移到 moviepy 2.x：用 `subclipped()` / `resized()` 等 2.x 命名，**不要**回退到 1.x 的 `subclip` / `resize`。

### 8. 已修复的回归坑（不要重新引入）

- `session_id` 用 `uuid4`，**不要**用 `int(time.time())`（同秒冲突）
- `OllamaClient` 必须在客户端层解析 SSE 协议，产出纯文本 delta，**不要**向 UI 泄漏 `{"message":{"content":...}}` JSON 碎片
- `ADVANCED_FEATURES_AVAILABLE` 必须**真实探测** moviepy/matplotlib/seaborn，不要硬编码为 `True`
- 内部哨兵 `__FULL_RESPONSE_END__` **不得**泄漏进最终报告 / Agent 输出
- `smart_extraction` 配置键名必须接入 `extract_smart_keyframes()`，不要被 worker 忽略
- `batch_runner` 已从 QObject + pyqtSignal 改纯 Python `_Signal`（v9 清理），**不要**回退到 pyqtSignal
- Electron 主进程 spawn Python 子进程后必须**健康探活**（`runtime-controller.js` 轮询 `/healthz`），探活失败要重试而非直接 loadURL

### 9. RTSP / 监控 / 远程访问

- `.env` 支持 `VAP_RTSP_URL` / `VAP_MONITOR_DIR` / `VAP_KEY_ITEM_IMAGE`（监控实时流 + 关键物品检测，运行时读取，不入库）
- `.env` 支持 `VAP_TUNNEL_PROVIDER`（`mock` / `tailscale` / `direct` / `cloudflare`）+ 对应凭据
- `.env` 支持 `VAP_IM_GATEWAY_MASTER_KEY`（IM 网关 AES 主密钥）+ `VAP_IM_ADAPTERS`（启用的 adapter 列表）
- `.env` 支持 `VAP_NV_API_KEYS`（NVIDIA 11 key 逗号分隔，provider_router 多 key 轮换）

---

## 四、配置与环境变量

### `.env`（复制 `.env.example`，已 gitignore）

| 变量 | 用途 |
|------|------|
| `VAP_LLM_PROVIDER` | LLM 提供商（`anthropic` / `ollama` / `openai` 格式） |
| `VAP_LLM_BASE_URL` | API 端点（如 `https://api.yjs.im/v1`） |
| `VAP_LLM_MODEL` | 模型名（如 `glm-5.3-flash`） |
| `VAP_LLM_API_KEY` | API Key（**绝不入库**） |
| `VAP_NV_API_KEYS` | NVIDIA 11 key 逗号分隔，provider_router 多 key 轮换 |
| `VAP_NV_BACKOFF_SEC` | 503 退避秒数（默认 1.5） |
| `VAP_NV_MAX_CONCURRENT_PER_KEY` | 单 key 并发上限（默认 2） |
| `VAP_MONITOR_DIR` | 监控目录 |
| `VAP_KEY_ITEM_IMAGE` | 关键物品参考图 |
| `VAP_RTSP_URL` | RTSP 流地址 |
| `VAP_IM_GATEWAY_MASTER_KEY` | IM 网关 AES 主密钥（缺失则降级明文 + 告警） |
| `VAP_IM_ADAPTERS` | 启用的 IM adapter 列表（`mock` / `wechat` / `telegram` / `discord`） |
| `VAP_TUNNEL_PROVIDER` | 远程访问方案（`mock` / `tailscale` / `direct` / `cloudflare`） |
| `VAP_TUNNEL_TOKEN` | Tailscale / Cloudflare Access 凭据 |
| `VAP_PLUGIN_DIR` | 插件加载目录（默认 `plugins/`） |
| `VAP_IP_RATE_LIMIT_PER_MIN` | headless IP 限流（默认 10，0=禁用） |
| `VAP_HEADLESS_TOKEN` | headless Bearer Token 鉴权（空=禁用） |

### `config/app_config.ini`（运行时，已 gitignore）

存 theme / version / venv_path / client_type / api_url / api_key / model_name。`api_key` 字段实际由密钥环覆盖，ini 只留标记位。

### 提示词模板（入库）

`config/prompts/frame_analysis/` 下三个 `.txt`：`describe.txt` / `frame_analysis.txt` / `video_summary.txt`。`PromptLoader` 默认指向此目录。新增模板放此处。

### 插件声明（入库）

`plugins/<plugin-name>/plugin.yaml`：声明式 patch，描述挂载点 / 依赖 / 版本。`loader.py` 扫此目录加载。

---

## 五、禁止提交的文件（见 `.gitignore`）

| 路径 | 类型 |
|------|------|
| `venv/` `.venv/` | Python 虚拟环境 |
| `__pycache__/` `.pytest_cache/` `.coverage` `htmlcov/` | Python 缓存 |
| `logs/` `cache/` | 运行时日志/缓存 |
| `config/chroma_db/` `config/history.db` `config/runs.db*` `config/app_config.ini` | 运行时数据/配置 |
| `.env` | 密钥 |
| `E2E实测结果/` | E2E 产物 |
| `desktop/node_modules/` `desktop/dist/` `desktop/out/` | Electron 构建产物 |
| `webapp/node_modules/` `webapp/out/` `webapp/.next/` | Next.js 构建产物 |
| `website/` | 独立官网项目，不属于本仓库主体 |
| `.claude/` `.codegraph/` `.code-review-graph/` graft/ | AI 工具产物 |
| `plugins/*/node_modules/` `plugins/*/__pycache__/` | 插件缓存 |

> 注：历史上有过 `chroma.sqlite3` / `history.db` 误入库后被清理的提交，新增运行时数据文件务必先加 gitignore。

---

## 六、知识工具（已配置）

- **graft 代码图谱**：`.mcp.json` + `.claude/helpers/graft-hooks.cjs`，PostToolUse 自动索引。结构性问题先 `graft ask` / `graft callers`，再回退 Grep/Read。
- **CodeGraph**：`.codegraph/`，MCP 工具 `codegraph_*`。
- **code-review-graph**：`.code-review-graph/`，需先 `code-review-graph build`。
- MCP 服务器可能因网络超时连不上——视为连接失败而非未配置，提示用户重试即可。

---

## 七、Skill 使用

本项目已安装 superpowers-zh 技能框架（见 `.claude/skills/` 与 `skills-lock.json`）。匹配时优先用：

- **brainstorming** — 任何创造性工作前先做需求分析
- **test-driven-development** — 写实现前先写测试
- **systematic-debugging** — 任何 bug/测试失败/异常前先用
- **verification-before-completion** — 声称完成前必须跑验证命令并确认输出
- **receiving-code-review** / **requesting-code-review** — 审查反馈闭环

---

## 八、常见任务速查

| 想做 | 怎么做 |
|------|--------|
| 加一个 Agent 工具 | `src/core/tools/definition.py` 用 `ToolDefinition` + 四 waterfall + 注册到 registry，`adapter.py` 自动桥接；在 `tests/test_agent_framework.py` 补测试 |
| 加一个 Agent 提示词模板 | `config/prompts/frame_analysis/` 放 `.txt`，前端模板下拉自动收录 |
| 加一个前端页面 | `webapp/app/<route>/page.tsx` 新建，Next 16 App Router；遵守 Tailwind v4 + 玻璃拟态设计系统 |
| 加一个 IM adapter | `src/core/im_gateway/adapters/` 新建，实现 `IMAdapter` 协议（send/receive），注册到 `gateway.py`；在 `tests/test_im_gateway.py` 补测试 |
| 加一个远程 tunnel 方案 | `src/remote/tunnels/` 新建，实现 `Tunnel` 抽象（connect/health/close），注册到 `manager.py`；在 `tests/test_remote_tunnel.py` 补测试 |
| 加一个插件 | `plugins/<name>/plugin.yaml` 声明式 patch + `main.py` 入口，`loader.py` 扫描自动加载 |
| 加一个 subagent 角色 | `src/core/subagent/role_template.py` 加 RoleTemplate，`director.py` 路由表加 entry |
| 改 LLM 接入 | `src/core/logic.py` 的 `VideoAnalyzer` / `OllamaClient`；API 客户端测试在 `tests/test_api_clients.py` |
| 改版本号 | `src/utils/constants.py:APP_VERSION` + `CHANGELOG.md` 顶部加条目 + `desktop/package.json` version + `webapp/package.json` version + `config/app_config.ini` 的 `version`（运行时文件，勿入库） |
| 加运行时数据文件 | 先加 `.gitignore` 再创建，避免误入库 |

---

*最后更新：2026-09-11（v10.3.0 / v10.3.1 P0 批次）*
