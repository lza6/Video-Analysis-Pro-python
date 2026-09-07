<p align="center"><img src="resources/logo_256.png" alt="TingFeng Hermes" width="160"/></p>

# TingFeng Hermes（听风·赫尔墨斯）
## 通用全能 AI Agent 桌面平台 | Your Universal AI Agent Desktop Platform
### 由 听风公司 (Tingfeng) 出品

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)
[![Web](https://img.shields.io/badge/Web-Electron-black.svg)](https://www.electronjs.org/)
[![AI](https://img.shields.io/badge/AI-Agent%20Framework-orange.svg)](./src/core/agent)

> **"跑在你设备里的全能 AI Agent。通过 IM 渠道连接万物，内置视频分析能力——不只是工具，是一个会思考、会协作、会进化的数字伙伴。"**

> **截图待补**：v10.0.0 完成自进化+可观测+真实化迭代后，webapp UI（Next.js 16 + React 19 + Tailwind v4 玻璃拟态）截图即将更新。

---

## 目录

1. [项目简介](#项目简介)
2. [核心功能](#核心功能)
3. [快速开始](#快速开始)
4. [架构总览](#架构总览)
5. [安装与开发](#安装与开发)
6. [打包发布](#打包发布)
7. [开源协议](#开源协议)

---

## 项目简介

**TingFeng Hermes** 是听风公司出品的通用全能 AI Agent 桌面平台。它从 v8.0.0 的 Web UI 视频分析工具升级而来，v9.0.0 完成"通用全能 Agent 桌面平台"转型，v10.0.0 迭代到"自进化 + 可观测 + 真实化"：

- **跑在你设备里**：Electron 桌面壳 + 本地 Python FastAPI 后端，数据不出设备
- **通过 IM 渠道连接万物**：内置 IM 网关（微信 / Telegram / Discord），把 Agent 能力投递到聊天渠道
- **内置视频分析能力**：三阶段流水线（抽帧/转录/检测 → LLM 推理 → 媒体生成）作为 Agent 工具集之一
- **DSH 式 Agent 框架**：ReactLoopAgent + SessionEvent append-only log + 工具四 waterfall + subagent director + 插件框架
- **远程访问**：Tailscale Serve / Direct / Cloudflare Access tunnel，从任意设备访问你的 Agent

### v10.0.0 新增能力

- **自进化**：Session 持久化记忆（崩溃可恢复）+ 工具异常分级自动重试/降级 + Agent 经验沉淀进 skill
- **可观测**：结构化 JSON 日志 + trace_id 贯穿 + GC 长跑守护 + dashboard 总览页
- **真实化**：提示注入守卫（零宽字符 + system: 伪冒检测）+ 凭据审计与轮换 + VLM 视觉理解接入
- **质量门禁**：CI 覆盖率门禁 + 前端 Playwright E2E + 旧产物清理

### 核心价值观

- **隐私至上**：本地运行，数据不出设备；IM 网关只投递摘要，不投递原始视频
- **Agent 形态**：对话驱动 + 自动 plan + 工具自动调用 + 长程任务追踪 + 每步可追溯
- **可扩展**：插件框架（声明式 YAML patch）+ subagent 角色 + skill 库，社区可贡献能力
- **开源精神**：GPL-3.0，修改必开源

---

## 核心功能

### Agent 对话
- ReactLoopAgent（ReAct 循环 + 思考链 + 工具调用）
- SessionEvent append-only log，每步可追溯
- Turn 15 phase 事件链（user_msg → plan → tool_pre → tool_exec → tool_post → tool_result → … → assistant_msg）
- subagent director 三模式（foreground / background / continuable）+ 四级路由（call > role > default > inherit）

### 视频分析（内置能力）
- 三阶段流水线：OpenCV 智能抽帧 + Whisper 音频转录 + YOLOv11 物体检测 → LLM 推理 → MoviePy 剪辑 + GIF 摘要 + 数据图表
- 跨视频向量知识库（ChromaDB 全局 collection + 自然语言搜索）
- 模型支持：Ollama 本地 / OpenAI 格式 API / NVIDIA Integrate 多 key 轮换

### 批量监控分析
- motion_detector 1fps 抽帧 + scenedetect + 帧差分 + 昼夜自适应阈值，只送变化时段给 AI，省 99% API 调用
- 二次验证防误判 + 断点续跑 + 跨会话记忆
- skills 蒸馏（稀疏走廊 / 人多密集 / 夜间自适应），agent 按场景自动选 skill

### RTSP 实时流
- RTSP 拉流 + 帧差预筛 + VLM 确认 + 命中回调投到 agent 对话

### IM 网关
- 微信 / Telegram / Discord adapter + IMMailbox SQLite lease/ack 投递箱
- GatewayCipher AES 加密（条件依赖，缺失降级明文 + 告警）
- Agent 可主动推消息到 IM 渠道（命中提醒 / 任务完成 / 异常告警）

### 远程访问
- Tailscale Serve / Direct / Cloudflare Access tunnel 三方案
- RemoteManager 健康探活 + 自动重连
- 从手机/平板访问桌面 Agent

### 插件生态
- 声明式 YAML patch（`plugin.yaml`），社区可贡献插件
- contextvars 替 Cordis fiber，轻量无重依赖
- 运行时热插拔

---

## 快速开始

### 1. 准备工作
- **Python**：3.10 或更高（[下载](https://www.python.org/downloads/)）
- **Node.js**：18+（用于 Electron 壳和 webapp，[下载](https://nodejs.org/)）
- **FFmpeg**：视频处理核心（Windows 软件自带 imageio-ffmpeg，可免装；Mac `brew install ffmpeg`，Linux `sudo apt install ffmpeg`）

### 2. 下载项目
```bash
git clone https://github.com/lza6/Video-Analysis-Pro-python.git
cd Video-Analysis-Pro-python
```

### 3. 一键启动

**Windows 桌面版（唯一入口）**：双击根目录 `start-desktop.bat`
- 自动检测 Node.js + 首次装 desktop 依赖（electron + electron-builder）
- 清理上次残留进程（netstat 查 8000-8019 占用，只杀 python，不动系统服务）
- 启动 Electron 壳 → spawn Python FastAPI 后端子进程 → BrowserWindow loadURL
- 单实例 + 系统托盘 + 关窗口最小化到托盘

**Mac / Linux / 开发模式**：
```bash
cd desktop && npm start           # Electron 壳（会自动 spawn 后端）
# 或单独后端（开发用）：
python -m src.web.serve
# 或前端独立开发：
cd webapp && npm run dev
```

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    Electron 桌面壳 (desktop/)               │
│  main.js + runtime-controller.js + preload.js + 托盘 + 更新   │
└────────────────────────┬────────────────────────────────────┘
                         │ spawn python src/web/serve.py
                         │ + 健康探活 /healthz + 端口转发
┌────────────────────────▼────────────────────────────────────┐
│              FastAPI 后端 (src/web/ 12 router)               │
│  health / analyze / metrics / media / models / agent /     │
│  config / logs / decisions / skills / batch / surveillance │
└──────┬───────────────────────────────────┬─────────────────┘
       │                                   │
       ▼                                   ▼
┌──────────────────┐              ┌──────────────────────┐
│  Agent 框架      │              │  Next.js 前端        │
│  (src/core/      │              │  (webapp/ 13 页)     │
│   agent/tools/   │              │  Next 16 + React 19  │
│   subagent/      │              │  + Tailwind v4       │
│   credentials/   │              │  玻璃拟态            │
│   plugins/)      │              └──────────────────────┘
└────┬──────┬──────┬─────────────┘
     │      │      │
     ▼      ▼      ▼
┌──────┐ ┌──────┐ ┌──────────────┐
│ IM   │ │ 远程 │ │ 视频分析核心 │
│ 网关 │ │ 访问 │ │ (logic.py)   │
│      │ │      │ │ 三阶段流水线 │
│ 微信 │ │ TS   │ └──────────────┘
│ TG   │ │ CF   │
│ DC   │ │ Dir  │
└──────┘ └──────┘
```

| 层 | 目录 | 说明 |
|----|------|------|
| Electron 壳 | `desktop/` | 主进程 + Python 子进程托管 + 单实例 + 托盘 + 自动更新 |
| FastAPI 后端 | `src/web/` | 12 router + SSE 流式 + uvicorn |
| Next.js 前端 | `webapp/` | 13 页 + 静态导出 + 玻璃拟态 |
| Agent 框架 | `src/core/agent/` | ReactLoopAgent + Session + Turn |
| 工具系统 | `src/core/tools/` | ToolDefinition + 四 waterfall + ParallelExecutor |
| Subagent | `src/core/subagent/` | Director + RoleTemplate + 三模式 |
| 凭据 | `src/core/credentials/` | env > keyring > ini 分层 |
| 插件 | `src/core/plugins/` | contextvars + 声明式 YAML patch |
| IM 网关 | `src/core/im_gateway/` | Mailbox + Cipher + 三 adapter |
| 远程访问 | `src/remote/` | Tunnel 抽象 + 三方案 + RemoteManager |
| 视频核心 | `src/core/logic.py` | 三阶段流水线（内置 Agent 工具集） |

---

## 安装与开发

### 后端（Python）
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt          # core
# 或全量（含可选 OCR）：
pip install -r requirements-ocr.txt
```

### 前端（webapp/）
```bash
cd webapp
npm install
npm run dev          # 开发模式 http://localhost:3000
npm run build        # 静态导出 → webapp/out/
```

### Electron 壳（desktop/）
```bash
cd desktop
npm install
npm run dev          # 开发模式（需后端已启动）
npm run dist         # 打包 Windows 安装包
```

### 配置
复制 `.env.example` 为 `.env`，填入：
- `VAP_LLM_API_KEY` / `VAP_LLM_BASE_URL` / `VAP_LLM_MODEL`（LLM 凭据）
- `VAP_NV_API_KEYS`（NVIDIA 11 key，逗号分隔，可选）
- `VAP_IM_GATEWAY_MASTER_KEY`（IM 网关 AES 主密钥，可选）
- `VAP_IM_ADAPTERS`（启用的 IM adapter，可选）
- `VAP_TUNNEL_PROVIDER`（远程访问方案，可选）

API Key 优先存 OS 密钥环（Windows DPAPI / macOS Keychain / Linux SecretService），降级 ini + 告警，**绝不入库**。

---

## 打包发布

```bash
# Electron 桌面安装包（推荐分发形态）
cd desktop && npm run dist

# Web 版 Docker 镜像
docker build -t tingfeng-hermes .           # CPU
docker build -f Dockerfile.cuda -t tingfeng-hermes:cuda .   # GPU
docker compose up

# 前端静态导出
cd webapp && npm run build
```

CI 在打 `v*` tag 时触发 `build-windows` job：全量测试 + electron-builder + 上传 artifact。

---

## 开源协议

本项目采用 **GNU General Public License v3.0 (GPL-3.0)**。

- 您可以免费使用、复制、修改本项目
- 如果您修改了代码并发布，**您也必须开源您的修改代码**（传染性）
- 让我们一起维护开源社区的繁荣

---

**感谢您的阅读！** 如果觉得这个项目有趣，请给一个 Star 鼓励一下。

*(Project maintained by lza6 · 听风公司 (Tingfeng) 出品)*

---

*文档最后更新：2026-09-06 (v10.0.0)*
