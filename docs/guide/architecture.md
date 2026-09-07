# 架构总览

v9.0.0 完成"通用全能 Agent 桌面平台"转型：Electron 桌面壳 + Python FastAPI 后端 + Next.js 前端 + DSH 式 Agent 框架 + IM 网关 + 远程访问 + 插件生态。视频分析能力降为内置 Agent 工具集之一。

## 四层分层

| 层 | 目录 | 职责 |
|----|------|------|
| Electron 桌面壳 | `desktop/` | `main.js` 主进程 + `runtime-controller.js`（spawn Python 子进程 + 健康探活 + 端口转发）+ `preload.js` IPC 桥 + electron-builder 打包 |
| FastAPI 后端 | `src/web/` | `app.py` 装配 17 router，`serve.py` 跑 uvicorn，SSE 流式 |
| Next.js 前端 | `webapp/` | Next 16 + React 19 + Tailwind v4，静态导出，13 页路由，玻璃拟态设计系统 |
| Agent 框架 | `src/core/agent/` | ReactLoopAgent（ReAct 循环 + 工具调用 + 思考链）+ Session + Turn 15 phase |

## 三阶段视频流水线

`src/core/logic.py` 三阶段：

1. **Phase 1 数据提取** — OpenCV 智能抽帧 + Whisper 转录 + YOLO 检测 → ChromaDB 全局 collection
2. **Phase 2 AI 分析** — 多模型推理（Ollama / OpenAI 格式 / NVIDIA 多 key）+ 提示词模板 → Markdown 报告
3. **Phase 3 媒体生成** — MoviePy 2.x 智能剪辑高光 + GIF 摘要 + 数据图表

## IM 网关

`src/core/im_gateway/`：`IMMailbox`（SQLite lease/ack 投递箱，消息不丢）+ `GatewayCipher`（AES 条件依赖，缺失降级明文 + 告警）+ 三 adapter（Mock / 微信 / TG / Discord）+ `IMGateway` 路由。

## 远程访问

`src/remote/`：`Tunnel` 抽象 + 四方案（Mock / Tailscale Serve / Direct / Cloudflare Access）+ `RemoteManager` 健康探活 + 自动重连。

## 配置与环境变量

| 变量 | 用途 |
|------|------|
| `VAP_LLM_PROVIDER` | LLM 提供商 |
| `VAP_LLM_API_KEY` | API Key（绝不入库） |
| `VAP_NV_API_KEYS` | NVIDIA 11 key 轮换 |
| `VAP_IM_ADAPTERS` | IM adapter 启用列表 |
| `VAP_TUNNEL_PROVIDER` | 远程访问方案 |

完整列表见仓库根 `CLAUDE.md` 第四节。
