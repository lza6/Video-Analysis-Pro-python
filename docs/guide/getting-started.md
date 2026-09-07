# 快速开始

本页 5 分钟带你把 TingFeng Hermes 在 Windows 本地跑起来。

## 环境要求

| 工具 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.10 | CI 矩阵测 3.10 / 3.11，venv 子目录名 `venv` |
| Node.js | 18 | 前端 Next 16 + 桌面 Electron |
| FFmpeg | 任意 | 视频抽帧 / 剪辑依赖（可选，缺失自动跳过相关功能） |
| Git | 任意 | clone 仓库 |

## 一键准备（推荐）

```powershell
git clone https://github.com/lza6/Video-Analysis-Pro-python
cd Video-Analysis-Pro-python
.\scripts\dev-setup.ps1
```

> TODO: `scripts/dev-setup.ps1` 当前由 12.2 onboarding 任务落地，脚本未就绪前请按下方手动步骤。

## 手动步骤

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd webapp; npm install; cd ..
cd desktop; npm install; cd ..
copy .env.example .env
```

## 配置 .env

打开 `.env`，至少填：

```ini
VAP_LLM_PROVIDER=openai
VAP_LLM_BASE_URL=https://api.yjs.im/v1
VAP_LLM_MODEL=glm-5.3-flash
VAP_LLM_API_KEY=sk-xxx   # 绝不入库
```

详见 [架构 - 配置与环境变量](./architecture#配置与环境变量)。

## 启动

`.\start-desktop.bat`（Electron 壳 + 自动 spawn FastAPI 后端）。打开 `http://localhost:8000/api/docs` 看 Swagger，前端窗口由 Electron 自动加载。

## 下一步

- [架构总览](./architecture) — 四层分层与三阶段流水线
- [插件开发](./plugin-development) — 声明式 patch 接入
