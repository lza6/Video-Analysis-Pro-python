# API 参考

TingFeng Hermes 后端 FastAPI 提供 17 个 router 前缀，Swagger UI **自动生成**，无需手写。

## Swagger / OpenAPI

启动后端后访问：

```
http://localhost:8000/api/docs      # Swagger UI
http://localhost:8000/openapi.json  # OpenAPI Schema
```

本页只列端点前缀与职责，详细请求 / 响应字段以 Swagger 实时生成为准。

## Router 前缀

| 前缀 | tag | 目录 | 职责 |
|------|-----|------|------|
| `/api` | health | `routers/health.py` | 健康探活 `/healthz` |
| `/api` | analyze | `routers/analyze.py` | 视频分析三阶段流水线入口 |
| `/api` | metrics | `routers/metrics.py` | 运行指标 |
| `/api` | media | `routers/media.py` | 媒体文件 / 帧图 |
| `/api/models` | models | `routers/models.py` | 模型列表 / 下载 |
| `/api/agent` | agent | `routers/agent.py` | Agent 会话 SSE 流 |
| `/api/config` | config | `routers/config.py` | 运行时配置读写 |
| `/api/providers` | providers | `routers/providers.py` | LLM provider 路由 |
| `/api/logs` | logs | `routers/logs.py` | 日志流 |
| `/api/decisions` | decisions | `routers/decisions.py` | 决策记录 |
| `/api/skills` | skills | `routers/skills.py` | Skill 列表 |
| `/api` | batch | `routers/batch.py` | 批量任务编排 |
| `/api/surveillance` | surveillance | `routers/surveillance.py` | 监控 RTSP 流 |
| `/api/im` | im-gateway | `routers/im_gateway.py` | IM 网关消息投递 |
| `/api/remote` | remote-access | `routers/remote.py` | 远程 tunnel 管理 |
| `/api/requests` | requests | `routers/requests.py` | 请求追踪 |

> 路由前缀以 `src/web/app.py` 的 `include_router` 为准，本页随版本同步。

## 鉴权

- 桌面模式：本地无鉴权（loopback）
- Headless 模式：`VAP_HEADLESS_TOKEN` Bearer Token + `VAP_IP_RATE_LIMIT_PER_MIN` IP 限流

详见 [快速开始](./getting-started) 的 `.env` 配置段。
