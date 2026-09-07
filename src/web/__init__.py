"""TingFeng Hermes — Web 后端包(FastAPI)。

复用 src/core/ 全部分析能力,通过 HTTP + SSE 暴露:

  GET  /api/health               → 能力矩阵 + 磁盘余量 + keyring 状态
  POST /api/analyze              → 创建分析作业,返回 job_id
  GET  /api/jobs/{id}/stream     → SSE: phase/progress/frame/transcript/report-token
  GET  /api/jobs                 → 列出作业
  GET  /api/jobs/{id}            → 作业详情
  GET  /api/jobs/{id}/frames     → 关键帧列表
  GET  /api/frames/{sanitized}   → 帧图片(路径消毒后 FileResponse)
  POST /api/agent/chat          → SSE: thought/action/observation/answer(ReAct)
  POST /api/models/download      → 模型下载 + SHA256 校验(SSE 进度)
  ... (完整路由见 src/web/routers/)

启动: uvicorn src.web.app:app --host 0.0.0.0 --port 8000
"""
__version__ = "0.1.0"
