"""Pydantic 请求/响应模型 + SSE 事件类型。

与前端 webapp/src/lib/types.ts 对齐(前端由 codegen 或手动同步)。
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================ 请求 ============================

class AnalyzeRequest(BaseModel):
    """POST /api/analyze 的 JSON 配置部分(配合 multipart 文件)。

    multipart: file=<video>; config=<AnalyzeRequest JSON>
    """
    density: float = Field(0.3, ge=0.0, le=1.0, description="抽帧密度 0.1-1.0")
    smart_extraction: bool = Field(True, description="智能关键帧(scenedetect)")
    enable_audio: bool = Field(True, description="Whisper 音频转录")
    enable_yolo: bool = Field(False, description="YOLO 物体检测(重,默认关)")
    enable_ocr: bool = Field(False, description="OCR 文字识别(需 PaddleOCR)")
    model: str = Field("qwen2.5:3b", description="Ollama 模型名")
    custom_prompt: Optional[str] = Field(None, description="自定义提示词模板")
    # 本地部署:也允许传本机视频路径(绕过上传)
    local_path: Optional[str] = Field(
        None, description="本机视频绝对路径(本地模式,绕过上传)"
    )


# ============================ 响应 ============================

class JobCreated(BaseModel):
    job_id: str
    status: str
    video_name: str


class JobSummary(BaseModel):
    job_id: str
    video_name: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration: float = 0.0
    frame_count: int = 0
    transcript_preview: str = ""
    report_preview: str = ""
    error: Optional[str] = None


class FrameInfo(BaseModel):
    timestamp: float
    metrics: dict[str, float] = {}
    url: str
    vision_content: str = ""
    ocr_text: str = ""


class JobDetail(BaseModel):
    """GET /api/jobs/{id} 完整结果。"""
    job_id: str
    video_name: str
    status: str
    duration: float
    frame_count: int
    transcript: str
    report: str
    frames: list[FrameInfo]
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    capabilities: dict[str, Any]
    disk_free_gb: float
    keyring_available: bool


# ============================ SSE 事件载荷 ============================

# 事件类型枚举(前端按 type 分发渲染)
class SSEEvent:
    PHASE = "phase"              # {"phase": "extraction"|"audio"|"analysis"|"done"}
    PROGRESS = "progress"        # {"label": "...", "value": 0-100}
    FRAME = "frame"              # FrameInfo
    TRANSCRIPT = "transcript"    # {"text": "..."}
    REPORT_TOKEN = "report-token"  # {"token": "..."}
    LOG = "log"                  # {"level": "info", "msg": "..."}
    DONE = "done"                # {"duration": ..., "frame_count": ..., "report_preview": ...}
    ERROR = "error"              # {"message": "..."}
    # 内部哨兵,不发出给客户端
    _CLOSE = "__close__"
