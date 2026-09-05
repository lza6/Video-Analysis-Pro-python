/**
 * 与后端 src/web/schemas.py 对齐的 TS 类型。
 * 改后端 schema 时同步改这里(字段名一致,snake_case 保留,减少映射层)。
 */

export type JobStatus = "pending" | "running" | "done" | "failed" | "canceled";

export interface JobCreated {
  job_id: string;
  status: JobStatus;
  video_name: string;
}

export interface JobSummary {
  job_id: string;
  video_name: string;
  status: JobStatus;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  duration: number;
  frame_count: number;
  transcript_preview: string;
  report_preview: string;
  error: string | null;
}

export interface FrameInfo {
  timestamp: number;
  metrics: Record<string, number>;
  url: string;
  vision_content: string;
  ocr_text: string;
}

export interface JobDetail {
  job_id: string;
  video_name: string;
  status: JobStatus;
  duration: number;
  frame_count: number;
  transcript: string;
  report: string;
  frames: FrameInfo[];
  error: string | null;
}

export interface HealthResponse {
  status: string;
  capabilities: {
    clip_semantic: boolean;
    nvidia_gpu: boolean;
    advanced_media: boolean;
    ffmpeg: boolean;
    ocr: boolean;
    llm_backend: string;
  };
  disk_free_gb: number;
  keyring_available: boolean;
}

export interface AnalyzeConfig {
  density: number;
  smart_extraction: boolean;
  enable_audio: boolean;
  enable_yolo: boolean;
  enable_ocr: boolean;
  model: string;
  custom_prompt?: string | null;
  local_path?: string | null;
}

/** SSE 事件(后端 src/web/schemas.py SSEEvent)。 */
export type SSEType =
  | "phase"
  | "progress"
  | "frame"
  | "transcript"
  | "report-token"
  | "log"
  | "done"
  | "error";

export interface PhaseEvent { phase: string; }
export interface ProgressEvent { label: string; value: number; }
export interface LogEvent { level: string; msg: string; }
export interface TranscriptEvent { text: string; }
export interface ReportTokenEvent { token: string; }
export interface DoneEvent {
  duration: number;
  frame_count: number;
  report_preview: string;
}
export interface ErrorEvent { message: string; }
