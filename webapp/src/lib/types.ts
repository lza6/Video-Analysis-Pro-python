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

// ============================ models ============================

export interface ModelCard {
  id: string;
  name: string;
  desc: string;
  size_hint: string;
  exists: boolean;
  path: string | null;
  size_mb: number;
  sha256_expected: boolean;
}

export interface ModelsResponse {
  cards: ModelCard[];
  local_models: { name: string; type: string }[];
}

export interface DownloadProgress { value: number; label: string; }
export interface DownloadVerify { ok: boolean; msg?: string; }

// ============================ config ============================

export interface ProviderConfigOut {
  client_type: number;
  api_url: string;
  model_name: string;
  has_key: boolean;
  keyring_available: boolean;
  nvidia_keys: number;
}

export interface Preset {
  name: string;
  api_url: string;
  model?: string;
  notes?: string;
}

export interface PromptTemplate {
  name: string;
  content: string;
}

export interface TestResult {
  ok: boolean;
  models?: string[];
  count?: number;
  via?: string;
  key_used?: string;
  total_keys?: number;
  error?: string;
}

// ============================ metrics / media ============================

export interface MetricsResponse {
  job_id: string;
  avg: Record<string, number>;
  series?: {
    timestamps: number[];
    brightness: number[];
    saturation: number[];
    sharpness: number[];
  };
  chart_url?: string | null;
  generating?: boolean;
}

export interface MediaResponse {
  job_id: string;
  clips: string[];
  summary_video: string | null;
  gif: string | null;
}

// ============================ agent ============================

export interface PlanStep {
  step_id: string;
  description: string;
  tool?: string;
  args?: Record<string, unknown>;
}

export interface AgentChatResponse {
  intent: string;
  skill_name?: string;
  plan_steps?: PlanStep[];
  reply: string;
}

export interface AgentRunStep {
  description: string;
  tool: string | null;
  result: string | null;
  status: string | null;
}

export interface AgentRunResponse {
  done: boolean;
  step: AgentRunStep | null;
}
