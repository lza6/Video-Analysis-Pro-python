"use client";

import { useCallback, useRef, useState } from "react";
import { apiPostMultipart } from "@/lib/api";
import { streamSSE } from "@/lib/sse";
import type {
  AnalyzeConfig,
  DoneEvent,
  ErrorEvent,
  FrameInfo,
  JobCreated,
  LogEvent,
  PhaseEvent,
  ProgressEvent,
  ReportTokenEvent,
  TranscriptEvent,
} from "@/lib/types";

export type JobPhase = "idle" | "submitting" | "running" | "done" | "failed";

export interface AnalyzeState {
  phase: JobPhase;
  stage: string;
  progress: number;
  progressLabel: string;
  frames: FrameInfo[];
  transcript: string;
  report: string;
  logs: LogEvent[];
  jobId: string | null;
  error: string | null;
}

const INITIAL: AnalyzeState = {
  phase: "idle",
  stage: "",
  progress: 0,
  progressLabel: "",
  frames: [],
  transcript: "",
  report: "",
  logs: [],
  jobId: null,
  error: null,
};

/**
 * 分析作业生命周期 hook:提交 → SSE 消费 → 状态聚合。
 * 取代 PyQt6 的 ExtractionWorker/AnalysisWorker + 信号槽。
 */
export function useAnalyzeJob() {
  const [state, setState] = useState<AnalyzeState>(INITIAL);
  const abortRef = useRef<AbortController | null>(null);

  const start = useCallback(async (config: AnalyzeConfig, file?: File | null) => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    setState({ ...INITIAL, phase: "submitting", stage: "提交作业" });

    let created: JobCreated;
    try {
      created = await apiPostMultipart<JobCreated>(
        "/api/analyze",
        { config: JSON.stringify(config) },
        file,
      );
    } catch (err) {
      setState((s) => ({
        ...s,
        phase: "failed",
        error: err instanceof Error ? err.message : String(err),
      }));
      return;
    }

    setState((s) => ({ ...s, phase: "running", jobId: created.job_id, stage: "启动中" }));

    await streamSSE(`/api/jobs/${created.job_id}/stream`, {
      signal: ac.signal,
      onEvent: (msg) => {
        setState((s) => {
          switch (msg.type) {
            case "phase": {
              const d = msg.data as PhaseEvent;
              return { ...s, stage: d.phase };
            }
            case "progress": {
              const d = msg.data as ProgressEvent;
              return { ...s, progress: d.value, progressLabel: d.label };
            }
            case "frame":
              return { ...s, frames: [...s.frames, msg.data as FrameInfo] };
            case "transcript": {
              const d = msg.data as TranscriptEvent;
              return { ...s, transcript: d.text };
            }
            case "report-token": {
              const d = msg.data as ReportTokenEvent;
              return { ...s, report: s.report + d.token };
            }
            case "log": {
              const d = msg.data as LogEvent;
              return { ...s, logs: [...s.logs.slice(-199), d] };
            }
            case "done": {
              void (msg.data as DoneEvent); // 标记消费
              return {
                ...s,
                phase: "done",
                stage: "完成",
                progress: 100,
                progressLabel: "分析完成",
              };
            }
            case "error": {
              const d = msg.data as ErrorEvent;
              return { ...s, phase: "failed", error: d.message };
            }
            default:
              return s;
          }
        });
      },
      onError: (err) => {
        setState((s) =>
          s.phase === "running" || s.phase === "submitting"
            ? {
                ...s,
                phase: "failed",
                error: err instanceof Error ? err.message : String(err),
              }
            : s,
        );
      },
    });
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(INITIAL);
  }, []);

  return { state, start, reset };
}
