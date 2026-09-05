"use client";

import { useState } from "react";
import { VideoPicker } from "@/components/analyze/VideoPicker";
import { ConfigPanel } from "@/components/analyze/ConfigPanel";
import { PhaseIndicator } from "@/components/analyze/PhaseIndicator";
import { FrameStrip } from "@/components/analyze/FrameStrip";
import { ReportView } from "@/components/analyze/ReportView";
import { LogConsole } from "@/components/analyze/LogConsole";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useAnalyzeJob } from "@/hooks/useAnalyzeJob";
import type { AnalyzeConfig } from "@/lib/types";

const DEFAULT_CONFIG: AnalyzeConfig = {
  density: 0.3,
  smart_extraction: true,
  enable_audio: false,
  enable_yolo: false,
  enable_ocr: false,
  model: "qwen2.5:3b",
  custom_prompt: null,
  local_path: null,
};

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [localPath, setLocalPath] = useState("");
  const [config, setConfig] = useState<AnalyzeConfig>(DEFAULT_CONFIG);
  const { state, start, reset } = useAnalyzeJob();

  const busy = state.phase === "submitting" || state.phase === "running";
  const hasSource = Boolean(file) || localPath.trim().length > 0;

  const patch = (p: Partial<AnalyzeConfig>) =>
    setConfig((c) => ({ ...c, ...p }));

  const run = () => {
    const finalConfig: AnalyzeConfig = {
      ...config,
      local_path: file ? null : localPath.trim() || null,
    };
    start(finalConfig, file);
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">
          AI <span className="text-gradient">视频分析</span>
        </h1>
        <p className="text-sm text-mute mt-1.5">
          三阶段流水线:数据提取 → AI 分析 → 生成报告。全程本地运行,隐私不出本机。
        </p>
      </header>

      <div className="grid lg:grid-cols-2 gap-6 items-start">
        <div className="space-y-6">
          <VideoPicker
            file={file}
            localPath={localPath}
            onFile={setFile}
            onLocalPath={setLocalPath}
            disabled={busy}
          />
          <ConfigPanel config={config} onChange={patch} disabled={busy} />

          <div className="flex items-center gap-3">
            <Button size="lg" onClick={run} disabled={!hasSource || busy}>
              {busy ? "分析中…" : "开始分析"}
            </Button>
            {(state.phase === "done" || state.phase === "failed") && (
              <Button variant="chip" onClick={reset}>
                重新开始
              </Button>
            )}
          </div>

          {state.error && (
            <Card className="p-4 border-danger/40">
              <p className="text-sm text-danger">
                <span className="font-medium">错误:</span> {state.error}
              </p>
            </Card>
          )}
        </div>

        <div className="space-y-6">
          <PhaseIndicator state={state} />
          <FrameStrip frames={state.frames} />
          <ReportView report={state.report} />
          {state.transcript && (
            <Card className="p-5">
              <h3 className="text-sm font-medium text-white mb-3">音频转录</h3>
              <p className="text-sm text-mute leading-relaxed max-h-48 overflow-y-auto whitespace-pre-wrap">
                {state.transcript}
              </p>
            </Card>
          )}
          <LogConsole logs={state.logs} />
        </div>
      </div>
    </div>
  );
}
