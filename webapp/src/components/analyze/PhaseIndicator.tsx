"use client";

import { Card } from "@/components/ui/Card";
import type { AnalyzeState } from "@/hooks/useAnalyzeJob";
import { cn } from "@/lib/utils";

const STAGE_LABEL: Record<string, string> = {
  "": "等待",
  submitting: "提交作业",
  extraction: "Phase 1 · 数据提取",
  audio: "Phase 1.5 · 音频转录",
  analysis: "Phase 2 · AI 分析",
  done: "完成",
};

/** 三阶段进度指示器:extraction → audio → analysis → done,带进度条。 */
export function PhaseIndicator({ state }: { state: AnalyzeState }) {
  const order = ["extraction", "audio", "analysis", "done"];
  const currentIdx = state.stage ? order.indexOf(state.stage) : -1;
  const active = state.phase === "running" || state.phase === "submitting";

  return (
    <Card className="p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-white">
          {state.phase === "failed"
            ? "分析失败"
            : STAGE_LABEL[state.stage] ?? state.stage}
        </span>
        <span className="text-xs text-mute font-mono">
          {state.progressLabel || `${state.progress}%`}
        </span>
      </div>

      <div className="h-1.5 rounded-full bg-white/10 overflow-hidden mb-4">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500",
            state.phase === "failed" ? "bg-danger" : "bg-gradient-to-r from-accent-2 to-accent",
          )}
          style={{ width: `${state.progress}%` }}
        />
      </div>

      <ol className="flex items-center gap-2 text-[11px]">
        {order.map((s, i) => {
          const done = currentIdx > i || state.phase === "done";
          const cur = currentIdx === i && active;
          return (
            <li key={s} className="flex items-center gap-2 flex-1">
              <span
                className={cn(
                  "w-2 h-2 rounded-full shrink-0",
                  done ? "bg-ok" : cur ? "bg-accent animate-[pulse-glow_1.5s_ease-in-out_infinite]" : "bg-white/20",
                )}
              />
              <span className={cn(done || cur ? "text-mist" : "text-mute/60", "truncate")}>
                {STAGE_LABEL[s]}
              </span>
              {i < order.length - 1 && (
                <span className="flex-1 h-px bg-white/10 hidden sm:block" />
              )}
            </li>
          );
        })}
      </ol>
    </Card>
  );
}
