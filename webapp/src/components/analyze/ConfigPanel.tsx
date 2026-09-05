"use client";

import { Card } from "@/components/ui/Card";
import type { AnalyzeConfig } from "@/lib/types";

interface ConfigPanelProps {
  config: AnalyzeConfig;
  onChange: (patch: Partial<AnalyzeConfig>) => void;
  disabled?: boolean;
}

const STAGES: { key: keyof AnalyzeConfig; label: string; hint: string }[] = [
  { key: "smart_extraction", label: "智能关键帧", hint: "按场景切换抽帧,而非固定密度" },
  { key: "enable_audio", label: "音频转录", hint: "Whisper 听写,耗时较长" },
  { key: "enable_yolo", label: "物体检测", hint: "YOLOv11 标注画面物体(重)" },
  { key: "enable_ocr", label: "文字识别", hint: "PaddleOCR 提取画面文字(需装 OCR)" },
];

/** 分析参数面板:抽帧密度 + 阶段开关 + 模型 + 自定义提示词。 */
export function ConfigPanel({ config, onChange, disabled }: ConfigPanelProps) {
  return (
    <Card className="p-5 space-y-5">
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="density" className="text-xs text-mute">
            抽帧密度
          </label>
          <span className="text-xs text-accent font-mono">
            {(config.density * 100).toFixed(0)}%
          </span>
        </div>
        <input
          id="density"
          type="range"
          min={0.1}
          max={1}
          step={0.05}
          value={config.density}
          disabled={disabled}
          onChange={(e) => onChange({ density: Number(e.target.value) })}
          className="w-full accent-[oklch(0.8_0.12_205)]"
        />
        <p className="text-[11px] text-mute/70 mt-1">
          越高越细致(1.0 ≈ 每秒 1 帧),智能关键帧开启时此项作参考
        </p>
      </div>

      <div className="space-y-2">
        {STAGES.map((s) => (
          <label
            key={s.key}
            className="flex items-start gap-3 cursor-pointer group"
          >
            <input
              type="checkbox"
              checked={Boolean(config[s.key])}
              disabled={disabled}
              onChange={(e) => onChange({ [s.key]: e.target.checked })}
              className="mt-0.5 w-4 h-4 accent-[oklch(0.68_0.19_295)] shrink-0"
            />
            <span>
              <span className="block text-sm text-white group-hover:text-accent transition-colors">
                {s.label}
              </span>
              <span className="block text-[11px] text-mute">{s.hint}</span>
            </span>
          </label>
        ))}
      </div>

      <div>
        <label htmlFor="model" className="block text-xs text-mute mb-1.5">
          分析模型(Ollama)
        </label>
        <input
          id="model"
          type="text"
          value={config.model}
          disabled={disabled}
          onChange={(e) => onChange({ model: e.target.value })}
          className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-accent"
        />
      </div>

      <div>
        <label htmlFor="prompt" className="block text-xs text-mute mb-1.5">
          自定义提示词(可选)
        </label>
        <textarea
          id="prompt"
          rows={3}
          value={config.custom_prompt ?? ""}
          disabled={disabled}
          onChange={(e) => onChange({ custom_prompt: e.target.value || null })}
          placeholder="留空使用默认模板"
          className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white placeholder:text-mute/50 focus:outline-none focus:ring-2 focus:ring-accent resize-y"
        />
      </div>
    </Card>
  );
}
