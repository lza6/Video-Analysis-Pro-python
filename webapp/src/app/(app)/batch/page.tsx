"use client";

import { useEffect, useState } from "react";
import { apiGet, apiPostJson } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

interface BatchConfig {
  video_dir: string;
  key_item_image: string;
  item_description: string;
  segment_sec: number;
  resume: boolean;
  model: string;
  confidence_threshold: number;
}

interface RunItem {
  run_id: string;
  video_name: string;
  status: string;
  hits_count: number;
  segments_total: number;
  segments_ok: number;
}

export default function BatchPage() {
  const [cfg, setCfg] = useState<BatchConfig>({
    video_dir: "",
    key_item_image: "",
    item_description: "关键物品",
    segment_sec: 120,
    resume: true,
    model: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    confidence_threshold: 0.7,
  });
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [runs, setRuns] = useState<RunItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refreshRuns = () => {
    apiGet<{ runs: RunItem[]; available: boolean }>("/api/runs?limit=100").then((r) => setRuns(r.runs || []));
  };
  useEffect(() => { refreshRuns(); }, []);

  const start = async () => {
    if (!cfg.video_dir) { setError("请填写视频目录"); return; }
    setRunning(true);
    setError(null);
    setProgress(null);
    try {
      await apiPostJson("/batch/run", cfg);
    } catch (e) {
      setError(String(e));
      setRunning(false);
    }
  };

  // SSE 进度流
  useEffect(() => {
    if (!running) return;
    const es = new EventSource("/api/batch/stream");
    es.addEventListener("batch_progress", (e) => {
      const d = JSON.parse(e.data);
      setProgress({ done: d.done, total: d.total });
    });
    es.addEventListener("batch_finished", () => {
      es.close();
      setRunning(false);
      refreshRuns();
    });
    es.addEventListener("error", () => {
      if (es.readyState === EventSource.CLOSED) return;
      es.close();
      setRunning(false);
    });
    return () => es.close();
  }, [running]);

  const cancel = () => apiPostJson("/batch/cancel", {});

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">批量处理</h1>
        <p className="text-sm text-mute mt-1.5">整目录视频批量跑 NVIDIA 视频模型分析,断点续跑。</p>
      </header>

      <Card className="p-5 space-y-4">
        <div>
          <label className="block text-xs text-mute mb-1.5">视频目录(本机绝对路径)</label>
          <input
            type="text"
            value={cfg.video_dir}
            onChange={(e) => setCfg({ ...cfg, video_dir: e.target.value })}
            placeholder="D:\监控视频\ 或 /home/user/videos"
            className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white font-mono"
          />
        </div>
        <div className="grid sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-mute mb-1.5">关键物品图(可选)</label>
            <input
              type="text"
              value={cfg.key_item_image}
              onChange={(e) => setCfg({ ...cfg, key_item_image: e.target.value })}
              className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-mute mb-1.5">物品描述</label>
            <input
              type="text"
              value={cfg.item_description}
              onChange={(e) => setCfg({ ...cfg, item_description: e.target.value })}
              className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
            />
          </div>
        </div>
        <div className="grid sm:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-mute mb-1.5">分片时长(秒)</label>
            <input
              type="number"
              value={cfg.segment_sec}
              onChange={(e) => setCfg({ ...cfg, segment_sec: Number(e.target.value) })}
              className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-mute mb-1.5">置信度阈值</label>
            <input
              type="number"
              step="0.05"
              value={cfg.confidence_threshold}
              onChange={(e) => setCfg({ ...cfg, confidence_threshold: Number(e.target.value) })}
              className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
            />
          </div>
          <div className="flex items-end">
            <label className="flex items-center gap-2 text-sm text-mist">
              <input
                type="checkbox"
                checked={cfg.resume}
                onChange={(e) => setCfg({ ...cfg, resume: e.target.checked })}
                className="w-4 h-4 accent-[oklch(0.68_0.19_295)]"
              />
              断点续跑
            </label>
          </div>
        </div>

        {error && <p className="text-sm text-danger">{error}</p>}

        <div className="flex gap-2">
          <Button onClick={start} disabled={running}>开始批量</Button>
          {running && <Button variant="danger" onClick={cancel}>取消</Button>}
        </div>

        {progress && (
          <div className="space-y-1">
            <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-accent-2 to-accent"
                style={{ width: `${progress.total ? (progress.done / progress.total) * 100 : 0}%` }}
              />
            </div>
            <p className="text-[11px] text-mute">{progress.done} / {progress.total}</p>
          </div>
        )}
      </Card>

      <Card className="p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-white">历史运行</h3>
          <button onClick={refreshRuns} className="text-xs text-mute hover:text-white">刷新</button>
        </div>
        <div className="space-y-1 max-h-[40vh] overflow-y-auto">
          {runs.length === 0 ? (
            <p className="text-mute/60 text-sm">暂无运行记录</p>
          ) : (
            runs.map((r) => (
              <div key={r.run_id} className="flex items-center justify-between glass-chip rounded-card-sm px-3 py-2 text-xs">
                <span className="text-white font-mono truncate">{r.video_name}</span>
                <span className={
                  r.status === "done" ? "text-ok" :
                  r.status === "failed" ? "text-danger" : "text-warn"
                }>
                  {r.status} · {r.hits_count} hits · {r.segments_ok}/{r.segments_total}
                </span>
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  );
}
