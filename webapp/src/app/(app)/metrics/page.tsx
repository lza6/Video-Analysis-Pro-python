"use client";

import { useEffect, useState } from "react";
import { apiGet, apiPostJson } from "@/lib/api";
import type { MetricsResponse } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { MetricTile } from "@/components/charts";
import { BarStrip } from "@/components/charts";

export default function MetricsPage() {
  const [jobs, setJobs] = useState<{ job_id: string; video_name: string; status: string }[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<{ job_id: string; video_name: string; status: string }[]>("/api/jobs").then((j) => {
      setJobs(j);
      if (j[0]) setSelected(j[0].job_id);
    });
  }, []);

  const generate = async () => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      const r = await apiPostJson<MetricsResponse>(`/api/jobs/${selected}/metrics`, {});
      setData(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const avgEntries = data?.avg ? Object.entries(data.avg) : [];

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">元数据与画质</h1>
        <p className="text-sm text-mute mt-1.5">
          视频技术参数与画质指标趋势:亮度、饱和度、清晰度随时间变化。
        </p>
      </header>

      <Card className="p-5 space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs text-mute mb-1.5">选择作业</label>
            <select
              value={selected || ""}
              onChange={(e) => setSelected(e.target.value)}
              className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
            >
              {jobs.length === 0 && <option value="">暂无作业</option>}
              {jobs.map((j) => (
                <option key={j.job_id} value={j.job_id}>
                  {j.video_name} · {j.status}
                </option>
              ))}
            </select>
          </div>
          <Button onClick={generate} disabled={!selected || busy}>
            {busy ? "生成中…" : "生成指标"}
          </Button>
        </div>

        {error && <p className="text-sm text-danger">{error}</p>}
      </Card>

      {data && (
        <>
          {avgEntries.length > 0 && (
            <div className="grid sm:grid-cols-3 gap-4">
              {avgEntries.map(([k, v]) => {
                const series =
                  data.series && data.series.timestamps.length > 0
                    ? k === "brightness"
                      ? data.series.brightness
                      : k === "saturation"
                        ? data.series.saturation
                        : k === "sharpness"
                          ? data.series.sharpness
                          : undefined
                    : undefined;
                return (
                  <MetricTile
                    key={k}
                    label={k}
                    value={typeof v === "number" ? v.toFixed(1) : String(v)}
                    hint="平均"
                    tone={typeof v === "number" && v >= 0.6 ? "ok" : "mute"}
                    trend={series}
                  />
                );
              })}
            </div>
          )}

          {data.chart_url && (
            <Card className="p-5">
              <h3 className="text-sm font-medium text-white mb-3">画质趋势图</h3>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={data.chart_url} alt="画质趋势" className="w-full rounded-card-sm" />
            </Card>
          )}

          {data.series && data.series.timestamps.length > 0 && (
            <Card className="p-5">
              <h3 className="text-sm font-medium text-white mb-3">时间序列</h3>
              <div className="text-xs text-mute font-mono mb-4">
                {data.series.timestamps.length} 个采样点
              </div>
              <BarStrip
                items={[
                  {
                    key: "brightness",
                    label: "亮度",
                    value: data.series.brightness[data.series.brightness.length - 1] ?? 0,
                    max: 1,
                    tone: "accent",
                    hint: "末采样点",
                  },
                  {
                    key: "saturation",
                    label: "饱和度",
                    value: data.series.saturation[data.series.saturation.length - 1] ?? 0,
                    max: 1,
                    tone: "dim",
                    hint: "末采样点",
                  },
                  {
                    key: "sharpness",
                    label: "清晰度",
                    value: data.series.sharpness[data.series.sharpness.length - 1] ?? 0,
                    max: 1,
                    tone: "ok",
                    hint: "末采样点",
                  },
                ]}
              />
            </Card>
          )}
        </>
      )}
    </div>
  );
}
