"use client";

import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";
import type { JobDetail } from "@/lib/types";
import { Card } from "@/components/ui/Card";

export default function GalleryPage() {
  const [jobs, setJobs] = useState<{ job_id: string; video_name: string }[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<JobDetail | null>(null);

  useEffect(() => {
    apiGet<{ job_id: string; video_name: string }[]>("/api/jobs").then((j) => {
      setJobs(j);
      if (j[0]) setSelected(j[0].job_id);
    });
  }, []);

  useEffect(() => {
    if (!selected) return;
    apiGet<JobDetail>(`/api/jobs/${selected}`).then(setDetail);
  }, [selected]);

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">关键帧画廊</h1>
        <p className="text-sm text-mute mt-1.5">
          浏览已提取的关键帧,查看画质指标与视觉描述。
        </p>
      </header>

      {jobs.length > 0 && (
        <Card className="p-4">
          <select
            value={selected || ""}
            onChange={(e) => setSelected(e.target.value)}
            className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
          >
            {jobs.map((j) => (
              <option key={j.job_id} value={j.job_id}>{j.video_name}</option>
            ))}
          </select>
        </Card>
      )}

      {detail && (
        <>
          {detail.frames.length === 0 ? (
            <Card className="p-8 text-center text-mute">该作业无帧</Card>
          ) : (
            <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-4">
              {detail.frames.map((f, i) => (
                <Card key={i} className="p-3 space-y-2">
                  <div className="aspect-video rounded-card-sm overflow-hidden glass-chip">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={f.url} alt={`帧 ${f.timestamp}s`} loading="lazy" className="w-full h-full object-cover" />
                  </div>
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-white font-mono">{f.timestamp.toFixed(2)}s</span>
                  </div>
                  {Object.keys(f.metrics).length > 0 && (
                    <div className="grid grid-cols-2 gap-1 text-[10px] text-mute font-mono">
                      <span>亮 {f.metrics.brightness?.toFixed(0)}</span>
                      <span>饱 {f.metrics.saturation?.toFixed(0)}</span>
                      <span>对 {f.metrics.contrast?.toFixed(0)}</span>
                      <span>锐 {f.metrics.sharpness?.toFixed(0)}</span>
                    </div>
                  )}
                </Card>
              ))}
            </div>
          )}
          {detail.report && (
            <Card className="p-5">
              <h3 className="text-sm font-medium text-white mb-3">AI 摘要报告</h3>
              <p className="text-sm text-mute leading-relaxed whitespace-pre-wrap">{detail.report}</p>
            </Card>
          )}
          {detail.transcript && (
            <Card className="p-5">
              <h3 className="text-sm font-medium text-white mb-3">音频转录</h3>
              <p className="text-sm text-mute leading-relaxed whitespace-pre-wrap">{detail.transcript}</p>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
