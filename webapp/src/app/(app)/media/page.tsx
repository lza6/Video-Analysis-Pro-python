"use client";

import { useEffect, useState } from "react";
import { apiGet, apiPostJson } from "@/lib/api";
import type { MediaResponse } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

export default function MediaPage() {
  const [jobs, setJobs] = useState<{ job_id: string; video_name: string }[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [media, setMedia] = useState<MediaResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<{ job_id: string; video_name: string }[]>("/api/jobs").then((j) => {
      setJobs(j);
      if (j[0]) setSelected(j[0].job_id);
    });
  }, []);

  const generate = async () => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      const r = await apiPostJson<MediaResponse>(`/api/jobs/${selected}/media`, {
        make_video: true,
        make_gif: false,
        num_clips: 5,
      });
      setMedia(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">摘要媒体</h1>
        <p className="text-sm text-mute mt-1.5">
          AI 挑选精彩片段,自动拼接高光集锦短视频与 GIF 动图。
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
                  {j.video_name}
                </option>
              ))}
            </select>
          </div>
          <Button onClick={generate} disabled={!selected || busy}>
            {busy ? "生成中…(最长 2 分钟)" : "生成集锦"}
          </Button>
        </div>
        {error && <p className="text-sm text-danger">{error}</p>}
      </Card>

      {media && (
        <div className="space-y-4">
          {media.summary_video && (
            <Card className="p-5">
              <h3 className="text-sm font-medium text-white mb-3">集锦短视频</h3>
              <video
                src={media.summary_video}
                controls
                className="w-full rounded-card-sm"
              />
            </Card>
          )}
          {media.clips.length > 0 && (
            <Card className="p-5">
              <h3 className="text-sm font-medium text-white mb-3">单片段</h3>
              <div className="grid sm:grid-cols-2 gap-3">
                {media.clips.map((c) => (
                  <video key={c} src={c} controls className="w-full rounded-card-sm" />
                ))}
              </div>
              <p className="text-[11px] text-mute mt-2">{media.clips.length} 个片段</p>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
