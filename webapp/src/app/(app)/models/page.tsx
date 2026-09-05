"use client";

import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";
import type { ModelsResponse, ModelCard } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card, Chip } from "@/components/ui/Card";
import { cn } from "@/lib/utils";

export default function ModelsPage() {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [progress, setProgress] = useState<Record<string, number>>({});
  const [verifyMsg, setVerifyMsg] = useState<Record<string, string>>({});

  const refresh = () => {
    apiGet<ModelsResponse>("/api/models")
      .then(setData)
      .catch((e) => setError(String(e)));
  };
  useEffect(() => { refresh(); }, []);

  const download = async (id: string) => {
    setDownloading(id);
    setProgress((p) => ({ ...p, [id]: 0 }));
    try {
      const res = await fetch(`/api/models/${id}/download`, { method: "POST" });
      if (!res.ok) throw new Error(`${res.status}`);
      const { job_id } = await res.json();
      // SSE 消费下载进度
      const es = new EventSource(`/api/models/${id}/download/stream`);
      es.addEventListener("progress", (e) => {
        const d = JSON.parse(e.data);
        setProgress((p) => ({ ...p, [id]: d.value }));
      });
      es.addEventListener("verify", (e) => {
        const d = JSON.parse(e.data);
        setVerifyMsg((m) => ({ ...m, [id]: d.msg || "校验完成" }));
      });
      es.addEventListener("done", () => {
        es.close();
        setDownloading(null);
        refresh();
      });
      es.addEventListener("error", () => {
        if (es.readyState === EventSource.CLOSED) return;
        es.close();
        setDownloading(null);
        setVerifyMsg((m) => ({ ...m, [id]: "下载失败,见日志" }));
      });
      void job_id;
    } catch (e) {
      setDownloading(null);
      setVerifyMsg((m) => ({ ...m, [id]: `启动失败: ${e}` }));
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">模型管理</h1>
        <p className="text-sm text-mute mt-1.5">
          下载与管理本地模型(YOLOv11 / Whisper / Sentence-Transformer / FFmpeg),SHA256 完整性校验防篡改。
        </p>
      </header>

      {error && (
        <Card className="p-4 border-danger/40">
          <p className="text-sm text-danger">{error}</p>
        </Card>
      )}

      <div className="grid sm:grid-cols-2 gap-4">
        {data?.cards.map((c: ModelCard) => (
          <Card key={c.id} className="p-5 space-y-3">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-white font-bold">{c.name}</h3>
                <p className="text-xs text-mute">{c.desc}</p>
              </div>
              {c.exists ? (
                <Chip className="text-ok">已就绪</Chip>
              ) : (
                <Chip className="text-warn">未下载</Chip>
              )}
            </div>
            <div className="text-[11px] text-mute font-mono">
              {c.size_hint}
              {c.exists && c.size_mb > 0 && ` · ${c.size_mb} MB`}
              {c.sha256_expected && " · SHA256"}
            </div>
            {downloading === c.id && (
              <div className="space-y-1">
                <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-accent-2 to-accent transition-all"
                    style={{ width: `${progress[c.id] || 0}%` }}
                  />
                </div>
                <p className="text-[11px] text-mute">下载中 {progress[c.id] || 0}%</p>
              </div>
            )}
            {verifyMsg[c.id] && (
              <p className="text-[11px] text-accent">{verifyMsg[c.id]}</p>
            )}
            <div className="flex gap-2">
              <Button
                size="sm"
                variant={c.exists ? "chip" : "primary"}
                onClick={() => download(c.id)}
                disabled={downloading !== null}
              >
                {c.exists ? "重新下载" : "下载"}
              </Button>
            </div>
          </Card>
        ))}
      </div>

      {data && data.local_models.length > 0 && (
        <Card className="p-5">
          <h3 className="text-sm font-medium text-white mb-3">本地模型</h3>
          <div className="grid sm:grid-cols-2 gap-2 text-xs">
            {data.local_models.map((m) => (
              <div key={m.name} className="flex items-center justify-between glass-chip rounded-card-sm px-3 py-2">
                <span className="text-mist font-mono truncate">{m.name}</span>
                <span className={cn("text-[10px]", m.type.includes("VL") ? "text-accent" : "text-mute")}>
                  {m.type}
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
