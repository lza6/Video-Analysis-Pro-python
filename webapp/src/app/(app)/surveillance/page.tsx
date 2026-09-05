"use client";

import { useEffect, useState } from "react";
import { apiPostJson } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";

interface HitEvent {
  timestamp: number;
  kind: string;
  detail: string;
  confidence: number;
  frame_url: string;
}

export default function SurveillancePage() {
  const [rtspUrl, setRtspUrl] = useState("");
  const [keyImage, setKeyImage] = useState("");
  const [description, setDescription] = useState("关键物品");
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<HitEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [es, setEs] = useState<EventSource | null>(null);

  const start = async () => {
    if (!rtspUrl) { setError("请填 RTSP URL"); return; }
    setError(null);
    try {
      await apiPostJson("/surveillance/start", {
        rtsp_url: rtspUrl,
        key_item_image: keyImage,
        item_description: description,
        fps: 1.0,
      });
      setRunning(true);
      // SSE 命中事件流
      const s = new EventSource("/api/surveillance/stream");
      s.addEventListener("hit", (e) => {
        const d = JSON.parse(e.data);
        setEvents((ev) => [...ev.slice(-99), d]);
      });
      s.addEventListener("error", () => {
        if (s.readyState === EventSource.CLOSED) return;
      });
      setEs(s);
    } catch (e) {
      setError(String(e));
    }
  };

  const stop = async () => {
    es?.close();
    setEs(null);
    await apiPostJson("/surveillance/stop", {});
    setRunning(false);
  };

  useEffect(() => () => { es?.close(); }, [es]);

  const hits = events.filter((e) => e.kind === "hit");

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">监控分析</h1>
        <p className="text-sm text-mute mt-1.5">RTSP 实时流,运动检测 + VLM 判断,命中即告警。</p>
      </header>

      <Card className="p-5 space-y-4">
        <div>
          <label className="block text-xs text-mute mb-1.5">RTSP URL</label>
          <input
            type="text"
            value={rtspUrl}
            onChange={(e) => setRtspUrl(e.target.value)}
            placeholder="rtsp://user:pass@192.168.1.100:554/stream"
            className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white font-mono"
          />
        </div>
        <div className="grid sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-mute mb-1.5">关键物品图(本机路径,可选)</label>
            <input
              type="text"
              value={keyImage}
              onChange={(e) => setKeyImage(e.target.value)}
              className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-mute mb-1.5">物品描述</label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
            />
          </div>
        </div>
        {error && <p className="text-sm text-danger">{error}</p>}
        <div className="flex gap-2">
          <Button onClick={start} disabled={running}>开始监控</Button>
          {running && <Button variant="danger" onClick={stop}>停止</Button>}
        </div>
        <p className="text-[11px] text-mute">
          VLM backend 走 NVIDIA 多 key 路由(.env VAP_NV_API_KEYS)。
        </p>
      </Card>

      <Card className="p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-white">命中事件</h3>
          <span className={cn("text-xs", running ? "text-ok" : "text-mute")}>
            {running ? "● 监控中" : "○ 已停止"} · {hits.length} 命中 · {events.length} 事件
          </span>
        </div>
        <div className="space-y-2 max-h-[45vh] overflow-y-auto">
          {events.length === 0 ? (
            <p className="text-mute/60 text-sm">暂无事件</p>
          ) : (
            [...events].reverse().map((e, i) => (
              <div
                key={i}
                className={cn(
                  "flex items-center gap-3 rounded-card-sm px-3 py-2 text-xs",
                  e.kind === "hit" ? "glass-strong border-accent/40" : "glass-chip",
                )}
              >
                <span className={cn("w-2 h-2 rounded-full shrink-0",
                  e.kind === "hit" ? "bg-accent" : "bg-mute")} />
                <span className="text-white font-mono">
                  {new Date(e.timestamp * 1000).toLocaleTimeString()}
                </span>
                <span className={e.kind === "hit" ? "text-accent" : "text-mute"}>{e.kind}</span>
                {e.confidence > 0 && (
                  <span className="text-mute">{(e.confidence * 100).toFixed(0)}%</span>
                )}
                <span className="text-mist truncate flex-1">{e.detail || "运动检测"}</span>
                {e.frame_url && (
                  <a href={e.frame_url} target="_blank" rel="noreferrer" className="text-accent text-[11px]">
                    查看帧
                  </a>
                )}
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  );
}
