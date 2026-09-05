"use client";

import { useEffect, useState } from "react";
import { apiUrl } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * 底部状态条:能力矩阵 + 磁盘余量 + 资源占用(轮询 /api/health)。
 * 取代 PyQt6 的 StatusConsole。
 */
export function StatusBar() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const res = await fetch(apiUrl("/api/health"), { cache: "no-store" });
        if (!res.ok) throw new Error(String(res.status));
        const data = (await res.json()) as HealthResponse;
        if (alive) {
          setHealth(data);
          setOnline(true);
        }
      } catch {
        if (alive) setOnline(false);
      }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const caps = health?.capabilities;

  return (
    <footer className="shrink-0 border-t border-white/5 px-4 py-2 text-xs text-mute flex items-center gap-4 flex-wrap">
      <span className="flex items-center gap-1.5">
        <span
          className={cn(
            "w-2 h-2 rounded-full",
            online === null
              ? "bg-mute"
              : online
                ? "bg-ok animate-[pulse-glow_2s_ease-in-out_infinite]"
                : "bg-danger",
          )}
        />
        {online === null ? "连接中…" : online ? "后端在线" : "后端离线"}
      </span>

      {caps && (
        <div className="flex items-center gap-3">
          <Cap label="GPU" on={caps.nvidia_gpu} />
          <Cap label="FFmpeg" on={caps.ffmpeg} />
          <Cap label="CLIP" on={caps.clip_semantic} />
          <Cap label="高级媒体" on={caps.advanced_media} />
          <Cap label="OCR" on={caps.ocr} />
        </div>
      )}

      {health && (
        <span className="ml-auto">磁盘余量 {health.disk_free_gb} GB</span>
      )}
    </footer>
  );
}

function Cap({ label, on }: { label: string; on: boolean }) {
  return (
    <span className={cn("flex items-center gap-1", on ? "text-mist" : "text-mute/50")}>
      <span className={cn("w-1.5 h-1.5 rounded-full", on ? "bg-accent" : "bg-mute/40")} />
      {label}
    </span>
  );
}
