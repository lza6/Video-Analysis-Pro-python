"use client";

import { useEffect, useRef, useState } from "react";
import { apiUrl } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * 底部状态条:能力矩阵 + 磁盘余量 + 资源占用(轮询 /api/health)。
 * StatusConsole 的 Web 等价物。
 *
 * 轮询策略(用户启动日志实证:空闲时 /api/health 每页 mounted 都打):
 *  - 标签页可见时 15s 一跳(原 5s 过频,空闲也持续打后端);
 *  - 标签页隐藏时暂停(visibilitychange),回来立即补一次;
 *  - 失败时指数退避(15→30→60s 封顶),避免后端宕时疯狂重连。
 */
const BASE_INTERVAL_MS = 15_000;
const MAX_INTERVAL_MS = 60_000;

export function StatusBar() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const backoffRef = useRef<number>(BASE_INTERVAL_MS);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async () => {
      // 标签页隐藏时不轮询,等 visibility 回来再补
      if (typeof document !== "undefined" && document.hidden) {
        schedule();
        return;
      }
      try {
        const res = await fetch(apiUrl("/api/health"), { cache: "no-store" });
        if (!res.ok) throw new Error(String(res.status));
        const data = (await res.json()) as HealthResponse;
        if (alive) {
          setHealth(data);
          setOnline(true);
          backoffRef.current = BASE_INTERVAL_MS; // 成功则重置退避
        }
      } catch {
        if (alive) setOnline(false);
        // 失败指数退避,封顶 MAX_INTERVAL_MS
        backoffRef.current = Math.min(backoffRef.current * 2, MAX_INTERVAL_MS);
      }
      schedule();
    };

    const schedule = () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(tick, backoffRef.current);
    };

    const onVisible = () => {
      if (!document.hidden) {
        backoffRef.current = BASE_INTERVAL_MS;
        if (timer) clearTimeout(timer);
        void tick();
      }
    };

    void tick();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
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
