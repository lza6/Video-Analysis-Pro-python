"use client";

import { useEffect, useRef } from "react";
import { Card } from "@/components/ui/Card";
import type { LogEvent } from "@/lib/types";

/** 实时日志面板(滚动到底)。 */
export function LogConsole({ logs }: { logs: LogEvent[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs.length]);

  return (
    <Card className="p-4">
      <h3 className="text-sm font-medium text-white mb-3">运行日志</h3>
      <div
        ref={ref}
        className="h-40 overflow-y-auto font-mono text-[11px] leading-relaxed space-y-0.5"
      >
        {logs.length === 0 ? (
          <p className="text-mute/60">暂无日志</p>
        ) : (
          logs.map((l, i) => (
            <p
              key={i}
              className={
                l.level === "warning"
                  ? "text-warn"
                  : l.level === "error"
                    ? "text-danger"
                    : "text-mute"
              }
            >
              <span className="text-mute/50">[{l.level}]</span> {l.msg}
            </p>
          ))
        )}
      </div>
    </Card>
  );
}
