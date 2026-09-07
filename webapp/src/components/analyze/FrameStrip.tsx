"use client";

import { useEffect, useRef } from "react";
import { Card } from "@/components/ui/Card";
import type { FrameInfo } from "@/lib/types";
import { apiUrl } from "@/lib/api";

/** 实时帧条:分析中逐帧追加,横向滚动。CarouselWidget 实时部分的 Web 等价物。 */
export function FrameStrip({ frames }: { frames: FrameInfo[] }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [frames.length]);

  if (frames.length === 0) return null;

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-white">关键帧</h3>
        <span className="text-xs text-mute">{frames.length} 帧</span>
      </div>
      <div ref={ref} className="flex gap-3 overflow-x-auto pb-2">
        {frames.map((f, i) => (
          <figure key={`${f.url}-${i}`} className="shrink-0 w-40">
            <div className="aspect-video rounded-card-sm overflow-hidden glass-chip">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={apiUrl(f.url)}
                alt={`帧 ${f.timestamp}s`}
                loading="lazy"
                className="w-full h-full object-cover"
              />
            </div>
            <figcaption className="mt-1.5 text-[11px] text-mute font-mono">
              {f.timestamp.toFixed(2)}s
            </figcaption>
          </figure>
        ))}
      </div>
    </Card>
  );
}
