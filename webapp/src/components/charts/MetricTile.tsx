/**
 * MetricTile — KPI 磁贴(数值 + 增量箭头 + 趋势)。纯 SVG,无第三方依赖。
 *
 * 设计语言(P3-2):
 * - 数值永远 JetBrains Mono(数据字体记忆点)
 * - 增量箭头用三角形 (SVG) 而非 emoji,语义色 ok/danger
 * - 3px 圆角条形 = 锐角 > 胶囊,反"均匀圆角"AI-slop
 * - 下缘基线刻度线(全部走 var(--color-border-glass))——有节奏而非千篇一律卡片
 * - hover:translate-y 抬升 + 电光青边框亮起 + 300ms normal 过渡
 * - 动效 token:--duration-normal / --ease-out-expo
 */

"use client";

import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface MetricTileProps {
  label: string;
  value: string;
  /** 增量百分比,如 +12.4 或 -3.1;缺省隐藏 */
  delta?: number | null;
  /** 可选迷你趋势数据(传则渲染底部 Sparkline) */
  trend?: number[];
  tone?: "ok" | "danger" | "mute";
  hint?: string;
  className?: string;
  children?: ReactNode;
}

function Arrow({ up }: { up: boolean }) {
  return (
    <svg
      width="10"
      height="10"
      viewBox="0 0 10 10"
      className={cn("shrink-0", up ? "text-ok" : "text-danger")}
      aria-hidden="true"
    >
      <polygon
        points={up ? "5,1 9,6 6.5,6 6.5,9 3.5,9 3.5,6 1,6" : "5,9 1,4 3.5,4 3.5,1 6.5,1 6.5,4 9,4"}
        fill="currentColor"
      />
    </svg>
  );
}

export function MetricTile({
  label,
  value,
  delta,
  trend,
  tone = "mute",
  hint,
  className,
  children,
}: MetricTileProps) {
  const trendEnd = Array.isArray(trend) && trend.length > 0 ? trend[trend.length - 1] : null;
  const trendStart = trendEnd !== null && Array.isArray(trend) && trend.length > 0 ? trend[0] : null;
  const flat = trendStart !== null && trendEnd !== null && trendStart === trendEnd;
  const min = trendEnd !== null && trendStart !== null ? Math.min(trendStart, trendEnd) : null;
  const max = trendEnd !== null && trendStart !== null ? Math.max(trendStart, trendEnd) : null;
  const cross = flat ? null : min !== null && max !== null ? max : null;

  return (
    <div
      className={cn(
        "glass-strong glass-edge rounded-card relative overflow-hidden",
        "transition-transform duration-[var(--duration-normal)] ease-[var(--ease-out-expo)]",
        "hover:-translate-y-1 hover:border-accent/40",
        className,
      )}
    >
      {/* 顶部电光青高光(8%,data-viz 即设计系统的记忆点) */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent/60 to-transparent" />

      <div className="p-4">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] uppercase tracking-[0.14em] text-mute font-medium">
            {label}
          </span>
          {delta !== null && delta !== undefined ? (
            <span className="flex items-center gap-1 font-mono text-[11px]">
              <Arrow up={delta >= 0} />
              <span className={delta >= 0 ? "text-ok" : "text-danger"}>
                {delta >= 0 ? "+" : ""}
                {delta.toFixed(1)}%
              </span>
            </span>
          ) : null}
        </div>

        <div
          className={cn(
            "mt-1.5 text-2xl font-black leading-none font-[var(--font-data)] tracking-tight",
            tone === "danger" ? "text-danger" : tone === "ok" ? "text-ok" : "text-white",
          )}
        >
          {value}
        </div>

        {hint ? <div className="mt-1 text-xs text-mute truncate">{hint}</div> : null}
        {children ? <div className="mt-2">{children}</div> : null}
      </div>

      {trend && trend.length > 0 ? (
        <div className="px-4 pb-3" aria-label="迷你趋势">
          <svg
            width="100%"
            height="10"
            viewBox="0 0 100 10"
            preserveAspectRatio="none"
            className="block"
          >
            {cross !== null ? (
              <line
                x1="0"
                y1="8.5"
                x2="100"
                y2="8.5"
                stroke="var(--color-border-glass)"
                strokeWidth="0.75"
                strokeDasharray="2 2"
              />
            ) : null}
            {trend.map((v, i) => {
              const maxV = Math.max(...trend);
              const minV = Math.min(...trend);
              const span = maxV - minV || 1;
              const x = trend.length === 1 ? 50 : (i / (trend.length - 1)) * 100;
              const y = 9.5 - ((v - minV) / span) * 8;
              return (
                <rect
                  key={i}
                  x={x - 0.5}
                  y={y}
                  width="1"
                  height={9.5 - y}
                  fill="var(--color-accent-dim)"
                  rx="0.5"
                />
              );
            })}
          </svg>
        </div>
      ) : null}
    </div>
  );
}
