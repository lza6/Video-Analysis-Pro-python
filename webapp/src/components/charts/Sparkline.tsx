/**
 * Sparkline — 迷你趋势线(纯 SVG,无第三方依赖)。
 *
 * 设计语言(P3-2):
 * - 电光青主线 + 数据字体刻度(数字永远是 JetBrains Mono)
 * - 末端动态点:上升 ok / 下降 danger / 平 mute,语义色
 * - 4px 半径末端(锐角 > 胶囊,反"均匀圆角"AI-slop)
 * - hover 放大 1.1 + 160ms fast 过渡,显示值浮层
 * - 动效 token:--duration-fast / --ease-out-expo
 */

"use client";

import { useId } from "react";

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  className?: string;
  /** 显示格式(默认原值,空数组时显示"—") */
  format?: (v: number) => string;
}

function defaultFormat(v: number): string {
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(1);
}

/** 归一化数据到画布坐标(全部为同一值时在垂直中线绘制平线)。 */
function normalize(
  data: number[],
  w: number,
  h: number,
  pad: number,
): { points: string; end: { x: number; y: number } } {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min;
  const innerW = w - pad * 2;
  const innerH = h - pad * 2;
  const coords = data.map((v, i) => {
    const x = pad + (data.length === 1 ? innerW : (i / (data.length - 1)) * innerW);
    const y =
      span === 0
        ? pad + innerH / 2
        : pad + innerH - ((v - min) / span) * innerH;
    return { x, y };
  });
  const pts = coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ");
  return { points: pts, end: coords[coords.length - 1] };
}

export function Sparkline({
  data,
  width = 160,
  height = 44,
  className,
  format = defaultFormat,
}: SparklineProps) {
  const gid = useId();
  const pad = 3;
  const safe = Array.isArray(data) && data.length > 0 ? data : [];

  if (safe.length === 0) {
    return (
      <div
        className={`font-mono text-[11px] leading-[44px] text-mute/60 text-center ${className ?? ""}`}
        style={{ width, height }}
      >
        —
      </div>
    );
  }

  const { points, end } = normalize(safe, width, height, pad);
  const last = safe[safe.length - 1];
  const first = safe[0];
  const trend = last > first ? "up" : last < first ? "down" : "flat";
  const dot =
    trend === "up" ? "var(--color-ok)" : trend === "down" ? "var(--color-danger)" : "var(--color-mute)";

  const label = format(last);

  return (
    <span className={`inline-block ${className ?? ""}`}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        role="img"
        aria-label={`趋势: ${label}`}
        className="block transition-transform duration-[var(--duration-fast)] ease-[var(--ease-out-expo)] hover:scale-110 group/spark"
      >
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="var(--color-accent-dim)" />
            <stop offset="100%" stopColor="var(--color-accent-strong)" />
          </linearGradient>
        </defs>
        <polyline
          points={points}
          fill="none"
          stroke={`url(#${gid})`}
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="chart-spark-shadow"
        />
        <line
          x1={end.x - 7}
          y1={end.y}
          x2={end.x}
          y2={end.y}
          stroke="var(--color-accent-strong)"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <circle
          cx={end.x}
          cy={end.y}
          r="3.5"
          fill="var(--color-ink)"
          stroke={dot}
          strokeWidth="1.5"
        />
      </svg>
      <span className="sr-only">结束值 {label}</span>
    </span>
  );
}
