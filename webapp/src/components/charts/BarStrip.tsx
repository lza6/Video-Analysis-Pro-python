/**
 * BarStrip — 横向条形分布(纯 div,无第三方依赖)。
 *
 * 设计语言(P3-2):
 * - track 有 depth 层次(chart-bar-track 双渐变),fill 语义色非列表紫蓝
 * - 强(accent)/ 弱(dim)/ 危险(danger)/ 好(ok) 四种语义色阶
 * - 标签左对齐 + 值右对齐 = 双端定标,避免千篇一律 card-grid
 * - hover:整行提示 + fill 提亮到 strong(150ms fast 过渡)
 * - 动效 token:--duration-fast / --ease-out-expo
 */

'use client';

import { cn } from '@/lib/utils';

interface Item {
  key: string;
  label: string;
  value: number;
  max?: number;
  tone?: 'accent' | 'dim' | 'danger' | 'ok';
  hint?: string;
}

interface BarStripProps {
  items: Item[];
  className?: string;
  /** 值格式,默认字符串原样 */
  format?: (v: number) => string;
}

function defaultFormat(v: number): string {
  return v.toLocaleString();
}

export function BarStrip({ items, className, format = defaultFormat }: BarStripProps) {
  const safe = Array.isArray(items) ? items : [];
  if (safe.length === 0) {
    return (
      <div className={cn('text-xs text-mute/60 py-6 text-center font-mono', className)}>
        —
      </div>
    );
  }

  const globalMax = Math.max(...safe.map((it) => it.value)) || 1;

  return (
    <ul
      className={cn('space-y-2.5', className)}
      role="list"
      aria-label="横向条形分布"
    >
      {safe.map((it, idx) => {
        const cap = it.max && it.max > 0 ? it.max : globalMax;
        const width = Math.max(2, Math.min(100, (it.value / cap) * 100));
        const tone =
          it.tone ?? (idx % 3 === 0 ? 'accent' : idx % 3 === 1 ? 'dim' : 'accent');
        const fillVar =
          tone === 'danger'
            ? 'var(--color-danger)'
            : tone === 'ok'
              ? 'var(--color-ok)'
              : tone === 'dim'
                ? 'var(--color-accent-dim)'
                : 'var(--color-accent)';
        const fillStrong =
          tone === 'danger'
            ? 'var(--color-danger)'
            : tone === 'ok'
              ? 'var(--color-ok)'
              : 'var(--color-accent-strong)';

        return (
          <li
            key={it.key}
            className="group relative rounded-[3px] bg-white/[0.03] px-2 py-1.5 transition-colors duration-[var(--duration-fast)] ease-[var(--ease-out-expo)] hover:bg-white/[0.06]"
            title={it.label}
          >
            <div className="relative z-10 flex items-center justify-between gap-2 text-xs">
              <span className="truncate text-mist">{it.label}</span>
              <span className="shrink-0 font-mono text-[11px] text-mute group-hover:text-accent-strong">
                {format(it.value)}
              </span>
            </div>
            <div className="chart-bar-track mt-1 h-1" aria-hidden="true">
              <div
                className="chart-bar-fill h-full"
                style={{
                  width: `${width}%`,
                  background: `linear-gradient(to right, var(--color-accent-dim), ${fillVar})`,
                }}
              />
              <div
                className="pointer-events-none absolute inset-0 h-full transition-opacity duration-[var(--duration-fast)] opacity-0 group-hover:opacity-100"
                style={{ width: `${width}%`, background: fillStrong }}
                aria-hidden="true"
              />
            </div>
            {it.hint ? (
              <div className="mt-0.5 text-[11px] text-mute/70">{it.hint}</div>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}