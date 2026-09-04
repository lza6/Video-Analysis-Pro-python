"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { useReducedMotion } from "motion/react";
import ErrorBoundary from "./ErrorBoundary";

/* 静态降级层 —— Canvas 2D 渲染失败时兜底 */
function StaticAurora(): ReactNode {
  return (
    <div className="relative w-full h-full" aria-hidden="true">
      <div className="aurora w-72 h-72 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-[oklch(0.68_0.19_295/0.5)]" />
      <div className="aurora w-56 h-56 left-1/3 top-2/3 bg-[oklch(0.8_0.12_205/0.4)]" />
      <div className="glass-strong absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-52 h-52 rounded-[2.5rem]" />
    </div>
  );
}

/* oklch 调色板 —— Canvas fillStyle 现代浏览器原生支持 oklch 字符串 */
const PALETTE = {
  ink: "oklch(0.16 0.028 265)",
  ink2: "oklch(0.21 0.035 268)",
  ink3: "oklch(0.27 0.04 270)",
  scanline: "oklch(0.4 0.02 265)",
  accent: "oklch(0.8 0.12 205)",
  accent2: "oklch(0.68 0.19 295)",
  accent3: "oklch(0.75 0.15 340)",
  text: "oklch(0.88 0.015 260)",
  textDim: "oklch(0.6 0.02 260)",
} as const;

const TINTS = [PALETTE.accent, PALETTE.accent2, PALETTE.accent3] as const;

const PAD = 20;
const STRIP_TOP = 20;
const FRAME_COUNT = 4;
const SCROLL_SPEED = 16; // px/s，胶片上滚速度
const BOX_SPEED = 0.8; // rad/s，检测框横向正弦角速度
const HIT_CYCLE = 3; // s，命中事件周期
const HIT_DURATION = 0.6; // s，命中高亮持续时长
const TIMELINE_HEIGHT = 30; // 底部时间轴预留高度

interface FrameThumb {
  y: number;
  h: number;
  tintIndex: number;
}

function formatTimecode(seconds: number): string {
  const s = Math.floor(seconds) % 60;
  return `00:${s.toString().padStart(2, "0")}`;
}

/* 圆角矩形路径（不 fill/stroke，交调用方）—— 兼容性优于 ctx.roundRect */
function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const radius = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

/* 绘制角标（L 形）—— 强化检测框四角的仪器感 */
function drawCornerMarks(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  corner: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x, y + corner);
  ctx.lineTo(x, y);
  ctx.lineTo(x + corner, y);
  ctx.moveTo(x + w - corner, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + corner);
  ctx.moveTo(x + w, y + h - corner);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w - corner, y + h);
  ctx.moveTo(x + corner, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + h - corner);
  ctx.stroke();
}

export default function HeroVisual() {
  const prefersReduced = useReducedMotion();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctxNullable = canvas.getContext("2d");
    if (!ctxNullable) return;
    const ctx: CanvasRenderingContext2D = ctxNullable;

    const reduced = prefersReduced === true;

    // 布局变量（resize 时重算）
    let width = 1;
    let height = 1;
    let dpr = 1;
    let stripW = 1;
    let frameH = 100;
    let gap = 12;
    let step = frameH + gap;
    let totalStep = step * FRAME_COUNT;
    let boxW = 140;
    let boxH = 90;
    let minX = PAD;
    let maxX = 0;

    const frames: FrameThumb[] = Array.from({ length: FRAME_COUNT }, (_, i) => ({
      y: 0,
      h: frameH,
      tintIndex: i % TINTS.length,
    }));

    function recompute(): void {
      stripW = Math.max(1, width - PAD * 2);
      frameH = Math.max(80, height * 0.22);
      gap = 12;
      step = frameH + gap;
      totalStep = step * FRAME_COUNT;
      boxW = Math.max(120, stripW * 0.42);
      boxH = Math.max(80, frameH * 0.92);
      minX = PAD;
      maxX = Math.max(minX, width - PAD - boxW);
      for (let i = 0; i < frames.length; i++) {
        frames[i].h = frameH;
        frames[i].tintIndex = i % TINTS.length;
      }
    }

    function drawFrame(tSec: number, hitActive: boolean): void {
      ctx.clearRect(0, 0, width, height);

      const stripBottom = height - TIMELINE_HEIGHT;

      // ── 1. 胶片帧列（裁剪到 strip 区域，超出不绘制）
      ctx.save();
      ctx.beginPath();
      ctx.rect(PAD, STRIP_TOP, stripW, stripBottom - STRIP_TOP);
      ctx.clip();

      const offset = (tSec * SCROLL_SPEED) % totalStep;
      for (let i = 0; i < frames.length; i++) {
        let y = STRIP_TOP + i * step - offset;
        if (y + frameH < STRIP_TOP) y += totalStep;
        frames[i].y = y;
        const f = frames[i];
        const fx = PAD;

        // 帧底
        ctx.fillStyle = PALETTE.ink2;
        roundRect(ctx, fx, y, stripW, f.h, 10);
        ctx.fill();

        // 帧内画面底
        const innerPad = 6;
        ctx.fillStyle = PALETTE.ink3;
        roundRect(ctx, fx + innerPad, y + innerPad, stripW - innerPad * 2, f.h - innerPad * 2, 6);
        ctx.fill();

        // 模拟场景色块（随帧 tint 偏移 accent 色系）
        ctx.fillStyle = TINTS[f.tintIndex];
        ctx.globalAlpha = 0.18;
        const blobW = stripW * 0.35;
        const blobH = f.h * 0.4;
        roundRect(ctx, fx + innerPad + 10, y + innerPad + 8, blobW, blobH, 4);
        ctx.fill();
        ctx.globalAlpha = 1;

        // 扫描线（模拟视频行）
        ctx.strokeStyle = PALETTE.scanline;
        ctx.lineWidth = 1;
        for (let k = 0; k < 3; k++) {
          const ly = y + innerPad + 6 + k * 8;
          ctx.beginPath();
          ctx.moveTo(fx + innerPad + 4, ly);
          ctx.lineTo(fx + stripW - innerPad - 4, ly);
          ctx.stroke();
        }

        // 帧时间码小条
        ctx.fillStyle = PALETTE.textDim;
        ctx.fillRect(fx + 14, y + f.h - 14, stripW * 0.28, 2);

        // 胶片孔（左右两侧）
        ctx.fillStyle = PALETTE.ink;
        for (let k = 0; k < 5; k++) {
          const dotY = y + 8 + (k * (f.h - 16)) / 4;
          ctx.beginPath();
          ctx.arc(fx + 4, dotY, 1.8, 0, Math.PI * 2);
          ctx.arc(fx + stripW - 4, dotY, 1.8, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();

      // ── 2. 检测框（x 正弦扫动，y 居中于 strip）
      const range = maxX - minX;
      const boxX = minX + range * (0.5 + 0.5 * Math.sin(tSec * BOX_SPEED));
      const boxY = STRIP_TOP + (stripBottom - STRIP_TOP - boxH) * 0.5;
      const boxColor = hitActive ? PALETTE.accent2 : PALETTE.accent;

      // 命中光晕（短暂）
      if (hitActive) {
        ctx.save();
        ctx.shadowColor = PALETTE.accent2;
        ctx.shadowBlur = 24;
        ctx.strokeStyle = PALETTE.accent2;
        ctx.lineWidth = 1.5;
        roundRect(ctx, boxX, boxY, boxW, boxH, 6);
        ctx.stroke();
        ctx.restore();
      }

      // 框体 + 四角强化
      ctx.save();
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 2;
      roundRect(ctx, boxX, boxY, boxW, boxH, 6);
      ctx.stroke();
      ctx.lineWidth = 3;
      drawCornerMarks(ctx, boxX, boxY, boxW, boxH, 14);
      ctx.restore();

      // ── 3. 时间码标签（框右上角）
      const tc = formatTimecode(tSec);
      ctx.save();
      ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
      const tcPad = 6;
      const tcW = ctx.measureText(tc).width;
      const tcBoxW = tcW + tcPad * 2;
      const tcBoxH = 16;
      const tcX = boxX + boxW - tcBoxW - 4;
      const tcY = boxY - tcBoxH - 2;
      ctx.fillStyle = boxColor;
      roundRect(ctx, tcX, tcY, tcBoxW, tcBoxH, 4);
      ctx.fill();
      ctx.fillStyle = PALETTE.ink;
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      ctx.fillText(tc, tcX + tcPad, tcY + tcBoxH / 2 + 0.5);
      ctx.restore();

      // ── 4. 底部时间轴 + 移动指针
      const tlY = height - 14;
      const tlX1 = PAD;
      const tlX2 = width - PAD;
      ctx.strokeStyle = PALETTE.ink3;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(tlX1, tlY);
      ctx.lineTo(tlX2, tlY);
      ctx.stroke();

      ctx.strokeStyle = PALETTE.textDim;
      const tickCount = 12;
      for (let i = 0; i <= tickCount; i++) {
        const x = tlX1 + (i / tickCount) * (tlX2 - tlX1);
        const h = i % 3 === 0 ? 6 : 3;
        ctx.beginPath();
        ctx.moveTo(x, tlY - h);
        ctx.lineTo(x, tlY + h);
        ctx.stroke();
      }

      const cycle = 60;
      const prog = (tSec % cycle) / cycle;
      const px = tlX1 + prog * (tlX2 - tlX1);
      ctx.fillStyle = boxColor;
      ctx.beginPath();
      ctx.moveTo(px, tlY - 7);
      ctx.lineTo(px - 4, tlY + 4);
      ctx.lineTo(px + 4, tlY + 4);
      ctx.closePath();
      ctx.fill();
    }

    function resize(): void {
      const parent = ctx.canvas.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      ctx.canvas.width = Math.floor(width * dpr);
      ctx.canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      recompute();
      if (reduced) drawFrame(0, false);
    }

    resize();
    const ro = new ResizeObserver(resize);
    const parentEl = ctx.canvas.parentElement;
    if (parentEl) ro.observe(parentEl);

    // reduced-motion：只画一帧静态（框停在中间），不启动 rAF
    if (reduced) {
      drawFrame(0, false);
      return () => {
        ro.disconnect();
      };
    }

    let rafId = 0;
    let startMs = 0;
    const tick = (now: number): void => {
      if (!startMs) startMs = now;
      const tSec = (now - startMs) / 1000;
      const hitActive = tSec % HIT_CYCLE > HIT_CYCLE - HIT_DURATION;
      try {
        drawFrame(tSec, hitActive);
      } catch {
        cancelAnimationFrame(rafId);
        return;
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
    };
  }, [prefersReduced]);

  return (
    <ErrorBoundary fallback={<StaticAurora />}>
      <div className="relative w-full h-full" aria-hidden="true">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
      </div>
    </ErrorBoundary>
  );
}
