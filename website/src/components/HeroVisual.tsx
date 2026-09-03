"use client";

import { useState, useEffect, type ReactNode } from "react";
import dynamic from "next/dynamic";
import { useReducedMotion } from "motion/react";
import ErrorBoundary from "./ErrorBoundary";

/* 3D 场景按需加载：SSR 输出静态光斑 fallback，WebGL 水合后替换 */
const HeroScene = dynamic(() => import("./three/HeroScene"), {
  ssr: false,
  loading: () => <StaticAurora />,
});

/* WebGL 能力预检 —— 在渲染 Canvas 前确认浏览器真有 WebGL 上下文 */
function useWebglAvailable(): boolean {
  const [ok, setOk] = useState(true);
  useEffect(() => {
    try {
      const canvas = document.createElement("canvas");
      const gl =
        canvas.getContext("webgl2") ||
        canvas.getContext("webgl") ||
        canvas.getContext("experimental-webgl");
      setOk(!!gl);
    } catch {
      setOk(false);
    }
  }, []);
  return ok;
}

function StaticAurora(): ReactNode {
  return (
    <div className="relative w-full h-full" aria-hidden="true">
      <div className="aurora w-72 h-72 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-[oklch(0.68_0.19_295/0.5)] animate-[pulse-glow_4s_ease-in-out_infinite]" />
      <div className="aurora w-56 h-56 left-1/3 top-2/3 bg-[oklch(0.8_0.12_205/0.4)]" />
      <div className="glass-strong absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-52 h-52 rounded-[2.5rem] animate-[floaty_5s_ease-in-out_infinite]" />
    </div>
  );
}

export default function HeroVisual() {
  const prefersReduced = useReducedMotion();
  const webglOk = useWebglAvailable();

  // reduced-motion 或 WebGL 不可用 → 纯 CSS 静态氛围层
  if (prefersReduced || !webglOk) {
    return <StaticAurora />;
  }

  return (
    <ErrorBoundary fallback={<StaticAurora />}>
      <div className="relative w-full h-full">
        <HeroScene />
        <div className="absolute inset-0 -z-10">
          <div className="aurora w-[28rem] h-[28rem] left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-[oklch(0.68_0.19_295/0.28)]" />
        </div>
      </div>
    </ErrorBoundary>
  );
}
