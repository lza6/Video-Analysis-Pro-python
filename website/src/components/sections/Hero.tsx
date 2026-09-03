"use client";

import { motion, useReducedMotion, type Variants } from "motion/react";
import dynamic from "next/dynamic";

const HeroVisual = dynamic(() => import("../HeroVisual"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full" aria-hidden="true">
      <div className="aurora w-72 h-72 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-[oklch(0.68_0.19_295/0.5)] animate-[pulse-glow_4s_ease-in-out_infinite]" />
    </div>
  ),
});

export default function Hero() {
  const prefersReduced = useReducedMotion();
  const container: Variants = {
    hidden: {},
    show: { transition: { staggerChildren: 0.12, delayChildren: 0.1 } },
  };
  const item: Variants = {
    hidden: prefersReduced ? {} : { opacity: 0, y: 28 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.7, ease: [0.16, 1, 0.3, 1] },
    },
  };

  return (
    <section
      id="top"
      className="relative min-h-[100svh] flex items-center pt-28 pb-16 overflow-hidden"
    >
      {/* 极光氛围层 */}
      <div className="absolute inset-0 -z-10 pointer-events-none" aria-hidden="true">
        <div
          className="aurora w-[42rem] h-[42rem] -left-40 -top-40 bg-[oklch(0.68_0.19_295/0.22)]"
          style={{ animation: "drift-a 18s ease-in-out infinite" }}
        />
        <div
          className="aurora w-[38rem] h-[38rem] right-[-12rem] top-1/3 bg-[oklch(0.8_0.12_205/0.18)]"
          style={{ animation: "drift-b 22s ease-in-out infinite" }}
        />
        <div className="aurora w-[30rem] h-[30rem] left-1/3 bottom-[-10rem] bg-[oklch(0.75_0.15_340/0.14)]" />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
        <div className="grid lg:grid-cols-2 gap-10 items-center">
          <motion.div
            variants={container}
            initial="hidden"
            animate="show"
            className="flex flex-col gap-6 max-w-2xl"
          >
            <motion.span
              variants={item}
              className="glass-chip glass-edge rounded-full px-4 py-2 self-start text-xs tracking-[0.2em] uppercase text-[oklch(0.8_0.12_205)] flex items-center gap-2"
            >
              <span className="w-2 h-2 rounded-full bg-[oklch(0.8_0.12_205)] animate-[pulse-glow_2s_ease-in-out_infinite]" />
              本地运行 · 隐私至上 · 开源
            </motion.span>

            <motion.h1
              variants={item}
              className="text-5xl sm:text-6xl lg:text-7xl font-black tracking-tight text-white leading-[1.05]"
            >
              让 AI 替你
              <br />
              <span className="text-gradient">「看」</span>完整个视频
            </motion.h1>

            <motion.p
              variants={item}
              className="text-lg sm:text-xl text-[oklch(0.72_0.02_260)] leading-relaxed max-w-xl"
            >
              基于 Python 的本地化视频分析工具。融合计算机视觉、语音识别与大语言模型，把 1 小时的视频浓缩成 3 分钟的精华报告——隐私不出本机，逻辑全透明。
            </motion.p>

            <motion.div
              variants={item}
              className="flex flex-wrap items-center gap-3 mt-2"
            >
              <a
                href="#download"
                className="group relative inline-flex items-center justify-center rounded-full bg-gradient-to-r from-[oklch(0.68_0.19_295)] to-[oklch(0.8_0.12_205)] px-7 py-3.5 text-white font-medium transition-transform hover:-translate-y-0.5 active:scale-[0.98]"
              >
                <span className="relative z-10">立即下载</span>
              </a>
              <a
                href="#how"
                className="glass glass-edge glass-hover rounded-full px-7 py-3.5 text-white font-medium"
              >
                查看工作原理
              </a>
            </motion.div>

            <motion.div
              variants={item}
              className="flex flex-wrap items-center gap-x-5 gap-y-2 mt-4 text-xs text-[oklch(0.72_0.02_260)]"
            >
              <span>Python 3.10+</span>
              <span className="w-1 h-1 rounded-full bg-current opacity-50" />
              <span>PyQt6 GUI</span>
              <span className="w-1 h-1 rounded-full bg-current opacity-50" />
              <span>YOLOv11 + Whisper</span>
              <span className="w-1 h-1 rounded-full bg-current opacity-50" />
              <span>Ollama / OpenAI API</span>
            </motion.div>
          </motion.div>

          <motion.div
            variants={item}
            className="relative h-[340px] sm:h-[420px] lg:h-[520px]"
          >
            <HeroVisual />
          </motion.div>
        </div>
      </div>
    </section>
  );
}
