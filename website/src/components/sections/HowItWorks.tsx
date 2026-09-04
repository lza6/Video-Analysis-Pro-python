"use client";

import { motion, useReducedMotion, type Variants } from "motion/react";
import { Container } from "@/components/ui/Container";
import { SectionHeading } from "@/components/SectionHeading";

const STEPS = [
  {
    num: "01",
    title: "数据提取",
    subtitle: "Phase 1",
    desc: "拖入视频 → OpenCV 抽帧、Whisper 听写音频、YOLO 识别物体。设置提取密度，想要细致就调高。",
    tags: ["OpenCV", "Whisper", "YOLOv11"],
  },
  {
    num: "02",
    title: "AI 分析",
    subtitle: "Phase 2",
    desc: "选择本地或云端模型与提示词模板，AI 阅读视频帧和字幕，输出内容总结、技术分析或情感识别报告。",
    tags: ["Ollama", "DeepSeek-V3", "Prompt 模板"],
  },
  {
    num: "03",
    title: "媒体生成",
    subtitle: "Phase 3",
    desc: "一键生成高光集锦短视频与 GIF 动图，直接分享到社交媒体或插入演示文稿。",
    tags: ["集锦剪辑", "GIF 导出"],
  },
];

export default function HowItWorks() {
  const prefersReduced = useReducedMotion();
  const line: Variants = {
    hidden: prefersReduced ? { scaleY: 1 } : { scaleY: 0 },
    show: {
      scaleY: 1,
      transition: { duration: 1.2, ease: [0.16, 1, 0.3, 1] },
    },
  };
  const card: Variants = {
    hidden: prefersReduced ? {} : { opacity: 0, y: 30 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] },
    },
  };

  return (
    <section id="how" className="relative py-24 sm:py-32">
      <Container size="wide">
        <SectionHeading
          eyebrow="How it works"
          title="三阶段，从视频到洞察"
          subtitle="逻辑清晰的工作流：提取 → 分析 → 生成。"
        />

        <div className="relative">
          {/* 中间连接线（桌面） */}
          <motion.div
            variants={line}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, amount: 0.4 }}
            className="hidden lg:block absolute left-1/2 top-12 bottom-12 w-px -translate-x-1/2 origin-top bg-gradient-to-b from-accent via-accent-2 to-accent-3 opacity-40"
          />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {STEPS.map((s, i) => (
              <motion.div
                key={s.num}
                variants={card}
                initial="hidden"
                whileInView="show"
                viewport={{ once: true, amount: 0.4 }}
                transition={{ delay: i * 0.15 }}
                className={`relative glass glass-edge glass-hover rounded-3xl p-8 ${
                  i === 1 ? "lg:translate-y-8 glass-strong" : ""
                }`}
              >
                <div className="flex items-center justify-between mb-6">
                  <span className="text-5xl font-black text-gradient">
                    {s.num}
                  </span>
                  <span className="text-xs tracking-[0.2em] uppercase text-mute">
                    {s.subtitle}
                  </span>
                </div>
                <h3 className="text-2xl font-bold text-white mb-3">{s.title}</h3>
                <p className="text-mute leading-relaxed mb-5">
                  {s.desc}
                </p>
                <div className="flex flex-wrap gap-2">
                  {s.tags.map((t) => (
                    <span
                      key={t}
                      className="glass-chip rounded-full px-3 py-1 text-xs text-mist"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </Container>
    </section>
  );
}
