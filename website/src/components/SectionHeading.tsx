"use client";

import { motion, useReducedMotion, type Variants } from "motion/react";

/* 章节标题统一组件 — 入场编排 + reduced-motion 降级 */
export function SectionHeading({
  eyebrow,
  title,
  subtitle,
  align = "center",
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  align?: "center" | "left";
}) {
  const prefersReduced = useReducedMotion();

  const container: Variants = {
    hidden: {},
    show: {
      transition: { staggerChildren: 0.12, delayChildren: 0.05 },
    },
  };
  const item: Variants = {
    hidden: prefersReduced ? {} : { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] } },
  };

  return (
    <motion.div
      variants={container}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, amount: 0.3 }}
      className={`flex flex-col gap-4 mb-14 ${
        align === "center" ? "items-center text-center" : "items-start text-left"
      }`}
    >
      {eyebrow && (
        <motion.span
          variants={item}
          className="glass-chip rounded-full px-4 py-1.5 text-xs tracking-[0.2em] uppercase text-[oklch(0.8_0.12_205)]"
        >
          {eyebrow}
        </motion.span>
      )}
      <motion.h2
        variants={item}
        className="text-4xl sm:text-5xl font-black tracking-tight text-white max-w-3xl"
      >
        {title}
      </motion.h2>
      {subtitle && (
        <motion.p
          variants={item}
          className="text-lg text-[oklch(0.72_0.02_260)] max-w-2xl leading-relaxed"
        >
          {subtitle}
        </motion.p>
      )}
    </motion.div>
  );
}
