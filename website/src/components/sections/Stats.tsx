"use client";

import { motion, useReducedMotion, type Variants } from "motion/react";
import { Container } from "@/components/ui/Container";

const STATS = [
  { value: "3 阶段", label: "提取 → 分析 → 生成流水线" },
  { value: "100%", label: "本地运行，隐私不出本机" },
  { value: "9 类", label: "Agent 工具：截图 / 剪辑 / 搜索…" },
  { value: "∞", label: "无限调用，无需订阅" },
];

export default function Stats() {
  const prefersReduced = useReducedMotion();
  const item: Variants = {
    hidden: prefersReduced ? {} : { opacity: 0, scale: 0.9, y: 16 },
    show: {
      opacity: 1,
      scale: 1,
      y: 0,
      transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] },
    },
  };

  return (
    <section className="relative py-20">
      <Container size="default">
        <motion.div
          variants={{ show: { transition: { staggerChildren: 0.1 } } }}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.3 }}
          className="glass-strong glass-edge rounded-3xl p-8 sm:p-12 grid grid-cols-2 lg:grid-cols-4 gap-8"
        >
          {STATS.map((s) => (
            <motion.div key={s.label} variants={item} className="text-center">
              <div className="text-4xl sm:text-5xl font-black text-gradient mb-2">
                {s.value}
              </div>
              <div className="text-sm text-mute leading-relaxed">
                {s.label}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </Container>
    </section>
  );
}

