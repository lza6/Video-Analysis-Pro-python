"use client";

import { motion, useReducedMotion, type Variants } from "motion/react";
import { Container } from "@/components/ui/Container";

const REPO_URL = "https://github.com/lza6/Video-Analysis-Pro-python";

const PLATFORMS = [
  {
    os: "Windows",
    icon: "🪟",
    hint: "双击「启动应用.bat」",
    primary: true,
  },
  { os: "macOS", icon: "🍎", hint: "python launcher.py" },
  { os: "Linux", icon: "🐧", hint: "python launcher.py" },
  { os: "源码", icon: "⌨️", hint: "git clone + 手动安装" },
];

export default function Download() {
  const prefersReduced = useReducedMotion();
  const item: Variants = {
    hidden: prefersReduced ? {} : { opacity: 0, y: 24 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.55, ease: [0.16, 1, 0.3, 1] },
    },
  };

  return (
    <section id="download" className="relative py-24 sm:py-32">
      <Container size="default">
        <motion.div
          variants={{ show: { transition: { staggerChildren: 0.1 } } }}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
          className="glass-strong glass-edge rounded-[2.5rem] p-10 sm:p-16 text-center relative overflow-hidden"
        >
          <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-[30rem] h-[30rem] bg-[oklch(0.68_0.19_295/0.25)] rounded-full blur-[100px] pointer-events-none" />
          <motion.span
            variants={item}
            className="glass-chip rounded-full px-4 py-1.5 text-xs tracking-[0.2em] uppercase text-accent"
          >
            Download
          </motion.span>
          <motion.h2
            variants={item}
            className="mt-6 text-4xl sm:text-6xl font-black tracking-tight text-white"
          >
            让 AI 替你<span className="text-gradient">「看」</span>完世界
          </motion.h2>
          <motion.p
            variants={item}
            className="mt-5 text-lg text-mute max-w-xl mx-auto leading-relaxed"
          >
            开源、本地、可离线。选择你的平台，三步即可运行。
          </motion.p>

          <motion.div
            variants={item}
            className="mt-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
          >
            {PLATFORMS.map((p) => (
              <a
                key={p.os}
                href={p.os === "源码" ? `${REPO_URL}#readme` : REPO_URL}
                target="_blank"
                rel="noopener noreferrer"
                className={`glass glass-edge glass-hover rounded-2xl p-6 flex flex-col items-center gap-3 ${
                  p.primary ? "glass-strong border-white/25" : ""
                }`}
              >
                <span className="text-3xl">{p.icon}</span>
                <span className="text-white font-bold">{p.os}</span>
                <span className="text-xs text-mute font-mono">
                  {p.hint}
                </span>
              </a>
            ))}
          </motion.div>

          <motion.p
            variants={item}
            className="mt-8 text-xs text-mute"
          >
            需要 Python 3.10+ 与 FFmpeg · GPL v3 开源协议 · 由{" "}
            <span className="text-white">听风公司</span> 出品
          </motion.p>
        </motion.div>
      </Container>
    </section>
  );
}
