"use client";

import { useState } from "react";
import { motion, AnimatePresence, useReducedMotion } from "motion/react";

const FAQS = [
  {
    q: "这个工具会联网吗？我的视频安全吗？",
    a: "默认完全离线。使用 Ollama 本地模型时，视频帧、音频、字幕全程不离开你的电脑。仅当你主动选择云端 API（如 DeepSeek、GPT-4o）时才会上传对应文本内容。",
  },
  {
    q: "对电脑配置有要求吗？",
    a: "基础运行：Python 3.10+、FFmpeg、8GB 内存即可使用云端 API 模式。若想本地跑大模型（Ollama），建议 16GB 内存 + 独立 GPU。YOLOv11 目标检测与 Whisper 转录在 CPU 上也能跑，GPU 会更快。",
  },
  {
    q: "支持哪些视频格式？",
    a: "通过 FFmpeg 与 OpenCV，支持 MP4、MKV、MOV、AVI、WebM 等主流容器格式。关键帧提取基于 OpenCV，音频转录基于 Whisper。",
  },
  {
    q: "不会写代码能用吗？",
    a: "能。Windows 用户双击根目录的「启动应用.bat」，脚本会自动创建虚拟环境、安装依赖、启动图形界面，无需任何命令行操作。Mac/Linux 用户运行 python launcher.py 即可。",
  },
  {
    q: "Agent 面板能做什么？",
    a: "基于 ReAct（思考→行动→观察）循环，可用自然语言指挥 AI：视觉搜索「哪里出现了猫」、截图第 10 秒画面、分析拍摄手法问题、跨视频语义检索等。",
  },
];

function Item({
  q,
  a,
}: {
  q: string;
  a: string;
}) {
  const [open, setOpen] = useState(false);
  const prefersReduced = useReducedMotion();
  return (
    <div className="glass glass-edge rounded-2xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full px-6 py-5 flex items-center justify-between gap-4 text-left"
        aria-expanded={open}
      >
        <span className="text-white font-medium text-base sm:text-lg">{q}</span>
        <span
          className={`glass-chip rounded-full w-8 h-8 flex items-center justify-center text-white transition-transform duration-300 ${
            open ? "rotate-45" : ""
          }`}
        >
          +
        </span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={prefersReduced ? { height: "auto" } : { height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={prefersReduced ? { height: "auto" } : { height: 0, opacity: 0 }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <p className="px-6 pb-5 text-[oklch(0.72_0.02_260)] leading-relaxed">
              {a}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function FAQ() {
  return (
    <section id="faq" className="relative py-24 sm:py-32">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-14">
          <span className="glass-chip rounded-full px-4 py-1.5 text-xs tracking-[0.2em] uppercase text-[oklch(0.8_0.12_205)]">
            FAQ
          </span>
          <h2 className="mt-5 text-4xl sm:text-5xl font-black tracking-tight text-white">
            常见问题
          </h2>
        </div>
        <div className="space-y-3">
          {FAQS.map((f) => (
            <Item key={f.q} {...f} />
          ))}
        </div>
      </div>
    </section>
  );
}
