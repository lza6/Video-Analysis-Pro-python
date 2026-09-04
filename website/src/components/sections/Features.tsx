"use client";

import { motion, useReducedMotion, type Variants } from "motion/react";
import { Container } from "@/components/ui/Container";
import { SectionHeading } from "@/components/SectionHeading";

type Feature = {
  icon: string;
  title: string;
  desc: string;
  span?: "wide";
};

const FEATURES: Feature[] = [
  {
    icon: "🧠",
    title: "多模态 AI 分析",
    desc: "画面（YOLOv11 识别物体）、声音（Whisper 转录文本）、语义（LLM 总结）三路融合，像人一样「看懂」视频。",
    span: "wide",
  },
  {
    icon: "🤖",
    title: "智能 Agent 面板",
    desc: "类 DeepSeek R1 的思维链组件，多轮对话，可直接指挥 AI 截图、剪辑、搜索画面。",
  },
  {
    icon: "🎞️",
    title: "智能关键帧",
    desc: "告别机械截图，算法自动识别画面变化最显著的关键时刻。",
  },
  {
    icon: "📊",
    title: "可视化报表",
    desc: "亮度、清晰度、饱和度趋势的专业图表，一眼看懂视频质量。",
  },
  {
    icon: "🎬",
    title: "自动集锦生成",
    desc: "AI 挑选精彩片段，自动拼接短视频或 GIF，便于分享与演示。",
  },
  {
    icon: "🔌",
    title: "灵活模型支持",
    desc: "本地 Ollama（Llama3、Qwen2.5）或云端 OpenAI 格式 API（DeepSeek、GPT-4o、Claude）。",
  },
];

export default function Features() {
  const prefersReduced = useReducedMotion();
  const container: Variants = {
    hidden: {},
    show: { transition: { staggerChildren: 0.08 } },
  };
  const item: Variants = {
    hidden: prefersReduced ? {} : { opacity: 0, y: 24 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.55, ease: [0.16, 1, 0.3, 1] },
    },
  };

  return (
    <section id="features" className="relative py-24 sm:py-32">
      <Container size="wide">
        <SectionHeading
          eyebrow="Features"
          title="一个工具，看透整段视频"
          subtitle="从数据提取到智能总结，再到媒体生成——三阶段流水线把 1 小时视频浓缩成 3 分钟精华。"
        />

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 auto-rows-[1fr]"
        >
          {FEATURES.map((f) => (
            <motion.article
              key={f.title}
              variants={item}
              className={`glass glass-edge glass-hover rounded-3xl p-7 flex flex-col gap-4 ${
                f.span === "wide" ? "lg:col-span-2" : ""
              }`}
            >
              <div className="w-12 h-12 rounded-2xl glass-chip flex items-center justify-center text-2xl">
                {f.icon}
              </div>
              <h3 className="text-xl font-bold text-white">{f.title}</h3>
              <p className="text-mute leading-relaxed">
                {f.desc}
              </p>
            </motion.article>
          ))}
        </motion.div>
      </Container>
    </section>
  );
}
