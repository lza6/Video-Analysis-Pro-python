"use client";

import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

interface PlaceholderProps {
  title: string;
  desc: string;
  /** 该模块将提供的能力清单 */
  capabilities: string[];
  /** 对应原 PyQt6 文件,便于追溯 */
  origin: string;
}

/**
 * 未落地模块的占位页 — 诚实标注"待实现",不伪装成功能可用。
 * 每个占位页说明:该模块将做什么、对应原桌面端哪个文件。
 */
export function Placeholder({ title, desc, capabilities, origin }: PlaceholderProps) {
  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-12 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">{title}</h1>
        <p className="text-sm text-mute mt-2">{desc}</p>
      </header>

      <Card className="p-6 space-y-4">
        <span className="inline-block glass-chip rounded-full px-3 py-1 text-xs tracking-[0.2em] uppercase text-warn">
          待实现
        </span>
        <p className="text-sm text-mute leading-relaxed">
          此模块在当前交付轮次尚未接入后端能力。下方列出它将提供的功能,对应原桌面端{" "}
          <code className="font-mono text-[12px] text-accent bg-white/5 px-1 rounded">
            {origin}
          </code>
          。
        </p>
        <ul className="space-y-2">
          {capabilities.map((c) => (
            <li key={c} className="flex items-start gap-2.5 text-sm text-mist">
              <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
              {c}
            </li>
          ))}
        </ul>
      </Card>

      <Button href="/" variant="glass">
        返回 AI 分析
      </Button>
    </div>
  );
}
