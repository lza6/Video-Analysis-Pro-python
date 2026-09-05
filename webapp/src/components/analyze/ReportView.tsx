"use client";

import { useMemo, type ReactNode } from "react";
import { Card } from "@/components/ui/Card";

/**
 * 极简 Markdown 渲染(无第三方依赖)。
 * 支持:标题 / 无序有序列表 / 粗体 / 行内代码 / 围栏代码块 / 段落。
 * 报告是 LLM 流式输出的 markdown,这里实时渲染。
 */
export function ReportView({ report }: { report: string }) {
  const blocks = useMemo(() => parseMarkdown(report), [report]);

  return (
    <Card className="p-6">
      <h3 className="text-sm font-medium text-white mb-4">AI 摘要报告</h3>
      {blocks.length === 0 ? (
        <p className="text-mute/60 text-sm">等待分析结果…</p>
      ) : (
        <div className="prose-vap space-y-3 text-sm leading-relaxed text-mist">
          {blocks.map((b, i) => (
            <Block key={i} block={b} />
          ))}
        </div>
      )}
    </Card>
  );
}

type Block =
  | { kind: "h"; level: number; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] }
  | { kind: "code"; text: string }
  | { kind: "p"; text: string };

function parseMarkdown(md: string): Block[] {
  const lines = md.replace(/\r/g, "").split("\n");
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === "") {
      i++;
      continue;
    }

    // 围栏代码块
    if (line.startsWith("```")) {
      const buf: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        buf.push(lines[i]);
        i++;
      }
      i++; // 跳过闭合 ```
      blocks.push({ kind: "code", text: buf.join("\n") });
      continue;
    }

    // 标题
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) {
      blocks.push({ kind: "h", level: h[1].length, text: h[2] });
      i++;
      continue;
    }

    // 无序列表
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ""));
        i++;
      }
      blocks.push({ kind: "ul", items });
      continue;
    }

    // 有序列表
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i++;
      }
      blocks.push({ kind: "ol", items });
      continue;
    }

    // 段落(聚合连续非空非特殊行)
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !lines[i].startsWith("```") &&
      !/^(#{1,4})\s+/.test(lines[i]) &&
      !/^\s*[-*]\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i])
    ) {
      para.push(lines[i]);
      i++;
    }
    blocks.push({ kind: "p", text: para.join(" ") });
  }

  return blocks;
}

function Block({ block }: { block: Block }) {
  switch (block.kind) {
    case "h": {
      const Tag = (`h${Math.min(block.level + 2, 6)}`) as "h3" | "h4" | "h5" | "h6";
      return (
        <Tag className="font-bold text-white mt-2 first:mt-0">
          {renderInline(block.text)}
        </Tag>
      );
    }
    case "ul":
      return (
        <ul className="list-disc pl-5 space-y-1">
          {block.items.map((it, i) => (
            <li key={i}>{renderInline(it)}</li>
          ))}
        </ul>
      );
    case "ol":
      return (
        <ol className="list-decimal pl-5 space-y-1">
          {block.items.map((it, i) => (
            <li key={i}>{renderInline(it)}</li>
          ))}
        </ol>
      );
    case "code":
      return (
        <pre className="rounded-card-sm glass-chip p-3 overflow-x-auto text-[12px] font-mono text-mist">
          <code>{block.text}</code>
        </pre>
      );
    case "p":
      return <p>{renderInline(block.text)}</p>;
  }
}

/** 行内:**bold** 和 `code`。 */
function renderInline(text: string): ReactNode {
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return tokens.map((t, i) => {
    if (t.startsWith("**") && t.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold text-white">
          {t.slice(2, -2)}
        </strong>
      );
    }
    if (t.startsWith("`") && t.endsWith("`")) {
      return (
        <code key={i} className="font-mono text-[12px] text-accent bg-white/5 px-1 rounded">
          {t.slice(1, -1)}
        </code>
      );
    }
    return <span key={i}>{t}</span>;
  });
}
