"use client";

import type { ReactNode } from "react";
import { GLOSSARY } from "@/lib/glossary";

/**
 * 术语气泡(v10.6.0 P1-1):术语以 <abbr> 呈现,悬浮显示解释。
 * 未收录的词直接原样渲染(不炸),便于渐进扩充词表。
 */
export function GlossaryTerm({ term }: { term: string }) {
  const explain = GLOSSARY[term];
  if (!explain) {
    return <span>{term}</span>;
  }
  return (
    <abbr
      title={explain}
      className="cursor-help border-b border-dotted border-mute/50"
    >
      {term}
    </abbr>
  );
}

/** 段落内嵌术语:把文本里的「术语」自动包成气泡(按词表精确匹配)。 */
export function GlossaryText({ text }: { text: string }): ReactNode {
  const terms = Object.keys(GLOSSARY).sort((a, b) => b.length - a.length);
  const parts: ReactNode[] = [];
  let rest = text;
  let key = 0;
  while (rest.length > 0) {
    let hit: string | null = null;
    let idx = rest.length;
    for (const t of terms) {
      const at = rest.indexOf(t);
      if (at >= 0 && at < idx) {
        idx = at;
        hit = t;
      }
    }
    if (hit === null) {
      parts.push(rest);
      break;
    }
    if (idx > 0) parts.push(rest.slice(0, idx));
    parts.push(<GlossaryTerm key={key++} term={hit} />);
    rest = rest.slice(idx + hit.length);
  }
  return <>{parts}</>;
}
