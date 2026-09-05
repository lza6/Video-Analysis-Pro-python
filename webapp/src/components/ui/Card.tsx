import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type CardTone = "default" | "strong";

interface CardProps {
  tone?: CardTone;
  interactive?: boolean;
  className?: string;
  children: ReactNode;
  onClick?: () => void;
}

/**
 * 全站唯一卡片基元 — 统一玻璃层 + 半径 + 内边距。
 * 取代散落的 `glass glass-edge glass-hover rounded-3xl p-7` 手写组合。
 */
export function Card({
  tone = "default",
  interactive = false,
  className,
  children,
  onClick,
}: CardProps) {
  return (
    <div
      onClick={onClick}
      className={cn(
        tone === "strong" ? "glass-strong" : "glass",
        "glass-edge rounded-card",
        interactive && "glass-hover cursor-pointer",
        className,
      )}
    >
      {children}
    </div>
  );
}

interface ChipProps {
  className?: string;
  children: ReactNode;
}

/** 小标签(chip)。统一 uppercase tracking 风格由调用方决定。 */
export function Chip({ className, children }: ChipProps) {
  return (
    <span className={cn("glass-chip rounded-full px-3 py-1 text-xs", className)}>
      {children}
    </span>
  );
}
