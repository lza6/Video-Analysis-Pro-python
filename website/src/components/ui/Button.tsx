import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "glass" | "chip";
type ButtonSize = "md" | "lg";

interface ButtonProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  href?: string;
  className?: string;
  children: ReactNode;
  target?: string;
  rel?: string;
  onClick?: () => void;
  "aria-label"?: string;
}

const VARIANT_CLASS: Record<ButtonVariant, string> = {
  primary:
    "bg-gradient-to-r from-accent-2 to-accent text-white rounded-full transition-transform hover:-translate-y-0.5 active:scale-[0.98]",
  glass: "glass glass-edge glass-hover rounded-full text-white",
  chip: "glass-chip glass-edge rounded-full text-white hover:border-white/30 transition-colors",
};

const SIZE_CLASS: Record<ButtonSize, string> = {
  md: "px-5 py-2 text-sm",
  lg: "px-7 py-3.5",
};

/**
 * 按钮基元：primary 渐变实心 / glass 玻璃 / chip 标签。
 * href 给定渲染 <a>，否则 <button type="button">。
 * 内层 <span className="relative z-10"> 给 primary 的渐变背景留 z 层级。
 * 无 "use client" — 需 onClick 的消费者在 client 组件中使用。
 */
export function Button({
  variant = "primary",
  size = "md",
  href,
  className,
  children,
  target,
  rel,
  onClick,
  "aria-label": ariaLabel,
}: ButtonProps) {
  const classes = cn(
    "inline-flex items-center justify-center",
    VARIANT_CLASS[variant],
    SIZE_CLASS[size],
    className,
  );
  const inner = <span className="relative z-10">{children}</span>;

  if (href) {
    return (
      <a
        href={href}
        target={target}
        rel={rel}
        onClick={onClick}
        aria-label={ariaLabel}
        className={classes}
      >
        {inner}
      </a>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      className={classes}
    >
      {inner}
    </button>
  );
}
