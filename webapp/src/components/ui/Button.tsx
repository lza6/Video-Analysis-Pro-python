import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "glass" | "chip" | "danger";
type ButtonSize = "sm" | "md" | "lg";

interface ButtonProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  href?: string;
  className?: string;
  children: ReactNode;
  target?: string;
  rel?: string;
  onClick?: () => void;
  disabled?: boolean;
  type?: "button" | "submit";
  "aria-label"?: string;
}

const VARIANT_CLASS: Record<ButtonVariant, string> = {
  primary:
    "bg-gradient-to-r from-accent-2 to-accent text-white rounded-full transition-transform hover:-translate-y-0.5 active:scale-[0.98]",
  glass: "glass glass-edge glass-hover rounded-full text-white",
  chip: "glass-chip glass-edge rounded-full text-white hover:border-white/30 transition-colors",
  danger:
    "bg-danger/90 text-white rounded-full hover:bg-danger transition-colors",
};

const SIZE_CLASS: Record<ButtonSize, string> = {
  sm: "px-3.5 py-1.5 text-xs",
  md: "px-5 py-2 text-sm",
  lg: "px-7 py-3.5 text-base",
};

/**
 * 全站唯一按钮基元。href 给定渲染 <a>,否则 <button>。
 * disabled 时降透明度 + 屏蔽指针,并透传原生 disabled(button)。
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
  disabled,
  type = "button",
  "aria-label": ariaLabel,
}: ButtonProps) {
  const classes = cn(
    "inline-flex items-center justify-center gap-2 font-medium",
    VARIANT_CLASS[variant],
    SIZE_CLASS[size],
    disabled && "opacity-40 pointer-events-none",
    className,
  );
  const inner = <span className="relative z-10">{children}</span>;

  if (href && !disabled) {
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
      type={type}
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      className={classes}
    >
      {inner}
    </button>
  );
}
