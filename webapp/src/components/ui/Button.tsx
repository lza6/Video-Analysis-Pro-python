"use client";

import type { ReactNode } from "react";
import { useState } from "react";
import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "glass" | "chip" | "danger";
type ButtonSize = "sm" | "md" | "lg";
type ButtonFeedbackState = "idle" | "success" | "error";

interface ButtonProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  href?: string;
  className?: string;
  children: ReactNode;
  target?: string;
  rel?: string;
  onClick?: () => void | Promise<void>;
  disabled?: boolean;
  loading?: boolean;
  type?: "button" | "submit";
  "aria-label"?: string;
  /**
   * v10.1.0:点击反馈态。传 true 启用自动 success/error 反馈(onClick 完成后
   * 短暂显示绿色对勾 / 红色 ×,1.5s 后回 idle)。不传 = 旧行为(只 loading 态)。
   */
  feedback?: boolean;
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
 *
 * 五态(v10.1.0 扩展):
 * - default:正常可用
 * - hover:VARIANT_CLASS 内置 hover 类(primary 位移 / glass 边框变亮)
 * - active:active:scale 略缩(primary)
 * - disabled:降透明度 + 屏蔽指针,透传原生 disabled
 * - loading:spinner + 屏蔽点击,与 disabled 同不可触发
 * - success(反馈态):绿色对勾 + 短暂 1.5s,onClick 成功后显示
 * - error(反馈态):红色 × + 摇晃 0.4s,onClick 抛错后显示
 *
 * feedback=true 时,onClick 自动 try/finally 切 success/error 态。
 * 不传 feedback = 旧行为(调用方自管 loading),向后兼容。
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
  loading,
  type = "button",
  feedback = false,
  "aria-label": ariaLabel,
}: ButtonProps) {
  const [feedbackState, setFeedbackState] =
    useState<ButtonFeedbackState>("idle");
  const inert = disabled || loading || feedbackState !== "idle";

  const handleClick = async () => {
    if (inert || !onClick) return;
    if (feedback) setFeedbackState("idle");
    try {
      await onClick();
      if (feedback) {
        setFeedbackState("success");
        setTimeout(() => setFeedbackState("idle"), 1500);
      }
    } catch (e) {
      if (feedback) {
        setFeedbackState("error");
        setTimeout(() => setFeedbackState("idle"), 1500);
      }
      throw e;
    }
  };

  const feedbackIcon =
    feedbackState === "success" ? (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        strokeWidth="3" className="w-4 h-4 text-emerald-400"
        aria-hidden="true">
        <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ) : feedbackState === "error" ? (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        strokeWidth="3" className="w-4 h-4 text-danger"
        aria-hidden="true">
        <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ) : null;

  const classes = cn(
    "inline-flex items-center justify-center gap-2 font-medium",
    VARIANT_CLASS[variant],
    SIZE_CLASS[size],
    inert && "opacity-40 pointer-events-none",
    feedbackState === "error" && "animate-[shake_0.4s_ease-in-out]",
    className,
  );
  const inner = (
    <span className="relative z-10 inline-flex items-center gap-2">
      {loading && (
        <span
          className="inline-block w-3.5 h-3.5 rounded-full border-2 border-white/30 border-t-white animate-spin"
          aria-hidden="true"
        />
      )}
      {feedbackIcon}
      {children}
    </span>
  );

  if (href && !inert) {
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
      onClick={handleClick}
      disabled={inert}
      aria-busy={loading || undefined}
      aria-label={ariaLabel}
      className={classes}
    >
      {inner}
    </button>
  );
}
