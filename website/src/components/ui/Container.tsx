import type { ElementType, ReactNode } from "react";
import { cn } from "@/lib/utils";

type ContainerSize = "wide" | "default" | "narrow";

interface ContainerProps {
  as?: ElementType;
  size?: ContainerSize;
  className?: string;
  children: ReactNode;
}

const SIZE_MAX_W: Record<ContainerSize, string> = {
  wide: "max-w-7xl",
  default: "max-w-6xl",
  narrow: "max-w-3xl",
};

/**
 * 纯展示容器：固定水平内边距 + 居中 + 可选最大宽度。
 * 无状态、无 "use client" — Server Component。
 */
export function Container({
  as,
  size = "default",
  className,
  children,
}: ContainerProps) {
  const Component = (as ?? "div") as ElementType<{
    className?: string;
    children?: ReactNode;
  }>;
  return (
    <Component
      className={cn(
        "mx-auto px-4 sm:px-6 lg:px-8",
        SIZE_MAX_W[size],
        className,
      )}
    >
      {children}
    </Component>
  );
}
