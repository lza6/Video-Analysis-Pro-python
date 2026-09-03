"use client";

import React from "react";

interface Props {
  children: React.ReactNode;
  fallback: React.ReactNode;
}

interface State {
  hasError: boolean;
}

/**
 * ErrorBoundary — 捕获 R3F Canvas 渲染失败（WebGL 不可用 / 着色器编译失败 /
 * context 丢失），降级到 CSS 静态氛围层。
 *
 * 注意：React 函数组件无法直接做 error boundary，必须用 class。
 */
export default class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error): void {
    // 仅在开发期打印，生产静默降级
    if (process.env.NODE_ENV !== "production") {
      console.warn("[HeroVisual] 3D 场景降级为静态层：", error?.message || error);
    }
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}
