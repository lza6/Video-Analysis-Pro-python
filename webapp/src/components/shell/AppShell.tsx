"use client";

import { useSyncExternalStore, type ReactNode } from "react";
import { cn } from "@/lib/utils";

/** 订阅 location 变化(静态导出全页导航时其实不会触发,但保证 hook 契约)。 */
function subscribeLocation(cb: () => void): () => void {
  window.addEventListener("popstate", cb);
  return () => window.removeEventListener("popstate", cb);
}

interface NavItem {
  href: string;
  label: string;
  icon: ReactNode;
  phase?: number; // 仅阶段 0 标注用
}

/* 内联 SVG 图标(单一 stroke 风格,16px 网格)—— 取代 emoji,跨平台一致 */
const I = {
  dashboard: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg>
  ),
  analyze: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><path d="M15 10l5-3v10l-5-3M3 6h12v12H3z" strokeLinecap="round" strokeLinejoin="round"/></svg>
  ),
  gallery: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-5-5L5 21"/></svg>
  ),
  media: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><path d="M10 8l6 4-6 4V8z"/><rect x="2" y="4" width="20" height="16" rx="2"/></svg>
  ),
  metrics: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><path d="M3 3v18h18M8 15V8m5 7V5m5 10v-4"/></svg>
  ),
  chat: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" strokeLinejoin="round"/></svg>
  ),
  logs: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><path d="M4 6h16M4 12h16M4 18h10" strokeLinecap="round"/></svg>
  ),
  batch: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="3" width="8" height="8" rx="1"/><rect x="3" y="13" width="8" height="8" rx="1"/><rect x="13" y="13" width="8" height="8" rx="1"/></svg>
  ),
  models: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><path d="M21 8l-9-5-9 5 9 5 9-5zM3 8v8l9 5 9-5V8" strokeLinejoin="round"/></svg>
  ),
  surveillance: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>
  ),
  skills: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><path d="M12 2l2.4 7.2H22l-6 4.6 2.3 7.2-6.3-4.5-6.3 4.5L8 13.8 2 9.2h7.6z" strokeLinejoin="round"/></svg>
  ),
  decisions: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><path d="M9 6h11M9 12h11M9 18h11M4 6h1v4m-1 8h2l2 2" strokeLinecap="round"/></svg>
  ),
  requests: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><path d="M4 6h16M4 12h16M4 18h10" strokeLinecap="round"/><circle cx="20" cy="18" r="2"/><path d="M20 18v-3" strokeLinecap="round"/></svg>
  ),
  providers: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><path d="M8 3l-4 4 4 4M16 3l4 4-4 4M3 11v3a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-3" strokeLinecap="round" strokeLinejoin="round"/></svg>
  ),
  settings: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4.5 h-4.5"><circle cx="12" cy="12" r="3"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3M4.9 4.9l2.1 2.1m10 10l2.1 2.1m0-14.2l-2.1 2.1m-10 10l-2.1 2.1" strokeLinecap="round"/></svg>
  ),
};

const NAV: NavItem[] = [
  { href: "/dashboard/", label: "总览", icon: I.dashboard },
  { href: "/", label: "AI 分析", icon: I.analyze },
  { href: "/gallery/", label: "关键帧画廊", icon: I.gallery },
  { href: "/media/", label: "摘要媒体", icon: I.media },
  { href: "/metrics/", label: "元数据画质", icon: I.metrics },
  { href: "/agent/", label: "Agent 对话", icon: I.chat },
  { href: "/logs/", label: "系统日志", icon: I.logs },
  { href: "/batch/", label: "批量处理", icon: I.batch },
  { href: "/models/", label: "模型管理", icon: I.models },
  { href: "/surveillance/", label: "监控分析", icon: I.surveillance },
  { href: "/skills/", label: "Skills", icon: I.skills },
  { href: "/decisions/", label: "决策日志", icon: I.decisions },
  { href: "/requests/", label: "请求日志", icon: I.requests },
  { href: "/providers/", label: "提供商", icon: I.providers },
  { href: "/settings/", label: "设置", icon: I.settings },
];

/**
 * 应用外壳:左侧图标导航栏 + 主内容 + 底部状态条插槽。
 * 静态导出模式(output: export)下用 <a> 全页导航(本地工具可接受)。
 */
export function AppShell({
  children,
  footer,
}: {
  children: ReactNode;
  footer?: ReactNode;
}) {
  // 静态导出(output: export)全页导航:每次新页面 AppShell 重新挂载,
  // pathname 初值在客户端 hydration 时已知。用 useSyncExternalStore 读
  // location.pathname 避免 effect 内 setState(React 18 推荐外部 store 模式)。
  const path = useSyncExternalStore(
    subscribeLocation,
    () => window.location.pathname,
    () => "/",
  );

  const isActive = (href: string) =>
    href === "/" ? path === "/" || path === "" : path.startsWith(href.replace(/\/$/, ""));

  return (
    <div className="flex flex-1 min-h-0">
      <a
        href="#main"
        className="sr-only-focusable focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 glass-strong rounded-full px-4 py-2 text-sm text-white"
      >
        跳到主内容
      </a>

      <nav
        aria-label="模块导航"
        className="hidden sm:flex flex-col w-16 lg:w-56 shrink-0 border-r border-white/5 py-4 gap-1 px-2 lg:px-3 sticky top-0 h-screen"
      >
        <div className="flex items-center gap-2.5 px-2 lg:px-3 pb-4">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.png" alt="TingFeng Hermes" className="w-7 h-7 shrink-0" />
          <span className="hidden lg:block text-sm font-bold text-white tracking-tight">
            TingFeng Hermes
          </span>
        </div>

        {NAV.map((n) => (
          <a
            key={n.href}
            href={n.href}
            aria-current={isActive(n.href) ? "page" : undefined}
            className={cn(
              "flex items-center gap-3 rounded-card-sm px-2.5 lg:px-3 py-2.5 text-sm transition-colors",
              isActive(n.href)
                ? "bg-white/10 text-white"
                : "text-mute hover:text-white hover:bg-white/5",
            )}
          >
            <span className="shrink-0">{n.icon}</span>
            <span className="hidden lg:block truncate">{n.label}</span>
          </a>
        ))}
      </nav>

      <div className="flex-1 flex flex-col min-w-0">
        <main id="main" className="flex-1 min-h-0 overflow-y-auto">
          {children}
        </main>
        {footer}
      </div>
    </div>
  );
}
