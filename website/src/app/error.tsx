"use client";

import { useEffect } from "react";

/**
 * App Router 错误边界。
 *
 * 渲染失败时 Next.js 以此文件作为 fallback UI。
 * 生产环境只展示 error.digest（避免泄漏敏感细节），
 * 开发期展示 error.message 便于排查。
 */
export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  // 不在此 console.log/error —— Next 会把错误记录到服务端日志，
  // 客户端重复打日志会污染浏览器控制台。这里仅在错误变化时重置焦点。
  useEffect(() => {
    // 占位 effect：保证 error 变化时组件重新订阅。Next 自行记录错误。
  }, [error]);

  const detail =
    process.env.NODE_ENV !== "production"
      ? error.message
      : error.digest ?? "未知错误";

  return (
    <div className="min-h-[70vh] flex items-center justify-center px-4 py-20">
      <div className="glass-strong glass-edge rounded-3xl p-10 max-w-md w-full text-center">
        <p className="glass-chip rounded-full px-4 py-1.5 text-xs tracking-[0.2em] uppercase text-accent inline-block">
          Error
        </p>
        <h1 className="mt-6 text-4xl font-black tracking-tight text-white">
          <span className="text-gradient">出错了</span>
        </h1>
        <p className="mt-4 text-mute leading-relaxed">
          分析这一帧时遇到了意外。
        </p>

        <p
          className="mt-5 font-mono text-xs text-mute break-words"
          aria-live="polite"
        >
          {detail}
        </p>

        <div className="mt-8 flex items-center justify-center gap-3">
          <button
            type="button"
            onClick={() => reset()}
            className="glass glass-edge glass-hover rounded-full px-6 py-2.5 text-sm text-white"
          >
            重试
          </button>
          <a
            href="#top"
            className="bg-accent-2 text-white font-medium rounded-full px-6 py-2.5 text-sm hover:bg-accent-2-deep transition-colors"
          >
            返回首页
          </a>
        </div>
      </div>
    </div>
  );
}
