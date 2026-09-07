"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { apiPostJson } from "@/lib/api";

/**
 * (app) 路由组全局错误边界。
 * Next.js App Router 约定文件:上层抛出未捕获异常时自动渲染。
 * 重试调用 router.refresh() 重载当前路由段,回首页走 <a> 全页导航。
 *
 * v10.1.0:路由错误上报 /api/logs(接现有 logs router),便于前端按 trace 排查。
 */
export default function AppError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // 控制台留痕,便于排查(不外发,本地工具)
    console.error("[Hermes] 路由错误:", error);
    // 上报后端 /api/logs(失败静默,不阻断错误边界本身)
    void apiPostJson("/api/logs", {
      level: "error",
      message: error?.message || "unknown route error",
      digest: error?.digest || null,
      stack: error?.stack?.slice(0, 2000) || null,
      source: "frontend_error_boundary",
    }).catch(() => {
      /* 后端不可达时静默,错误边界不能因上报失败而二次崩溃 */
    });
  }, [error]);

  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-6 py-16">
      <Card tone="strong" className="p-8 text-center space-y-5">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-danger/15 text-danger mx-auto">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="w-7 h-7"
            aria-hidden="true"
          >
            <path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <div>
          <h1 className="text-2xl font-black text-white">出错了</h1>
          <p className="text-sm text-mute mt-2">
            页面在渲染时遇到意外。可重试,或返回首页继续。
          </p>
        </div>

        {error?.message && (
          <pre className="text-xs text-mute/80 font-mono whitespace-pre-wrap break-all rounded-card-sm glass-chip p-3 max-h-32 overflow-y-auto">
            {error.message}
            {error.digest ? `\n[digest: ${error.digest}]` : ""}
          </pre>
        )}

        <div className="flex items-center justify-center gap-3 pt-2">
          <Button onClick={reset}>重试</Button>
          <Button variant="glass" href="/">
            回首页
          </Button>
        </div>
      </Card>
    </div>
  );
}
