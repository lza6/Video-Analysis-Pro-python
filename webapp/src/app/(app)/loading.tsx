import { Card } from "@/components/ui/Card";

/**
 * (app) 路由级骨架屏 — 路由切换 / 流式渲染未就绪时显示。
 * 玻璃拟态 spinner + 三块占位卡,避免布局抖动(CLS)。
 */
export default function Loading() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      <div className="flex items-center gap-3">
        <Spinner />
        <div className="space-y-2">
          <div className="h-5 w-40 rounded-full bg-white/10 animate-pulse" />
          <div className="h-3 w-56 rounded-full bg-white/5 animate-pulse" />
        </div>
      </div>

      <div className="grid sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i} className="p-5 space-y-3">
            <div className="h-2 w-20 rounded-full bg-white/10 animate-pulse" />
            <div className="h-7 w-16 rounded-full bg-white/15 animate-pulse" />
            <div className="h-2 w-32 rounded-full bg-white/5 animate-pulse" />
          </Card>
        ))}
      </div>

      <Card className="p-6 space-y-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="h-6 w-16 rounded-full bg-white/10 animate-pulse" />
            <div className="h-3 flex-1 rounded-full bg-white/5 animate-pulse" />
            <div className="h-3 w-24 rounded-full bg-white/5 animate-pulse" />
          </div>
        ))}
      </Card>
    </div>
  );
}

function Spinner() {
  return (
    <span
      className="inline-block w-6 h-6 rounded-full border-2 border-white/15 border-t-accent animate-spin"
      role="status"
      aria-label="加载中"
    />
  );
}
