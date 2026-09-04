/**
 * App Router 路由段加载占位。
 *
 * 嵌入在 layout 内、page 外的 <Suspense> 边界。
 * 无状态、不使用客户端 API，保持为 Server Component。
 */
export default function Loading() {
  return (
    <div className="min-h-[70vh] flex items-center justify-center px-4 py-20">
      <div className="glass-strong glass-edge rounded-3xl p-10 max-w-md w-full">
        <div className="flex items-center gap-2 mb-6">
          <span className="w-2.5 h-2.5 rounded-full bg-accent animate-pulse" />
          <span className="text-xs tracking-[0.2em] uppercase text-mute">
            正在分析
          </span>
        </div>
        <div className="space-y-3">
          <div className="h-4 bg-white/10 rounded animate-pulse" />
          <div className="h-4 bg-white/10 rounded animate-pulse w-5/6" />
          <div className="h-4 bg-white/10 rounded animate-pulse w-2/3" />
        </div>
      </div>
    </div>
  );
}
