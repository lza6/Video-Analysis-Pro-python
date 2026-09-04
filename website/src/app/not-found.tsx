/**
 * App Router 404 兜底。
 *
 * 当 notFound() 被抛出或 URL 无匹配路由时渲染此文件。
 * 站点无多级路由，所有"返回首页"指向 #top（page 顶部 Hero）。
 */
export default function NotFound() {
  return (
    <div className="min-h-[70vh] flex items-center justify-center px-4 py-20">
      <div className="glass-strong glass-edge rounded-3xl p-10 max-w-md w-full text-center">
        <h1 className="text-7xl font-black tracking-tight">
          <span className="text-gradient">404</span>
        </h1>
        <p className="mt-4 text-mute leading-relaxed">
          这一帧没找到
        </p>
        <div className="mt-8 flex items-center justify-center">
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
