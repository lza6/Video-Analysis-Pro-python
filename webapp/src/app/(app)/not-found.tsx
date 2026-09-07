import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

/**
 * 404 页 — 路由组内未匹配路径进入此处。
 * 与 error.tsx 同款玻璃拟态视觉,保持一致语言。
 */
export default function NotFound() {
  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-6 py-16">
      <Card tone="strong" className="p-10 text-center space-y-6">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full glass-chip text-mute mx-auto">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="w-8 h-8"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="10" />
            <path d="M9 9l6 6m0-6l-6 6" strokeLinecap="round" />
          </svg>
        </div>

        <div>
          <div className="text-5xl font-black text-gradient mb-2">404</div>
          <h1 className="text-xl font-bold text-white">找不到这个页面</h1>
          <p className="text-sm text-mute mt-2">
            它可能已被移动,或从未存在。从下方回到常去的地方。
          </p>
        </div>

        <div className="flex items-center justify-center gap-3 pt-2">
          <Button href="/">回首页</Button>
          <Button variant="glass" href="/dashboard/">
            去总览
          </Button>
        </div>
      </Card>
    </div>
  );
}
