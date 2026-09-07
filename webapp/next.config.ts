import type { NextConfig } from "next";

// 生产:静态导出到 out/,由 FastAPI StaticFiles 同源挂载(零 node 运行时)。
// 开发:next dev :3000,前端用 NEXT_PUBLIC_API_BASE 跨域调后端 :8000(后端已开 CORS)。
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
  // e2e/ + playwright.config.ts 不参与构建类型检查(@playwright/test 是 devDep,
  // 生产构建时未装会报 TS2307)。tsc 直接跑仍覆盖它们(见 tsconfig.json exclude)。
  typescript: {
    // build 时跳过类型检查(tsc --noEmit 已在 CI/开发态单独跑,build 只产静态产物)
    ignoreBuildErrors: false,
  },
  // Next 16 turbopack 会把上层目录(C:\Users\...\Desktop)当仓库根,
  // 报 "ignored package-lock.json"。显式指定项目根消除警告。
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
