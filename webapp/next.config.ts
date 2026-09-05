import type { NextConfig } from "next";

// 生产:静态导出到 out/,由 FastAPI StaticFiles 同源挂载(零 node 运行时)。
// 开发:next dev :3000,前端用 NEXT_PUBLIC_API_BASE 跨域调后端 :8000(后端已开 CORS)。
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default nextConfig;
