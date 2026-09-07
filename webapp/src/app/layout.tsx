import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TingFeng Hermes — 控制台",
  description: "通用全能 AI Agent 桌面平台:视频分析 + Agent 对话 + IM 网关 + 远程访问。",
  // 内部工具,不被搜索引擎索引
  robots: { index: false, follow: false },
  icons: { icon: "/logo.png" },
};

export const viewport: Viewport = {
  themeColor: "#151726",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
