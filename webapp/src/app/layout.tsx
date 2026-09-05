import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Video Analysis Pro — 控制台",
  description: "本地 AI 视频分析工作台:三阶段流水线 + Agent 对话式操控。",
  // 内部工具,不被搜索引擎索引
  robots: { index: false, follow: false },
  icons: { icon: "/logo.svg" },
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
