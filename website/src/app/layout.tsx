import type { Metadata, Viewport } from "next";
import { Noto_Sans_SC, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const notoSans = Noto_Sans_SC({
  variable: "--font-noto",
  subsets: ["latin"],
  weight: ["400", "500", "700", "900"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jb",
  subsets: ["latin"],
  display: "swap",
});

// 站点 URL 可配置化：部署时注入 NEXT_PUBLIC_SITE_URL，本地默认指向 localhost
// 避免 .example.com 占位域名导致 OG/canonical/sitemap 全断
const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/$/, "") ||
  "http://localhost:3000";

const SITE_TITLE = "Video Analysis Pro — 您的私人 AI 视频深度分析专家";
const SITE_DESC =
  "本地运行的 AI 视频分析工具：多模态理解（YOLOv11 + Whisper + LLM）、智能关键帧提取、自动集锦生成、Agent 对话式操控。隐私至上，支持完全离线。";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: SITE_TITLE,
    template: "%s | Video Analysis Pro",
  },
  description: SITE_DESC,
  keywords: [
    "AI 视频分析",
    "本地视频理解",
    "视频内容总结",
    "Whisper 语音识别",
    "YOLO 目标检测",
    "Ollama 本地大模型",
    "离线 AI 工具",
    "视频摘要生成",
    "开源视频分析",
    "隐私保护 AI",
  ],
  authors: [{ name: "听风公司 (Tingfeng)" }],
  openGraph: {
    type: "website",
    locale: "zh_CN",
    url: SITE_URL,
    siteName: "Video Analysis Pro",
    title: SITE_TITLE,
    description: SITE_DESC,
    images: [{ url: "/opengraph-image", width: 1200, height: 630, alt: SITE_TITLE }],
  },
  twitter: {
    card: "summary_large_image",
    title: SITE_TITLE,
    description: SITE_DESC,
    images: ["/opengraph-image"],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true, "max-image-preview": "large" },
  },
  alternates: {
    canonical: "/",
    types: { "image/svg+xml": "/logo.svg" },
  },
  icons: {
    icon: [{ url: "/logo.svg", type: "image/svg+xml" }, { url: "/logo-64.png" }],
    apple: [{ url: "/logo.png", sizes: "256x256" }],
  },
};

export const viewport: Viewport = {
  themeColor: "#151726",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="zh-CN"
      className={`${notoSans.variable} ${jetbrainsMono.variable} h-full antialiased`}
    >
      <body className="grain min-h-full flex flex-col">{children}</body>
    </html>
  );
}
