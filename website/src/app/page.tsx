import Header from "@/components/Header";
import Hero from "@/components/sections/Hero";
import Features from "@/components/sections/Features";
import HowItWorks from "@/components/sections/HowItWorks";
import Stats from "@/components/sections/Stats";
import Download from "@/components/sections/Download";
import FAQ from "@/components/sections/FAQ";
import Footer from "@/components/sections/Footer";

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "SoftwareApplication",
      name: "Video Analysis Pro",
      applicationCategory: "MultimediaApplication",
      operatingSystem: "Windows, macOS, Linux",
      description:
        "本地运行的 AI 视频分析工具：多模态理解（YOLOv11 + Whisper + LLM）、智能关键帧提取、自动集锦生成、Agent 对话式操控。隐私至上，支持完全离线。",
      url: "https://github.com/lza6/Video-Analysis-Pro-python",
      downloadUrl: "https://github.com/lza6/Video-Analysis-Pro-python",
      license: "https://www.gnu.org/licenses/gpl-3.0",
      author: { "@type": "Organization", name: "听风公司 (Tingfeng)" },
      offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
    },
    {
      "@type": "Organization",
      name: "听风公司 (Tingfeng)",
      url: "https://github.com/lza6/Video-Analysis-Pro-python",
      logo: "https://github.com/lza6/Video-Analysis-Pro-python/raw/main/resources/logo_256.png",
    },
    {
      "@type": "WebSite",
      name: "Video Analysis Pro",
      url: "https://github.com/lza6/Video-Analysis-Pro-python",
      inLanguage: "zh-CN",
    },
    {
      "@type": "FAQPage",
      mainEntity: [
        {
          "@type": "Question",
          name: "这个工具会联网吗？我的视频安全吗？",
          acceptedAnswer: {
            "@type": "Answer",
            text: "默认完全离线。使用 Ollama 本地模型时，视频帧、音频、字幕全程不离开你的电脑。仅当你主动选择云端 API 时才会上传对应文本内容。",
          },
        },
        {
          "@type": "Question",
          name: "对电脑配置有要求吗？",
          acceptedAnswer: {
            "@type": "Answer",
            text: "基础运行：Python 3.10+、FFmpeg、8GB 内存即可使用云端 API 模式。若想本地跑大模型，建议 16GB 内存 + 独立 GPU。",
          },
        },
        {
          "@type": "Question",
          name: "不会写代码能用吗？",
          acceptedAnswer: {
            "@type": "Answer",
            text: "能。Windows 用户双击「启动应用.bat」，脚本自动创建虚拟环境、安装依赖、启动图形界面，无需任何命令行操作。",
          },
        },
      ],
    },
  ],
};

export default function Page() {
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <Header />
      <main className="flex-1">
        <Hero />
        <Features />
        <HowItWorks />
        <Stats />
        <Download />
        <FAQ />
      </main>
      <Footer />
    </>
  );
}
