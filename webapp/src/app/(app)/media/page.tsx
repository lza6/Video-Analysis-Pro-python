import { Placeholder } from "@/components/shell/Placeholder";

export default function MediaPage() {
  return (
    <Placeholder
      title="摘要媒体"
      desc="AI 挑选精彩片段,自动生成高光集锦短视频与 GIF 动图。"
      origin="src/ui/main_window.py tab_media + MediaWorker + create_summary_media_artifacts"
      capabilities={[
        "一键生成高光集锦短视频(MoviePy)",
        "生成 GIF 摘要便于分享",
        "播放器预览 + 下载产物",
        "生成进度 SSE 实时推送",
      ]}
    />
  );
}
