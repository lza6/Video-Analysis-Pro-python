import { Placeholder } from "@/components/shell/Placeholder";

export default function MetricsPage() {
  return (
    <Placeholder
      title="元数据与画质"
      desc="视频技术参数与画质指标趋势图表,一眼看懂视频质量。"
      origin="src/ui/main_window.py tab_metrics + plot_metrics/get_advanced_video_metrics"
      capabilities={[
        "亮度/清晰度/饱和度随时间趋势折线图",
        "分辨率、帧率、时长、编码等元数据卡片",
        "静音检测与音频波形可视化",
        "图表导出",
      ]}
    />
  );
}
