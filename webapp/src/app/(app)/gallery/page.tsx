import { Placeholder } from "@/components/shell/Placeholder";

export default function GalleryPage() {
  return (
    <Placeholder
      title="关键帧画廊"
      desc="浏览与检索已提取的关键帧,支持按时间戳定位、查看帧详情。"
      origin="src/ui/main_window.py tab_gallery + carousel_widget.py"
      capabilities={[
        "网格/轮播双视图浏览全部关键帧",
        "点击帧查看 metrics(亮度/对比度/饱和度/清晰度)与视觉描述",
        "按时间戳跳转到视频对应位置",
        "跨作业历史帧检索(接 ChromaDB 知识库)",
      ]}
    />
  );
}
