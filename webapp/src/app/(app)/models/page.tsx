import { Placeholder } from "@/components/shell/Placeholder";

export default function ModelsPage() {
  return (
    <Placeholder
      title="模型管理"
      desc="下载与管理 YOLOv11 / Whisper / Sentence-Transformer 等模型,SHA256 完整性校验。"
      origin="src/ui/model_manager_tab.py + ModelDownloadWorker + ModelManager"
      capabilities={[
        "4 张必下模型卡 + 本地模型扫描网格",
        "下载进度 SSE 实时推送",
        "SHA256 完整性校验(防 MITM 投毒)",
        "一键全部下载 + 就绪状态检测",
      ]}
    />
  );
}
