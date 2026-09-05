import { Placeholder } from "@/components/shell/Placeholder";

export default function BatchPage() {
  return (
    <Placeholder
      title="批量处理"
      desc="对整目录视频批量跑分析,断点续跑、逐分片进度、命中统计。"
      origin="src/ui/batch_tab.py + src/core/batch_runner.py + run_store.py"
      capabilities={[
        "配置视频目录 + 关键物品图 + 分片时长 + 帧证据策略",
        "批量运行 SSE 逐任务/逐分片进度 + ETA",
        "断点续跑(接 RunStore SQLite)",
        "历史记录树(视频→分片可展开)+ 运行详情弹窗",
        "Agent 每轮介入决策(命中即停/深看/继续)",
      ]}
    />
  );
}
