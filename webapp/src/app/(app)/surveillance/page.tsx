import { Placeholder } from "@/components/shell/Placeholder";

export default function SurveillancePage() {
  return (
    <Placeholder
      title="监控分析"
      desc="接入 RTSP 实时流,关键物品检测命中即告警。"
      origin="src/ui/surveillance_tab.py + src/core/rtsp_stream.py + surveillance_agent.py"
      capabilities={[
        "RTSP URL + 关键物品参考图 + 判图后端配置",
        "实时拉流检测,命中事件 SSE 推送(时间/置信度/描述)",
        "开始/停止监控控制",
        "昼夜自适应阈值(接 skills 蒸馏)",
      ]}
    />
  );
}
