import { Placeholder } from "@/components/shell/Placeholder";

export default function DecisionsPage() {
  return (
    <Placeholder
      title="决策日志"
      desc="Agent 黑匣子:每一步思考、工具调用、决策原因与产物的完整追溯。"
      origin="src/ui/decision_log_panel.py + src/core/decision_log.py"
      capabilities={[
        "决策表(时间/步骤/工具/摘要/状态/风险)",
        "选中行展开完整 decision/reason/args/产物路径/耗时",
        "导出 JSON",
        "实时流式追加(SSE entry_append)",
      ]}
    />
  );
}
