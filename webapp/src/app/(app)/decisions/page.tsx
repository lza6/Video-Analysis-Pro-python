"use client";

import { useEffect, useState } from "react";
import { apiGet, apiDelete } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

interface Decision {
  id?: string;
  timestamp?: string;
  step_name: string;
  action_type: string;
  decision: string;
  reason: string;
  status?: string;
  risk?: string;
  duration_ms?: number;
  output_path?: string;
  cause_id?: string;
}

export default function DecisionsPage() {
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [selected, setSelected] = useState<Decision | null>(null);

  const refresh = () => {
    apiGet<{ decisions: Decision[]; available: boolean }>("/api/decisions").then((r) => {
      setDecisions(r.decisions);
      if (r.decisions[0]) setSelected(r.decisions[0]);
    });
  };
  useEffect(() => { refresh(); }, []);

  const clearAll = async () => {
    await apiDelete("/api/decisions");
    refresh();
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-black tracking-tight text-white">决策日志</h1>
          <p className="text-sm text-mute mt-1.5">Agent 黑匣子:每步思考、工具调用、原因与产物追溯。</p>
        </div>
        <Button variant="chip" onClick={clearAll}>清空</Button>
      </header>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card className="p-4">
          <h3 className="text-sm font-medium text-white mb-3">决策表</h3>
          <div className="space-y-1 max-h-[60vh] overflow-y-auto">
            {decisions.length === 0 ? (
              <p className="text-mute/60 text-sm">暂无决策记录</p>
            ) : (
              decisions.map((d, i) => (
                <button
                  key={d.id || i}
                  onClick={() => setSelected(d)}
                  className={`w-full text-left rounded-card-sm px-3 py-2 text-xs transition-colors ${
                    selected === d ? "bg-white/10" : "hover:bg-white/5"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-white font-medium truncate">{d.step_name}</span>
                    <span className={
                      d.risk === "high" ? "text-danger" :
                      d.risk === "medium" ? "text-warn" : "text-mute"
                    }>{d.risk}</span>
                  </div>
                  <div className="text-mute truncate">{d.decision}</div>
                </button>
              ))
            )}
          </div>
        </Card>

        <Card className="p-5">
          <h3 className="text-sm font-medium text-white mb-4">详情</h3>
          {selected ? (
            <dl className="space-y-2 text-sm">
              <Field label="步骤" value={selected.step_name} />
              <Field label="动作类型" value={selected.action_type} />
              <Field label="决策" value={selected.decision} />
              <Field label="原因" value={selected.reason} />
              <Field label="状态" value={selected.status || ""} />
              <Field label="风险" value={selected.risk || ""} />
              <Field label="耗时" value={`${selected.duration_ms || 0} ms`} />
              <Field label="产物路径" value={selected.output_path || "—"} />
              <Field label="触发来源" value={selected.cause_id || "—"} />
            </dl>
          ) : (
            <p className="text-mute/60 text-sm">选中左侧记录查看详情</p>
          )}
        </Card>
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2">
      <dt className="text-mute w-20 shrink-0">{label}</dt>
      <dd className="text-mist break-words">{value}</dd>
    </div>
  );
}
