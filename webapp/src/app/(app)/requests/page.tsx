"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiDelete, apiGet } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card, Chip } from "@/components/ui/Card";
import type {
  ProviderStat,
  RequestClearResponse,
  RequestListResponse,
  RequestLog,
  RequestStatsResponse,
} from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * LLM 请求日志页(F7):KPI 总览 + 历史表 + 详情弹窗 + provider 筛选 + 清空。
 * 轮询刷新(与 dashboard 同源,避免 SSE 长连接)。
 * 标签页隐藏时暂停 + 失败指数退避,减少空闲打后端。
 */

const BASE_INTERVAL_MS = 30_000;
const MAX_INTERVAL_MS = 120_000;
const LIST_LIMIT = 200;

interface RequestState {
  logs: RequestLog[];
  stats: Record<string, ProviderStat>;
  loading: boolean;
  error: string | null;
}

const INITIAL: RequestState = {
  logs: [],
  stats: {},
  loading: true,
  error: null,
};

function sumStats(stats: Record<string, ProviderStat>) {
  let requests = 0;
  let total = 0;
  let latency = 0;
  let errors = 0;
  for (const s of Object.values(stats)) {
    requests += s.requests;
    total += s.total_tokens;
    latency += s.avg_latency_ms * s.requests;
    errors += s.error_count;
  }
  return {
    requests,
    total,
    avg: requests > 0 ? latency / requests : 0,
    errors,
  };
}

export default function RequestsPage() {
  const [state, setState] = useState<RequestState>(INITIAL);
  const [providerFilter, setProviderFilter] = useState<string>("");
  const [selected, setSelected] = useState<RequestLog | null>(null);
  const [clearing, setClearing] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const backoffRef = useRef<number>(BASE_INTERVAL_MS);

  const refresh = useCallback(async () => {
    const [logsR, statsR] = await Promise.allSettled([
      apiGet<RequestListResponse>(
        `/api/requests?limit=${LIST_LIMIT}${
          providerFilter ? `&provider=${encodeURIComponent(providerFilter)}` : ""
        }`,
      ),
      apiGet<RequestStatsResponse>("/api/requests/stats"),
    ]);

    setState((prev) => {
      const next: RequestState = {
        ...prev,
        logs: logsR.status === "fulfilled" ? logsR.value.requests ?? [] : prev.logs,
        stats: statsR.status === "fulfilled" ? statsR.value.providers ?? {} : prev.stats,
        loading: false,
        error:
          logsR.status === "rejected"
            ? logsR.reason instanceof Error
              ? logsR.reason.message
              : String(logsR.reason)
            : prev.error,
      };
      return next;
    });
  }, [providerFilter]);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;

    const schedule = () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(async () => {
        if (typeof document !== "undefined" && document.hidden) {
          schedule();
          return;
        }
        try {
          await refresh();
          backoffRef.current = BASE_INTERVAL_MS;
        } catch {
          backoffRef.current = Math.min(backoffRef.current * 2, MAX_INTERVAL_MS);
        }
        schedule();
      }, backoffRef.current);
    };

    const onVisible = () => {
      if (!document.hidden) {
        backoffRef.current = BASE_INTERVAL_MS;
        if (timer) clearTimeout(timer);
        void refresh().finally(schedule);
      }
    };

    void refresh().finally(schedule);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      if (timer) clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [refresh]);

  const clearAll = async () => {
    setClearing(true);
    try {
      await apiDelete<RequestClearResponse>("/api/requests");
      setConfirmClear(false);
      await refresh();
    } finally {
      setClearing(false);
    }
  };

  const totals = sumStats(state.stats);
  const providerNames = Object.keys(state.stats).sort();

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black tracking-tight text-white">
            请求<span className="text-gradient">日志</span>
          </h1>
          <p className="text-sm text-mute mt-1.5">
            每次 LLM 调用的状态码 / token 用量 / 延迟 / 预览。15s 自动刷新。
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <select
            value={providerFilter}
            onChange={(e) => setProviderFilter(e.target.value)}
            className="glass-chip glass-edge rounded-full px-3 py-1.5 text-xs text-white"
            aria-label="按 provider 筛选"
          >
            <option value="">全部 provider</option>
            {providerNames.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
          <Button size="sm" variant="glass" onClick={() => void refresh()} disabled={state.loading}>
            刷新
          </Button>
          <Button
            size="sm"
            variant="danger"
            onClick={() => setConfirmClear(true)}
            disabled={clearing || totals.requests === 0}
            loading={clearing}
          >
            清空
          </Button>
        </div>
      </header>

      <section className="grid sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <KpiCard label="总请求" value={String(totals.requests)} hint="所有 provider 合计" />
        <KpiCard label="总 token" value={totals.total.toLocaleString()} hint={`输入+输出`} />
        <KpiCard
          label="平均延迟"
          value={`${totals.avg.toFixed(0)} ms`}
          hint="端到端含重试"
        />
        <KpiCard
          label="错误数"
          value={String(totals.errors)}
          hint="非 2xx 或网络异常"
          tone={totals.errors > 0 ? "danger" : "ok"}
        />
      </section>

      {providerNames.length > 0 && (
        <Card className="p-5">
          <h3 className="text-sm font-medium text-white mb-3">Provider 用量分布</h3>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {providerNames.map((p) => {
              const s = state.stats[p];
              if (!s) return null;
              return (
                <div key={p} className="glass-chip rounded-card-sm px-3 py-2 text-xs">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-white font-medium truncate">{p}</span>
                    <Chip
                      className={cn(s.error_count > 0 ? "text-danger" : "text-mute")}
                    >
                      {s.requests} 次
                    </Chip>
                  </div>
                  <div className="text-mute font-mono text-[11px]">
                    {s.total_tokens.toLocaleString()} tok · {s.avg_latency_ms.toFixed(0)} ms
                    {s.error_count > 0 && ` · ${s.error_count} 错`}
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      <Card className="p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-white">请求历史</h3>
          <span className="text-xs text-mute">{state.logs.length} 条</span>
        </div>
        {state.error ? (
          <p className="text-sm text-danger">{state.error}</p>
        ) : state.logs.length === 0 ? (
          <p className="text-sm text-mute/60 py-8 text-center">
            {state.loading ? "加载中…" : "暂无请求记录"}
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-mute border-b border-white/5">
                  <th className="py-2 pr-3 font-normal">时间</th>
                  <th className="py-2 pr-3 font-normal">Provider</th>
                  <th className="py-2 pr-3 font-normal">Model</th>
                  <th className="py-2 pr-3 font-normal">状态</th>
                  <th className="py-2 pr-3 font-normal text-right">耗时</th>
                  <th className="py-2 pr-3 font-normal text-right">Token</th>
                </tr>
              </thead>
              <tbody>
                {state.logs.map((l) => (
                  <tr
                    key={l.log_id}
                    onClick={() => setSelected(l)}
                    className="border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors"
                  >
                    <td className="py-2.5 pr-3 text-xs text-mute font-mono whitespace-nowrap">
                      {l.timestamp}
                    </td>
                    <td className="py-2.5 pr-3 text-xs text-mist">{l.provider}</td>
                    <td className="py-2.5 pr-3 text-xs text-mist font-mono truncate max-w-[200px]">
                      {l.model || "—"}
                    </td>
                    <td className="py-2.5 pr-3">
                      <StatusBadge code={l.status_code} />
                    </td>
                    <td className="py-2.5 pr-3 text-xs text-mute text-right whitespace-nowrap">
                      {l.latency_ms.toFixed(0)} ms
                    </td>
                    <td className="py-2.5 pr-3 text-xs text-mist text-right whitespace-nowrap">
                      {l.total_tokens.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* 详情弹窗 */}
      {selected && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          onClick={() => setSelected(null)}
        >
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
          <Card
            tone="strong"
            className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto p-6 space-y-4"
          >
            <div onClick={(e: { stopPropagation: () => void }) => e.stopPropagation()}>
              <div className="flex items-start justify-between gap-4 mb-4">
                <div className="min-w-0">
                  <h3 className="text-lg font-bold text-white truncate">{selected.provider}</h3>
                  <p className="text-xs text-mute font-mono mt-1 break-all">
                    {selected.model || "—"}
                  </p>
                </div>
                <Button size="sm" variant="chip" onClick={() => setSelected(null)}>
                  关闭
                </Button>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs mb-4">
                <DetailField label="状态码" value={String(selected.status_code ?? "—")} />
                <DetailField label="延迟" value={`${selected.latency_ms.toFixed(0)} ms`} />
                <DetailField label="Prompt tok" value={selected.prompt_tokens.toLocaleString()} />
                <DetailField label="完成 tok" value={selected.completion_tokens.toLocaleString()} />
              </div>

              {selected.error && (
                <div className="rounded-card-sm bg-danger/10 border border-danger/30 px-3 py-2 mb-3">
                  <div className="text-[11px] text-danger font-medium mb-0.5">错误</div>
                  <div className="text-xs text-danger/90 break-words font-mono">{selected.error}</div>
                </div>
              )}

              <DetailSection label="请求预览" text={selected.request_preview} />
              <DetailSection label="响应预览" text={selected.response_preview} />
            </div>
          </Card>
        </div>
      )}

      {/* 清空确认弹窗 */}
      {confirmClear && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          onClick={() => setConfirmClear(false)}
        >
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
          <Card tone="strong" className="relative w-full max-w-md p-6 space-y-4">
            <div onClick={(e: { stopPropagation: () => void }) => e.stopPropagation()}>
              <h3 className="text-lg font-bold text-white">确认清空所有请求日志?</h3>
              <p className="text-sm text-mute mt-1">
                将删除 {totals.requests} 条记录,不可恢复。token 统计将一并重置。
              </p>
              <div className="flex justify-end gap-2 mt-4">
                <Button size="sm" variant="glass" onClick={() => setConfirmClear(false)}>
                  取消
                </Button>
                <Button size="sm" variant="danger" onClick={() => void clearAll()} loading={clearing}>
                  确认清空
                </Button>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

/* ---------- 子组件 ---------- */

type KpiTone = "ok" | "danger" | "mute";

function KpiCard({
  label,
  value,
  hint,
  tone = "mute",
}: {
  label: string;
  value: string;
  hint: string;
  tone?: KpiTone;
}) {
  const dot = tone === "danger" ? "bg-danger" : tone === "ok" ? "bg-ok" : "bg-mute";
  return (
    <Card tone="strong" className="p-5">
      <div className="flex items-center gap-2 mb-2">
        <span className={cn("w-2 h-2 rounded-full", dot)} />
        <span className="text-xs uppercase tracking-wider text-mute">{label}</span>
      </div>
      <div className="text-2xl font-black text-white">{value}</div>
      <div className="text-xs text-mute mt-1.5 truncate">{hint}</div>
    </Card>
  );
}

function StatusBadge({ code }: { code: number | null }) {
  let cls = "bg-mute/20 text-mute";
  let label = "—";
  if (code !== null) {
    label = String(code);
    if (code >= 200 && code < 300) cls = "bg-ok/20 text-ok";
    else if (code >= 400 && code < 500) cls = "bg-warn/20 text-warn";
    else if (code >= 500) cls = "bg-danger/20 text-danger";
  }
  return (
    <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium", cls)}>
      {label}
    </span>
  );
}

function DetailField({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass-chip rounded-card-sm px-2.5 py-1.5">
      <div className="text-[10px] text-mute uppercase tracking-wider">{label}</div>
      <div className="text-sm text-white font-mono mt-0.5">{value}</div>
    </div>
  );
}

function DetailSection({ label, text }: { label: string; text: string }) {
  return (
    <div>
      <div className="text-xs text-mute uppercase tracking-wider mb-1.5">{label}</div>
      <pre className="glass-chip rounded-card-sm p-3 text-xs text-mist font-mono whitespace-pre-wrap break-words max-h-48 overflow-y-auto">
        {text || "(空)"}
      </pre>
    </div>
  );
}
