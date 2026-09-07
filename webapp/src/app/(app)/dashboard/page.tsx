"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiGet, apiUrl } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card, Chip } from "@/components/ui/Card";
import type { HealthResponse, JobSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * 全局总览页:聚合后端健康度 / 作业 / IM 网关 / 远程隧道四路状态。
 * 轮询刷新(本地工具可接受,避免 SSE 长连接的资源占用)。
 *
 * 轮询策略(用户启动日志实证:空闲时每页都打 /api/health + 4 路并行):
 *  - 标签页可见时 30s 一跳(原 15s 过频,空闲也持续打四路);
 *  - 标签页隐藏时暂停(visibilitychange),回来立即补一次;
 *  - 失败时指数退避(30→60→120s 封顶),后端宕时不疯狂重连。
 */


interface JobsResponse {
  jobs: JobSummary[];
}

/** IM 网关状态(与后端 src/web/routes/im.py 对齐,字段宽松)。 */
interface ImGatewayStatus {
  online: boolean;
  accounts?: number;
  pending_messages?: number;
  backend?: string;
}

/** 远程隧道状态(与后端 src/web/routes/remote.py 对齐,字段宽松)。 */
interface TunnelStatus {
  active: boolean;
  url?: string;
  provider?: string;
  expires_at?: string | null;
}

interface DashState {
  health: HealthResponse | null;
  jobs: JobSummary[];
  im: ImGatewayStatus | null;
  tunnel: TunnelStatus | null;
  online: boolean;
  lastError: string | null;
}

const INITIAL_STATE: DashState = {
  health: null,
  jobs: [],
  im: null,
  tunnel: null,
  online: true,
  lastError: null,
};

const BASE_INTERVAL_MS = 30_000;
const MAX_INTERVAL_MS = 120_000;

export default function DashboardPage() {
  const [state, setState] = useState<DashState>(INITIAL_STATE);
  const [loading, setLoading] = useState(true);
  const backoffRef = useRef<number>(BASE_INTERVAL_MS);

  const refresh = useCallback(async () => {
    // 四路并行拉取,任一失败不拖垮其他
    const [healthR, jobsR, imR, tunnelR] = await Promise.allSettled([
      apiGet<HealthResponse>("/api/health"),
      apiGet<JobsResponse>("/api/jobs").then((r) => r.jobs ?? []),
      apiGet<ImGatewayStatus>("/api/im/gateway/status").catch(() => null),
      apiGet<TunnelStatus>("/api/remote/tunnel/status").catch(() => null),
    ]);

    setState((prev) => {
      const next: DashState = {
        ...prev,
        health: healthR.status === "fulfilled" ? healthR.value : prev.health,
        jobs: jobsR.status === "fulfilled" ? jobsR.value : prev.jobs,
        im: imR.status === "fulfilled" ? imR.value : prev.im,
        tunnel: tunnelR.status === "fulfilled" ? tunnelR.value : prev.tunnel,
        online: healthR.status === "fulfilled",
        lastError:
          healthR.status === "rejected"
            ? healthR.reason instanceof Error
              ? healthR.reason.message
              : String(healthR.reason)
            : prev.lastError,
      };
      return next;
    });
    setLoading(false);
  }, []);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;

    const schedule = () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(async () => {
        // 标签页隐藏时跳过本次,等可见再补(避免空闲打四路)
        if (typeof document !== "undefined" && document.hidden) {
          schedule();
          return;
        }
        try {
          await refresh();
          backoffRef.current = BASE_INTERVAL_MS; // 成功重置退避
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

  const running = state.jobs.filter((j) => j.status === "running" || j.status === "pending");
  const done = state.jobs.filter((j) => j.status === "done");
  const failed = state.jobs.filter((j) => j.status === "failed");
  const caps = state.health?.capabilities;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black tracking-tight text-white">
            控制<span className="text-gradient">总览</span>
          </h1>
          <p className="text-sm text-mute mt-1.5">
            Agent 会话 / 视频分析 / IM 网关 / 远程隧道 — 一屏全知。
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Chip className="text-mute">
            {loading ? "初次拉取…" : `${new Date().toLocaleTimeString("zh-CN")}`}
          </Chip>
          <Button size="sm" variant="glass" onClick={() => void refresh()} disabled={loading}>
            刷新
          </Button>
        </div>
      </header>

      {/* 顶部四张状态卡 */}
      <section className="grid sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard
          label="后端健康"
          tone={state.online ? "ok" : "danger"}
          value={state.online ? "在线" : "离线"}
          hint={caps ? `LLM:${caps.llm_backend}` : "—"}
        />
        <StatCard
          label="分析作业"
          tone={running.length > 0 ? "warn" : "ok"}
          value={String(state.jobs.length)}
          hint={`进行 ${running.length} · 完成 ${done.length} · 失败 ${failed.length}`}
        />
        <StatCard
          label="IM 网关"
          tone={state.im?.online ? "ok" : state.im ? "danger" : "mute"}
          value={state.im?.online ? "在线" : "离线"}
          hint={state.im ? `账号 ${state.im.accounts ?? 0} · 待发 ${state.im.pending_messages ?? 0}` : "未配置"}
        />
        <StatCard
          label="远程隧道"
          tone={state.tunnel?.active ? "ok" : "mute"}
          value={state.tunnel?.active ? "已建立" : "未建立"}
          hint={state.tunnel?.provider ?? state.tunnel?.url ?? "本地访问"}
        />
      </section>

      {/* 能力矩阵 */}
      {caps && (
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-semibold text-white">能力矩阵</h2>
            {state.health && (
              <span className="text-xs text-mute">磁盘余量 {state.health.disk_free_gb} GB</span>
            )}
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <CapTile label="GPU 加速" on={caps.nvidia_gpu} />
            <CapTile label="FFmpeg" on={caps.ffmpeg} />
            <CapTile label="CLIP 语义" on={caps.clip_semantic} />
            <CapTile label="高级媒体" on={caps.advanced_media} />
            <CapTile label="OCR" on={caps.ocr} />
            <CapTile label="密钥环" on={state.health?.keyring_available ?? false} />
          </div>
        </Card>
      )}

      {/* 最近作业列表 */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-white">最近分析作业</h2>
          <a
            href="/batch/"
            className="text-xs text-accent hover:text-accent-2 transition-colors"
          >
            查看批量 →
          </a>
        </div>
        {state.jobs.length === 0 ? (
          <p className="text-sm text-mute/60 py-8 text-center">
            {loading ? "加载中…" : "暂无作业,去"}
            {!loading && (
              <a href="/analyze/" className="text-accent hover:text-accent-2 ml-1">发起分析</a>
            )}
          </p>
        ) : (
          <ul className="divide-y divide-white/5">
            {state.jobs.slice(0, 8).map((j) => (
              <li key={j.job_id} className="flex items-center gap-3 py-3 text-sm">
                <JobBadge status={j.status} />
                <span className="font-mono text-xs text-mute truncate">{j.job_id.slice(0, 12)}</span>
                <span className="text-mist truncate flex-1">{j.video_name}</span>
                <span className="text-xs text-mute shrink-0">
                  {j.frame_count} 帧 · {(j.duration || 0).toFixed(1)}s
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      {/* 快捷入口 */}
      <section className="grid sm:grid-cols-3 gap-4">
        <QuickLink href="/analyze/" title="AI 视频分析" desc="三阶段流水线,本地运行" />
        <QuickLink href="/agent/" title="Agent 对话" desc="自然语言指挥多 key 路由" />
        <QuickLink href="/surveillance/" title="监控分析" desc="RTSP 实时流 + 关键物品检测" />
      </section>

      {state.lastError && (
        <Card className="p-4 border-danger/40">
          <p className="text-sm text-danger">
            <span className="font-medium">后端连接异常:</span> {state.lastError}
            <span className="text-mute ml-2">— 检查后端服务是否启动于 {apiUrl("/api/health")}</span>
          </p>
        </Card>
      )}
    </div>
  );
}

/* ---------- 子组件 ---------- */

type StatTone = "ok" | "warn" | "danger" | "mute";

const TONE_DOT: Record<StatTone, string> = {
  ok: "bg-ok",
  warn: "bg-warn",
  danger: "bg-danger",
  mute: "bg-mute",
};

function StatCard({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: string;
  hint: string;
  tone: StatTone;
}) {
  return (
    <Card interactive tone="strong" className="p-5">
      <div className="flex items-center gap-2 mb-2">
        <span className={cn("w-2 h-2 rounded-full", TONE_DOT[tone])} />
        <span className="text-xs uppercase tracking-wider text-mute">{label}</span>
      </div>
      <div className="text-2xl font-black text-white">{value}</div>
      <div className="text-xs text-mute mt-1.5 truncate">{hint}</div>
    </Card>
  );
}

function CapTile({ label, on }: { label: string; on: boolean }) {
  return (
    <div
      className={cn(
        "rounded-card-sm px-3 py-2.5 border",
        on ? "border-accent/30 bg-accent/5" : "border-white/5 bg-white/2",
      )}
    >
      <div className="flex items-center gap-1.5">
        <span className={cn("w-1.5 h-1.5 rounded-full", on ? "bg-ok" : "bg-mute/40")} />
        <span className={cn("text-xs font-medium", on ? "text-mist" : "text-mute/50")}>
          {label}
        </span>
      </div>
    </div>
  );
}

function JobBadge({ status }: { status: JobSummary["status"] }) {
  const map: Record<JobSummary["status"], string> = {
    pending: "bg-warn/20 text-warn",
    running: "bg-accent/20 text-accent",
    done: "bg-ok/20 text-ok",
    failed: "bg-danger/20 text-danger",
    canceled: "bg-mute/20 text-mute",
  };
  return (
    <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium shrink-0", map[status])}>
      {status}
    </span>
  );
}

function QuickLink({ href, title, desc }: { href: string; title: string; desc: string }) {
  return (
    <a href={href} className="block group">
      <Card interactive className="p-5 h-full">
        <div className="text-base font-semibold text-white group-hover:text-accent transition-colors">
          {title}
        </div>
        <div className="text-xs text-mute mt-1.5">{desc}</div>
      </Card>
    </a>
  );
}
