"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiGet } from "@/lib/api";
import { streamSSE } from "@/lib/sse";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";
import type { DesktopLogEntry, DesktopDiagnosticsResult } from "@/types/vapDesktop";

interface LogEntry {
  ts: number;
  level: string;
  logger: string;
  msg: string;
}

const LEVELS = ["", "info", "warning", "error", "debug"];

export default function LogsPage() {
  // ---- 来源状态:后端(SSE + /api/logs)与主进程(IPC)合并展示 ----
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [live, setLive] = useState(false);
  const [filter, setFilter] = useState("");
  const [levelFilter, setLevelFilter] = useState<string>("");
  const [desktopConnected, setDesktopConnected] = useState(false);
  const [exporting, setExporting] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const desktopApi = typeof window !== "undefined" ? window.vapDesktop : undefined;
  const setLogsRef = useRef(setLogs);

  // 向后端拉初始日志
  const loadInitial = useCallback(() => {
    apiGet<{ logs: LogEntry[] }>("/api/logs?limit=200").then((r) => setLogs(r.logs));
  }, []);

  // 主进程(黑匣子)日志:订阅 append + 查询,合并进同一列表
  const pushDesktop = useCallback((e: DesktopLogEntry) => {
    const item: LogEntry = {
      ts: e.ts,
      level: e.level === "warn" ? "warning" : e.level,
      logger: `desktop:${e.source}`,
      msg: e.msg,
    };
    setLogsRef.current((l) => [...l.slice(-899), item]);
  }, []);

  // 同步 ref(在 effect 内,避免 render 期访问 ref)
  useEffect(() => {
    setLogsRef.current = setLogs;
  }, []);

  useEffect(() => {
    loadInitial();
    if (!desktopApi) return;
    let disposed = false;
    void desktopApi
      .logsSubscribe()
      .then(() => {
        if (disposed) return;
        setDesktopConnected(true);
      })
      .catch(() => setDesktopConnected(false));
    desktopApi.onLogsAppend((entry: DesktopLogEntry) => {
      if (!disposed) pushDesktop(entry);
    });
    // 初始从主进程拉一段(前 200 条,合并去重由渲染层做)
    void desktopApi.logsQuery({ limit: 200 }).then((r) => {
      if (!disposed && r.logs) {
        for (const e of r.logs) pushDesktop(e);
      }
    });
    return () => {
      disposed = true;
    };
  }, [desktopApi, loadInitial, pushDesktop]);

  // 后端 SSE(与已有行为一致)
  const toggleLive = () => {
    if (live) {
      abortRef.current?.abort();
      setLive(false);
      return;
    }
    const ac = new AbortController();
    abortRef.current = ac;
    setLive(true);
    streamSSE("/api/logs/stream", {
      signal: ac.signal,
      onEvent: (msg) => {
        if (msg.type === "log") {
          const entry = msg.data as LogEntry;
          setLogs((l) => [...l.slice(-499), entry]);
        }
      },
      onError: () => setLive(false),
    });
  };

  // 自动滚底
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs.length]);

  const filtered = logs.filter((l) => {
    if (levelFilter && l.level !== levelFilter) return false;
    if (filter && !l.msg.includes(filter) && !l.logger.includes(filter)) return false;
    return true;
  });

  // 复制(去重合并后文本)
  const copyVisible = () => {
    const text = filtered
      .map((l) => `[${new Date(l.ts * 1000).toISOString()}] [${l.logger}] ${l.msg}`)
      .join("\n");
    if (navigator.clipboard) {
      void navigator.clipboard.writeText(text);
    }
  };

  // 清空(仅前端视图,不清后端)
  const clearView = () => setLogs([]);

  // 导出诊断包:优先走 Electron IPC(返回 zip Buffer → Blob 下载);
  // 无 Electron 时降级导出当前可见文本为 txt。
  const exportDiagnostics = async () => {
    setExporting(true);
    try {
      if (desktopApi?.collectDiagnostics) {
        const res = (await desktopApi.collectDiagnostics()) as DesktopDiagnosticsResult;
        if (res.ok && res.buffer) {
          const blob = new Blob([res.buffer as unknown as BlobPart], {
            type: "application/zip",
          });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `tingfeng-hermes-diagnostics-${Date.now()}.zip`;
          a.click();
          URL.revokeObjectURL(url);
          return;
        }
        throw new Error(res.error || "诊断收集失败");
      }
      // 降级:导出可见文本
      const blob = new Blob(
        [filtered.map((l) => `[${l.logger}] ${l.msg}`).join("\n")],
        { type: "text/plain" }
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `logs-${Date.now()}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("[logs] export failed", e);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-black tracking-tight text-white">系统日志</h1>
          <p className="text-sm text-mute mt-1.5">
            后端运行日志实时流,含分析各阶段输出。含 Electron 主进程黑匣子日志。
          </p>
        </div>
        <Button
          onClick={toggleLive}
          /* v10.3.1 (D4 修复):此前 disabled={live} 导致开启后按钮永久
             禁用,无法点"停止"。现在始终可点,按钮即开关。 */
          className={cn(
            "glass-chip glass-edge rounded-full px-4 py-2 text-sm flex items-center gap-2",
            live ? "text-ok" : "text-mist",
          )}
        >
          <span className={cn("w-2 h-2 rounded-full", live ? "bg-ok animate-[pulse-glow_1.5s]" : "bg-mute")} />
          {live ? "实时(点击停止)" : "启动实时"}
        </Button>
      </header>

      {/* 主进程黑匣子连接状态 */}
      <div className="flex flex-wrap items-center gap-3 text-xs text-mute">
        <span className={cn(
          "inline-flex items-center gap-1.5",
          desktopConnected ? "text-ok" : (desktopApi ? "text-warn" : "text-mute/60")
        )}>
          <span className={cn("w-2 h-2 rounded-full",
            desktopConnected ? "bg-ok animate-[pulse-glow_2s_ease-in-out_infinite]" :
            (desktopApi ? "bg-warn" : "bg-mute/60"))} />
          {desktopConnected ? "主进程黑匣子已连接" :
            (desktopApi ? "主进程日志未连接(重新加载页面重试)" : "浏览器模式(无主进程日志)")}
        </span>
        {desktopApi && (
          <button
            onClick={exportDiagnostics}
            disabled={exporting}
            className="glass-chip glass-edge rounded-full px-4 py-2 text-sm text-mist hover:text-white transition-colors disabled:opacity-40"
          >
            {exporting ? "打包中…" : "导出诊断包"}
          </button>
        )}
      </div>

      <Card className="p-4 space-y-3">
        <div className="flex flex-wrap gap-2">
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="关键字过滤…"
            className="flex-1 min-w-[200px] rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
          />
          <select
            value={levelFilter}
            onChange={(e) => setLevelFilter(e.target.value)}
            className="rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
          >
            {LEVELS.map((lv) => (
              <option key={lv || "all"} value={lv}>
                {lv ? lv : "全部级别"}
              </option>
            ))}
          </select>
          <button onClick={loadInitial} className="glass-chip rounded-full px-4 py-2 text-sm text-mist">
            刷新
          </button>
          <button onClick={copyVisible} className="glass-chip rounded-full px-4 py-2 text-sm text-mist">
            复制
          </button>
          <button onClick={clearView} className="glass-chip rounded-full px-4 py-2 text-sm text-danger">
            清空
          </button>
        </div>

        <div ref={ref} className="h-[55vh] overflow-y-auto font-mono text-[11px] leading-relaxed space-y-0.5">
          {filtered.length === 0 ? (
            <p className="text-mute/60">暂无日志</p>
          ) : (
            filtered.map((l, i) => (
              <p
                key={i}
                className={
                  l.level === "error" ? "text-danger" :
                  l.level === "warning" ? "text-warn" : "text-mute"
                }
              >
                <span className="text-mute/40">[{l.logger}]</span> {l.msg}
              </p>
            ))
          )}
        </div>
        <p className="text-[11px] text-mute">{filtered.length} 条 · 视图环形缓冲 1000</p>
      </Card>
    </div>
  );
}