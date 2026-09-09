/**
 * Electron preload(contextBridge)暴露的桌面 API 类型声明。
 *
 * 渲染层(Next.js)不直接触达 Electron,只有 preload.js 显式 exposeInMainWorld
 * 的 `window.vapDesktop` 可用。此文件让 TS 认识该全局对象。
 *
 * 对应实现:desktop/preload.js(仅当 Electron 壳运行时存在;浏览器/静态导出
 * 场景该全局为 undefined,使用方需判空)。
 */

export interface DesktopLogEntry {
  ts: number;
  source: string;
  level: string;
  msg: string;
}

export interface DesktopLogQueryOpts {
  level?: string;
  search?: string;
  limit?: number;
}

export interface DesktopDiagnosticsResult {
  ok: boolean;
  buffer?: Uint8Array;
  entries?: string[];
  error?: string;
}

export interface VapDesktopApi {
  getAppVersion: () => string;
  platform: string;
  /** 主进程日志查询(LogStore)。返回 { ok, logs: DesktopLogEntry[] }。 */
  logsQuery: (opts: DesktopLogQueryOpts) => Promise<{
    ok: boolean;
    logs?: DesktopLogEntry[];
    error?: string;
  }>;
  /** 订阅主进程日志追加(通过 logs:append IPC 广播)。 */
  logsSubscribe: () => Promise<{ ok: boolean }>;
  onLogsAppend: (cb: (entry: DesktopLogEntry) => void) => void;
  /** 一键收集诊断 zip(主进程 collectDiagnostics)。 */
  collectDiagnostics: () => Promise<DesktopDiagnosticsResult>;
}

declare global {
  interface Window {
    vapDesktop?: VapDesktopApi;
  }
}

export {};