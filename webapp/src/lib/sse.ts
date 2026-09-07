"use client";
/**
 * 基于 fetch 的 SSE 客户端(不用 EventSource,以便支持 Last-Event-ID / Authorization 头)。
 *
 * 解析 text/event-stream:
 *   `event:` 行设类型,`data:` 行设载荷,`id:` 行设事件 seq,
 *   空行分帧(触发 onEvent)。
 *   以冒号开头的行(如 `: keepalive`)是注释,忽略。
 *
 * v10.2.0:断线续连。
 *   - 解析 `id:` 行,记下 lastEventId
 *   - 重连时带 `Last-Event-ID` header,服务端从环形缓冲重放漏掉的事件
 *   - 指数退避:1s/2s/4s/8s/16s/30s(上限 30s)
 *   - 收到 done/error 事件后流正常结束,不再重连;异常断开(网络中断、
 *     reader 提前 done)才重连
 *   - 外部 AbortSignal abort 时立即停止,不重连
 *
 * 同时导出 useSSE React hook,封装连接生命周期 + 自动重连。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { apiUrl } from "./api";
import type { SSEType } from "./types";

export interface SSEMessage {
  type: SSEType;
  data: unknown;
  /** 该事件的 seq(后端 put_event 分配,用于断线续连)。 */
  id?: string;
}

export interface StreamOptions {
  signal?: AbortSignal;
  onEvent: (msg: SSEMessage) => void;
  onError?: (err: unknown) => void;
  onClose?: () => void;
  /** 自动重连(默认 false,保持向后兼容)。useSSE hook 默认 true。 */
  autoReconnect?: boolean;
  /** 重连前回调(通知 UI 显示"重连中…"等)。 */
  onReconnect?: (attempt: number, delayMs: number) => void;
}

// 指数退避序列:1s/2s/4s/8s/16s/30s(上限 30s,6 档后固定 30s)。
const RECONNECT_BACKOFF_MS = [1000, 2000, 4000, 8000, 16000, 30000];

/**
 * 流式订阅 SSE。单次 fetch + ReadableStream 解析,可选自动重连。
 *
 * 用法(命令式,适合需要 AbortController 显式控制的场景):
 *   await streamSSE("/api/jobs/abc/stream", {
 *     signal: ac.signal,
 *     onEvent: (msg) => console.log(msg.type, msg.data, msg.id),
 *   });
 */
export async function streamSSE(path: string, opts: StreamOptions): Promise<void> {
  const {
    signal,
    onEvent,
    onError,
    onClose,
    autoReconnect = false,
    onReconnect,
  } = opts;

  let lastEventId: string | null = null;
  let stop = false; // 收到 done/error 终结事件,不再重连
  let attempt = 0;

  while (!stop && !(signal?.aborted ?? false)) {
    let connectionClosed = false; // 本次连接正常结束(reader done)
    try {
      const headers: Record<string, string> = { Accept: "text/event-stream" };
      if (lastEventId) headers["Last-Event-ID"] = lastEventId;

      const res = await fetch(apiUrl(path), {
        headers,
        signal,
        cache: "no-store",
      });
      if (!res.ok || !res.body) {
        throw new Error(`SSE 连接失败: ${res.status} ${res.statusText}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      // 帧状态(空行触发 onEvent 后重置)
      let frameType = "message";
      let frameId: string | null = null;
      let frameData: string[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          connectionClosed = true;
          break;
        }
        buffer += decoder.decode(value, { stream: true });

        let nl: number;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nl).replace(/\r$/, "");
          buffer = buffer.slice(nl + 1);

          if (line === "") {
            // 帧结束:有 data 才触发(SSE 规范允许空帧作为注释)
            if (frameData.length > 0) {
              const raw = frameData.join("\n");
              let parsed: unknown = raw;
              try {
                parsed = JSON.parse(raw);
              } catch {
                /* 非 JSON data,原样传 */
              }
              if (frameId !== null) lastEventId = frameId;
              onEvent({
                type: frameType as SSEType,
                data: parsed,
                id: frameId ?? undefined,
              });
              // 收到终结事件,标记不再重连(流正常结束)
              if (frameType === "done" || frameType === "error") {
                stop = true;
              }
            }
            // 重置帧状态(SSE 规范:event/data/id 在空行后归零)
            frameType = "message";
            frameId = null;
            frameData = [];
            continue;
          }
          if (line.startsWith(":")) continue; // 注释/keepalive

          if (line.startsWith("id:")) {
            frameId = line.slice(3).trim();
          } else if (line.startsWith("event:")) {
            frameType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            const d = line.slice(5).replace(/^ /, "");
            frameData.push(d);
          }
          // 其他字段(field:)按 SSE 规范忽略
        }
      }

      // 连接正常结束(reader done):若已收到终结事件或不开重连,则收尾
      if (stop || !autoReconnect) {
        onClose?.();
        return;
      }
      // 否则:服务端关闭但未发 done/error → 视为异常断开,重连
    } catch (err) {
      if ((err as Error)?.name === "AbortError") {
        // 外部主动 abort,不重连
        onClose?.();
        return;
      }
      if (!autoReconnect) {
        onError?.(err);
        return;
      }
      // 网络错误,继续重连循环
    }

    if (stop || (signal?.aborted ?? false)) {
      onClose?.();
      return;
    }

    // 指数退避等待(可被 signal abort 提前打断)
    const delayMs = RECONNECT_BACKOFF_MS[
      Math.min(attempt, RECONNECT_BACKOFF_MS.length - 1)
    ];
    onReconnect?.(attempt + 1, delayMs);
    await new Promise<void>((resolve) => {
      const t = setTimeout(resolve, delayMs);
      signal?.addEventListener(
        "abort",
        () => {
          clearTimeout(t);
          resolve();
        },
        { once: true },
      );
    });
    if (signal?.aborted ?? false) {
      onClose?.();
      return;
    }
    attempt += 1;
    void connectionClosed; // 标记消费(本连接是否正常结束,仅用于调试)
  }
  onClose?.();
}

// ============================ useSSE hook ============================

export interface UseSSEOptions {
  autoReconnect?: boolean;
  onReconnect?: (attempt: number, delayMs: number) => void;
  onError?: (err: unknown) => void;
  onClose?: () => void;
}

export interface UseSSEReturn {
  /** 当前是否处于活跃连接中(重连等待期视为 false)。 */
  connected: boolean;
  /** 主动断开(abort),触发后不再重连。 */
  stop: () => void;
}

/**
 * React hook 封装 streamSSE。url 为 null 时不连接。
 *
 * onEvent 用 ref 持有,避免每次 render 重启连接;只有 url 变化才重连。
 * 默认 autoReconnect=true(与 streamSSE 默认 false 相反,hook 场景更需要自动重连)。
 *
 * 用法:
 *   const { connected, stop } = useSSE(
 *     `/api/jobs/${jid}/stream`,
 *     (type, data, id) => { ... },
 *   );
 */
export function useSSE(
  url: string | null,
  onEvent: (type: SSEType, data: unknown, id?: string) => void,
  options?: UseSSEOptions,
): UseSSEReturn {
  const [connected, setConnected] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const onEventRef = useRef(onEvent);
  const optsRef = useRef(options);
  onEventRef.current = onEvent;
  optsRef.current = options;

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setConnected(false);
  }, []);

  useEffect(() => {
    if (!url) {
      setConnected(false);
      return;
    }
    const ac = new AbortController();
    abortRef.current = ac;
    setConnected(true);
    let cancelled = false;

    streamSSE(url, {
      signal: ac.signal,
      autoReconnect: optsRef.current?.autoReconnect ?? true,
      onReconnect: optsRef.current?.onReconnect,
      onError: optsRef.current?.onError,
      onClose: () => {
        if (!cancelled) setConnected(false);
      },
      onEvent: (msg) => {
        if (cancelled) return;
        onEventRef.current(msg.type, msg.data, msg.id);
      },
    }).catch(() => {
      // streamSSE 内部已通过 onError 回调上报错误,这里防 unhandled rejection
    });

    return () => {
      cancelled = true;
      ac.abort();
    };
  }, [url]);

  return { connected, stop };
}
