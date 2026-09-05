/**
 * 基于 fetch 的 SSE 客户端(不用 EventSource,以便未来支持 Authorization 头)。
 *
 * 解析 text/event-stream:`event:` 行设类型,`data:` 行设载荷,空行分帧。
 * 以冒号开头的行(如 `: keepalive`)是注释,忽略。
 */
import { apiUrl } from "./api";
import type { SSEType } from "./types";

export interface SSEMessage {
  type: SSEType;
  data: unknown;
}

export interface StreamOptions {
  signal?: AbortSignal;
  onEvent: (msg: SSEMessage) => void;
  onError?: (err: unknown) => void;
  onClose?: () => void;
}

export async function streamSSE(path: string, opts: StreamOptions): Promise<void> {
  const { signal, onEvent, onError, onClose } = opts;
  try {
    const res = await fetch(apiUrl(path), {
      headers: { Accept: "text/event-stream" },
      signal,
      cache: "no-store",
    });
    if (!res.ok || !res.body) {
      throw new Error(`SSE 连接失败: ${res.status} ${res.statusText}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let currentType = "message";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // 按行处理,保留未完成的尾行在 buffer
      let nl: number;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).replace(/\r$/, "");
        buffer = buffer.slice(nl + 1);

        if (line === "") {
          // 帧结束(空行分隔),重置类型
          currentType = "message";
          continue;
        }
        if (line.startsWith(":")) continue; // 注释/keepalive

        if (line.startsWith("event:")) {
          currentType = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          const raw = line.slice(5).trim();
          if (!raw) continue;
          let parsed: unknown = raw;
          try {
            parsed = JSON.parse(raw);
          } catch {
            /* 非 JSON data,原样传 */
          }
          onEvent({ type: currentType as SSEType, data: parsed });
        }
      }
    }
    onClose?.();
  } catch (err) {
    if ((err as Error)?.name === "AbortError") {
      onClose?.();
      return;
    }
    onError?.(err);
  }
}
