/**
 * API 基址 + fetch 封装。
 *
 * v10.6.0 (P1-6): 统一超时/取消(AbortController) + GET/DELETE 幂等重试。
 * 此前无超时——后端挂起时前端无限等待(黑盒)。现在默认 15s 超时,
 * 超时抛 ApiError(408);GET/DELETE 幂等请求网络失败最多重试 1 次。
 *
 * 开发态: NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000(.env.local),跨域带 CORS。
 * 生产态: 静态导出由 FastAPI 同源挂载,留空走相对路径。
 */
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/$/, "");

export function apiUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const DEFAULT_TIMEOUT_MS = 15_000;

/** fetch 包装:统一超时(AbortController)。超时抛 ApiError(408)。 */
export async function fetchWithTimeout(
  path: string,
  init?: RequestInit,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(apiUrl(path), { ...init, signal: ctrl.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new ApiError(
        408,
        `请求超时(${Math.round(timeoutMs / 1000)}s): ${path}`,
      );
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

/** 幂等请求网络失败重试(HTTP 错误码不重试)。 */
async function withRetry(fn: () => Promise<Response>, retries: number): Promise<Response> {
  let lastErr: unknown;
  for (let i = 0; i <= retries; i++) {
    try {
      return await fn();
    } catch (e) {
      lastErr = e;
      if (i < retries) await new Promise((r) => setTimeout(r, 400));
    }
  }
  throw lastErr;
}

async function parseError(res: Response): Promise<ApiError> {
  let detail: unknown = null;
  try {
    detail = await res.json();
  } catch {
    /* 非 JSON 响应体 */
  }
  const msg =
    (detail && typeof detail === "object" && "detail" in detail
      ? String((detail as { detail: unknown }).detail)
      : null) ?? `${res.status} ${res.statusText}`;
  return new ApiError(res.status, msg, detail);
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await withRetry(
    () => fetchWithTimeout(path, { cache: "no-store" }),
    1,
  );
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

export async function apiPostJson<T>(path: string, body: unknown, method: "POST" | "PUT" = "POST"): Promise<T> {
  const res = await fetchWithTimeout(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

/** PUT JSON(用于更新配置类端点)。 */
export async function apiPutJson<T>(path: string, body: unknown): Promise<T> {
  return apiPostJson<T>(path, body, "PUT");
}

/** POST multipart(config 字段为 JSON 字符串,与后端 Form(...) 对齐)。 */
export async function apiPostMultipart<T>(
  path: string,
  fields: Record<string, string>,
  file?: File | null,
): Promise<T> {
  const fd = new FormData();
  for (const [k, v] of Object.entries(fields)) fd.append(k, v);
  if (file) fd.append("file", file);
  const res = await fetchWithTimeout(path, { method: "POST", body: fd });
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await withRetry(() => fetchWithTimeout(path, { method: "DELETE" }), 1);
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}
