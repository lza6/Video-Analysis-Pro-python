/**
 * API 基址 + fetch 封装。
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
  const res = await fetch(apiUrl(path), { cache: "no-store" });
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

export async function apiPostJson<T>(path: string, body: unknown, method: "POST" | "PUT" = "POST"): Promise<T> {
  const res = await fetch(apiUrl(path), {
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
  const res = await fetch(apiUrl(path), { method: "POST", body: fd });
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path), { method: "DELETE" });
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}
