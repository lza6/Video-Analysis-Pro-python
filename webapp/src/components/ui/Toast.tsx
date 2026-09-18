"use client";

import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { cn } from "@/lib/utils";

/**
 * 全局轻量 Toast(v10.6.0 P1-6)。
 * 统一操作反馈:成功/信息/错误,4s 自动消失,aria-live polite。
 * 用法: const { toast } = useToast(); toast("已保存", "success");
 */
type ToastKind = "info" | "success" | "error";

interface ToastItem {
  id: number;
  kind: ToastKind;
  text: string;
}

interface ToastApi {
  toast: (text: string, kind?: ToastKind) => void;
}

const ToastCtx = createContext<ToastApi>({ toast: () => {} });

export const useToast = () => useContext(ToastCtx);

const KIND_CLS: Record<ToastKind, string> = {
  info: "glass-chip text-mist border-white/10",
  success: "bg-ok/15 text-ok border-ok/30",
  error: "bg-danger/15 text-danger border-danger/30",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const nextId = useRef(1);

  const toast = useCallback((text: string, kind: ToastKind = "info") => {
    const id = nextId.current++;
    setToasts((prev) => [...prev, { id, kind, text }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  return (
    <ToastCtx.Provider value={{ toast }}>
      {children}
      <div
        aria-live="polite"
        className="fixed bottom-16 right-4 z-50 flex w-full max-w-sm flex-col gap-2 px-4"
      >
        {toasts.map((t) => (
          <div
            key={t.id}
            className={cn(
              "rounded-card-sm border px-4 py-2.5 text-sm shadow-lg backdrop-blur",
              KIND_CLS[t.kind],
            )}
          >
            {t.text}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}
