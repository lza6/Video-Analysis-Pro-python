"use client";

import { useCallback, useEffect, useState } from "react";
import { apiDelete, apiGet, apiPostJson, apiPutJson } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card, Chip } from "@/components/ui/Card";
import type {
  ActiveProviderResponse,
  ProviderPresetOut,
  ProviderType,
} from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Provider 管理页(cc-switch 风格):preset 列表 + 新增/编辑/删除/激活 + 活跃高亮。
 * api_key 永不回传,只在表单输入新值时写入(留空=不改)。
 */

const PROVIDER_TYPES: { value: ProviderType; placeholder: string }[] = [
  { value: "openai", placeholder: "https://api.openai.com/v1" },
  { value: "anthropic", placeholder: "https://api.anthropic.com" },
  { value: "gemini", placeholder: "https://generativelanguage.googleapis.com/v1beta" },
  { value: "ollama", placeholder: "http://127.0.0.1:11434" },
  { value: "custom", placeholder: "https://your-llm-endpoint/v1" },
];

interface FormState {
  name: string;
  provider_type: ProviderType;
  base_url: string;
  model: string;
  api_key: string;
}

const EMPTY_FORM: FormState = {
  name: "",
  provider_type: "custom",
  base_url: "",
  model: "",
  api_key: "",
};

export default function ProvidersPage() {
  const [presets, setPresets] = useState<ProviderPresetOut[]>([]);
  const [active, setActive] = useState<ActiveProviderResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<ProviderPresetOut | null>(null);

  const refresh = useCallback(async () => {
    const [listR, activeR] = await Promise.allSettled([
      apiGet<ProviderPresetOut[]>("/api/providers"),
      apiGet<ActiveProviderResponse>("/api/providers/active"),
    ]);
    if (listR.status === "fulfilled") setPresets(listR.value);
    if (listR.status === "rejected") {
      setError(listR.reason instanceof Error ? listR.reason.message : String(listR.reason));
    }
    if (activeR.status === "fulfilled") setActive(activeR.value);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const openCreate = () => {
    setForm(EMPTY_FORM);
    setEditingId(null);
    setShowForm(true);
  };

  const openEdit = (p: ProviderPresetOut) => {
    setForm({
      name: p.name,
      provider_type: (p.provider_type as ProviderType) || "custom",
      base_url: p.base_url,
      model: p.model,
      api_key: "",
    });
    setEditingId(p.id);
    setShowForm(true);
  };

  const save = async () => {
    if (!form.name.trim()) {
      setError("name 不能为空");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const body = {
        name: form.name.trim(),
        provider_type: form.provider_type,
        base_url: form.base_url.trim(),
        model: form.model.trim(),
        api_key: form.api_key || undefined,
      };
      if (editingId) {
        await apiPutJson<ProviderPresetOut>(`/api/providers/${editingId}`, body);
      } else {
        await apiPostJson<ProviderPresetOut>("/api/providers", body);
      }
      setShowForm(false);
      setEditingId(null);
      setForm(EMPTY_FORM);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const activate = async (p: ProviderPresetOut) => {
    try {
      await apiPostJson<ProviderPresetOut>(`/api/providers/${p.id}/activate`, {});
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const remove = async (p: ProviderPresetOut) => {
    try {
      await apiDelete<{ ok: boolean }>(`/api/providers/${p.id}`);
      setConfirmDelete(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const placeholder =
    PROVIDER_TYPES.find((t) => t.value === form.provider_type)?.placeholder ?? "";

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black tracking-tight text-white">
            提供<span className="text-gradient">商</span>
          </h1>
          <p className="text-sm text-mute mt-1.5">
            多 provider 预设(cc-switch 风格),api_key 走 OS 密钥环,永不回传明文。
          </p>
        </div>
        <Button onClick={openCreate}>新增 provider</Button>
      </header>

      {/* 当前活跃高亮 */}
      {active && active.id && (
        <Card tone="strong" className="p-4 border-accent/40">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="w-2 h-2 rounded-full bg-ok animate-[pulse-glow_1.5s]" />
            <span className="text-xs uppercase tracking-wider text-accent">活跃 provider</span>
            <span className="text-sm font-bold text-white">{active.name}</span>
            {active.has_key ? (
              <Chip className="text-ok">已配 key</Chip>
            ) : (
              <Chip className="text-warn">无 key</Chip>
            )}
          </div>
        </Card>
      )}

      {error && (
        <Card className="p-4 border-danger/40">
          <p className="text-sm text-danger">{error}</p>
        </Card>
      )}

      {/* preset 列表 */}
      {presets.length === 0 ? (
        <Card className="p-8">
          <p className="text-sm text-mute/60 text-center">
            暂无 provider,点击右上角“新增 provider”创建。
          </p>
        </Card>
      ) : (
        <div className="grid sm:grid-cols-2 gap-4">
          {presets.map((p) => (
            <Card
              key={p.id}
              tone={p.is_active ? "strong" : "default"}
              className={cn("p-5 space-y-3", p.is_active && "border-accent/40")}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="text-white font-bold truncate">{p.name}</h3>
                    {p.is_active && <Chip className="text-accent">活跃</Chip>}
                  </div>
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <Chip className="text-mute">{p.provider_type}</Chip>
                    {p.has_key ? (
                      <Chip className="text-ok">● key</Chip>
                    ) : (
                      <Chip className="text-warn">○ 无 key</Chip>
                    )}
                    {!p.enabled && <Chip className="text-mute">已禁用</Chip>}
                  </div>
                </div>
              </div>

              <div className="text-[11px] text-mute font-mono space-y-0.5 break-all">
                <div>{p.base_url || "(无 base_url)"}</div>
                <div className="text-mist">{p.model || "(无 model)"}</div>
              </div>

              <div className="flex gap-2 flex-wrap">
                {!p.is_active && (
                  <Button size="sm" variant="chip" onClick={() => void activate(p)}>
                    设为活跃
                  </Button>
                )}
                <Button size="sm" variant="glass" onClick={() => openEdit(p)}>
                  编辑
                </Button>
                <Button
                  size="sm"
                  variant="danger"
                  onClick={() => setConfirmDelete(p)}
                  disabled={p.is_active}
                >
                  删除
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* 新增/编辑表单弹窗 */}
      {showForm && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          onClick={() => setShowForm(false)}
        >
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
          <Card tone="strong" className="relative w-full max-w-lg p-6 space-y-4">
            <div onClick={(e: { stopPropagation: () => void }) => e.stopPropagation()}>
              <h3 className="text-lg font-bold text-white mb-4">
                {editingId ? "编辑 provider" : "新增 provider"}
              </h3>

              <div className="space-y-3">
                <FormRow label="名称">
                  <input
                    type="text"
                    value={form.name}
                    onChange={(e) => setForm({ ...form, name: e.target.value })}
                    placeholder="my-nvidia"
                    className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
                  />
                </FormRow>

                <FormRow label="provider 类型">
                  <select
                    value={form.provider_type}
                    onChange={(e) =>
                      setForm({ ...form, provider_type: e.target.value as ProviderType })
                    }
                    className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
                  >
                    {PROVIDER_TYPES.map((t) => (
                      <option key={t.value} value={t.value}>
                        {t.value}
                      </option>
                    ))}
                  </select>
                </FormRow>

                <FormRow label="base_url">
                  <input
                    type="text"
                    value={form.base_url}
                    onChange={(e) => setForm({ ...form, base_url: e.target.value })}
                    placeholder={placeholder}
                    className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white font-mono"
                  />
                </FormRow>

                <FormRow label="model">
                  <input
                    type="text"
                    value={form.model}
                    onChange={(e) => setForm({ ...form, model: e.target.value })}
                    placeholder="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
                    className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white font-mono"
                  />
                </FormRow>

                <FormRow label={editingId ? "api_key (留空不改)" : "api_key (可选,Ollama 可空)"}>
                  <input
                    type="password"
                    value={form.api_key}
                    onChange={(e) => setForm({ ...form, api_key: e.target.value })}
                    placeholder={editingId && presets.find((p) => p.id === editingId)?.has_key ? "••••••••" : "sk-..."}
                    className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white font-mono"
                  />
                </FormRow>
              </div>

              <div className="flex justify-end gap-2 mt-5">
                <Button size="sm" variant="glass" onClick={() => setShowForm(false)}>
                  取消
                </Button>
                <Button size="sm" onClick={() => void save()} loading={saving}>
                  {editingId ? "保存" : "创建"}
                </Button>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* 删除确认弹窗 */}
      {confirmDelete && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          onClick={() => setConfirmDelete(null)}
        >
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
          <Card tone="strong" className="relative w-full max-w-md p-6 space-y-4">
            <div onClick={(e: { stopPropagation: () => void }) => e.stopPropagation()}>
              <h3 className="text-lg font-bold text-white">确认删除 provider?</h3>
              <p className="text-sm text-mute mt-1">
                将删除预设 <span className="text-mist font-medium">{confirmDelete.name}</span>,
                并 best-effort 清理 keyring 中的对应 key。不可恢复。
              </p>
              <div className="flex justify-end gap-2 mt-4">
                <Button size="sm" variant="glass" onClick={() => setConfirmDelete(null)}>
                  取消
                </Button>
                <Button
                  size="sm"
                  variant="danger"
                  onClick={() => void remove(confirmDelete)}
                >
                  确认删除
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

function FormRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs text-mute mb-1.5">{label}</label>
      {children}
    </div>
  );
}
