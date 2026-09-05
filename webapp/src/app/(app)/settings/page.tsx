"use client";

import { useEffect, useState } from "react";
import { apiGet, apiDelete, apiPostJson, apiPutJson } from "@/lib/api";
import type { ProviderConfigOut, Preset, PromptTemplate, TestResult } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

export default function SettingsPage() {
  const [cfg, setCfg] = useState<ProviderConfigOut | null>(null);
  const [apiUrl, setApiUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [testing, setTesting] = useState(false);
  const [saved, setSaved] = useState(false);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [prompts, setPrompts] = useState<PromptTemplate[]>([]);

  const refresh = () => {
    apiGet<ProviderConfigOut>("/api/config").then((c) => {
      setCfg(c);
      setApiUrl(c.api_url);
      setModel(c.model_name);
    });
    apiGet<Preset[]>("/api/config/presets").then(setPresets);
    apiGet<PromptTemplate[]>("/api/config/prompts").then(setPrompts);
  };
  useEffect(() => { refresh(); }, []);

  const save = async () => {
    setSaved(false);
    await apiPutJson<unknown>("/api/config", {
      client_type: 1,
      api_url: apiUrl,
      api_key: apiKey || undefined,
      model_name: model,
    });
    setApiKey("");
    setSaved(true);
    refresh();
    setTimeout(() => setSaved(false), 2000);
  };

  const test = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const r = await apiPostJson<TestResult>("/api/config/test", {
        api_url: apiUrl,
        api_key: apiKey,
        model,
      });
      setTestResult(r);
    } catch (e) {
      setTestResult({ ok: false, error: String(e) });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">设置</h1>
        <p className="text-sm text-mute mt-1.5">
          配置 LLM Provider。凭据走 OS 密钥环(不可用时降级 ini + 告警)。
        </p>
      </header>

      {cfg && (
        <Card className="p-5 space-y-4">
          <div className="flex items-center gap-2 text-xs">
            <span className={cfg.has_key ? "text-ok" : "text-warn"}>
              {cfg.has_key ? "● 已配置凭据" : "○ 未配置凭据"}
            </span>
            {cfg.nvidia_keys > 0 && (
              <span className="text-accent">· NVIDIA 多 key 路由({cfg.nvidia_keys} keys,从 .env 读)</span>
            )}
            <span className="text-mute">
              · keyring {cfg.keyring_available ? "可用" : "不可用"}
            </span>
          </div>

          <div>
            <label className="block text-xs text-mute mb-1.5">API URL</label>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="https://integrate.api.nvidia.com/v1 或 https://api.openai.com/v1"
              className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-mute mb-1.5">
              API Key{cfg.has_key && " (已存,留空则不修改)"}
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={cfg.has_key ? "••••••••" : "sk-..."}
              className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-mute mb-1.5">模型名</label>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
            />
          </div>

          <div className="flex gap-2">
            <Button onClick={save}>保存</Button>
            <Button variant="glass" onClick={test} disabled={testing || (!apiKey && !cfg.has_key)}>
              {testing ? "测试中…" : "测活性"}
            </Button>
            {saved && <span className="text-xs text-ok self-center">已保存</span>}
          </div>

          {testResult && (
            <div className={testResult.ok ? "text-ok text-sm" : "text-danger text-sm"}>
              {testResult.ok
                ? `✓ 连接成功 · ${testResult.count} 个模型可用${
                    testResult.via === "router" ? ` · via ${testResult.key_used}` : ""
                  }`
                : `✗ ${testResult.error}`}
            </div>
          )}
        </Card>
      )}

      {presets.length > 0 && (
        <Card className="p-5">
          <h3 className="text-sm font-medium text-white mb-3">API 预设</h3>
          <div className="space-y-2">
            {presets.map((p) => (
              <div key={p.name} className="flex items-center justify-between glass-chip rounded-card-sm px-3 py-2">
                <div>
                  <span className="text-sm text-white">{p.name}</span>
                  <span className="text-[11px] text-mute ml-2 font-mono">{p.api_url}</span>
                </div>
                <Button
                  size="sm"
                  variant="chip"
                  onClick={() => apiDelete(`/api/config/presets/${p.name}`).then(refresh)}
                >
                  删除
                </Button>
              </div>
            ))}
          </div>
        </Card>
      )}

      {prompts.length > 0 && (
        <Card className="p-5">
          <h3 className="text-sm font-medium text-white mb-3">提示词模板</h3>
          <div className="space-y-2">
            {prompts.map((p) => (
              <div key={p.name} className="glass-chip rounded-card-sm px-3 py-2">
                <div className="text-sm text-white">{p.name}</div>
                <div className="text-[11px] text-mute mt-1 line-clamp-2">{p.content}</div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
