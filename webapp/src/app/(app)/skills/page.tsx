"use client";

import { useEffect, useState } from "react";
import { apiGet, apiPostJson } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

interface Skill {
  name: string;
  description: string;
  triggers: string[];
  path: string;
  enabled: boolean;
}

export default function SkillsPage() {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [selected, setSelected] = useState<Skill | null>(null);
  const [genText, setGenText] = useState("停车场监控");
  const [genResult, setGenResult] = useState<string | null>(null);

  const refresh = () => {
    apiGet<{ skills: Skill[]; count: number }>("/api/skills").then((r) => {
      setSkills(r.skills);
      if (r.skills[0]) setSelected(r.skills[0]);
    });
  };
  useEffect(() => { refresh(); }, []);

  const toggle = async (name: string, enabled: boolean) => {
    await apiPostJson(`/api/skills/${name}/toggle`, { enabled: !enabled });
    refresh();
  };

  const generate = async () => {
    const r = await apiPostJson<{ ok: boolean; skill_name?: string; error?: string }>(
      "/api/skills/generate", { text: genText }
    );
    setGenResult(r.ok ? `✓ 生成 ${r.skill_name}` : `✗ ${r.error}`);
    if (r.ok) refresh();
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">Skills 管理</h1>
        <p className="text-sm text-mute mt-1.5">本地沉淀的分析技能,启用/禁用/自动生成。</p>
      </header>

      <Card className="p-5 space-y-3">
        <h3 className="text-sm font-medium text-white">自动生成 Skill</h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={genText}
            onChange={(e) => setGenText(e.target.value)}
            placeholder="场景描述,如 停车场监控 / 人脸识别 / 火焰检测"
            className="flex-1 rounded-card-sm glass-chip px-3 py-2 text-sm text-white"
          />
          <Button onClick={generate}>生成</Button>
        </div>
        {genResult && <p className="text-xs text-accent">{genResult}</p>}
        <p className="text-[11px] text-mute">纯规则模板生成,不真实调付费 LLM。</p>
      </Card>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card className="p-4">
          <h3 className="text-sm font-medium text-white mb-3">Skills 列表</h3>
          <div className="space-y-1 max-h-[50vh] overflow-y-auto">
            {skills.length === 0 ? (
              <p className="text-mute/60 text-sm">暂无 skills(可自动生成)</p>
            ) : (
              skills.map((s) => (
                <button
                  key={s.name}
                  onClick={() => setSelected(s)}
                  className={`w-full text-left rounded-card-sm px-3 py-2 text-xs transition-colors ${
                    selected?.name === s.name ? "bg-white/10" : "hover:bg-white/5"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-white font-medium">{s.name}</span>
                    <span className={s.enabled ? "text-ok" : "text-mute"}>
                      {s.enabled ? "✓ 启用" : "✗ 禁用"}
                    </span>
                  </div>
                  <div className="text-mute truncate">{s.description}</div>
                </button>
              ))
            )}
          </div>
        </Card>

        {selected && (
          <Card className="p-5 space-y-4">
            <h3 className="text-sm font-medium text-white">{selected.name}</h3>
            <p className="text-sm text-mute">{selected.description}</p>
            {selected.triggers.length > 0 && (
              <div>
                <div className="text-xs text-mute mb-1.5">触发词</div>
                <div className="flex flex-wrap gap-1.5">
                  {selected.triggers.map((t) => (
                    <span key={t} className="glass-chip rounded-full px-2.5 py-0.5 text-[11px] text-mist">{t}</span>
                  ))}
                </div>
              </div>
            )}
            <div className="text-[11px] text-mute font-mono break-all">{selected.path}</div>
            <Button
              size="sm"
              variant={selected.enabled ? "chip" : "primary"}
              onClick={() => toggle(selected.name, selected.enabled)}
            >
              {selected.enabled ? "禁用" : "启用"}
            </Button>
          </Card>
        )}
      </div>
    </div>
  );
}
