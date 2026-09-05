"use client";

import { useState } from "react";
import { apiPostJson } from "@/lib/api";
import type { AgentChatResponse } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

interface Msg {
  role: "user" | "agent";
  text: string;
}

export default function AgentPage() {
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  const [jobId, setJobId] = useState("");

  const send = async () => {
    if (!input.trim() || busy) return;
    const text = input.trim();
    setInput("");
    setMsgs((m) => [...m, { role: "user", text }]);
    setBusy(true);
    try {
      const r = await apiPostJson<AgentChatResponse>("/api/agent/chat", {
        text,
        job_id: jobId || null,
      });
      setMsgs((m) => [...m, {
        role: "agent",
        text: r.reply || `(意图:${r.intent}${r.skill_name ? ` · skill:${r.skill_name}` : ""})`,
      }]);
    } catch (e) {
      setMsgs((m) => [...m, { role: "agent", text: `错误: ${e}` }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">Agent 对话</h1>
        <p className="text-sm text-mute mt-1.5">
          用自然语言指挥 AI。意图解析 → 选 skill → plan → 工具调用。走 NVIDIA 多 key 路由。
        </p>
      </header>

      <Card className="p-4">
        <label className="block text-xs text-mute mb-1.5">关联作业(可选,工具调用用)</label>
        <input
          type="text"
          value={jobId}
          onChange={(e) => setJobId(e.target.value)}
          placeholder="作业 ID,如 b5806b43b09c"
          className="w-full rounded-card-sm glass-chip px-3 py-2 text-sm text-white font-mono"
        />
      </Card>

      <Card className="p-4 flex flex-col h-[55vh]">
        <div className="flex-1 overflow-y-auto space-y-3">
          {msgs.length === 0 ? (
            <p className="text-mute/60 text-sm">发送消息开始对话…</p>
          ) : (
            msgs.map((m, i) => (
              <div
                key={i}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-card-sm px-4 py-2.5 text-sm whitespace-pre-wrap ${
                    m.role === "user"
                      ? "bg-gradient-to-r from-accent-2 to-accent text-white"
                      : "glass-chip text-mist"
                  }`}
                >
                  {m.text}
                </div>
              </div>
            ))
          )}
          {busy && <p className="text-mute text-xs">思考中…</p>}
        </div>
        <div className="flex gap-2 pt-3 border-t border-white/5 mt-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="问我任何关于视频的问题…"
            className="flex-1 rounded-card-sm glass-chip px-4 py-2.5 text-sm text-white"
          />
          <Button onClick={send} disabled={busy || !input.trim()}>
            发送
          </Button>
        </div>
      </Card>
    </div>
  );
}
