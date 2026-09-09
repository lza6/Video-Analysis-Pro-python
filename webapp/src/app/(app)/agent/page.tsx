"use client";

import { useState } from "react";
import { apiPostJson, apiUrl, ApiError } from "@/lib/api";
import type {
  AgentApprovalRequest,
  AgentChatResponse,
  AgentRunStreamStepEvent,
  AgentRunStreamDoneEvent,
} from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

interface Msg {
  role: "user" | "agent";
  text: string;
}

/** 写工具审批决定结果(调 POST /api/agent/approval/{pin}/decide)。 */
interface ApprovalDecisionResponse {
  decided: boolean;
}

/** 待审批弹窗状态:null=无弹窗;非 null 显示工具名/参数/原因 + 允许/拒绝。 */
interface PendingApproval {
  pin: string;
  tool: string;
  args: Record<string, unknown>;
  reason: string;
  priority: string;
  timeout: number;
}

/** 流式读取 SSE,逐个解析 event。 */
async function* readSSE(
  res: Response,
): AsyncGenerator<{ event: string; data: string }> {
  const reader = res.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE 帧以 \n\n 分隔
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      yield { event, data };
    }
  }
}

export default function AgentPage() {
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  const [jobId, setJobId] = useState("");
  const [approval, setApproval] = useState<PendingApproval | null>(null);
  const [approvalBusy, setApprovalBusy] = useState(false);

  const appendAgent = (text: string) =>
    setMsgs((m) => [...m, { role: "agent", text }]);

  /** 处理 run_stream 收到的 approval-request 事件:弹审批 UI。 */
  const handleApproval = (data: string) => {
    let payload: unknown = null;
    try {
      payload = data ? JSON.parse(data) : null;
    } catch {
      payload = null;
    }
    if (!payload || typeof payload !== "object") return;
    const p = payload as Partial<AgentApprovalRequest>;
    if (typeof p.pin !== "string" || !p.pin) return;
    setApproval({
      pin: p.pin,
      tool: p.tool ?? "(未知工具)",
      args: (p.args ?? {}) as Record<string, unknown>,
      reason: p.reason ?? "",
      priority: p.priority ?? "write",
      timeout: typeof p.timeout === "number" ? p.timeout : 60,
    });
  };

  /** 允许/拒绝当前审批(调后端 decide 端点,超时默认 deny 兜底)。 */
  const decideApproval = async (allow: boolean): Promise<void> => {
    if (!approval || approvalBusy) return;
    setApprovalBusy(true);
    const pin = approval.pin;
    try {
      const r = await apiPostJson<ApprovalDecisionResponse>(
        `/api/agent/approval/${encodeURIComponent(pin)}/decide`,
        { allow },
      );
      appendAgent(
        `${allow ? "✅" : "⛔"} 审批[${r.decided ? "已生效" : "未生效(重复/已超时)"}] 工具=${approval.tool}`,
      );
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : String(e);
      appendAgent(`⚠️ 审批回调失败: ${msg}`);
    } finally {
      setApprovalBusy(false);
      setApproval(null);
    }
  };

  /** chat 后自动循环执行 plan(SSE 流式),每步实时投到对话流。 */
  const autoRunPlan = async (jobIdParam: string): Promise<void> => {
    const url = apiUrl(
      `/api/agent/run_stream${jobIdParam ? `?job_id=${encodeURIComponent(jobIdParam)}` : ""}`,
    );
    const res = await fetch(url, { method: "GET" });
    if (!res.ok) {
      const err = await res.text();
      appendAgent(`执行失败: ${err}`);
      return;
    }
    try {
      for await (const { event, data } of readSSE(res)) {
        if (event === "approval-request") {
          handleApproval(data);
          continue;
        }
        if (event !== "step" && event !== "done") continue;
        let payload:
          | AgentRunStreamStepEvent
          | AgentRunStreamDoneEvent
          | null = null;
        try {
          payload = data ? (JSON.parse(data) as typeof payload) : null;
        } catch {
          payload = null;
        }
        if (event === "step" && payload && "index" in payload) {
          const p = payload as AgentRunStreamStepEvent;
          const statusIcon =
            p.status === "done"
              ? "✅"
              : p.status === "error"
                ? "⚠️"
                : p.status === "skipped"
                  ? "⏭️"
                  : "⏳";
          appendAgent(
            `${statusIcon} 步骤${p.index}: ${p.description}\n` +
              `工具: ${p.tool ?? "(无)"}\n` +
              `结果: ${p.result ?? "(空)"}`,
          );
        } else if (event === "done") {
          const p = payload as AgentRunStreamDoneEvent | null;
          const reason = p?.reason ? `(${p.reason})` : "";
          appendAgent(`🏁 执行完成,共 ${p?.total ?? 0} 步 ${reason}`);
          return;
        }
      }
    } catch (e) {
      appendAgent(`流式执行中断: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

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
      appendAgent(
        r.reply ||
          `(意图:${r.intent}${r.skill_name ? ` · skill:${r.skill_name}` : ""})`,
      );
      // auto_run 标记为 true → 自动闭环执行(SSE 流式逐步投递)
      if (r.auto_run && r.plan_steps && r.plan_steps.length > 0) {
        await autoRunPlan(jobId);
      }
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : String(e);
      appendAgent(`错误: ${msg}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">Agent 对话</h1>
        <p className="text-sm text-mute mt-1.5">
          用自然语言指挥 AI。意图解析 → 选 skill → plan → 自动执行工具闭环。走 NVIDIA 多 key 路由。
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

      {approval && (
        <Card className="p-5 border border-accent/40 shadow-lg">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-base font-bold text-white">
                审批请求: {approval.tool}
              </h3>
              <p className="text-xs text-mute mt-0.5">
                {approval.reason || "写操作需人工审批"}
                <span className="ml-2 opacity-70">
                  (priority: {approval.priority} · {approval.timeout}s 内决定, 超时默认拒绝)
                </span>
              </p>
            </div>
          </div>
          {Object.keys(approval.args).length > 0 && (
            <pre className="mt-3 rounded-card-sm glass-chip px-3 py-2 text-xs text-mist whitespace-pre-wrap overflow-x-auto max-h-32">
              {JSON.stringify(approval.args, null, 2)}
            </pre>
          )}
          <div className="flex gap-3 mt-4">
            <Button
              variant="primary"
              onClick={() => decideApproval(true)}
              disabled={approvalBusy}
            >
              允许
            </Button>
            <Button
              variant="danger"
              onClick={() => decideApproval(false)}
              disabled={approvalBusy}
            >
              拒绝
            </Button>
            <span className="text-xs text-mute self-center ml-auto">
              {approvalBusy ? "提交中…" : "超时(60s)将默认拒绝"}
            </span>
          </div>
        </Card>
      )}
    </div>
  );
}
