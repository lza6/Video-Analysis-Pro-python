"use client";

import { useEffect, useRef, useState } from "react";
import { apiDelete, apiGet, apiPostJson, apiUrl, ApiError } from "@/lib/api";
import type {
  AgentApprovalRequest,
  AgentChatResponse,
  AgentRunStreamStepEvent,
  AgentRunStreamDoneEvent,
} from "@/lib/types";
import {
  APPROVAL_TOOL_CN,
  APPROVAL_CONSEQUENCE,
} from "@/lib/approvalMaps";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { GlossaryText } from "@/components/ui/GlossaryTerm";
import { useToast } from "@/components/ui/Toast";

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

interface SessionItem {
  session_id: string;
  updated_at?: string;
  size?: number;
}

interface SessionEventDto {
  type: string;
  payload: Record<string, unknown>;
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

/** 工具中文名(缺失时退回裸名——approvalMaps 守护测试会拦住新增遗漏)。 */
const toolCn = (tool: string) => APPROVAL_TOOL_CN[tool] || tool;

export default function AgentPage() {
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  const [jobId, setJobId] = useState("");
  const [approval, setApproval] = useState<PendingApproval | null>(null);
  const [approvalBusy, setApprovalBusy] = useState(false);
  // v10.5.0 (P1-8): 审批剩余秒数(实时倒计时,归零自动关闭)
  const [remaining, setRemaining] = useState<number>(0);
  // v10.5.0 (P1-9): 会话管理
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const { toast } = useToast();

  // v10.5.0 (P1-7): 流式打字机——当前正在追加的 agent 消息下标
  const streamRef = useRef<number | null>(null);
  // 已由 tool_done 实时渲染的工具调用 id(去重 tool_result 兼容补发)
  const toolDoneIds = useRef<Set<string>>(new Set());

  const appendAgent = (text: string) =>
    setMsgs((m) => [...m, { role: "agent", text }]);

  /** 把增量文本追加到当前流式消息(打字机)。 */
  const appendStream = (delta: string) => {
    setMsgs((prev) => {
      if (streamRef.current != null && streamRef.current < prev.length) {
        const next = [...prev];
        next[streamRef.current] = {
          role: "agent",
          text: next[streamRef.current].text + delta,
        };
        return next;
      }
      const idx = prev.length;
      streamRef.current = idx;
      return [...prev, { role: "agent", text: delta }];
    });
  };

  const finalizeStream = () => {
    streamRef.current = null;
  };

  /** 会话列表刷新。 */
  const loadSessions = async () => {
    try {
      const r = await apiGet<{ sessions: SessionItem[] }>("/api/agent/sessions");
      setSessions(r.sessions || []);
    } catch {
      /* 后端不可用/未就绪,静默 */
    }
  };

  useEffect(() => {
    void loadSessions();
  }, []);

  const startNewSession = () => {
    setActiveSessionId("");
    setMsgs([]);
    setJobId("");
    finalizeStream();
  };

  const openSession = async (sid: string) => {
    try {
      const r = await apiGet<{ events: SessionEventDto[] }>(
        `/api/agent/sessions/${sid}`,
      );
      const rebuilt: Msg[] = [];
      for (const e of r.events) {
        if (e.type === "user") {
          rebuilt.push({ role: "user", text: String(e.payload.content ?? "") });
        } else if (e.type === "assistant" && !e.payload.tool_calls) {
          rebuilt.push({ role: "agent", text: String(e.payload.content ?? "") });
        }
      }
      setMsgs(rebuilt);
      setActiveSessionId(sid);
    } catch (e) {
      appendAgent(`加载会话失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const deleteSession = async (sid: string) => {
    if (!window.confirm(`删除会话 ${sid.slice(0, 8)}…?此操作不可恢复。`)) return;
    try {
      await apiDelete<unknown>(`/api/agent/sessions/${sid}`);
      if (activeSessionId === sid) startNewSession();
      await loadSessions();
    } catch (e) {
      appendAgent(`删除会话失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

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
        `${allow ? "✅" : "⛔"} 审批[${r.decided ? "已生效" : "未生效(重复/已超时)"}] 工具=${toolCn(approval.tool)}`,
      );
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : String(e);
      appendAgent(`⚠️ 审批回调失败: ${msg}`);
      toast(`审批回调失败: ${msg}`, "error");
    } finally {
      setApprovalBusy(false);
      setApproval(null);
    }
  };

  // v10.5.0 (P1-8): 审批实时倒计时
  useEffect(() => {
    if (!approval) return;
    const total = approval.timeout > 0 ? approval.timeout : 60;
    const startedAt = Date.now();
    setRemaining(total);
    const timer = setInterval(() => {
      const left = total - (Date.now() - startedAt) / 1000;
      if (left <= 0) {
        clearInterval(timer);
        setRemaining(0);
        setApproval(null);
        appendAgent(`⏰ 审批超时(${total}s),已自动拒绝`);
      } else {
        setRemaining(left);
      }
    }, 250);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [approval?.pin]);

  // v10.5.0 (P1-8): 键盘快捷键——Enter=允许 / Esc=拒绝(排除中文输入法组字)
  useEffect(() => {
    if (!approval) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.isComposing) return;
      if (e.key === "Enter") {
        e.preventDefault();
        void decideApproval(true);
      } else if (e.key === "Escape") {
        e.preventDefault();
        void decideApproval(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [approval, approvalBusy]);

  /** chat 后自动循环执行 plan(SSE 流式),每步实时投到对话流。 */
  const autoRunPlan = async (
    jobIdParam: string,
    sessionIdParam?: string,
    textParam?: string,
  ): Promise<void> => {
    const params = new URLSearchParams();
    if (jobIdParam) params.set("job_id", jobIdParam);
    if (sessionIdParam) params.set("session_id", sessionIdParam);
    if (textParam) params.set("text", textParam);
    const qs = params.toString();
    const url = apiUrl(`/api/agent/run_stream${qs ? `?${qs}` : ""}`);
    const res = await fetch(url, { method: "GET" });
    if (!res.ok) {
      const err = await res.text();
      appendAgent(`执行失败: ${err}`);
      return;
    }
    try {
      for await (const { event, data } of readSSE(res)) {
        // v10.5.0 (P1-7): 实时打字机
        if (event === "assistant_delta") {
          let p: { delta?: string } | null = null;
          try {
            p = data ? (JSON.parse(data) as { delta?: string }) : null;
          } catch {
            p = null;
          }
          if (p?.delta) appendStream(p.delta);
          continue;
        }
        // v10.5.0 (P1-7): 工具实时状态
        if (event === "tool_start") {
          finalizeStream();
          let p: { tool?: string } | null = null;
          try {
            p = data ? (JSON.parse(data) as { tool?: string }) : null;
          } catch {
            p = null;
          }
          appendAgent(`⚙️ 执行 ${toolCn(p?.tool ?? "(未知工具)")}…`);
          continue;
        }
        if (event === "tool_done") {
          finalizeStream();
          let p: {
            tool?: string;
            tool_call_id?: string;
            ok?: boolean;
            error?: string | null;
            human?: string;
          } | null = null;
          try {
            p = data ? JSON.parse(data) : null;
          } catch {
            p = null;
          }
          if (p?.tool_call_id) toolDoneIds.current.add(p.tool_call_id);
          const name = toolCn(p?.tool ?? "(未知工具)");
          if (p?.ok) {
            appendAgent(`✅ ${p.human || `${name} 完成`}`);
          } else {
            appendAgent(`⚠️ ${name} 失败: ${p?.error ?? "未知错误"}`);
          }
          continue;
        }
        if (event === "supervise") {
          finalizeStream();
          let p: { describe?: string } | null = null;
          try {
            p = data ? (JSON.parse(data) as { describe?: string }) : null;
          } catch {
            p = null;
          }
          if (p?.describe) appendAgent(`🧠 监督: ${p.describe}`);
          continue;
        }
        if (event === "approval-request") {
          handleApproval(data);
          continue;
        }
        if (event !== "step" && event !== "done" && event !== "tool_result")
          continue;
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
          const human = (payload as Record<string, unknown>)?.human as
            | string
            | undefined;
          appendAgent(
            human
              ? `${statusIcon} ${human}`
              : `${statusIcon} 步骤${p.index}: ${p.description}\n` +
                  `工具: ${p.tool ?? "(无)"}\n` +
                  `结果: ${p.result ?? "(空)"}`,
          );
        } else if (event === "tool_result") {
          // 兼容补发:实时 tool_done 已渲染过的工具跳过(去重)
          try {
            const p = JSON.parse(data) as {
              human?: string;
              tool_call_id?: string;
            };
            if (p?.tool_call_id && toolDoneIds.current.has(p.tool_call_id)) {
              continue;
            }
            if (p?.human) appendAgent(`🔧 ${p.human}`);
          } catch {
            /* 忽略解析失败 */
          }
          continue;
        } else if (event === "done") {
          finalizeStream();
          const p = payload as AgentRunStreamDoneEvent | null;
          const reason = p?.reason ? `(${p.reason})` : "";
          appendAgent(`🏁 执行完成,共 ${p?.total ?? 0} 步 ${reason}`);
          await loadSessions();
          return;
        }
      }
    } catch (e) {
      finalizeStream();
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
      if (r.session_id) setActiveSessionId(r.session_id);
      if (r.auto_run && r.session_id) {
        await autoRunPlan(jobId, r.session_id, text);
      } else if (r.auto_run && r.plan_steps && r.plan_steps.length > 0) {
        await autoRunPlan(jobId);
      }
      await loadSessions();
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : String(e);
      appendAgent(`错误: ${msg}`);
      toast(msg, "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-black tracking-tight text-white">Agent 对话</h1>
        <p className="text-sm text-mute mt-1.5">
          用自然语言指挥 AI。意图解析 → 选 <GlossaryText text="skill" /> → plan → 自动执行工具闭环。<GlossaryText text="写操作需审批,会话可续接。" />
        </p>
      </header>

      {/* v10.5.0 (P1-9): 会话管理(新建/切换/删除历史) */}
      <Card className="p-3 flex items-center gap-2 flex-wrap">
        <Button variant="chip" onClick={startNewSession}>
          ＋ 新建会话
        </Button>
        <select
          value={activeSessionId}
          onChange={(e) =>
            e.target.value ? void openSession(e.target.value) : startNewSession()
          }
          className="rounded-card-sm glass-chip px-3 py-2 text-sm text-white max-w-[16rem]"
          aria-label="历史会话"
        >
          <option value="">（未保存会话）</option>
          {sessions.map((s) => (
            <option key={s.session_id} value={s.session_id}>
              {s.session_id.slice(0, 8)} · {s.updated_at?.replace("T", " ") ?? ""}
            </option>
          ))}
        </select>
        {activeSessionId && (
          <Button variant="danger" onClick={() => void deleteSession(activeSessionId)}>
            删除
          </Button>
        )}
        <span className="text-xs text-mute ml-auto">{sessions.length} 个历史会话</span>
      </Card>

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
        <Card
          className={
            "p-5 border shadow-lg " +
            (approval.priority === "dangerous_write"
              ? "border-danger/60"
              : "border-accent/40")
          }
        >
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                {approval.priority === "dangerous_write" && (
                  <span className="text-danger">⚠️ 危险操作</span>
                )}
                审批请求: {toolCn(approval.tool)}
              </h3>
              <p className="text-xs text-mute mt-0.5">
                {approval.reason ||
                  (approval.priority === "dangerous_write"
                    ? "危险写操作,需人工审批"
                    : "写操作需人工审批")}
              </p>
              <p className="text-xs text-mute/70 mt-1">
                {APPROVAL_CONSEQUENCE[approval.tool] ||
                  "该操作会改动本地数据。"}
                <span className="ml-2 opacity-70">
                  按 Enter 允许 / Esc 拒绝 ·
                  {remaining > 0
                    ? ` ${Math.ceil(remaining)}s 后自动拒绝`
                    : " 等待决定…"}
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
              onClick={() => void decideApproval(true)}
              disabled={approvalBusy}
            >
              允许本次
            </Button>
            <Button
              variant="danger"
              onClick={() => void decideApproval(false)}
              disabled={approvalBusy}
            >
              拒绝
            </Button>
            <span className="text-xs text-mute self-center ml-auto">
              {approvalBusy ? "提交中…" : ""}
            </span>
          </div>
        </Card>
      )}
    </div>
  );
}