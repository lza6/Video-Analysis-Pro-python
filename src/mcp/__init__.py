"""MCP server —— 把 Agent 工具发成 MCP 协议（双形态之一：对外）。

双形态说明：
  - 对内：工具仍是 `src/core/tools/adapter.py` 的 ToolDefinition 工具面
    （17 legacy + 4 media_gen + 12 web_auto），由 ReactLoopAgent / 主控调度。
  - 对外：`src.mcp.server` 把这些工具白名单发成 **MCP stdio server**，
    供 Claude Code / 其它 MCP 客户端调用。**不引入 fastmcp/mcp 第三方库**，
    纯 stdlib asyncio + json 实现 JSON-RPC 2.0 子集
    （initialize / tools/list / tools/call）。

付费红线：本 server 只暴露白名单内的安全/读/已授权工具；写工具默认不在
白名单（除非 allowlist 显式加且 `VAP_MCP_ALLOW_WRITE=1`）。所有真实付费
API 调用仍守红线（Mock provider 照常标注），本模块不做任何真实付费调用。

白名单策略（`src/config/mcp/tools_allowlist.json`，入库）：
  默认只允许读工具（get_* / search_* / kb_* / scan_videos /
  web_browser_snapshot / web_browser_screenshot / cdp_list_targets /
  cdp_attach / cdp_evaluate）+ media_gen 的 mock 安全项（make_subtitle /
  make_voiceover —— 前者显式 segments 免费本地排版，后者本地 TTS/mock 占位）。
  写工具（delete_*/highlight_cut/trigger_*/start_*/generate_skill/
  create_cut_clip/make_short_video/web_browser_trigger/update/close/
  cdp_eval_write/cdp_close 等）默认不在白名单。

审批边界：`tools/call` 调 registry.execute 走完整 waterfall（pre/execute/
post/result），scope_guard 的 Ask 信号在 MCP 无前端审批人时超时默认 deny；
白名单外的工具名直接返回 permission 错误，不进入执行。
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["server", "run", "__version__"]
