# TingFeng Hermes MCP server

把 Agent 工具发成 **MCP server**（stdio），供 Claude Code / 其它 MCP 客户端调用。

## 双形态

| 形态 | 载体 | 说明 |
|------|------|------|
| 对内 | `src/core/tools/adapter.py` 的 ToolDefinition 工具面 | 17 legacy + 4 media_gen + 12 web_auto，由 ReactLoopAgent / 主控调度，scope_guard 审批 |
| 对外 | `src/mcp/server.py` MCP stdio server | 白名单子集，JSON-RPC 2.0（initialize / tools/list / tools/call），零第三方依赖 |

对内工具面**不动**（agent.py / loop.py / scope_guard.py / registry.py /
waterfall.py / adapter.py 零回归）；对外只是把它投影成 MCP 协议。

## 启动

```bash
# stdio server（供 MCP 客户端 spawn）
venv/Scripts/python.exe -m src.mcp.run

# 单发冒烟
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | \
    venv/Scripts/python.exe -m src.mcp.run
```

## 协议子集

- `initialize` —— 握手（protocolVersion / capabilities / serverInfo）
- `tools/list` —— 白名单工具 schema（OpenAI function-calling 子集）
- `tools/call` —— 调白名单工具，走 `registry.execute` 完整四层 waterfall

## 白名单策略

`src/config/mcp/tools_allowlist.json`（入库，默认）：

- 读工具：`get_video_meta` `get_frame_details` `search_web` `search_visual`
  `search_kb` `search_by_image` `scan_videos` `summarize_hits` `trace_item`
  `point_at_object` `web_browser_snapshot` `web_browser_screenshot`
  `cdp_list_targets` `cdp_attach` `cdp_evaluate`
- media_gen mock 安全项：`make_subtitle`（显式 segments 免费本地排版）、
  `make_voiceover`（本地 TTS / Mock 占位，绝不真调付费 TTS）
- **写工具默认不在白名单**：`delete_*` `highlight_cut` `trigger_batch`
  `start_rtsp_monitor` `generate_skill` `create_cut_clip` `make_short_video`
  `web_browser_trigger/update/close` `cdp_eval_write` `cdp_close` 等。

环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `VAP_MCP_ALLOW_WRITE` | `0` | `1` 且 allowlist 显式加该写工具才可执行（双闸） |
| `VAP_MCP_TOOLS` | 空 | 逗号白名单覆盖（替换 default allowlist） |

## 审批边界（安全）

1. **白名单硬边界**：allowlist 外的工具 `tools/call` 直接返回
   `-32003 permission denied`，不进入 execute（不泄露注册信息）。
2. **写工具双闸**：allowlist 显式含 AND `VAP_MCP_ALLOW_WRITE=1`。
3. **waterfall + scope_guard**：`tools/call` 走 `registry.execute` 完整
   pre/execute/post/result。MCP 无前端审批人时，Ask 信号无 handler →
   `ToolNeedsApproval` → 归一成错误返回（默认拒绝，安全侧）。
4. **付费红线**：对外暴露的媒体工具（make_subtitle / make_voiceover）
   本地免费路径真实执行、付费路径 Mock 占位并标注，**绝不真调付费 API**。

## 测试

```bash
QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe \
    -m pytest tests/test_mcp_server.py -q -p no:cacheprovider
```
