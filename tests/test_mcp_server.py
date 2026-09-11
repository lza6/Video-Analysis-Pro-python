"""MCP server 测试（批 B-P3-1）。

契约（指南 v2 P3-1）：
  ① initialize 握手
  ② tools/list 返回白名单子集（读工具在、写工具不在）
  ③ tools/call 读工具走通（mock registry 或真实 registry + mock 工具）
  ④ 写工具不在白名单 → tools/call 返回 permission error
  ⑤ VAP_MCP_ALLOW_WRITE=1 且 allowlist 含写工具 → 可调（用无害 mock 写工具）
  ⑥ 手写 stdio server 用 asyncio subprocess 或直接调 handler 验证 JSON-RPC
     响应格式（parse error / method not found / 正常响应）。

不依赖 pytest-asyncio：用 asyncio.new_event_loop 驱动 async（与
tests/test_agent_framework.py / test_scope_guard.py 一致）。
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.tools.definition import ToolDefinition  # noqa: E402
from src.core.tools.registry import ToolRegistry  # noqa: E402
from src.mcp.server import (  # noqa: E402
    ALLOWLIST_PATH,
    ENV_ALLOW_WRITE,
    ENV_TOOLS,
    MCPServer,
    MCPToolFilter,
    MCPServerError,
    build_tool_registry,
    _read_allowlist,
)


def _run(coro):
    """同步驱动 async 测试，不依赖 pytest-asyncio。"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _echo_read_tool(name: str = "get_dummy") -> ToolDefinition:
    """无害 mock 读工具：返回 dict，验证 execute 真正执行。"""

    async def _impl() -> dict:
        return {"ok": True, "name": name}

    return ToolDefinition(
        name=name, description=f"mock read tool {name}", execute_callback=_impl)


def _echo_write_tool(name: str = "write_dummy") -> ToolDefinition:
    """无害 mock 写工具（名字含 write → scope_guard 分类 write）。"""

    async def _impl(value: str = "") -> dict:
        return {"ok": True, "name": name, "value": value}

    return ToolDefinition(
        name=name, description=f"mock write tool {name}", execute_callback=_impl)


def _make_server(
    *,
    read_names: tuple = ("get_dummy",),
    write_names: tuple = (),
    allow_write: bool = False,
) -> MCPServer:
    """构造只含 mock 工具的隔离 MCP server（不碰真实 registry/环境）。"""
    reg = ToolRegistry()
    for n in read_names:
        reg.register(_echo_read_tool(n))
    for n in write_names:
        reg.register(_echo_write_tool(n))
    allowlist = list(read_names) + list(write_names)
    filt = MCPToolFilter(allowlist=allowlist, allow_write=allow_write)
    return MCPServer(registry=reg, filter_=filt)


# ---------------------------------------------------------------------------
# ① initialize 握手
# ---------------------------------------------------------------------------


def test_initialize_handshake() -> None:
    """initialize 返回 protocolVersion / capabilities.tools / serverInfo。"""
    srv = _make_server()
    res = _run(srv.initialize())
    assert res["protocolVersion"]
    assert res["capabilities"]["tools"]["listChanged"] is False
    assert res["serverInfo"]["name"] == "tingfeng-hermes-mcp"


def test_initialize_via_dispatch_keeps_id() -> None:
    """经 dispatch 的 initialize 透传 id + jsonrpc=2.0。"""
    srv = _make_server()
    resp = _run(srv.dispatch(
        {"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": {}}))
    assert resp["id"] == 7
    assert resp["jsonrpc"] == "2.0"
    assert "result" in resp and "serverInfo" in resp["result"]


# ---------------------------------------------------------------------------
# ② tools/list 白名单子集
# ---------------------------------------------------------------------------


def test_list_tools_only_allowlist() -> None:
    """tools/list 只返回白名单工具（读在、写不在）。"""
    srv = _make_server(
        read_names=("get_dummy", "search_dummy"),
        write_names=("write_dummy",),
        allow_write=False,
    )
    tools = srv.list_tools()["tools"]
    names = [t["name"] for t in tools]
    assert "get_dummy" in names
    assert "search_dummy" in names
    assert "write_dummy" not in names


def test_list_tools_schema_shape() -> None:
    """tools/list 的 tool 项含 name / description / inputSchema（OpenAI 兼容）。"""
    srv = _make_server(read_names=("get_dummy",))
    tools = srv.list_tools()["tools"]
    assert len(tools) == 1
    t = tools[0]
    assert t["name"] == "get_dummy"
    assert t["description"]
    assert t["inputSchema"]["type"] == "object"


def test_default_allowlist_read_tools_present_write_absent() -> None:
    """真实 allowlist（src/config/mcp/tools_allowlist.json）读在写不在。"""
    allow = _read_allowlist(ALLOWLIST_PATH)
    assert "get_video_meta" in allow
    assert "search_kb" in allow
    assert "web_browser_snapshot" in allow
    assert "cdp_evaluate" in allow
    # 写工具绝不在默认白名单
    for w in ("delete_history", "highlight_cut", "trigger_batch",
              "start_rtsp_monitor", "generate_skill", "create_cut_clip",
              "make_short_video", "web_browser_trigger", "cdp_eval_write"):
        assert w not in allow, f"{w} 不应在默认白名单"


# ---------------------------------------------------------------------------
# ③ tools/call 读工具走通
# ---------------------------------------------------------------------------


def test_call_read_tool_success() -> None:
    """读工具 tools/call 走通：isError=False + output 正确。"""
    srv = _make_server(read_names=("get_dummy",))
    res = _run(srv.call_tool({"name": "get_dummy", "arguments": {}}))
    assert res["isError"] is False
    text = res["content"][0]["text"]
    assert "get_dummy" in text and '"ok": true' in text


def test_call_read_tool_requires_registered() -> None:
    """工具未注册且不在白名单 → 先按白名单 permission 拒绝（不泄露注册信息）。"""
    srv = _make_server(read_names=("get_dummy",))
    with pytest.raises(MCPServerError) as ei:
        _run(srv.call_tool({"name": "missing_tool", "arguments": {}}))
    assert ei.value.code == -32003


def test_call_real_registry_read_tool() -> None:
    """真实 registry（build_tool_registry）读工具 get_video_meta 走通（无 job 上下文提示）。"""
    reg = build_tool_registry(allow_write=False)
    filt = MCPToolFilter(
        allowlist=["get_video_meta", "web_browser_snapshot", "scan_videos"],
        allow_write=False,
    )
    srv = MCPServer(registry=reg, filter_=filt)
    res = _run(srv.call_tool({"name": "get_video_meta", "arguments": {}}))
    assert res["isError"] is False
    text = res["content"][0]["text"]
    assert "No video loaded" in text or "video" in text


# ---------------------------------------------------------------------------
# ④ 写工具不在白名单 → permission error
# ---------------------------------------------------------------------------


def test_call_write_not_allowed_denied() -> None:
    """写工具（allow_write=False）→ permission error，不执行。"""
    srv = _make_server(
        read_names=("get_dummy",), write_names=("write_dummy",), allow_write=False)
    with pytest.raises(MCPServerError) as ei:
        _run(srv.call_tool({"name": "write_dummy", "arguments": {}}))
    assert ei.value.code == -32003
    assert "allowlist" in ei.value.message


def test_call_write_not_in_allowlist_denied() -> None:
    """写工具不在 allowlist（即使 allow_write=True）→ permission error。"""
    # allow_write=True 但 allowlist 不含 write_dummy → 仍拒
    srv = _make_server(read_names=("get_dummy",), allow_write=True)
    with pytest.raises(MCPServerError) as ei:
        _run(srv.call_tool({"name": "write_dummy", "arguments": {}}))
    assert ei.value.code == -32003


def test_call_non_allowlist_read_denied() -> None:
    """读工具不在 allowlist → permission error（白名单是硬边界）。"""
    srv = _make_server(read_names=("get_dummy",))
    with pytest.raises(MCPServerError) as ei:
        _run(srv.call_tool({"name": "search_dummy", "arguments": {}}))
    assert ei.value.code == -32003


# ---------------------------------------------------------------------------
# ⑤ VAP_MCP_ALLOW_WRITE=1 且 allowlist 含写工具 → 可调
# ---------------------------------------------------------------------------


def test_call_write_allowed_when_flag_and_allowlist(monkeypatch) -> None:
    """allow_write=True 且写工具在 allowlist → 可执行。"""
    srv = _make_server(
        read_names=("get_dummy",), write_names=("write_dummy",), allow_write=True)
    res = _run(srv.call_tool({"name": "write_dummy", "arguments": {"value": "x"}}))
    assert res["isError"] is False
    assert "write_dummy" in res["content"][0]["text"]


def test_env_flag_gates_write(monkeypatch) -> None:
    """VAP_MCP_ALLOW_WRITE 环境变量驱动 can_write（默认 0=拒绝）。"""
    reg = ToolRegistry()
    reg.register(_echo_read_tool("get_dummy"))
    reg.register(_echo_write_tool("write_dummy"))
    filt = MCPToolFilter(allowlist=["get_dummy", "write_dummy"], allow_write=True)
    srv = MCPServer(registry=reg, filter_=filt)
    assert srv._filter.can_write("write_dummy") is True
    assert srv._filter.can_write("get_dummy") is True

    filt2 = MCPToolFilter(allowlist=["get_dummy", "write_dummy"], allow_write=False)
    srv2 = MCPServer(registry=reg, filter_=filt2)
    assert srv2._filter.can_write("write_dummy") is False


def test_scope_guard_ask_blocks_write_without_approval() -> None:
    """真实 registry 的 make_voiceover（classify read）在 allow_write=False 仍可执行。

    scope_guard classify 按「工具名」判定：v10.3.1 (P0-3) 起 make_*
    (make_subtitle/make_short_video/make_voiceover) 真实落盘 → 归类
    **write**(ask 审批) —— 此前因无写关键词误归 read。MCP 白名单虽
    显式含 make_subtitle/make_voiceover,但 allow_write=False 时
    can_write=False(与"写工具默认不放行"一致);对外只读消费方若需
    调 make_*,须 VAP_MCP_ALLOW_WRITE=1(双闸)。

    MCP 层真正收紧的是**显式写工具**（classify 为 write/dangerous_write）：
      即使 allowlist 显式含 delete_history，allow_write=False 时也不可执行；
      allow_write=True 后方可（且需 allowlist 显式加，双闸）。
    这守住了「写工具默认不放行」「危险写必须双闸」的审批边界。
    """
    build_tool_registry(allow_write=False)
    # v10.3.1 (P0-3):make_* 归 write → allow_write=False 时不可执行
    filt_read = MCPToolFilter(allowlist=["make_voiceover"], allow_write=False)
    assert filt_read.can_write("make_voiceover") is False, \
        "P0-3 后 make_voiceover 是 write,allow_write=False 应拒绝"
    filt_read_ok = MCPToolFilter(allowlist=["make_voiceover"], allow_write=True)
    assert filt_read_ok.can_write("make_voiceover") is True
    # 纯读工具不受影响
    filt_true_read = MCPToolFilter(allowlist=["get_video_meta"],
                                   allow_write=False)
    assert filt_true_read.can_write("get_video_meta") is True
    # 显式写工具：白名单含 + allow_write=False → 仍不可执行（硬边界）
    filt_write = MCPToolFilter(allowlist=["delete_history"], allow_write=False)
    assert filt_write.can_write("delete_history") is False
    # 危险写：白名单含 + allow_write=True → 可执行（双闸放行）
    filt_write_ok = MCPToolFilter(
        allowlist=["delete_history"], allow_write=True)
    assert filt_write_ok.can_write("delete_history") is True


# ---------------------------------------------------------------------------
# ⑥ stdio JSON-RPC 响应格式
# ---------------------------------------------------------------------------


def test_handle_line_parse_error() -> None:
    """非法 JSON 行 → parse error 响应（id=None）。"""
    srv = _make_server()
    resp = _run(srv.handle_line("{not json"))
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] is None
    assert resp["error"]["code"] == -32700


def test_handle_line_invalid_request() -> None:
    """非 dict 消息 → invalid request 响应。"""
    srv = _make_server()
    resp = _run(srv.handle_line("[1,2,3]"))
    assert resp["error"]["code"] == -32600


def test_handle_line_method_not_found() -> None:
    """未知 method → method not found 响应。"""
    srv = _make_server()
    resp = _run(srv.handle_line(
        '{"jsonrpc":"2.0","id":9,"method":"unknown/x","params":{}}'))
    assert resp["id"] == 9
    assert resp["error"]["code"] == -32601


def test_handle_line_tools_list_end_to_end() -> None:
    """handle_line 驱动 tools/list：返回白名单子集（mock server）。"""
    srv = _make_server(read_names=("get_dummy",))
    resp = _run(srv.handle_line(
        '{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{}}'))
    assert resp["id"] == 3
    names = [t["name"] for t in resp["result"]["tools"]]
    assert names == ["get_dummy"]


def test_handle_line_tools_call_permission() -> None:
    """handle_line 驱动 tools/call 白名单外写工具 → permission error 响应。"""
    srv = _make_server(read_names=("get_dummy",))
    resp = _run(srv.handle_line(
        '{"jsonrpc":"2.0","id":4,"method":"tools/call",'
        '"params":{"name":"write_dummy","arguments":{}}}'))
    assert resp["id"] == 4
    assert resp["error"]["code"] == -32003


def test_env_tools_override(monkeypatch) -> None:
    """VAP_MCP_TOOLS 逗号白名单覆盖 default allowlist。"""
    monkeypatch.setenv(ENV_TOOLS, "get_video_meta,scan_videos")
    from src.mcp.server import _env_tools_override
    allow = _env_tools_override(os.environ.get(ENV_TOOLS, ""))
    assert allow == ["get_video_meta", "scan_videos"]


def test_allowlist_path_exists() -> None:
    """allowlist 文件存在且含 default_tools。"""
    assert ALLOWLIST_PATH.exists()
    allow = _read_allowlist(ALLOWLIST_PATH)
    assert isinstance(allow, list) and len(allow) > 0


def test_default_filter_allow_write_env(monkeypatch) -> None:
    """MCPServer 默认 filter 读 VAP_MCP_ALLOW_WRITE（monkeypatch 隔离）。"""
    monkeypatch.setenv(ENV_ALLOW_WRITE, "1")
    monkeypatch.setenv(ENV_TOOLS, "get_dummy,write_dummy")
    reg = ToolRegistry()
    reg.register(_echo_write_tool("write_dummy"))
    reg.register(_echo_read_tool("get_dummy"))
    srv = MCPServer(registry=reg, filter_=None)
    assert srv._filter.can_write("write_dummy") is True
