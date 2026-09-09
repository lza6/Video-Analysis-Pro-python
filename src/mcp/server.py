"""最小 stdio JSON-RPC 2.0 MCP server（MCP 协议子集）。

**零第三方依赖**：纯 stdlib asyncio + json。不引入 fastmcp / mcp 库。

支持的 MCP 方法：
  - `initialize`        —— 握手（返回 protocolVersion / capabilities / serverInfo）
  - `tools/list`        —— 返回白名单工具 schema（OpenAI function-calling 子集）
  - `tools/call`        —— 调白名单工具（走 registry.execute 完整 waterfall）

权限模型（对外只暴露安全/读/已授权工具）：
  1. 白名单 `src/config/mcp/tools_allowlist.json` 默认只含读工具 + mock 安全项。
  2. `VAP_MCP_TOOLS`（可选，逗号分隔）覆盖 allowlist 的最终工具集合。
  3. `VAP_MCP_ALLOW_WRITE`（默认 0）：`1` 时 allowlist 里显式加到的写工具可执行；
     否则白名单外的工具 `tools/call` 直接返回 permission 错误（不进入执行）。

审批边界：`tools/call` 走 `registry.execute` 的完整四层 waterfall；scope_guard
的 Ask 信号在 MCP 无前端审批人时超时默认 deny（安全侧）。对白名单外 / 未授权
写工具绝不执行（与付费 API 红线一致）。

设计（browser-use / video-use 范式）：对内 = adapter 工具面（不动），对外 =
本 MCP server。测试见 `tests/test_mcp_server.py`。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.tools.definition import ToolDefinition
from src.core.tools.registry import ToolCall, ToolRegistry
from src.core.tools.scope_guard import ScopeGuard

log = logging.getLogger("src.mcp.server")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: MCP 协议版本（本实现支持的子集）
PROTOCOL_VERSION = "2025-03-26"

#: 项目根目录（allowlist 定位）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 白名单默认路径
ALLOWLIST_PATH = PROJECT_ROOT / "src" / "config" / "mcp" / "tools_allowlist.json"

#: 环境变量
ENV_ALLOW_WRITE = "VAP_MCP_ALLOW_WRITE"
ENV_TOOLS = "VAP_MCP_TOOLS"

#: JSON-RPC 错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
TOOL_NOT_FOUND = -32002
PERMISSION_DENIED = -32003
TOOL_ERROR = -32004


# ---------------------------------------------------------------------------
# 白名单加载
# ---------------------------------------------------------------------------


def _read_allowlist(path: Path) -> List[str]:
    """读 allowlist JSON 的 default_tools。缺失/损坏返回空列表（严格默认拒绝）。"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        tools = data.get("default_tools", [])
        if isinstance(tools, list):
            return [str(t) for t in tools if isinstance(t, str)]
    except Exception as exc:  # noqa: BLE001 — 配置损坏按拒绝处理（安全侧）
        log.warning("allowlist 读取失败(%s)，按空白名单处理: %s", path, exc)
    return []


def _env_tools_override(raw: str) -> Optional[List[str]]:
    """解析 `VAP_MCP_TOOLS`（逗号分隔）覆盖白名单。None=未设置。"""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    return parts


# ---------------------------------------------------------------------------
# registry 装配
# ---------------------------------------------------------------------------


def build_tool_registry(
    *,
    allow_write: Optional[bool] = None,
    allowlist_path: Optional[Path] = None,
) -> ToolRegistry:
    """构造 MCP 专用 ToolRegistry，挂 scope_guard 但**不**装配前端审批。

    - `install_tool_guard` 挂 pre_execute 钩子：读工具 allow，写工具 ask。
    - 不装配 ApprovalBus approval_fn → Ask 信号在 execute 内抛
      ToolNeedsApproval → 归一成 ToolResult.error（默认拒绝，MCP 无前端审批人）。
    - 白名单外的写工具根本不进 execute（见 `MCPServer.call_tool`）。
    """
    from src.core.tools.adapter import (
        register_legacy_tools,
        register_media_gen_tools,
        register_web_auto_tools,
    )
    from src.core.tools.scope_guard import install_tool_guard

    registry = ToolRegistry()

    # 内部轻量 context：legacy 工具的 app_context_getter。读工具在无 job
    # 上下文时返回可读的提示串（不崩溃），写工具多数要求 app 上下文。
    class _MCPContext:
        video_path = None
        video_duration = 0.0
        output_dir = None
        frames = []
        history_manager = None

        def seek_video(self, ts: float) -> None:
            log.info("MCP 下 seek 仅记录意图: %s", ts)

    _ctx = _MCPContext()
    _ctx_getter = lambda: _ctx  # noqa: E731 — 匹配 adapter 期望的 app_context_getter

    register_legacy_tools(registry, _ctx_getter)
    register_media_gen_tools(registry)
    register_web_auto_tools(registry)

    # 读工具 allow / 写工具 ask；ask 无 handler → 默认拒绝（安全侧）。
    install_tool_guard(
        registry,
        approval_fn=None,
        sandbox=None,
        enabled=False,
        guard=ScopeGuard(
            allow_write_general=allow_write,
            allow_write_cut=allow_write,
            allow_write_delete=allow_write,
        ),
    )
    return registry


# ---------------------------------------------------------------------------
# 工具过滤：白名单
# ---------------------------------------------------------------------------


class MCPToolFilter:
    """白名单过滤器：决定哪些工具对 MCP 客户端可见/可调。

    规则（`allowed(name)` / `can_write(name)`）：
      - `name in allowlist` → 可见。
      - allowlist 外 → 不可见（tools/list 不返回，tools/call 返回 permission 错误）。
      - 写工具（ScopeGuard.classify != read）→ `can_write` 才可执行；
        `can_write` = allowlist 显式含该写工具 AND `VAP_MCP_ALLOW_WRITE=1`。
    """

    def __init__(
        self,
        allowlist: List[str],
        allow_write: bool = False,
    ) -> None:
        self._allowlist = set(allowlist)
        self._allow_write = allow_write
        self._guard = ScopeGuard(
            allow_write_general=allow_write,
            allow_write_cut=allow_write,
            allow_write_delete=allow_write,
        )

    def allowed(self, name: str) -> bool:
        return name in self._allowlist

    def can_write(self, name: str) -> bool:
        """写工具需 allowlist 显式含 + VAP_MCP_ALLOW_WRITE=1 才放行。"""
        if self._guard.classify(name) == "read":
            return True
        return self._allow_write and name in self._allowlist

    def filter_tools(self, defs: Dict[str, ToolDefinition]) -> List[Dict[str, Any]]:
        """从 registry 全量定义投影白名单子集（MCP tool 格式）。

        只暴露「已授权」工具：白名单内且可执行。写工具在 allow_write=False 时
        即使白名单显式含也不列出（对外只暴露安全/读/已授权的工具）。
        """
        out: List[Dict[str, Any]] = []
        for name, defn in defs.items():
            if not self.allowed(name):
                continue
            if not self.can_write(name):
                continue
            schema = defn.to_llm_schema()
            out.append({
                "name": schema["function"]["name"],
                "description": schema["function"]["description"],
                "inputSchema": schema["function"]["parameters"],
            })
        return out


# ---------------------------------------------------------------------------
# JSON-RPC 处理器
# ---------------------------------------------------------------------------


class MCPServer:
    """MCP stdio server 核心（无 IO，纯 handler，便于测试直调）。"""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        filter_: Optional[MCPToolFilter] = None,
    ) -> None:
        self._registry = registry if registry is not None else build_tool_registry()
        self._filter = filter_ if filter_ is not None else self._default_filter()
        self._initialized = False

    def _default_filter(self) -> MCPToolFilter:
        allow_write = os.environ.get(ENV_ALLOW_WRITE, "0").strip().lower() == "1"
        allowlist = _read_allowlist(ALLOWLIST_PATH)
        override = _env_tools_override(os.environ.get(ENV_TOOLS, ""))
        if override is not None:
            allowlist = override
        return MCPToolFilter(allowlist=allowlist, allow_write=allow_write)

    # ---- MCP 方法 ----

    async def initialize(self) -> Dict[str, Any]:
        self._initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": "tingfeng-hermes-mcp",
                "version": "0.1.0",
            },
        }

    def list_tools(self) -> Dict[str, Any]:
        """tools/list：返回白名单子集。"""
        return {
            "tools": self._filter.filter_tools(
                {n: self._registry.get(n) for n in self._registry.list_names()}
            ),
        }

    async def call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """tools/call：走 registry.execute（完整 waterfall + 审批边界）。"""
        name = params.get("name", "")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            raise MCPServerError(
                INVALID_PARAMS, "arguments must be an object", name)

        if not self._filter.allowed(name):
            raise MCPServerError(
                PERMISSION_DENIED, f"tool not in allowlist: {name}", name)

        defn = self._registry.get(name)
        if defn is None:
            raise MCPServerError(TOOL_NOT_FOUND, f"tool not found: {name}", name)
        if not self._filter.can_write(name):
            raise MCPServerError(
                PERMISSION_DENIED,
                f"write tool requires VAP_MCP_ALLOW_WRITE=1 and allowlist entry: {name}",
                name,
            )

        result = await self._registry.execute(ToolCall(name=name, args=args))
        if result.is_error():
            # 审批拒绝 / 运行错误统一归为工具错误（不按 JSON-RPC 异常抛，
            # 保留 MCP 的 isError 语义，让客户端能看到结构化的错误正文）。
            return {
                "content": [{"type": "text", "text": str(result.error or "")}],
                "isError": True,
            }
        text = _serialize(result.output)
        return {"content": [{"type": "text", "text": text}], "isError": False}

    # ---- 分派 ----

    async def dispatch(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """解析单条 JSON-RPC 2.0 请求，返回响应（含 id 透传）。"""
        rid = message.get("id")
        method = message.get("method", "")
        params = message.get("params") or {}

        if method == "initialize":
            return {"jsonrpc": "2.0", "id": rid, "result": await self.initialize()}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": rid, "result": self.list_tools()}
        if method == "tools/call":
            if not isinstance(params, dict):
                raise MCPServerError(INVALID_PARAMS, "params must be object")
            result = await self.call_tool(params)
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        raise MCPServerError(
            METHOD_NOT_FOUND, f"method not supported: {method}")

    async def handle_line(self, line: str) -> Optional[Dict[str, Any]]:
        """从 stdin 读一行 JSON，返回响应 dict；解析失败返回 parse error 响应。"""
        line = line.strip()
        if not line:
            return None
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": PARSE_ERROR, "message": f"parse error: {exc}"},
            }
        if not isinstance(message, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": INVALID_REQUEST,
                    "message": "request must be a JSON object",
                },
            }
        try:
            return await self.dispatch(message)
        except MCPServerError as exc:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": exc.code, "message": exc.message},
            }
        except Exception as exc:  # noqa: BLE001 — 兜底归一成内部错误
            log.exception("MCP dispatch error")
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": INTERNAL_ERROR, "message": f"{type(exc).__name__}: {exc}"},
            }


# ---------------------------------------------------------------------------
# stdio 循环
# ---------------------------------------------------------------------------


class _StdinReader(asyncio.Protocol):
    """stdin 读取 protocol（无 IO 传输，仅标记 EOF）。"""

    def __init__(self, on_line: Any) -> None:
        self._on_line = on_line

    def connection_made(self, transport: Any) -> None:  # noqa: D401
        pass

    def connection_lost(self, exc: Optional[Exception]) -> None:
        pass


async def serve_stdio(server: MCPServer) -> None:
    """stdio JSON-RPC 循环（Windows 管道兼容，零 asyncio 传输）。

    Windows proactor 事件循环对「重定向的 stdin/stdout 管道」绑 IOCP 会抛
    OSError [WinError 6]（句柄不可绑定）；`connect_read_pipe` 还会要求
    StreamReader 协议（on_stdout 场景不适用）。因此**不在 asyncio 传输层
    碰 stdin/stdout**：stdin 用后台线程同步逐行读，经
    `loop.call_soon_threadsafe` 抛给事件循环处理；stdout 直接
    `sys.stdout.buffer.write` + flush（线程安全，无缓冲丢行）。
    """
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_line(line: str) -> None:
        loop.create_task(_handle_line(server, line))

    def _reader_loop() -> None:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            loop.call_soon_threadsafe(_on_line, line)
        loop.call_soon_threadsafe(stop_event.set)

    threading.Thread(target=_reader_loop, daemon=True).start()
    await stop_event.wait()


async def _handle_line(server: MCPServer, line: str) -> None:
    """处理一行 JSON-RPC 请求并写 stdout（同步缓冲+flush，线程安全）。"""
    response = await server.handle_line(line)
    if response is None:
        return
    try:
        sys.stdout.buffer.write(
            (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()
    except Exception:  # noqa: BLE001 — stdout 关闭时静默
        pass


class _StdoutWriter(asyncio.Protocol):
    """写 stdout 的 asyncio.Protocol（缓冲 + flush，兼容 Windows 管道）。"""

    def __init__(self) -> None:
        self._buf = bytearray()

    def write(self, data: bytes) -> None:
        self._buf.extend(data)
        try:
            sys.stdout.buffer.write(bytes(self._buf))
            sys.stdout.buffer.flush()
            self._buf.clear()
        except Exception:  # noqa: BLE001 — stdout 关闭时静默
            pass

    def connection_made(self, transport: Any) -> None:  # noqa: D401
        self._transport = transport

    def connection_lost(self, exc: Optional[Exception]) -> None:
        pass


def main() -> int:
    """`python -m src.mcp.run` 入口。"""
    server = MCPServer()
    try:
        asyncio.run(serve_stdio(server))
    except KeyboardInterrupt:
        pass
    return 0


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


class MCPServerError(Exception):
    """JSON-RPC 协议级错误（带 code / message，可选 tool name）。"""

    def __init__(self, code: int, message: str, tool: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.tool = tool


def _serialize(output: Any) -> str:
    """工具输出序列化成文本。dict/list → pretty JSON；其它 → str。"""
    if isinstance(output, (dict, list, tuple)):
        try:
            return json.dumps(output, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            pass
    return str(output)
