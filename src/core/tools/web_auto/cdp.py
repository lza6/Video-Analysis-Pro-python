"""CDP（Chrome DevTools Protocol）调试工具——P2-3。

把 CDP 调试能力暴露为 Agent 工具：

  - `cdp_list_targets`      列出所有调试目标（页面 / 扩展 / 其它）
  - `cdp_attach`            附加到指定 target（保存 WebSocket 连接）
  - `cdp_evaluate`          在附加 target 上执行**只读** JS 表达式（默认放行）
  - `cdp_eval_write`        执行 JS 表达式（含写入/导航，写操作走审批）
  - `cdp_close`             关闭调试连接

真实连接策略（`VAP_CDP_URL`）：
  - 显式提供 CDP 端点 URL（如 `http://127.0.0.1:9222`）→ 尝试真实连接
    （经 Chrome DevTools HTTP /json 列表 + WebSocket 调试）。连接失败 → mock
    降级并明确标注（不伪造）。
  - 不提供 → mock 后端（返回占位 target 列表 + 标注"未连真实调试器"）。

付费红线：CDP 仅连本机调试端口，不调用任何付费云端浏览器服务。
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional

from src.core.tools.definition import ToolDefinition

log = logging.getLogger("core.tools.web_auto.cdp")

#: CDP 端点环境变量（如 http://127.0.0.1:9222）。缺省 → mock。
CDP_URL_ENV = "VAP_CDP_URL"
#: evaluate 超时秒数
_EVALUATE_TIMEOUT = 8.0


class _CdpUnavailableError(RuntimeError):
    """CDP 端点不可用（未提供 URL / 连接失败 / 依赖缺失）。"""


class _BaseCdpBackend:
    mode = "unknown"

    def list_targets(self) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def attach(self, target_id: Optional[str]) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def evaluate(self, expression: str, *, write: bool) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


class MockCdpBackend(_BaseCdpBackend):
    """Mock 后端：无真实调试连接，返回带 `mode: "mock"` 标注的占位结果。

    不伪造 evaluate 的真实执行结果——`evaluate` 永远返回
    `mock-evaluated: <expr>` 字符串，且带 `mock: true` 标注。
    """

    def __init__(self, note: str) -> None:
        super().__init__()
        self.mode = "mock"
        self._note = note
        self._attached: Optional[str] = None

    def _stamp(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "mode": "mock", "note": self._note, **data}

    def list_targets(self) -> Dict[str, Any]:
        return self._stamp({"targets": [
            {"id": "mock-target-1", "type": "page",
             "title": "mock 页面（占位）", "url": "about:blank",
             "mock": True},
        ]})

    def attach(self, target_id: Optional[str]) -> Dict[str, Any]:
        self._attached = target_id or "mock-target-1"
        return self._stamp({"target_id": self._attached,
                           "description": "mock 附加（未建立真实 WS）"})

    def evaluate(self, expression: str, *, write: bool) -> Dict[str, Any]:
        if self._attached is None:
            return {"ok": False, "mode": "mock", "error": "尚未 attach"}
        return self._stamp({
            "result": "mock-evaluated",
            "value": f"mock-evaluated: {expression[:60]}",
            "mock": True,
        })

    def close(self) -> Dict[str, Any]:
        r = self._stamp({"closed": True})
        self._attached = None
        return r


class CdpBackend(_BaseCdpBackend):
    """真实 CDP 后端：Chrome DevTools HTTP 发现 + WebSocket 调试连接。

    依赖：`requests` + `websockets`（本项目 core requirements 已有）。
    连接失败 / 依赖缺失 → 抛出 `_CdpUnavailableError`（由 CdpTool 降级 mock）。
    """

    def __init__(self, endpoint: str) -> None:
        super().__init__()
        self.mode = "cdp"
        self._endpoint = endpoint.rstrip("/")
        self._ws: Any = None
        self._target_id: Optional[str] = None
        self._send_lock = threading.Lock()

    # ---- 发现 / 连接 ----
    def _fetch_targets(self) -> List[Dict[str, Any]]:
        try:
            import requests  # noqa: PLC0415 - 延迟导入避免 core 启动成本
        except ImportError as e:
            raise _CdpUnavailableError(f"requests 不可用: {e}") from e
        try:
            r = requests.get(f"{self._endpoint}/json/list", timeout=3)
            r.raise_for_status()
            return list(r.json())
        except Exception as e:  # noqa: BLE001
            raise _CdpUnavailableError(
                f"CDP 端点 {self._endpoint} 连接失败: {e}") from e

    def _connect_ws(self, ws_url: str) -> Any:
        try:
            from websockets.sync.client import connect  # noqa: PLC0415
        except ImportError as e:
            raise _CdpUnavailableError(f"websockets 不可用: {e}") from e
        try:
            return connect(ws_url, open_timeout=4)
        except Exception as e:  # noqa: BLE001
            raise _CdpUnavailableError(f"WebSocket 连接失败: {e}") from e

    # ---- 后端接口 ----
    def list_targets(self) -> Dict[str, Any]:
        targets = self._fetch_targets()
        slim = [
            {"id": t.get("id"), "type": t.get("type"),
             "title": t.get("title"), "url": t.get("url")}
            for t in targets
        ]
        return {"ok": True, "mode": "cdp", "targets": slim}

    def attach(self, target_id: Optional[str]) -> Dict[str, Any]:
        targets = self._fetch_targets()
        chosen = None
        if target_id:
            chosen = next((t for t in targets if t.get("id") == target_id), None)
        else:
            chosen = next((t for t in targets if t.get("type") == "page"), None)
        if chosen is None:
            raise ValueError(f"target 不存在: {target_id or '<page>'}")
        ws_url = chosen.get("webSocketDebuggerUrl")
        if not ws_url:
            raise ValueError("target 无 webSocketDebuggerUrl，无法附加")
        self._ws = self._connect_ws(ws_url)
        self._target_id = chosen.get("id")
        return {"ok": True, "mode": "cdp",
                "target_id": self._target_id,
                "title": chosen.get("title"), "url": chosen.get("url")}

    def _send(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._ws is None:
            raise ValueError("尚未 attach")
        with self._send_lock:
            msg = {"id": 1, "method": method, "params": params or {}}
            self._ws.send(json.dumps(msg))
            resp = json.loads(self._ws.recv())
        if "error" in resp:
            raise RuntimeError(
                f"CDP {method} error: {resp['error']}")
        return resp.get("result", {})

    def evaluate(self, expression: str, *, write: bool) -> Dict[str, Any]:
        result = self._send("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": False,
        })
        exc = result.get("exceptionDetails")
        if exc:
            return {"ok": False, "mode": "cdp",
                    "error": f"evaluate 异常: {exc.get('text', '')}"}
        value = result.get("result", {}).get("value")
        return {"ok": True, "mode": "cdp", "result": value,
                "value": value if isinstance(value, str) else json.dumps(
                    value, ensure_ascii=False) if value is not None else None}

    def close(self) -> Dict[str, Any]:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        self._ws = None
        self._target_id = None
        return {"ok": True, "mode": "cdp", "closed": True}


class CdpTool:
    """CdpTool 工厂：按 VAP_CDP_URL 解析后端，产出 CDP 工具定义。

    真实连接可选：提供 `VAP_CDP_URL` 且依赖可用 → 真实；否则 mock 且标注。
    """

    def __init__(self, cdp_url: Optional[str] = None, *,
                 force_mock: bool = False) -> None:
        self._lock = threading.Lock()
        self._url = cdp_url if cdp_url is not None else os.environ.get(CDP_URL_ENV, "")
        self._force_mock = force_mock
        self._backend: _BaseCdpBackend = self._resolve_backend()

    def _resolve_backend(self) -> _BaseCdpBackend:
        if self._force_mock:
            return MockCdpBackend("VAP_CDP_URL 未配置（或 force_mock）→ mock 模式")
        url = self._url.strip().rstrip("/")
        if not url:
            return MockCdpBackend(
                "未配置 VAP_CDP_URL → mock 模式（未连真实调试器，不伪造）")
        if not (importlib.util.find_spec("requests")
                and importlib.util.find_spec("websockets")):
            return MockCdpBackend(
                "requests/websockets 不可用 → mock 降级")
        return CdpBackend(url)

    @property
    def mode(self) -> str:
        return self._backend.mode

    # ---- 底层驱动 ----
    def _exec(self, fn: Callable[..., Dict[str, Any]],
              *args: Any, **kwargs: Any) -> Dict[str, Any]:
        with self._lock:

            def _target() -> Dict[str, Any]:
                try:
                    return fn(*args, **kwargs)
                except _CdpUnavailableError as e:
                    self._backend = MockCdpBackend(
                        f"真实 CDP 连接失败 → mock 降级: {e}")
                    return self._backend.list_targets()
                except Exception as e:  # noqa: BLE001 - 工具错误归一
                    return {"ok": False, "mode": self._backend.mode,
                            "error": f"{type(e).__name__}: {e}"}

            return _target()

    # ---- 公开 async 方法（测试直接驱动） ----
    async def list_targets(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self._exec, self._backend.list_targets)

    async def attach(self, target_id: Optional[str] = None) -> Dict[str, Any]:
        return await asyncio.to_thread(self._exec, self._backend.attach, target_id)

    async def evaluate(self, expression: str, *, write: bool = False) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._exec, self._backend.evaluate, expression, write=write)

    async def close(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self._exec, self._backend.close)

    # ---- ToolDefinition 装配 ----
    def definitions(self) -> List[ToolDefinition]:
        """产出 5 个 CDP 工具定义。

        读操作 `cdp_list_targets` / `cdp_attach` / `cdp_evaluate`（只读命名）
        默认 allow；写操作 `cdp_eval_write`（含 write 关键词 → 分类 write）走 ask。
        """

        async def _list() -> Dict[str, Any]:
            return await self.list_targets()

        async def _attach(target_id: Optional[str] = None) -> Dict[str, Any]:
            return await self.attach(target_id=target_id)

        async def _evaluate(expression: str) -> Dict[str, Any]:
            return await self.evaluate(expression, write=False)

        async def _execute(expression: str) -> Dict[str, Any]:
            return await self.evaluate(expression, write=True)

        async def _close() -> Dict[str, Any]:
            return await self.close()

        return [
            ToolDefinition(
                name="cdp_list_targets",
                description=(
                    "列出 CDP 调试目标（页面/扩展/其它）。"
                    "只读，直接放行。"
                ),
                execute_callback=_list,
                input_schema={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="cdp_attach",
                description=(
                    "附加到指定 CDP target（省略 target_id 选第一个 page 目标）。"
                    "连接类操作，放行。"
                ),
                execute_callback=_attach,
                input_schema={"type": "object",
                              "properties": {"target_id": {"type": "string"}}},
            ),
            ToolDefinition(
                name="cdp_evaluate",
                description=(
                    "在已附加 target 上执行**只读** JS 表达式（如读取 DOM 值）。"
                    "只读 evaluate，直接放行。注意：若表达式有写入/导航副作用，"
                    "请改用 cdp_eval_write（会走审批）。"
                ),
                execute_callback=_evaluate,
                input_schema={"type": "object",
                              "properties": {"expression": {"type": "string"}},
                              "required": ["expression"]},
            ),
            ToolDefinition(
                name="cdp_eval_write",
                description=(
                    "在已附加 target 上执行 JS 表达式（**允许写入 / 导航副作用**）。"
                    "写操作（工具名含 write，scope_guard 分类 write），需人工审批。"
                ),
                execute_callback=_execute,
                input_schema={"type": "object",
                              "properties": {"expression": {"type": "string"}},
                              "required": ["expression"]},
            ),
            ToolDefinition(
                name="cdp_close",
                description="关闭 CDP 调试连接，释放资源。",
                execute_callback=_close,
                input_schema={"type": "object", "properties": {}},
            ),
        ]


def build_cdp_tool(*, cdp_url: Optional[str] = None,
                   force_mock: bool = False) -> CdpTool:
    """构造 CdpTool（真实/mock 解析见类 docstring）。"""
    return CdpTool(cdp_url=cdp_url, force_mock=force_mock)


__all__ = ["CdpTool", "build_cdp_tool", "MockCdpBackend", "CdpBackend",
           "CDP_URL_ENV"]