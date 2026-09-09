"""工具定义与 DSL。

参考 DSH `packages/core/tool-host/src/tool.ts` 的 ToolDefinition 概念：
工具由 name / description / input_schema / output_schema / execute_callback
组成。input_schema 用 JSON Schema 子集（type/properties/required），execute_callback
是 async 回调。

`define_tool` 是声明式 DSL：装饰器式或函数式都可，本实现走函数式 dataclass +
build helper，避免装饰器对类型的扭曲。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Union

# 工具回调签名：接受 kwargs，返回任意 JSON 可序列化结果
ToolCallback = Callable[..., Awaitable[Any]]
SyncToolCallback = Callable[..., Any]


@dataclass(frozen=True)
class ToolDefinition:
    """单个工具的不可变定义。

    Attributes:
        name: 工具名（LLM 调用时引用，唯一）。
        description: 给 LLM 看的自然语言描述。
        input_schema: JSON Schema 子集，约束 kwargs。{"type": "object",
            "properties": {...}, "required": [...]}。
        output_schema: 可选输出形状描述，供 post_execute 校验。
        execute_callback: async 回调，实际跑工具逻辑。
    """

    name: str
    description: str
    execute_callback: ToolCallback
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None

    def to_llm_schema(self) -> Dict[str, Any]:
        """投影成 LLM tool schema（OpenAI function-calling 格式）。

        NVIDIA integrate 的 OpenAI 兼容端点要求 tools[].type="function" +
        tools[].function.{name,description,parameters}；缺 type 会 400
        "missing field `type`"（实测 2026-09-07）。VAP 本地用 name/input_schema
        简化投影，此处补齐 OpenAI 兼容包装。
        """
        parameters = dict(self.input_schema) if self.input_schema else {
            "type": "object",
            "properties": {},
        }
        parameters.setdefault("type", "object")
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


def define_tool(
    name: str,
    description: str,
    *,
    input_schema: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    callback: Optional[ToolCallback] = None,
) -> Callable[[ToolCallback], ToolDefinition] | ToolDefinition:
    """声明式 DSL：`define_tool("x", "...", input_schema=...)(async def cb) -> ToolDefinition`。

    也支持直接传 callback 一次成形。返回 ToolDefinition（不可变）。
    """
    if callback is not None:
        return ToolDefinition(
            name=name,
            description=description,
            execute_callback=callback,
            input_schema=input_schema or {},
            output_schema=output_schema,
        )

    def _wrap(cb: ToolCallback) -> ToolDefinition:
        return ToolDefinition(
            name=name,
            description=description,
            execute_callback=cb,
            input_schema=input_schema or {},
            output_schema=output_schema,
        )

    return _wrap


def _coerce_async(cb: Union[ToolCallback, SyncToolCallback]) -> ToolCallback:
    """把同步回调包装成 async（轻量，不引入 extra deps）。"""
    import inspect

    if inspect.iscoroutinefunction(cb):
        return cb  # type: ignore[return-value]

    async def _wrapped(*args: Any, **kwargs: Any) -> Any:
        return cb(*args, **kwargs)

    return _wrapped
