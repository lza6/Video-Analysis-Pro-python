"""示例插件：演示 plugins/<name>/main.py 的最小可用写法。

这个插件**不做任何有害的事**，只展示插件的两个真实生效面：

  1. ``ctx.register_tool(...)`` —— 往 agent 的**同一个** ToolRegistry 注册一个
     只读工具 ``example_hello``（无副作用、不触网、不写盘）。它与内置工具享有
     同一套 scope_guard 治理：``example_hello`` 按名字判定为 read → 放行；
     若插件注册的是写工具，同样会被 ask（超时默认拒绝）。
  2. ``ctx.append_system_prompt(...)`` —— 往 system prompt 追加一段说明，
     经 ``src/web/routers/agent.py`` 拼接到 build_agent_system_prompt 的结果后面。

它是 ``docs/guide/plugin-development.md`` 描述的目录布局的**可运行参照**：
v10.4.0 之前 loader 只会 importlib 已安装的模块、PluginLoader 在 src/ 里零调用，
这个布局实际上跑不起来。

启用方式：把同目录 ``plugin.yaml`` 的 ``enabled`` 改成 true（或设
``VAP_PLUGIN_DIR`` 指向另一个目录），重启后端；用 ``GET /api/plugins``
可看到 ``loaded`` 里出现 ``example-hello``。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.core.plugins.context import PluginContext
from src.core.tools.definition import ToolDefinition

PLUGIN_ID = "example-hello"
PLUGIN_NAME = "示例插件（Hello）"

TOOL_NAME = "example_hello"


async def _example_hello(greeting: str = "你好", **_: Any) -> Dict[str, Any]:
    """只读工具实现：把问候语回显为结构化结果（无副作用）。"""
    return {"ok": True, "greeting": greeting, "plugin": PLUGIN_ID}


def register(
    ctx: PluginContext, config: Optional[Dict[str, Any]] = None
) -> Optional[Callable[[], None]]:
    """注册副作用并返回 disposer。

    Args:
        ctx: 插件运行时上下文（可注册工具 / 追加 system prompt / 读写 setting）。
        config: 来自 plugin.yaml 的 ``config:`` 段。
    """
    greeting = str((config or {}).get("greeting", "你好"))

    # 面 1：注册工具（进入与内置工具相同的 registry + scope_guard 治理）
    tool_def = ToolDefinition(
        name=TOOL_NAME,
        description=(
            "示例插件提供的只读工具：回显问候语，用于验证插件工具面已接通。"
            "无副作用、不访问网络。"
        ),
        execute_callback=_example_hello,
        input_schema={
            "type": "object",
            "properties": {
                "greeting": {"type": "string", "description": "要回显的问候语"},
            },
        },
    )
    ctx.register_tool(tool_def, plugin_id=PLUGIN_ID)

    # 面 2：追加 system prompt（agent 路由会把它拼进模型上下文）
    ctx.append_system_prompt(
        f"（{PLUGIN_NAME} 已加载）你有一个只读工具 {TOOL_NAME} 可用；"
        f"请在回答开头用「{greeting}」打招呼。",
        plugin_id=PLUGIN_ID,
    )
    ctx.set_setting("example_hello.greeting", greeting, plugin_id=PLUGIN_ID)

    def _dispose() -> None:
        # 真实插件在这里释放资源（关连接、注销工具等）。本示例无资源。
        pass

    return _dispose
