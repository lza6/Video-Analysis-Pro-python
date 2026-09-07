# 插件开发

TingFeng Hermes 插件框架用 **contextvars**（Python 标准库）替代 Cordis 重依赖，声明式 YAML patch，运行时热插拔。

## 目录结构

```
plugins/<plugin-name>/
├── plugin.yaml      # 声明式 patch（挂载点 / 依赖 / 版本）
├── main.py          # 入口（register(ctx) 函数或 Plugin 子类）
└── README.md        # 插件说明
```

`PluginLoader` 扫描顺序：`config/plugins.yml` 声明 → `src/core/plugins/builtin/` 内置 → 第三方 entry_points。

## plugin.yaml 声明

```yaml
name: my-plugin
version: 1.0.0
description: 示例插件
depends_on: []          # 依赖的其他插件
permissions:            # 权限模型
  - tool:execute        # 调用 Agent 工具
  - im:send             # 发送 IM 消息
mounts:
  - point: agent.tool.register
    handler: main.register_tool
  - point: im.adapter.register
    handler: main.register_adapter
```

## PluginContext

`src/core/plugins/context.py` 的 `PluginContext` 是 contextvars.ContextVar：

```python
from src.core.plugins.context import PluginContext

def register(ctx: PluginContext) -> None:
    # 注册 Agent 工具 / IM adapter / 远程 tunnel
    disposer = ctx.tools.register(my_tool)
    ctx.on_dispose(disposer)   # 卸载时逆序调用
```

- **不要**回写或删除已有 SessionEvent
- **不要**让插件直接写 Python hot-patch，必须走 `loader.py` + `patch.py`
- **不要**引入 Cordis 重依赖

## 权限模型

插件在 `plugin.yaml` 声明 `permissions`，加载器校验后注入 `PluginContext`。未声明的权限调用会被拒绝并记录到 `logs/`。

## 下一步

- [API 参考](./api-reference) — Agent / IM / 插件相关 HTTP 端点
