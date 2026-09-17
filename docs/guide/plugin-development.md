# 插件开发

TingFeng Hermes 的插件框架用 **contextvars**（Python 标准库）替代 Cordis 重依赖：插件通过
`PluginContext` 注册工具 / 系统提示词 / 运行时配置，`PluginLoader` 负责发现、加载与逆序卸载。

> **v10.4.0 修正**：这份文档此前描述过 `permissions` / `mounts` / `depends_on` /
> entry_points / `ctx.tools.register` / `ctx.on_dispose` 等**代码里并不存在**的契约，
> 而且 loader 当时只能 `importlib` 已安装模块 —— 文档里的目录布局从未真正跑起来，
> `PluginLoader` 在 `src/` 里零调用。现在文档描述的每一项都对应真实代码路径，
> 并附带可执行的自查方式。

## 目录结构

```
plugins/<plugin-name>/
├── plugin.yaml      # 可选：id / name / enabled / config
├── main.py          # 必需：register(ctx[, config]) 或 PLUGIN_CLASS
└── README.md        # 可选：插件说明
```

内置插件另有一套：`src/core/plugins/builtin/`，模块路径由
`src/core/plugins/loader.py` 的 `BUILTIN_PLUGIN_MODULES` 声明，启动时由 `load_builtin()` 加载。

### 发现顺序

1. `config/plugins.yml` 声明的模块（`PluginLoader.load_from_specs`，只接受**可 import** 的模块路径）
2. `src/core/plugins/builtin/` 内置插件（`load_builtin`）
3. `plugins/<name>/main.py` 目录型插件（`load_from_dir`，目录可用 `VAP_PLUGIN_DIR` 覆盖，默认 `plugins`）

第 3 条是第三方插件推荐的落地形态：**不需要 pip 安装，也不需要进 sys.path**。

## plugin.yaml

```yaml
id: my-plugin            # 省略则用目录名
name: 我的插件            # 展示名
enabled: true            # false = 发现但不加载（默认 true）
config:                  # 原样传给 register(ctx, config) 的第二个参数
  greeting: 你好
```

字段就这 4 个。**未实现的**（不要照抄旧版文档）：`version` / `description` / `permissions` /
`depends_on` / `mounts` / entry_points。传了会被忽略，不会报错。

## 入口约定

`main.py` 二选一：

```python
# 约定 1：register(ctx) 或 register(ctx, config)
from typing import Any, Callable, Dict, Optional
from src.core.plugins.context import PluginContext

PLUGIN_ID = "my-plugin"


def register(ctx: PluginContext, config: Optional[Dict[str, Any]] = None) -> Optional[Callable[[], None]]:
    ...
    return None  # 或返回一个清理闭包
```

```python
# 约定 2：PLUGIN_CLASS 指向 Plugin 子类（apply(ctx, config) 返回 disposer 或 None）
from src.core.plugins.patch import Plugin

class MyPlugin(Plugin):
    def apply(self, ctx, config=None):
        return None

PLUGIN_CLASS = MyPlugin
```

两者都没有 → 该插件被跳过并记 warning（不影响其他插件）。

## PluginContext API（实际存在的方法）

| 方法 | 作用 |
| --- | --- |
| `ctx.register_tool(defn, plugin_id=...)` | 注册 `ToolDefinition`，返回 disposer |
| `ctx.append_system_prompt(text, plugin_id=...)` | 追加系统提示词片段（拼在 `build_agent_system_prompt` 结果之后） |
| `ctx.prepend_system_prompt(text, plugin_id=...)` | 前置提示词片段 |
| `ctx.set_setting(key, value, plugin_id=...)` / `ctx.get_setting(key, default=None)` | 插件运行时配置 |
| `ctx.on(phase, hook, plugin_id=...)` | 注册 turn 阶段钩子（记录进 effect 审计） |
| `ctx.get_state(plugin_id)` / `ctx.plugins` | 查看某个插件注册的副作用与 disposer |
| `PluginContext.current()` | 取当前 active context（contextvars） |

`plugin_id` 建议每个插件都传：它是**卸载时的归属键**，也是 effect 审计的分组键。

### 工具会走同一套治理

插件工具注册进的是**与内置工具相同的** `ToolRegistry`，因此自动受
`src/core/tools/scope_guard.py` 管控：按工具名分类，读放行 / 写 ask（超时默认拒绝）/ 危险写 ask。
插件**无法**绕过审批 —— 想加一个「删除文件」的工具，它会和内置的危险写一样需要人工确认。

## 卸载语义

`PluginLoader.dispose_all()` 逆序调用每个插件的卸载器。卸载器由 loader 组合而成：

1. 先跑插件 `register()` 自己 return 的 disposer；
2. 再逆序调 `ctx.register_tool()` 等注册的副作用 disposer。

所以**插件作者不必**手写「注销我刚才注册的工具」，也不该假定自己不清理就没问题。

## 自查：我的插件到底加载了吗

```bash
curl http://127.0.0.1:8000/api/plugins
```

返回：

| 字段 | 含义 |
| --- | --- |
| `dir` | 生效的插件目录（`VAP_PLUGIN_DIR` 或默认 `plugins`） |
| `discovered` | 目录扫描结果：`id/name/enabled/path/has_main/config_keys`（**不执行插件代码**） |
| `loaded` | 最近一次 react 运行中真正加载成功的插件 id |
| `loaded_effects` | 每个插件的副作用审计（tool / system_prompt / setting / hook） |
| `loaded_at_least_once` | 是否已经历过一次 agent 运行（false 时 `loaded` 为空是正常的） |

排错对照：

- `discovered` 为空 → 目录不存在，或 `VAP_PLUGIN_DIR` 指错了。
- `discovered` 里 `has_main: false` → 文件名不是 `main.py`（大小写敏感）。
- `enabled: false` → 声明里关掉了。
- 发现正常但 `loaded` 里没有 → 看后端日志 `[plugins] ...`：模块 import 失败 / `register` 抛异常都会被记 warning。

## 已知边界

- **只在 react 后端加载**（`VAP_AGENT_BACKEND=react`，v10.3.1 起为默认）。legacy 后端用的是
  另一套 `src/core/agent_tools.ToolRegistry`，**没有** scope_guard 审批；把插件工具灌进去会让
  插件绕过审批，因此有意不接（有反向测试守着这条决策）。
- **没有权限声明模型**。插件的运行权限 = 插件目录的写权限 + scope_guard 的运行时审批。
  不要在无审批期望的场景下把插件目录暴露给不可信来源。
- 插件在**首次 agent 运行**时加载（与请求级 registry 同生命周期），不是进程启动时加载。

## 下一步

- [API 参考](./api-reference) — Agent / IM / 插件相关 HTTP 端点
- [插件示例](../../plugins/example-hello) — 可运行的最小插件（工具 + 提示词两面）
