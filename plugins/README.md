# plugins/ — 第三方插件目录

本目录存放**目录型插件**。运行时 `PluginLoader.load_from_dir()` 会扫描这里的
每个子目录，加载其中带 `main.py` 的插件。

## 目录布局

```
plugins/<plugin-name>/
├── plugin.yaml      # 可选：声明 id / name / enabled / config
├── main.py          # 必需：导出 register(ctx) 或 PLUGIN_CLASS
└── README.md        # 可选：插件说明
```

## plugin.yaml（可选）

```yaml
id: my-plugin
name: 我的插件
enabled: true          # false 时跳过加载
config:
  key: value           # 通过 register(ctx, config) 的第二参传入
```

缺少 `plugin.yaml` 时用**目录名**作为插件 id，默认启用。

## main.py 两种写法

### 1. 函数式（推荐）

```python
from src.core.plugins.context import PluginContext


def register(ctx: PluginContext, config=None):
    """注册副作用，返回 disposer（或不返回）。"""
    ctx.append_system_prompt("（来自我的插件）请用中文回答。", plugin_id="my-plugin")

    def _dispose():
        pass  # 清理资源

    return _dispose
```

### 2. 类式

```python
from src.core.plugins.patch import Plugin


class MyPlugin(Plugin):
    def apply(self, ctx, config=None):
        ctx.append_system_prompt("...", plugin_id=self.spec.id)
        return None


PLUGIN_CLASS = MyPlugin
```

## 加载顺序与开关

| 来源 | 说明 |
|------|------|
| `config/plugins.yml` | 声明式列表（id/name/enabled/module/config），按顺序加载 |
| `src/core/plugins/builtin/` | 项目内置插件（每个模块导出 `register(ctx)` 或 `PLUGIN_CLASS`） |
| `plugins/<name>/` | 本目录（第三方落地形态） |

环境变量：

- `VAP_PLUGIN_DIR`（默认 `plugins`）：目录型插件的扫描根目录
- `VAP_PLUGIN_CONFIG`（默认 `config/plugins.yml`）：声明式插件清单路径

## 安全须知

插件是**代码执行**入口。只放你自己信任的插件到本目录；不要把本目录指向
可被第三方写入的位置。插件的工具注册会进入与内置工具**同一套**审批治理
（`scope_guard`：写操作 `ask`、危险写 `ask` 且超时默认拒绝）。

## 示例

`example-hello/` 是一个最小可运行示例（默认 `enabled: false`，不注册任何东西，
仅演示目录布局与 `register(ctx, config)` 契约）。
