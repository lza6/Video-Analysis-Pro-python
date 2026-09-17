"""插件加载器。

发现插件（扫 plugins/ 目录 + entry_points + config/plugins.yml 声明），
调 apply(ctx, config) 入口，收集 disposer。

加载顺序：
  1. config/plugins.yml 声明的（按顺序）
  2. src/core/plugins/builtin/ 下的内置插件（约定：每个模块导出
     register(ctx) 函数 或 Plugin 子类）
  3. plugins/<name>/main.py 目录型插件（第三方落地形态）

unloader：dispose_all() 调所有 disposer，逆序。

v10.4.0 (P0-3) 补齐 ``load_from_dir``
------------------------------------
docs/guide/plugin-development.md 一直描述着 ``plugins/<plugin-name>/main.py``
这2种目录布局，但 loader 此前只会 importlib.import_module(``spec.module``)
—— 只能加载**已安装且可 import** 的模块，文档里的目录布局实际上从未生效
（且 PluginLoader 当时在 src/ 里零调用）。现在补上基于文件路径的加载，
让文档与实现一致。
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

#: 内置插件模块（与 src/core/plugins/builtin/ 下的模块一一对应）。
BUILTIN_PLUGIN_MODULES: tuple[str, ...] = (
    "src.core.plugins.builtin.video_analysis",
)

#: 默认插件目录（可被 VAP_PLUGIN_DIR 覆盖）。
DEFAULT_PLUGIN_DIR = "plugins"


def builtin_plugin_modules() -> List[str]:
    """返回内置插件模块路径列表（供调用方显式传给 load_builtin）。"""
    return list(BUILTIN_PLUGIN_MODULES)


def describe_plugin_dir(plugin_dir: str) -> List[Dict[str, Any]]:
    """扫描插件目录，返回声明摘要（**不执行插件代码**）。

    与 ``PluginLoader.load_from_dir`` 共用同一套解析规则（plugin.yaml →
    PluginSpec，缺省 id = 目录名，默认 enabled=True），因此「列表里显示的启停
    状态」与「加载器实际行为」不会漂移。

    为什么需要它：v10.4.0 之前插件目录布局在文档里存在、在代码里跑不起来，
    用户无法自查「我的插件到底被认到了吗」。这个只读探测让 ``GET /api/plugins``
    在**首次 agent 运行之前**就能显示发现结果（含 enabled/main.py 缺失诊断）。

    Args:
        plugin_dir: 插件根目录（如 ``plugins``）。

    Returns:
        每个子目录一项：id / name / enabled / path / has_main / config_keys。
        目录不存在时返回空列表（不抛异常）。
    """
    base = Path(plugin_dir)
    if not base.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        main_py = entry / "main.py"
        spec = PluginLoader._spec_from_plugin_dir(entry)
        out.append({
            "id": spec.id,
            "name": spec.name,
            "enabled": bool(spec.enabled),
            "path": str(entry),
            "has_main": main_py.is_file(),
            "config_keys": sorted(spec.config.keys()),
        })
    return out

from src.core.plugins.context import PluginContext
from src.core.plugins.patch import Plugin, PluginSpec


@dataclass
class _LoadedPlugin:
    spec: PluginSpec
    disposer: Optional[Callable[[], None]] = None


class PluginLoader:
    """插件加载/卸载器。"""

    def __init__(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._loaded: List[_LoadedPlugin] = []

    def load_from_specs(self, specs: List[PluginSpec]) -> None:
        """按声明式 spec 列表加载。"""
        for spec in specs:
            if not spec.enabled:
                continue
            disposer = self._apply_spec(spec)
            self._loaded.append(_LoadedPlugin(spec=spec, disposer=disposer))

    def load_builtin(self, modules: List[str]) -> None:
        """加载内置插件模块（每个模块导出 register(ctx) 或 Plugin 子类）。

        Args:
            modules: 模块路径列表，如 ["src.core.plugins.builtin.video_analysis"]
        """
        for mod_path in modules:
            try:
                mod = importlib.import_module(mod_path)
            except Exception:
                continue
            # 约定 1：模块有 register(ctx) -> disposer 函数
            if hasattr(mod, "register"):
                fn = getattr(mod, "register")
                if callable(fn):
                    spec = PluginSpec(
                        id=getattr(mod, "PLUGIN_ID", mod_path.split(".")[-1]),
                        name=getattr(mod, "PLUGIN_NAME", mod_path.split(".")[-1]),
                        module=mod_path,
                    )
                    disposer = self._safe_register(fn, spec)
                    self._loaded.append(_LoadedPlugin(
                        spec=spec, disposer=disposer))
                    continue
            # 约定 2：模块有 Plugin 子类 PLUGIN_CLASS
            cls = getattr(mod, "PLUGIN_CLASS", None)
            if cls is not None and isinstance(cls, type) and issubclass(cls, Plugin):
                spec = PluginSpec(
                    id=getattr(mod, "PLUGIN_ID", mod_path.split(".")[-1]),
                    name=getattr(mod, "PLUGIN_NAME", mod_path.split(".")[-1]),
                    module=mod_path,
                )
                plugin = cls(spec)
                disposer = plugin.apply(self.ctx, spec.config) or None
                self._loaded.append(_LoadedPlugin(
                    spec=spec,
                    disposer=self._combine_disposers(spec.id, disposer)))

    def _apply_spec(self, spec: PluginSpec) -> Optional[Callable[[], None]]:
        if spec.module is None:
            return None
        try:
            mod = importlib.import_module(spec.module)
        except Exception:
            return None
        # 优先 Plugin 子类
        cls = getattr(mod, "PLUGIN_CLASS", None)
        if cls is not None and isinstance(cls, type) and issubclass(cls, Plugin):
            return self._combine_disposers(
                spec.id, cls(spec).apply(self.ctx, spec.config) or None)
        # 兜底：register(ctx)
        fn = getattr(mod, "register", None)
        if callable(fn):
            return self._safe_register(fn, spec)
        return None

    def _safe_register(self, fn: Callable, spec: PluginSpec) -> Optional[Callable[[], None]]:
        try:
            sig = inspect.signature(fn)
            if len(sig.parameters) == 1:
                disposer = fn(self.ctx) or None
            else:
                # 双参：(ctx, config)
                params = list(sig.parameters.values())
                if len(params) >= 2:
                    disposer = fn(self.ctx, spec.config) or None
                else:
                    disposer = fn(self.ctx) or None
        except Exception:
            return None
        return self._combine_disposers(spec.id, disposer)

    def _combine_disposers(
        self, plugin_id: str, disposer: Optional[Callable[[], None]]
    ) -> Callable[[], None]:
        """把插件返回的 disposer 与 ctx 记录的副作用 disposer 合成一个卸载器。

        v10.4.0 (P0-3 补漏) —— 真实缺陷：``PluginContext.register_tool()`` 返回的
        disposer 会被存进 ``PluginState.disposers``，但 **没有任何人调它**。
        ``dispose_all()`` 只跑插件 register() 自己 return 的那个闭包；而插件作者
        几乎不会在里面手写「注销我刚才注册的工具」（register 返回 None 更是常见）。
        结果：卸载/热重载后工具残留在 registry 里，同一 name 再注册会被覆盖成
        新、旧对象，旧闭包仍持着资源——「卸载」只是个说法。

        顺序：LIFO。插件自己的 disposer 先跑（它可能依赖工具还在，比如发一条
        告别消息），然后逆序注销 ctx 记录的副作用。
        """
        def _combined() -> None:
            if disposer is not None:
                try:
                    disposer()
                except Exception as e:  # noqa: BLE001 — 卸载失败不该阻断其他清理
                    log.warning("[plugins] %s 自带 disposer 抛异常: %s", plugin_id, e)
            state = self.ctx.get_state(plugin_id)
            if state is None:
                return
            for d in reversed(state.disposers):
                try:
                    d()
                except Exception as e:  # noqa: BLE001
                    log.warning("[plugins] %s 副作用 disposer 抛异常: %s", plugin_id, e)
            state.disposers.clear()

        return _combined

    # ---- v10.4.0 (P0-3)：目录型插件（plugins/<name>/main.py） ----

    def load_from_dir(self, plugin_dir: str) -> List[str]:
        """扫 plugins/<name>/main.py 并加载，返回成功加载的插件 id 列表。

        目录布局（与 docs/guide/plugin-development.md 一致）::

            plugins/<name>/
            ├── plugin.yaml   # 可选：id / name / enabled / config
            ├── main.py       # 必需：register(ctx) 或 PLUGIN_CLASS
            └── README.md

        防御性：目录不存在 / 单个插件报错 / 无 main.py 都只记日志、不影响其他插件，
        也不阻断启动（不抛异常）。
        """
        base = Path(plugin_dir)
        loaded: List[str] = []
        if not base.is_dir():
            log.info("[plugins] 插件目录不存在，跳过: %s", plugin_dir)
            return loaded
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            main_py = entry / "main.py"
            if not main_py.is_file():
                continue
            spec = self._spec_from_plugin_dir(entry)
            if not spec.enabled:
                log.info("[plugins] 插件 %s 已禁用，跳过", spec.id)
                continue
            plugin_id = self._apply_path_plugin(spec, main_py)
            if plugin_id:
                loaded.append(plugin_id)
        return loaded

    @staticmethod
    def _spec_from_plugin_dir(plugin_dir: Path) -> PluginSpec:
        """读 plugin.yaml（可选）得到 PluginSpec；读不到则用目录名当 id。"""
        spec = PluginSpec(id=plugin_dir.name, name=plugin_dir.name,
                          module=str(plugin_dir / "main.py"))
        yaml_path = plugin_dir / "plugin.yaml"
        if not yaml_path.is_file():
            return spec
        try:
            import yaml  # type: ignore[import-untyped]
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception as e:  # noqa: BLE001 — 声明文件坏了不该阻断加载
            log.warning("[plugins] %s 的 plugin.yaml 解析失败: %s", plugin_dir.name, e)
            return spec
        if not isinstance(data, dict):
            return spec
        return PluginSpec(
            id=str(data.get("id") or plugin_dir.name),
            name=str(data.get("name") or plugin_dir.name),
            config=data.get("config", {}) or {},
            enabled=bool(data.get("enabled", True)),
            module=str(plugin_dir / "main.py"),
        )

    def _apply_path_plugin(self, spec: PluginSpec, main_py: Path) -> Optional[str]:
        """从文件路径加载插件模块并应用；成功返回 plugin id，失败返回 None。"""
        module = self._load_module_from_path(spec.id, main_py)
        if module is None:
            return None
        # 约定 1：PLUGIN_CLASS（Plugin 子类）
        cls = getattr(module, "PLUGIN_CLASS", None)
        if cls is not None and isinstance(cls, type) and issubclass(cls, Plugin):
            try:
                disposer = cls(spec).apply(self.ctx, spec.config) or None
            except Exception as e:  # noqa: BLE001
                log.warning("[plugins] %s apply 失败: %s", spec.id, e)
                return None
            self._loaded.append(_LoadedPlugin(
                spec=spec, disposer=self._combine_disposers(spec.id, disposer)))
            return spec.id
        # 约定 2：register(ctx) / register(ctx, config)
        fn = getattr(module, "register", None)
        if callable(fn):
            disposer = self._safe_register(fn, spec)
            self._loaded.append(_LoadedPlugin(spec=spec, disposer=disposer))
            return spec.id
        log.warning("[plugins] %s/main.py 既无 PLUGIN_CLASS 也无 register()", spec.id)
        return None

    @staticmethod
    def _load_module_from_path(name: str, main_py: Path):
        """按文件路径加载 Python 模块（不要求插件目录在 sys.path 上）。"""
        mod_name = f"vap_plugin_{name}"
        try:
            file_spec = importlib.util.spec_from_file_location(mod_name, str(main_py))
            if file_spec is None or file_spec.loader is None:
                log.warning("[plugins] 无法为 %s 建 module spec", main_py)
                return None
            module = importlib.util.module_from_spec(file_spec)
        except Exception as e:  # noqa: BLE001
            log.warning("[plugins] 加载 %s 失败: %s", main_py, e)
            return None
        # 让插件内部 `import 同目录模块` 可用（加载完毕后立即还原 sys.path）
        plugin_dir = str(main_py.parent)
        inserted = plugin_dir not in sys.path
        if inserted:
            sys.path.insert(0, plugin_dir)
        try:
            file_spec.loader.exec_module(module)
        except Exception as e:  # noqa: BLE001
            log.warning("[plugins] 执行 %s 失败: %s", main_py, e)
            return None
        finally:
            if inserted:
                try:
                    sys.path.remove(plugin_dir)
                except ValueError:
                    pass
        return module

    def dispose_all(self) -> None:
        """逆序调所有 disposer。"""
        for lp in reversed(self._loaded):
            if lp.disposer is not None:
                try:
                    lp.disposer()
                except Exception:
                    pass
        self._loaded.clear()

    @property
    def loaded(self) -> List[PluginSpec]:
        return [lp.spec for lp in self._loaded]
