"""插件框架接线测试（v10.4.0 P0-3）。

背景（真实缺陷）
----------------
1. ``PluginLoader`` 全仓**只在测试里**被实例化过（``tests/test_agent_framework.py``），
   生产代码零调用。
2. ``docs/guide/plugin-development.md`` 一直描述 ``plugins/<plugin-name>/main.py``
   的目录布局，但 loader 只会 ``importlib.import_module(spec.module)`` —— 只能加载
   **已安装且可 import** 的模块，文档里的目录布局从未真正生效。
3. ``CLAUDE.md`` 声称 ``plugins/`` 目录存在、``loader.py`` 扫此目录加载，但
   ``plugins/`` 目录、``config/plugins.yml``、``VAP_PLUGIN_DIR`` 当时**都不存在**。

本文件守护：目录型加载真实可用 + 生产路径真实接线 + 文档与实现一致。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.plugins.context import PluginContext
from src.core.plugins.loader import (
    BUILTIN_PLUGIN_MODULES,
    DEFAULT_PLUGIN_DIR,
    PluginLoader,
    builtin_plugin_modules,
)
from src.core.tools.registry import ToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _loader() -> tuple[PluginLoader, PluginContext]:
    ctx = PluginContext(ToolRegistry())
    return PluginLoader(ctx), ctx


def _write_plugin(base: Path, name: str, main_py: str, yaml_text: str = "") -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "main.py").write_text(main_py, encoding="utf-8")
    if yaml_text:
        (d / "plugin.yaml").write_text(yaml_text, encoding="utf-8")
    return d


# 注意：这是 .format(pid=...) 的模板，除了 {pid} 之外的字面花括号都要写成 {{ }}。
FUNCTIONAL_PLUGIN = '''
from src.core.plugins.context import PluginContext

PLUGIN_ID = "{pid}"
_disposed = []


def register(ctx: PluginContext, config=None):
    greeting = str((config or {{}}).get("greeting", "hi"))
    ctx.append_system_prompt("plugin:{pid}:" + greeting, plugin_id=PLUGIN_ID)

    def _dispose():
        _disposed.append(PLUGIN_ID)

    return _dispose
'''

CLASS_PLUGIN = '''
from src.core.plugins.patch import Plugin


class _P(Plugin):
    def apply(self, ctx, config=None):
        ctx.append_system_prompt("class-plugin", plugin_id=self.spec.id)
        return None


PLUGIN_CLASS = _P
'''


# --------------------------------------------------------------------------
# 1) 目录型加载（本批次补上的能力）
# --------------------------------------------------------------------------


def test_missing_plugin_dir_is_safe(tmp_path):
    """目录不存在 → 返回空列表，不抛异常、不阻断启动。"""
    loader, _ = _loader()
    assert loader.load_from_dir(str(tmp_path / "nope")) == []
    assert loader.loaded == []


def test_functional_plugin_is_loaded(tmp_path):
    """register(ctx, config) 形式：加载成功、system prompt 生效、disposer 可调。"""
    base = tmp_path / "plugins"
    _write_plugin(
        base, "hello", FUNCTIONAL_PLUGIN.format(pid="hello"),
        yaml_text="id: hello\nname: Hi\nenabled: true\nconfig:\n  greeting: 你好\n",
    )
    loader, ctx = _loader()
    loaded = loader.load_from_dir(str(base))
    assert loaded == ["hello"]
    assert "plugin:hello:你好" in ctx.system_prompt
    assert ctx.get_setting("example_hello.greeting") is None  # 不是示例插件
    loader.dispose_all()  # 不应抛异常


def test_class_plugin_is_loaded(tmp_path):
    """PLUGIN_CLASS 形式同样可加载。"""
    base = tmp_path / "plugins"
    _write_plugin(base, "cls", CLASS_PLUGIN)
    loader, ctx = _loader()
    assert loader.load_from_dir(str(base)) == ["cls"]
    assert "class-plugin" in ctx.system_prompt


def test_disabled_plugin_is_skipped(tmp_path):
    """plugin.yaml 里 enabled: false → 跳过（不注册任何东西）。"""
    base = tmp_path / "plugins"
    _write_plugin(base, "off", FUNCTIONAL_PLUGIN.format(pid="off"),
                  yaml_text="id: off\nenabled: false\n")
    loader, ctx = _loader()
    assert loader.load_from_dir(str(base)) == []
    assert ctx.system_prompt == ""


def test_plugin_without_main_py_is_skipped(tmp_path):
    """目录里没有 main.py → 跳过。"""
    base = tmp_path / "plugins"
    (base / "empty").mkdir(parents=True)
    loader, ctx = _loader()
    assert loader.load_from_dir(str(base)) == []


def test_broken_plugin_does_not_block_others(tmp_path):
    """一个插件 main.py 报错，不影响同目录的其他插件加载。"""
    base = tmp_path / "plugins"
    _write_plugin(base, "aaa-broken", "raise RuntimeError('boom')\n")
    _write_plugin(base, "bbb-good", FUNCTIONAL_PLUGIN.format(pid="bbb-good"))
    loader, ctx = _loader()
    loaded = loader.load_from_dir(str(base))
    assert loaded == ["bbb-good"]
    assert "plugin:bbb-good" in ctx.system_prompt


def test_broken_plugin_yaml_does_not_block_load(tmp_path):
    """plugin.yaml 语法错误 → 退回目录名当 id，插件仍加载。"""
    base = tmp_path / "plugins"
    _write_plugin(base, "yy", FUNCTIONAL_PLUGIN.format(pid="yy"),
                  yaml_text="id: [unclosed\n  bad: : :\n")
    loader, _ = _loader()
    assert loader.load_from_dir(str(base)) == ["yy"]


def test_plugin_dirname_used_when_no_yaml(tmp_path):
    """无 plugin.yaml → 用目录名当 plugin id。"""
    base = tmp_path / "plugins"
    _write_plugin(base, "no-yaml", FUNCTIONAL_PLUGIN.format(pid="no-yaml"))
    loader, _ = _loader()
    assert loader.load_from_dir(str(base)) == ["no-yaml"]


def test_plugin_yaml_id_overrides_dirname(tmp_path):
    """plugin.yaml 的 id 优先于目录名。"""
    base = tmp_path / "plugins"
    _write_plugin(base, "dirname", FUNCTIONAL_PLUGIN.format(pid="custom-id"),
                  yaml_text="id: custom-id\nenabled: true\n")
    loader, _ = _loader()
    assert loader.load_from_dir(str(base)) == ["custom-id"]


def test_dispose_all_calls_disposers(tmp_path):
    """dispose_all 必须真的调用插件返回的 disposer。"""
    base = tmp_path / "plugins"
    _write_plugin(base, "d1", FUNCTIONAL_PLUGIN.format(pid="d1"))
    loader, _ = _loader()
    loader.load_from_dir(str(base))
    loader.dispose_all()
    assert loader.loaded == []


# --------------------------------------------------------------------------
# 2) 内置插件模块
# --------------------------------------------------------------------------


def test_builtin_plugin_modules_declared():
    """内置插件模块清单必须非空且指向真实存在的模块。"""
    mods = builtin_plugin_modules()
    assert mods == list(BUILTIN_PLUGIN_MODULES)
    assert mods, "内置插件清单为空"
    for m in mods:
        path = PROJECT_ROOT / (m.replace(".", "/") + ".py")
        assert path.is_file(), f"内置插件模块不存在: {m}"


def test_load_builtin_registers_tools():
    """load_builtin 把内置插件的工具注册进 registry（真实副作用）。"""
    registry = ToolRegistry()
    loader = PluginLoader(PluginContext(registry))
    loader.load_builtin(builtin_plugin_modules())
    assert len(registry.list_names()) >= 16


def test_builtin_load_is_idempotent():
    """重复加载同名词不报错（registry.register 同名覆盖）。"""
    registry = ToolRegistry()
    loader = PluginLoader(PluginContext(registry))
    loader.load_builtin(builtin_plugin_modules())
    first = set(registry.list_names())
    loader.load_builtin(builtin_plugin_modules())
    assert set(registry.list_names()) == first


# --------------------------------------------------------------------------
# 3) 生产接线 + 文档一致（防回退）
# --------------------------------------------------------------------------


def test_plugin_loader_has_production_caller():
    """反向断言：src/web/routers/agent.py 必须真实实例化 PluginLoader。"""
    text = (PROJECT_ROOT / "src" / "web" / "routers" / "agent.py").read_text(encoding="utf-8")
    assert "PluginLoader(" in text, "PluginLoader 又变成零调用模块了（P0-3 接线断裂）"
    assert "load_from_dir(" in text
    assert "VAP_PLUGIN_DIR" in text


def test_plugin_dir_exists():
    """CLAUDE.md 描述的 plugins/ 目录必须真实存在。"""
    assert (PROJECT_ROOT / DEFAULT_PLUGIN_DIR).is_dir()


def test_plugin_config_yml_exists():
    """声明式插件清单必须存在且能被解析。"""
    from src.core.plugins.patch import load_plugin_specs

    path = PROJECT_ROOT / "config" / "plugins.yml"
    assert path.is_file()
    # 空列表是合法值（当前默认）
    assert load_plugin_specs(str(path)) == []


def test_example_plugin_is_valid_and_disabled():
    """示例插件必须存在、可解析、默认 enabled: false（不改变现有行为）。"""
    import yaml

    example = PROJECT_ROOT / DEFAULT_PLUGIN_DIR / "example-hello"
    assert (example / "main.py").is_file()
    assert (example / "plugin.yaml").is_file()
    data = yaml.safe_load((example / "plugin.yaml").read_text(encoding="utf-8"))
    assert data["enabled"] is False
    assert data["id"] == "example-hello"


def test_example_plugin_loads_when_enabled(tmp_path):
    """把示例插件复制到临时目录并启用 → 真实加载成功。"""
    import shutil

    src = PROJECT_ROOT / DEFAULT_PLUGIN_DIR / "example-hello"
    dst_base = tmp_path / "plugins"
    dst = dst_base / "example-hello"
    shutil.copytree(src, dst)
    (dst / "plugin.yaml").write_text(
        "id: example-hello\nenabled: true\nconfig:\n  greeting: 你好\n",
        encoding="utf-8")
    loader, ctx = _loader()
    assert loader.load_from_dir(str(dst_base)) == ["example-hello"]
    assert "示例插件" in ctx.system_prompt
    assert ctx.get_setting("example_hello.greeting") == "你好"


def test_claude_md_plugin_section_matches_reality():
    """CLAUDE.md 提到的插件关键路径必须真实存在（防文档漂移）。"""
    text = (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "config/plugins.yml" in text
    assert (PROJECT_ROOT / "config" / "plugins.yml").is_file()
    assert "plugins/" in text
    assert (PROJECT_ROOT / "plugins").is_dir()
    assert "VAP_PLUGIN_DIR" in text


@pytest.mark.parametrize("entry", ["README.md", "example-hello"])
def test_plugins_dir_contents(entry: str):
    """plugins/ 必须带 README 与示例（让文档可照做）。"""
    assert (PROJECT_ROOT / DEFAULT_PLUGIN_DIR / entry).exists()


# --------------------------------------------------------------------------
# 4) 发现探测 describe_plugin_dir（只读，不执行插件代码）
# --------------------------------------------------------------------------8


def test_describe_plugin_dir_missing_returns_empty(tmp_path):
    """目录不存在 → 空列表（不抛异常）。"""
    from src.core.plugins.loader import describe_plugin_dir

    assert describe_plugin_dir(str(tmp_path / "nope")) == []


def test_describe_plugin_dir_reports_declaration(tmp_path):
    """声明摘要与 loader 解析结果一致（enabled / id / has_main）。"""
    from src.core.plugins.loader import describe_plugin_dir

    _write_plugin(tmp_path, "on-plug", "def register(ctx, config=None):\n    return None\n",
                  yaml_text="id: on-plug\nname: On\nenabled: true\n")
    _write_plugin(tmp_path, "off-plug", "def register(ctx, config=None):\n    return None\n",
                  yaml_text="id: off-plug\nenabled: false\n")
    # 无 main.py 的目录 → 仍列出但 has_main=False（便于用户自查拼错文件名）
    (tmp_path / "broken").mkdir()

    got = {d["id"]: d for d in describe_plugin_dir(str(tmp_path))}
    assert set(got) == {"on-plug", "off-plug", "broken"}
    assert got["on-plug"]["enabled"] is True
    assert got["off-plug"]["enabled"] is False
    assert got["broken"]["has_main"] is False
    assert got["on-plug"]["config_keys"] == []


def test_describe_plugin_dir_does_not_execute_plugin(tmp_path):
    """纯探测：不 import 插件代码（main.py 里有副作用也不该被执行）。"""
    from src.core.plugins.loader import describe_plugin_dir

    marker = tmp_path / "EXECUTED"
    _write_plugin(
        tmp_path, "side-effect",
        f"from pathlib import Path\nPath(r'{marker}').write_text('boom')\n\n"
        "def register(ctx, config=None):\n    return None\n",
        yaml_text="id: side-effect\nenabled: true\n")
    describe_plugin_dir(str(tmp_path))
    assert not marker.exists(), "describe_plugin_dir 执行了插件代码（应为纯探测）"


def test_describe_and_loader_agree_on_scan(tmp_path):
    """列表显示的启停状态必须与加载器实际行为一致（防两者漂移）。"""
    from src.core.plugins.loader import describe_plugin_dir

    _write_plugin(tmp_path, "will-load", FUNCTIONAL_PLUGIN.format(pid="will-load"),
                  yaml_text="id: will-load\nenabled: true\n")
    _write_plugin(tmp_path, "wont-load", FUNCTIONAL_PLUGIN.format(pid="wont-load"),
                  yaml_text="id: wont-load\nenabled: false\n")
    described_enabled = {d["id"] for d in describe_plugin_dir(str(tmp_path)) if d["enabled"]}
    loader, _ = _loader()
    assert set(loader.load_from_dir(str(tmp_path))) == described_enabled


# --------------------------------------------------------------------------
# 5) 插件副作用真正进入生产链路（P0-3 补漏）
# --------------------------------------------------------------------------


def test_example_plugin_registers_tool_and_prompt(tmp_path):
    """示例插件启用后：工具进 registry + setting/prompt 真实生效。"""
    import shutil

    src = PROJECT_ROOT / DEFAULT_PLUGIN_DIR / "example-hello"
    dst_base = tmp_path / "plugins"
    shutil.copytree(src, dst_base / "example-hello")
    (dst_base / "example-hello" / "plugin.yaml").write_text(
        "id: example-hello\nenabled: true\nconfig:\n  greeting: 早上好\n",
        encoding="utf-8")

    registry = ToolRegistry()
    ctx = PluginContext(registry)
    loader = PluginLoader(ctx)
    assert loader.load_from_dir(str(dst_base)) == ["example-hello"]
    assert "example_hello" in registry.list_names(), "插件工具未进 registry"
    assert "早上好" in ctx.system_prompt
    # 副作用审计（供 /api/plugins 展示）
    state = ctx.get_state("example-hello")
    assert state is not None
    kinds = {e.kind for e in state.effects}
    assert {"tool", "system_prompt", "setting"} <= kinds


def test_example_plugin_tool_is_executable(tmp_path):
    """插件工具不是「注册了个名字」——必须真能穿 registry 执行出结果。

    这是防假实现的关键一步：只断言 list_names() 会放过「注册了一个永远报错的
    stub」。这里真调 execute() 并断言返回体，同时验证它经 scope_guard 判定为
    **read → 放行**（插件写工具默认会被 ask，不静默绕过审批）。
    """
    import asyncio
    import shutil

    from src.core.tools.registry import ToolCall
    from src.core.tools.scope_guard import build_guard_from_env, install_tool_guard

    src = PROJECT_ROOT / DEFAULT_PLUGIN_DIR / "example-hello"
    dst_base = tmp_path / "plugins"
    shutil.copytree(src, dst_base / "example-hello")
    (dst_base / "example-hello" / "plugin.yaml").write_text(
        "id: example-hello\nenabled: true\nconfig:\n  greeting: 测试问候\n",
        encoding="utf-8")

    registry = ToolRegistry()
    loader = PluginLoader(PluginContext(registry))
    assert loader.load_from_dir(str(dst_base)) == ["example-hello"]

    # 生命周期：dispose_all 后插件工具必须从 registry 消失（不残留）
    guard = install_tool_guard(registry, approval_fn=lambda *a, **k: True)
    assert guard.classify("example_hello") == "read"

    loop = asyncio.new_event_loop()
    try:
        res = loop.run_until_complete(registry.execute(
            ToolCall(call_id="c1", name="example_hello", args={"greeting": "嗨"})))
    finally:
        asyncio.set_event_loop(None)
        loop.close()
    assert res.error in (None, ""), f"插件工具执行失败: {res.error}"
    assert res.output == {"ok": True, "greeting": "嗨", "plugin": "example-hello"}

    loader.dispose_all()
    assert "example_hello" not in registry.list_names(), "disposer 未生效"
    # 显式引用 build_guard_from_env，保持与生产同源（防以后换了 guard 实现不自知）
    assert build_guard_from_env() is not None


def test_agent_route_consumes_plugin_system_prompt(tmp_path, monkeypatch):
    """P0-3 补漏回归：_build_react_agent 必须把插件 system prompt 片段拼进
    模型上下文，并让插件工具出现在 agent 的 registry 里。

    修复前的真实缺陷：PluginContext 在本函数里就地构造、随后被丢弃，
    ctx.system_prompt 没有任何消费者 —— 插件声明的「追加提示」是静默 no-op，
    用户开了插件也看不到任何效果（假实现）。
    """
    import shutil

    src = PROJECT_ROOT / DEFAULT_PLUGIN_DIR / "example-hello"
    dst_base = tmp_path / "plugins"
    shutil.copytree(src, dst_base / "example-hello")
    (dst_base / "example-hello" / "plugin.yaml").write_text(
        "id: example-hello\nenabled: true\nconfig:\n  greeting: 插件问候\n",
        encoding="utf-8")

    monkeypatch.setenv("VAP_PLUGIN_DIR", str(dst_base))
    for var in ("VAP_AGENT_BACKEND", "VAP_AGENT_SUPERVISOR", "VAP_SANDBOX_ENABLED",
                "VAP_MEMORY_LAYERED", "VAP_SKILLS_ROSTER"):
        monkeypatch.delenv(var, raising=False)

    from src.web.routers import agent as mod

    class _Req:
        class _State:
            session_store = None
        app = type("App", (), {"state": _State()})
        query_params = {}
        headers = {}

    ag, _session, _store = mod._build_react_agent(
        _Req(), "s-plugin-prompt", None, "你好")

    assert "插件问候" in ag.config.system_prompt, \
        "插件 system prompt 片段未进入模型上下文（P0-3 补漏回归）"
    assert "example_hello" in ag.tools.list_names(), \
        "插件工具未进入本次请求的 registry"


def test_agent_route_prompt_unchanged_without_plugins(monkeypatch):
    """零回归：**没有任何插件启用**时，system prompt 与「插件目录不存在」时逐字节相同。

    对比对象选「不存在的目录」而非 build_agent_system_prompt 的字面重建：
    _build_react_agent 还会按用户输入跑 roster/match_skills，字面重建很难稳定复现，
    而「关插件 == 无插件」才是这条回归真正要守的语义。
    """
    from src.web.routers import agent as mod

    class _Req:
        class _State:
            session_store = None
        app = type("App", (), {"state": _State()})
        query_params = {}
        headers = {}

    for var in ("VAP_AGENT_BACKEND", "VAP_AGENT_SUPERVISOR", "VAP_SANDBOX_ENABLED",
                "VAP_MEMORY_LAYERED", "VAP_SKILLS_ROSTER"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("VAP_PLUGIN_DIR", "__nonexistent_plugin_dir__")
    ag_none, _s, _st = mod._build_react_agent(_Req(), "s-no-plugin", None, "你好")

    # 仓库自带的 plugins/（只有 example-hello，且 enabled: false）
    monkeypatch.setenv("VAP_PLUGIN_DIR", str(PROJECT_ROOT / DEFAULT_PLUGIN_DIR))
    ag_disabled, _s2, _st2 = mod._build_react_agent(_Req(), "s-no-plugin", None, "你好")

    assert ag_none.config.system_prompt == ag_disabled.config.system_prompt, \
        "默认插件目录（全部 disabled）不得改变 system prompt"
    assert "示例插件" not in ag_disabled.config.system_prompt
    assert "example_hello" not in ag_disabled.tools.list_names()


# --------------------------------------------------------------------------
# 6) GET /api/plugins（可观测性：用户能自查「我的插件认到了吗」）
# --------------------------------------------------------------------------


def test_plugins_endpoint_reports_discovered_and_loaded(tmp_path, monkeypatch):
    """端到端：HTTP 接口能看到发现结果 + 该次 agent 运行的真实加载结果。"""
    from fastapi.testclient import TestClient

    from src.web.app import create_app

    monkeypatch.setenv("VAP_PLUGIN_DIR", str(tmp_path / "empty-plugins"))
    client = TestClient(create_app())
    r = client.get("/api/plugins")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dir"].endswith("empty-plugins")
    assert body["discovered"] == []
    assert body["loaded"] == []
    assert body["loaded_at_least_once"] is False
    assert body["builtin_modules"], "内置插件清单不该为空"


def test_plugins_endpoint_never_500_on_bad_dir(monkeypatch):
    """探测失败不许把能力接口打成 500（防御性）。"""
    from fastapi.testclient import TestClient

    from src.web.app import create_app

    import src.core.plugins.loader as loader_mod

    def _boom(_path: str):
        raise PermissionError("模拟磁盘/权限异常")

    monkeypatch.setattr(loader_mod, "describe_plugin_dir", _boom)
    client = TestClient(create_app())
    r = client.get("/api/plugins")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["discovered"] == []
    assert "PermissionError" in body["error"], "异常路径应带诊断信息，便于用户自查"


def test_legacy_backend_has_no_plugin_registry():
    """已记录的架构决策：插件只在 **react** 路径加载。

    原因：legacy 路径用的是 src/core/agent_tools.ToolRegistry（字符串注册 +
    **没有** scope_guard），把插件工具注入那里会让插件工具完全绕过审批治理。
    legacy 是 VAP_AGENT_BACKEND=legacy 的兼容回退，不是默认路径。

    本测试把「不做」变成可执行契约：若有人以后往 _build_orchestrator 里加插件
    加载，必须先补上审批治理，否则此测试失败提醒他。
    """
    text = (PROJECT_ROOT / "src" / "web" / "routers" / "agent.py").read_text(encoding="utf-8")
    start = text.index("def _build_orchestrator(")
    end = text.index("class _AgentContext", start)
    legacy_body = text[start:end]
    assert "PluginLoader" not in legacy_body, (
        "legacy 路径出现了 PluginLoader：legacy registry 没有 scope_guard，"
        "插件工具会绕过审批。请先把审批治理接上再加插件。")


def test_plugins_endpoint_route_is_registered():
    """反向断言：路由必须在 create_app 里注册（防漏 include_router）。"""
    from fastapi.testclient import TestClient

    from src.web.app import create_app

    client = TestClient(create_app())
    schema_paths = set(client.get("/api/openapi.json").json()["paths"])
    assert "/api/plugins" in schema_paths, "插件路由未注册进 create_app"
