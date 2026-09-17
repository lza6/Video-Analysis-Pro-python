"""插件路由 —— 让插件框架「可观测」。

  GET /api/plugins → 插件目录发现结果 + 上一次 agent 运行时的实际加载结果

为什么需要这个端点（v10.4.0 / P0-3）：
    插件框架此前是「文档幻觉」——CLAUDE.md 与 docs/guide/plugin-development.md
    完整描述了 ``plugins/<name>/main.py``、``config/plugins.yml``、``VAP_PLUGIN_DIR``，
    但 loader 只 importlib 已安装模块、目录布局跑不起来，而且 PluginLoader 在
    ``src/`` 里零调用。接线之后，用户仍然无法自查「我的插件被认到了吗 /
    为什么没生效」——只能翻日志。这里把**发现**（只读扫描，不执行插件代码）
    与**加载结果**（agent 运行时写入 app.state.plugin_loader）都暴露出来。

安全：只读。不 import 插件代码、不返回 plugin.yaml 全文（配置可能含密钥，
仅返回 config 的 key 名）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request

from ..security import require_auth

router = APIRouter(prefix="/api/plugins", tags=["plugins"])


@router.get("", dependencies=[Depends(require_auth)])
async def list_plugins(request: Request) -> Dict[str, Any]:
    """返回插件发现 + 加载状态。

    字段：
      - dir: 生效的插件目录（VAP_PLUGIN_DIR 或默认 ``plugins``）
      - discovered: 目录扫描结果（id/name/enabled/path/has_main/config_keys）
      - loaded: 上一次 agent 运行中**真正成功加载**的插件 id 列表
      - loaded_effects: 每个插件的副作用审计（tool/system_prompt/setting/hook）
      - loaded_at_least_once: 是否至少经历过一次 agent 运行（false 时
        discovered 有值但 loaded 为空是正常的，不代表加载失败）

    发现扫描失败不抛异常（返回空 + error 字段），与 loader 的防御性风格一致：
    插件问题不该让整个能力列表接口 500。
    """
    plugin_dir = os.environ.get("VAP_PLUGIN_DIR", "plugins")
    discovered: List[Dict[str, Any]] = []
    error = ""
    try:
        from src.core.plugins.loader import describe_plugin_dir
        discovered = describe_plugin_dir(plugin_dir)
    except Exception as e:  # noqa: BLE001 — 探测失败不阻断接口
        error = f"{type(e).__name__}: {e}"

    loader = getattr(request.app.state, "plugin_loader", None)
    loaded: List[str] = []
    effects: Dict[str, List[Dict[str, str]]] = {}
    if loader is not None:
        try:
            loaded = [spec.id for spec in loader.loaded]
            ctx = getattr(loader, "ctx", None)
            if ctx is not None:
                for pid, state in ctx.plugins.items():
                    effects[pid] = [
                        {"kind": eff.kind, "target": eff.target, "detail": eff.detail}
                        for eff in state.effects
                    ]
        except Exception as e:  # noqa: BLE001 — 状态读取失败不该 500
            error = error or f"{type(e).__name__}: {e}"

    builtin: List[str] = []
    try:
        from src.core.plugins.loader import builtin_plugin_modules
        builtin = builtin_plugin_modules()
    except Exception:  # noqa: BLE001
        pass

    return {
        "dir": plugin_dir,
        "builtin_modules": builtin,
        "discovered": discovered,
        "loaded": loaded,
        "loaded_effects": effects,
        "loaded_at_least_once": loader is not None,
        "error": error,
    }
