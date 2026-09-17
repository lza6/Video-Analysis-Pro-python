"""Electron 壳接线守卫（v10.4.0 P0-4）。

背景
----
``desktop/cua-service.js``（3.7KB）真实存在并被 electron-builder 打进安装包，
但 ``desktop/main.js`` 里的 ``registerIpc()`` 从未引用它 —— 典型"死文件 + 
``VAP_CUA_ENABLED`` 无消费方"。本文件用两层守卫：

1. **源码级接线断言**（无需 node，永远可跑）：main.js 必须 require + 挂载 + 卸载；
2. **真实 node 单测**（node 不可用时 skip）：跑 ``desktop/test-cua-service.js``。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESKTOP = PROJECT_ROOT / "desktop"


def _read(rel: str) -> str:
    return (DESKTOP / rel).read_text(encoding="utf-8", errors="ignore")


# --------------------------------------------------------------------------
# 1) 源码级接线（永远可跑）
# --------------------------------------------------------------------------


def test_main_js_mounts_cua_service():
    """main.js 必须真实 require 并挂载 CUA 服务。"""
    src = _read("main.js")
    assert 'require("./cua-service")' in src, "main.js 未挂载 cua-service（死文件回归）"
    assert "mountCuaService(" in src
    assert "unmountCua" in src, "退出时未卸载 CUA handler"


def test_vap_cua_enabled_has_consumer():
    """VAP_CUA_ENABLED 必须被 main.js 读取（此前是无效开关）。"""
    assert "VAP_CUA_ENABLED" in _read("main.js")


def test_cua_disabled_by_default():
    """默认必须不启用：只有显式 VAP_CUA_ENABLED === '1' 才挂载。"""
    src = _read("main.js")
    assert 'process.env.VAP_CUA_ENABLED === "1"' in src


def test_cua_service_is_honest_about_mock():
    """CUA 的 click/type 必须显式标注 mock，不得伪造真实桌面操作。"""
    src = _read("cua-service.js")
    assert "mock: true" in src
    assert "绝不伪造" in src or "不伪造" in src


def test_cua_service_file_exists():
    assert (DESKTOP / "cua-service.js").is_file()


def test_desktop_package_check_covers_cua():
    """desktop/package.json 的 check 脚本必须语法校验 cua-service.js。"""
    pkg = (DESKTOP / "package.json").read_text(encoding="utf-8")
    assert "node --check cua-service.js" in pkg


# --------------------------------------------------------------------------
# 2) 真实 node 执行（node 不可用时 skip）
# --------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="node 未安装")
def test_cua_service_node_tests_pass():
    """跑 desktop/test-cua-service.js，必须零失败退出。"""
    proc = subprocess.run(
        ["node", "test-cua-service.js"],
        cwd=str(DESKTOP),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert proc.returncode == 0, f"CUA 单测失败:\n{proc.stdout}\n{proc.stderr}"
    assert "checks passed" in proc.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node 未安装")
@pytest.mark.parametrize("filename", ["main.js", "preload.js", "cua-service.js"])
def test_desktop_js_syntax(filename: str):
    """关键 Electron 文件必须语法正确（node --check）。"""
    proc = subprocess.run(
        ["node", "--check", filename],
        cwd=str(DESKTOP),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert proc.returncode == 0, f"{filename} 语法错误:\n{proc.stderr}"
