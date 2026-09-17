"""启动性能回归测试（v10.4.0 P0-6）。

为什么必须存在
--------------
v10.3.1 的 `import src.web.app` 实测 **29.18s**，其中：

  - `sentence_transformers`（模块顶层为算 CLIP_AVAILABLE 而真实 import）≈ 19.00s
  - `torch`（模块顶层 import）≈ 4.15s
  - `seaborn → scipy.stats → pandas`（_detect_advanced_features 真实 import）≈ 0.9s
  - `moviepy / matplotlib`（同上）≈ 0.7s

修复后实测 `import src.web.app` ≈ 1.3s。这些用例把"启动不得再被重依赖拖慢"
固化为可执行的 CI 断言——将来任何人把重依赖挪回模块顶层，这里会立刻变红。

计时口径
--------
一律用 **subprocess 起干净解释器**，避免被 pytest 进程里已加载的模块污染。
预算阈值可用环境变量 `VAP_STARTUP_BUDGET_SEC` 覆盖（CI 机器慢时放宽）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CI 机器可能明显慢于开发机：默认给 8s 宽松预算（本地实测 ~1.3s）。
_DEFAULT_BUDGET_SEC = 8.0


def _run_python(code: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """在项目根目录起一个干净解释器执行 code，返回完成结果。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # encoding="utf-8" + errors="replace"：Windows 控制台默认 GBK，
    # 子进程输出中文表头会触发 UnicodeDecodeError（曾导致本文件用例失败）。
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def _import_time(statement: str) -> float:
    """Measured import wall-time of `statement` in a fresh interpreter."""
    code = (
        "import time\n"
        "t = time.time()\n"
        f"{statement}\n"
        "print('%.3f' % (time.time() - t))\n"
    )
    proc = _run_python(code)
    assert proc.returncode == 0, f"子进程失败: {proc.stderr}"
    return float(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------
# 1) 重依赖不得在模块顶层被绑定
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "sentence_transformers",
        "torch",
        "seaborn",
        "moviepy",
        "scipy.stats",
    ],
)
def test_logic_import_does_not_bind_heavy_modules(module: str):
    """import src.core.logic 后，重依赖不得出现在 sys.modules 中。

    这是 P0-6 的核心契约：能力标志（CLIP_AVAILABLE / ADVANCED_FEATURES_AVAILABLE）
    必须用 find_spec 探测，而不是真实 import。
    """
    code = (
        "import sys\n"
        "import src.core.logic  # noqa: F401\n"
        f"print('LOADED' if '{module}' in sys.modules else 'NOT_LOADED')\n"
    )
    proc = _run_python(code)
    assert proc.returncode == 0, f"子进程失败: {proc.stderr}"
    assert proc.stdout.strip().splitlines()[-1] == "NOT_LOADED", (
        f"{module} 被 src.core.logic 在模块顶层加载了——启动性能回归"
    )


def test_web_app_import_does_not_bind_heavy_modules():
    """import src.web.app（后端入口）同样不得绑定重依赖。"""
    code = (
        "import sys\n"
        "import src.web.app  # noqa: F401\n"
        "heavy = [m for m in ('sentence_transformers', 'torch', 'seaborn', 'moviepy')\n"
        "         if m in sys.modules]\n"
        "print(','.join(heavy) if heavy else 'CLEAN')\n"
    )
    proc = _run_python(code)
    assert proc.returncode == 0, f"子进程失败: {proc.stderr}"
    assert proc.stdout.strip().splitlines()[-1] == "CLEAN"


# --------------------------------------------------------------------------
# 2) 启动耗时预算
# --------------------------------------------------------------------------


def test_logic_import_time_budget():
    """import src.core.logic 必须远快于 v10.3.1 的 19.6s 基线。"""
    if os.environ.get("VAP_SKIP_PERF_BUDGET", "0") == "1":
        pytest.skip("VAP_SKIP_PERF_BUDGET=1：本地内存/负载敏感环境跳过绝对耗时预算（语义守卫仍运行）")
    budget = float(os.environ.get("VAP_LOGIC_BUDGET_SEC", "4.0"))
    elapsed = _import_time("import src.core.logic")
    assert elapsed < budget, f"src.core.logic 启动耗时 {elapsed:.2f}s 超预算 ({budget}s)"


def test_web_app_import_time_budget():
    """import src.web.app（真实后端入口）必须在预算内。

    预算默认 8s（CI 宽松），可用 VAP_STARTUP_BUDGET_SEC 覆盖。
    本地实测约 1.3s，v10.3.1 基线为 29.18s。
    """
    if os.environ.get("VAP_SKIP_PERF_BUDGET", "0") == "1":
        pytest.skip("VAP_SKIP_PERF_BUDGET=1：本地内存/负载敏感环境跳过绝对耗时预算（语义守卫仍运行）")
    budget = float(os.environ.get("VAP_STARTUP_BUDGET_SEC", _DEFAULT_BUDGET_SEC))
    elapsed = _import_time("import src.web.app")
    assert elapsed < budget, (
        f"src.web.app 启动耗时 {elapsed:.2f}s 超出预算 {budget}s"
    )


def test_startup_is_faster_than_legacy_baseline():
    """守住 10x 以上的改善幅度（防未来悄悄退回重依赖顶层 import）。"""
    if os.environ.get("VAP_SKIP_PERF_BUDGET", "0") == "1":
        pytest.skip("VAP_SKIP_PERF_BUDGET=1：本地内存/负载敏感环境跳过绝对耗时预算（语义守卫仍运行）")
    x = float(os.environ.get("VAP_STARTUP_IMPROVEMENT_X", "10"))
    elapsed = _import_time("import src.web.app")
    assert elapsed < 29.18 / x, (
        f"启动耗时 {elapsed:.2f}s —— 相比 v10.3.1 基线(29.18s)改善不足 10x"
    )


# --------------------------------------------------------------------------
# 3) 能力标志语义不变（探测仍真实，不是硬编码 True）
# --------------------------------------------------------------------------


def test_clip_available_is_real_probe_not_hardcoded():
    """CLIP_AVAILABLE 必须与 sentence_transformers 的真实安装状态一致。"""
    import importlib.util

    from src.core.logic import CLIP_AVAILABLE

    installed = importlib.util.find_spec("sentence_transformers") is not None
    assert CLIP_AVAILABLE is installed


def test_probe_returns_false_for_missing_module():
    """_probe 对不存在的模块返回 False（不抛异常）。"""
    from src.core.logic import _probe

    assert _probe("definitely_not_a_real_module_xyz") is False


def test_probe_handles_submodule_of_missing_parent():
    """_probe 对'父模块不存在'的子模块查询不得抛异常（find_spec 会抛）。"""
    from src.core.logic import _probe

    assert _probe("definitely_missing_parent_xyz.child") is False


def test_advanced_features_flag_is_probed(monkeypatch):
    """ADVANCED_FEATURES_AVAILABLE 的计算必须是探测式（可被 _probe 结果驱动）。

    历史坑：曾硬编码 True；也曾为算这个 flag 真实 import 三个重包。
    """
    import src.core.logic as logic

    monkeypatch.setattr(logic, "_probe", lambda m: False)
    assert logic._detect_advanced_features() is False

    monkeypatch.setattr(logic, "_probe", lambda m: True)
    assert logic._detect_advanced_features() is True


def test_advanced_features_true_in_current_env():
    """当前环境（装了 Phase-3 依赖）应判定为可用。"""
    from src.core.logic import ADVANCED_FEATURES_AVAILABLE

    assert ADVANCED_FEATURES_AVAILABLE is True


# --------------------------------------------------------------------------
# 4) 惰性代理：写法不变、按需加载、功能不破
# --------------------------------------------------------------------------


def test_torch_is_lazy_proxy_before_use():
    """模块级 torch 是惰性代理，尚未触发真实 import。"""
    import src.core.logic as logic

    assert isinstance(logic.torch, logic._LazyModule)
    assert logic.torch._mod is None  # 未加载


def test_torch_proxy_loads_on_attribute_access():
    """访问 torch.cuda 时按需加载真实模块并返回正确属性。"""
    import src.core.logic as logic

    identity = logic.torch.cuda  # 触发加载
    assert identity is not None
    assert logic.torch._mod is not None


def test_torch_lazy_proxy_survives_monkeypatch_style_usage():
    """既有测试的写法 `logic.torch.cuda.is_available` 必须继续可用。

    参考 tests/test_b_fin2_coverage.py:599 的 monkeypatch 用法。
    """
    import src.core.logic as logic

    assert isinstance(logic.torch.cuda.is_available(), bool)


def test_check_cuda_health_runs_without_error():
    """check_cuda_health 在惰性代理下仍可正常执行（返回 bool，不抛异常）。"""
    from src.core.logic import check_cuda_health

    assert isinstance(check_cuda_health(), bool)


def test_sentence_transformers_util_lazy_helper():
    """_sentence_transformers_util() 按需返回带 cos_sim 的真实模块。"""
    from src.core.logic import _sentence_transformers_util

    util_mod = _sentence_transformers_util()
    assert hasattr(util_mod, "cos_sim")


def test_lazy_module_raises_on_missing_module():
    """惰性代理指向不存在的模块时，访问属性才抛 ImportError（而非启动时）。"""
    from src.core.logic import _LazyModule

    proxy = _LazyModule("definitely_not_a_real_module_xyz")
    with pytest.raises(ImportError):
        _ = proxy.anything


def test_no_module_level_torch_statement():
    """反向断言：src/core/logic.py 不得再出现模块级 `import torch`。

    防回退守卫——比计时断言更早、更明确地指出回归原因。
    """
    source = (PROJECT_ROOT / "src" / "core" / "logic.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        assert line.strip() != "import torch", (
            "src/core/logic.py 又出现了模块级 `import torch` —— 启动回归"
        )


def test_measure_script_reports_numbers():
    """scripts/measure_startup.py 必须能独立跑出启动数字（供人工复跑）。"""
    script = PROJECT_ROOT / "scripts" / "measure_startup.py"
    assert script.exists(), "缺少 scripts/measure_startup.py"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, f"measure_startup.py 执行失败: {proc.stderr}"
    assert "src.web.app" in proc.stdout
    assert "PASS" in proc.stdout or "OVER BUDGET" in proc.stdout
