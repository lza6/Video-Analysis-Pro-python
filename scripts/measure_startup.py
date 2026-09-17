#!/usr/bin/env python
"""启动耗时基准（v10.4.0 P0-6）。

用法::

    ./venv/Scripts/python.exe scripts/measure_startup.py

对每个被测目标起一个**干净解释器**测量 import 墙钟时间，输出对照表。

为什么需要独立脚本：`pytest` 进程里模块早已加载，测不出真实启动代价；
只有干净进程 + `time.time()` 才是用户真实感受到的"后端多久起来"。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (显示名, import 语句, v10.3.1 基线秒数)
TARGETS: list[tuple[str, str, float | None]] = [
    ("src.web.app", "import src.web.app", 29.18),
    ("src.core.logic", "import src.core.logic", 19.64),
    ("sentence_transformers", "import sentence_transformers", 19.00),
    ("torch", "import torch", 4.15),
    ("chromadb", "import chromadb", 2.37),
    ("fastapi", "import fastapi", None),
]


def _measure(statement: str) -> float | None:
    """在干净解释器中测 statement 的 import 耗时；不可导入返回 None。"""
    code = (
        "import time\n"
        "t = time.time()\n"
        f"{statement}\n"
        "print('%.3f' % (time.time() - t))\n"
    )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def main() -> int:
    # Windows 控制台默认 GBK，中文表头会乱码；强制 utf-8 输出。
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    print("=" * 62)
    print("TingFeng Hermes — 启动耗时基准 (clean interpreter)")
    print("=" * 62)
    print(f"{'目标':<24}{'实测':>10}{'v10.3.1 基线':>16}   说明")
    print("-" * 62)

    app_elapsed: float | None = None
    for label, statement, baseline in TARGETS:
        elapsed = _measure(statement)
        if elapsed is None:
            print(f"{label:<24}{'N/A':>10}{'':>16}   未安装/不可导入")
            continue
        base_txt = f"{baseline:.2f}s" if baseline is not None else "—"
        note = ""
        if baseline is not None and baseline > 0:
            note = f"{baseline / elapsed:.1f}x faster" if elapsed > 0 else ""
        print(f"{label:<24}{elapsed:>9.2f}s{base_txt:>16}   {note}")
        if label == "src.web.app":
            app_elapsed = elapsed

    print("-" * 62)
    if app_elapsed is not None:
        verdict = "PASS" if app_elapsed < 3.0 else "OVER BUDGET"
        print(f"后端冷启动 {app_elapsed:.2f}s (目标 ≤3.0s) → {verdict}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
