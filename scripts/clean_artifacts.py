#!/usr/bin/env python
"""构建产物 / 缓存清理（v10.4.0 P0-10）。

    python scripts/clean_artifacts.py              # 预览（dry-run，默认）
    python scripts/clean_artifacts.py --apply      # 真删（安全集）
    python scripts/clean_artifacts.py --apply --include-dist   # 连安装包一起清

安全设计（为什么默认 dry-run）
-----------------------------
清理是**不可撤销**的。所以：

1. **默认只预览**，必须显式 ``--apply`` 才动手；
2. 每个路径先跑 ``git check-ignore`` —— **不在 .gitignore 里的一律跳过**
   （绝不误删已跟踪的源码/配置/文档）；
3. 跳过 ``webapp/out/``（Electron 直接加载它，删了应用起不来）；
4. ``desktop/dist/``（安装包 + win-unpacked，数 GB）**必须显式** ``--include-dist``；
   且只删 ``win-unpacked/`` 与**非最新**版本的安装包，保留最新的 .exe。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 安全集：纯缓存/构建产物，删了可由构建重新生成
SAFE_TARGETS: Tuple[str, ...] = (
    "webapp/.next",
    "webapp/test-results",
    "webapp/playwright-report",
    "cache",
    "graft",
    ".pytest_cache",
    ".ruff_cache",
    ".coverage",
    "htmlcov",
    "logs",
)

#: 永不动的路径（删了会破坏开发/运行）
PROTECTED: Tuple[str, ...] = (
    "webapp/out",     # Electron 静态加载目标
    "venv",
    "models",
    "config",
    "src",
    "tests",
    "desktop/dist/TingFeng",   # 由 --include-dist 的专门逻辑处理
)

#: 递归清理的缓存目录名（安全集内）
CACHE_DIR_NAMES = ("__pycache__",)
CACHE_DIR_ROOTS = ("src", "tests", "scripts")


def _git_check_ignore(path: Path) -> bool:
    """该路径是否被 .gitignore 忽略（未跟踪的相对路径）。"""
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", str(path.relative_to(PROJECT_ROOT))],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=30,
        )
    except Exception:
        return False
    return proc.returncode == 0


def _is_git_tracked(path: Path) -> bool:
    """该路径下是否有被 git 跟踪的文件（有则绝不删）。"""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(PROJECT_ROOT))],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return True  # 判不出来就当作已跟踪（保守）
    return proc.returncode == 0


def _dir_size(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n:.1f}GB"


def _protected(rel: str) -> bool:
    return any(rel == p or rel.startswith(p + "/") for p in PROTECTED)


def collect_targets(include_dist: bool, include_pycache: bool) -> List[Path]:
    targets: List[Path] = []
    for rel in SAFE_TARGETS:
        p = PROJECT_ROOT / rel
        if not p.exists():
            continue
        if _protected(rel):
            print(f"  保护跳过: {rel}")
            continue
        if not _git_check_ignore(p):
            print(f"  非 gitignore 跳过（防误删）: {rel}")
            continue
        if _is_git_tracked(p):
            print(f"  已被 git 跟踪，跳过: {rel}")
            continue
        targets.append(p)

    if include_pycache:
        for root in CACHE_DIR_ROOTS:
            base = PROJECT_ROOT / root
            if not base.is_dir():
                continue
            for name in CACHE_DIR_NAMES:
                targets.extend(sorted(base.rglob(name)))

    if include_dist:
        dist = PROJECT_ROOT / "desktop" / "dist"
        if dist.is_dir():
            unpacked = dist / "win-unpacked"
            if unpacked.is_dir():
                targets.append(unpacked)
            # 保留最新 .exe，删掉旧版本安装包与 blockmap
            exes = sorted(dist.glob("*.exe"), key=lambda p: p.stat().st_mtime)
            for old in exes[:-1]:
                targets.append(old)
            for oldmap in dist.glob("*.exe.blockmap"):
                matching = [e for e in exes if e.name == oldmap.name.replace(".blockmap", "")]
                if matching and matching[0].name != (exes[-1].name if exes else ""):
                    targets.append(oldmap)
    return targets


def _delete(path: Path) -> int:
    size = _dir_size(path)
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            return 0
    return size if not path.exists() else 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建产物 / 缓存清理")
    parser.add_argument("--apply", action="store_true", help="真删（默认只预览）")
    parser.add_argument("--include-dist", action="store_true",
                        help="同时清理 desktop/dist（安装包 + win-unpacked，数 GB）")
    parser.add_argument("--include-pycache", action="store_true",
                        help="同时清理 src/tests/scripts 下的 __pycache__")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    print("== TingFeng Hermes 产物清理 ==")
    print(f"模式: {'APPLY（真删）' if args.apply else 'DRY-RUN（仅预览）'}")
    targets = collect_targets(args.include_dist, args.include_pycache)

    if not targets:
        print("没有需要清理的路径。")
        return 0

    total = 0
    freed = 0
    for path in targets:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        size = _dir_size(path)
        total += size
        if args.apply:
            freed += _delete(path)
            mark = "已删除" if not path.exists() else "删除失败"
        else:
            mark = "待删除"
        print(f"  {mark}: {rel}  ({_human(size)})")

    print("-" * 56)
    if args.apply:
        print(f"释放空间: {_human(freed)}（计划 {_human(total)}）")
        print("提示: 运行 `git status --short` 确认没有已跟踪文件被改动。")
    else:
        print(f"预计可释放: {_human(total)}")
        print("加 --apply 执行；安装包用 --apply --include-dist。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
