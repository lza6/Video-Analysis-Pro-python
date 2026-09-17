#!/usr/bin/env python
"""CI 三道门（v10.4.0 P0-7）。

    python scripts/check_wiring.py --check wiring,version,docs

为什么需要
----------
本项目反复出现三类"事后才发现"的问题，全部本可由 CI 拦住：

1. **定义了没人调用** —— v10.2 的 ``set_approval_fn``/``set_sandbox``；
   v10.3 的 ``set_error_policy``/``set_lock_resolver``/``eli5``/``validator``/
   ``SkillAdvisor``/``PluginLoader``/``cua-service.js``。每次都是靠人肉审计发现。
2. **版本号漂移** —— 版本号散落在 5 处，v10.3.1 才补齐同步。
3. **文档宣称与实现不符** —— ``CLAUDE.md`` 描述了 ``plugins/`` 目录与
   ``VAP_PLUGIN_DIR``，但它们当时都不存在；``getting-started.md`` 留着 TODO。

三道门
------
**门 1 · wiring（棘轮式）**：扫描 ``src/**/*.py`` 的模块级公开符号
（``def``/``class``，不含 ``_`` 前缀），若该符号在**非测试**代码里零引用，
且未标注 ``# not-wired: <原因>``，即视为"未接线"。
- 免阻塞机制：``scripts/wiring_baseline.txt`` 冻结现有存量（棘轮）。
  新增未接线符号 → 失败；把该死代码清掉 → 基线变"陈旧"（警告）。
- 重新生成基线：``--update-baseline``。

**门 2 · version**：校验 5 处版本号一致（constants / app.py / two package.json /
CHANGELOG 顶部）。

**门 3 · docs**：校验文档里提到的关键路径真实存在，且 ``docs/`` 下没有
``TODO:`` 残留。

退出码：0=全通过；1=有失败。

执行提示：全量 wiring 门在 CI 跑；本机快速验证用
  `python scripts/check_wiring.py --check version,docs` 或 `pytest tests/test_check_wiring.py`。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "scripts" / "wiring_baseline.txt"

#: 扫描范围（源码树）
SCAN_ROOTS = ("src",)
#: 额外扫描的"调用方来源"（跨语言也算接线）
CALLER_EXTRA_GLOBS = (
    "launcher.py",
    "desktop/*.js",
    "webapp/src/**/*.ts",
    "webapp/src/**/*.tsx",
    "scripts/*.py",
)
#: 不计入"调用方"的路径（测试 / 缓存）
SKIP_PARTS = ("__pycache__", "node_modules", ".next", "venv")
SKIP_PREFIXES = ("tests/", "graft/")

_DEF_RE = re.compile(r"^(?P<kind>def|class)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
_NOT_WIRED_RE = re.compile(r"#\s*not-wired\s*:\s*(?P<reason>.+)")


@dataclass
class Violation:
    gate: str
    subject: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - 仅用于打印
        return f"[{self.gate}] {self.subject}: {self.detail}"


@dataclass
class Report:
    failures: List[Violation] = field(default_factory=list)
    warnings: List[Violation] = field(default_factory=list)

    def fail(self, gate: str, subject: str, detail: str) -> None:
        self.failures.append(Violation(gate, subject, detail))

    def warn(self, gate: str, subject: str, detail: str) -> None:
        self.warnings.append(Violation(gate, subject, detail))


# ---------------------------------------------------------------- 工具


def _iter_python_files() -> List[Path]:
    out: List[Path] = []
    for root in SCAN_ROOTS:
        base = PROJECT_ROOT / root
        if not base.is_dir():
            continue
        for p in base.rglob("*.py"):
            if any(part in SKIP_PARTS for part in p.parts):
                continue
            out.append(p)
    return sorted(out)


def _iter_caller_files() -> List[Path]:
    out: List[Path] = []
    for glob in CALLER_EXTRA_GLOBS:
        out.extend(sorted(PROJECT_ROOT.glob(glob)))
    for p in _iter_python_files():
        out.append(p)
    seen = set()
    uniq: List[Path] = []
    for p in out:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        if rel in seen:
            continue
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        if rel.startswith(SKIP_PREFIXES):
            continue
        seen.add(rel)
        uniq.append(p)
    return uniq


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


# ---------------------------------------------------------------- 门 1


def collect_public_symbols() -> Dict[str, Tuple[str, int]]:
    """{'src/core/x.py::Foo': (path, lineno)} —— 模块级公开 def/class。"""
    symbols: Dict[str, Tuple[str, int]] = {}
    for path in _iter_python_files():
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        rel = _rel(path)
        for idx, line in enumerate(lines, start=1):
            if line.startswith((" ", "\t")):
                continue  # 只要模块级
            m = _DEF_RE.match(line)
            if not m:
                continue
            name = m.group("name")
            if name.startswith("_"):
                continue
            # `# not-wired: 原因` 标注视为显式豁免。
            # 接受三种位置：同一行（`def f():  # not-wired: ...`）、紧邻上一行、
            # 函数体前两行（`def f():\n    return 1  # not-wired: ...`）。
            if re.search(r"#\s*not-wired", line):
                continue
            if idx >= 2 and re.search(r"#\s*not-wired", lines[idx - 2]):
                continue
            if any("# not-wired" in nxt for nxt in lines[idx:idx + 2]):
                continue
            symbols[f"{rel}::{name}"] = (rel, idx)
    return symbols


def find_unwired(symbols: Dict[str, Tuple[str, int]]) -> List[str]:
    """返回零生产引用的符号 key（排序稳定）。"""
    caller_files = _iter_caller_files()
    # 预读所有调用方文件内容（含定义文件自身，用于排除定义行）
    contents: Dict[str, str] = {}
    for path in caller_files:
        rel = _rel(path)
        try:
            contents[rel] = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            contents[rel] = ""

    unwired: List[str] = []
    for key, (def_rel, _lineno) in symbols.items():
        name = key.split("::", 1)[1]
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        found = False
        for rel, text in contents.items():
            hits = len(pattern.findall(text))
            if rel == def_rel:
                # 定义文件自身也算调用方，但要减掉定义本身那一处
                hits -= 1
            if hits > 0:
                found = True
                break
        if not found:
            unwired.append(key)
    return sorted(unwired)


def load_baseline() -> List[str]:
    if not BASELINE_PATH.is_file():
        return []
    out: List[str] = []
    for raw in BASELINE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def write_baseline(entries: Sequence[str]) -> None:
    header = (
        "# 未接线符号基线（棘轮）—— v10.4.0 (P0-7)\n"
        "#\n"
        "# 这些公开符号目前在非测试代码里没有调用方。门 1 会阻止**新增**条目；\n"
        "# 清理掉某个符号后重新生成基线，棘轮就会收紧。\n"
        "#\n"
        "# 重新生成: python scripts/check_wiring.py --check wiring --update-baseline\n"
        "# 显式豁免: 在符号定义处写 `# not-wired: <原因>`\n"
    )
    BASELINE_PATH.write_text(header + "\n".join(entries) + "\n", encoding="utf-8")


def gate_wiring(report: Report, update_baseline: bool = False) -> None:
    symbols = collect_public_symbols()
    unwired = find_unwired(symbols)
    if update_baseline:
        write_baseline(unwired)
        print(f"  基线已更新: {len(unwired)} 个未接线符号 -> {_rel(BASELINE_PATH)}")
        return
    baseline = set(load_baseline())
    new = [k for k in unwired if k not in baseline]
    stale = sorted(baseline - set(unwired))
    for key in new:
        rel, lineno = symbols[key]
        report.fail("wiring", key, f"未接线且不在基线中（{rel}:{lineno}）—— 接线，或标注 `# not-wired: 原因`")
    for key in stale:
        report.warn("wiring", key, "基线里的条目已不存在或已接线，可收紧基线（--update-baseline）")
    print(f"  未接线符号 {len(unwired)} 个（基线 {len(baseline)}）；新增 {len(new)}，陈旧 {len(stale)}")


# ---------------------------------------------------------------- 门 2


def _read_version_at(path: str, pattern: str) -> str | None:
    p = PROJECT_ROOT / path
    if not p.is_file():
        return None
    m = re.search(pattern, p.read_text(encoding="utf-8", errors="ignore"), re.MULTILINE)
    return m.group(1) if m else None


def collect_versions() -> Dict[str, str | None]:
    return {
        "src/utils/constants.py:APP_VERSION": _read_version_at(
            "src/utils/constants.py", r'APP_VERSION\s*=\s*"([^"]+)"'),
        "src/web/app.py:version": _read_version_at(
            "src/web/app.py", r'version="([^"]+)"'),
        "desktop/package.json:version": _read_version_at(
            "desktop/package.json", r'"version"\s*:\s*"([^"]+)"'),
        "webapp/package.json:version": _read_version_at(
            "webapp/package.json", r'"version"\s*:\s*"([^"]+)"'),
        "CHANGELOG.md:top-entry": _read_version_at(
            "CHANGELOG.md", r"^##\s*\[([0-9]+\.[0-9]+\.[0-9]+)\]"),
    }


def gate_version(report: Report) -> None:
    versions = collect_versions()
    missing = [k for k, v in versions.items() if v is None]
    for k in missing:
        report.fail("version", k, "未找到版本号（路径或格式变了？）")
    values = {v for v in versions.values() if v}
    print("  " + ", ".join(f"{k.split(':')[-1]}={v}" for k, v in versions.items()))
    if len(values) > 1 and not missing:
        report.fail(
            "version", "版本号不一致",
            " / ".join(f"{k}={v}" for k, v in versions.items()),
        )


# ---------------------------------------------------------------- 门 3

#: (文档文件, 文档里提到的路径, 说明) —— 校验"文档说的东西真实存在"
DOC_PATH_CONTRACTS: Tuple[Tuple[str, str], ...] = (
    ("CLAUDE.md", "plugins"),
    ("CLAUDE.md", "config/plugins.yml"),
    ("CLAUDE.md", "config/prompts/frame_analysis"),
    ("CLAUDE.md", "src/core/agent/supervisor.py"),
    ("CLAUDE.md", "src/core/tools/scope_guard.py"),
    ("docs/guide/plugin-development.md", "plugins"),
    ("README.md", "src/web/serve.py"),
    ("README.md", "start-desktop.bat"),
)

#: 文档目录（禁止 TODO 残留）
DOC_TODO_ROOTS = ("docs/guide", "CLAUDE.md", "AGENTS.md")


def gate_docs(report: Report) -> None:
    for doc, path in DOC_PATH_CONTRACTS:
        doc_path = PROJECT_ROOT / doc
        if not doc_path.is_file():
            report.warn("docs", doc, "文档不存在，跳过其路径契约")
            continue
        if (PROJECT_ROOT / path).exists():
            continue
        report.fail("docs", f"{doc} -> {path}", "文档提到的路径不存在（文档与实现漂移）")

    todo_hits: List[str] = []
    for root in DOC_TODO_ROOTS:
        target = PROJECT_ROOT / root
        files = [target] if target.is_file() else (
            [p for p in target.rglob("*.md") if p.is_file()] if target.is_dir() else []
        )
        for p in files:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"TODO:", text):
                todo_hits.append(_rel(p))
    for hit in todo_hits:
        report.fail("docs", hit, "文档里仍有 `TODO:` 残留（要么补上，要么删掉）")
    docs_checked = len(DOC_PATH_CONTRACTS)
    print(f"  路径契约 {docs_checked} 条；TODO 残留 {len(todo_hits)} 处")


# ---------------------------------------------------------------- main


GATES = {
    "wiring": gate_wiring,
    "version": gate_version,
    "docs": gate_docs,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TingFeng Hermes CI 三道门")
    parser.add_argument(
        "--check", default="wiring,version,docs",
        help="要跑的门（逗号分隔），默认全跑",
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="重新生成未接线符号基线（棘轮收紧/放宽）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结论")
    args = parser.parse_args(argv)

    selected = [g.strip() for g in args.check.split(",") if g.strip()]
    unknown = [g for g in selected if g not in GATES]
    if unknown:
        print(f"未知的门: {unknown}（可用: {sorted(GATES)}）", file=sys.stderr)
        return 2

    report = Report()
    print("== TingFeng Hermes 检查门 ==")
    for gate in selected:
        print(f"[{gate}]")
        if gate == "wiring":
            gate_wiring(report, update_baseline=args.update_baseline)
        else:
            GATES[gate](report)

    if args.json:
        print(json.dumps({
            "failures": [str(v) for v in report.failures],
            "warnings": [str(v) for v in report.warnings],
        }, ensure_ascii=False, indent=2))

    for w in report.warnings:
        print(f"  warn — {w}")
    for f in report.failures:
        print(f"  FAIL — {f}")

    if report.failures:
        print(f"\n结果: 失败（{len(report.failures)} 项）")
        return 1
    print("\n结果: 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
