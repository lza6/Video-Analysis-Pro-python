"""CI 三道门自测（v10.4.0 P0-7）。

门本身也要有测试——否则"门坏了"会被误读成"代码没问题"。

覆盖：
  1. 门在当前 HEAD 上必须**通过**（否则 PR 无法合入）；
  2. 每一道门都能真的**发现**它该发现的问题（用临时夹具构造违规）；
  3. 棘轮基线存在且格式正确；
  4. CI 配置里真的挂上了这道门。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "check_wiring.py"
BASELINE = PROJECT_ROOT / "scripts" / "wiring_baseline.txt"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )


# ---------------------------------------------------------------- 门本体


def test_gates_pass_on_current_head():
    """三道门在 HEAD 上必须全通过。"""
    proc = _run("--check", "wiring,version,docs")
    assert proc.returncode == 0, (
        f"门未通过（退出码 {proc.returncode}）:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "结果: 通过" in proc.stdout


def test_script_exists_and_runs():
    assert SCRIPT.is_file()
    proc = _run("--help")
    assert proc.returncode == 0


def test_unknown_gate_is_rejected():
    """未知门名必须报错（防拼写错误让门静默不跑）。"""
    proc = _run("--check", "does-not-exist")
    assert proc.returncode == 2
    assert "未知的门" in proc.stderr


def test_single_gates_pass_individually():
    for gate in ("wiring", "version", "docs"):
        proc = _run("--check", gate)
        assert proc.returncode == 0, f"门 {gate} 单独跑失败:\n{proc.stdout}"


# ---------------------------------------------------------------- 棘轮基线


def test_baseline_exists_and_has_header():
    assert BASELINE.is_file(), "缺少 scripts/wiring_baseline.txt"
    text = BASELINE.read_text(encoding="utf-8")
    assert "# not-wired" in text  # 头部说明提到豁免写法
    assert "--update-baseline" in text


def test_baseline_entries_are_well_formed():
    entries = [
        line.strip()
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert entries, "基线为空——若确实已清空请重新生成"
    for entry in entries:
        assert "::" in entry, f"基线条目格式应为 path::Symbol: {entry}"
        path, _, symbol = entry.partition("::")
        assert path.endswith(".py")
        assert symbol and not symbol.startswith("_")


def test_baseline_does_not_contain_wired_symbols():
    """棘轮必须已收紧：我本次接线的 3 个符号不得留在基线里。"""
    text = BASELINE.read_text(encoding="utf-8")
    for wired in ("validate_skill", "SkillAdvisor", "PluginLoader"):
        assert wired not in text, f"{wired} 已接线，却仍在基线中（棘轮未收紧）"


# ---------------------------------------------------------------- 各门的检出能力


def test_wiring_gate_detects_new_unwired_symbol(tmp_path, monkeypatch):
    """门 1 必须能发现"新增了一个没人调用的公开符号"。"""
    import scripts.check_wiring as cw

    fake_src = tmp_path / "src"
    fake_src.mkdir()
    (fake_src / "ghost.py").write_text(
        "def totally_unused_public_symbol():\n    return 1\n", encoding="utf-8")
    (fake_src / "user.py").write_text("x = 1\n", encoding="utf-8")

    monkeypatch.setattr(cw, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cw, "SCAN_ROOTS", ("src",))
    monkeypatch.setattr(cw, "CALLER_EXTRA_GLOBS", ())
    monkeypatch.setattr(cw, "BASELINE_PATH", tmp_path / "baseline.txt")

    report = cw.Report()
    cw.gate_wiring(report)
    assert any("totally_unused_public_symbol" in f.subject for f in report.failures)


def test_wiring_gate_accepts_not_wired_annotation(tmp_path, monkeypatch):
    """`# not-wired: 原因` 必须能豁免（显式声明的未接线不算违规）。"""
    import scripts.check_wiring as cw

    fake_src = tmp_path / "src"
    fake_src.mkdir()
    (fake_src / "ghost.py").write_text(
        "def declared_unwired():\n    return 1  # not-wired: 留给 v11 的预留接口\n",
        encoding="utf-8")

    monkeypatch.setattr(cw, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cw, "SCAN_ROOTS", ("src",))
    monkeypatch.setattr(cw, "CALLER_EXTRA_GLOBS", ())
    monkeypatch.setattr(cw, "BASELINE_PATH", tmp_path / "baseline.txt")

    report = cw.Report()
    cw.gate_wiring(report)
    assert not report.failures


def test_wiring_gate_counts_same_module_usage(tmp_path, monkeypatch):
    """同一模块内被调用的符号必须算「已接线」（否则门会大量误报）。"""
    import scripts.check_wiring as cw

    fake_src = tmp_path / "src"
    fake_src.mkdir()
    (fake_src / "m.py").write_text(
        "def helper():\n    return 1\n\n\ndef main():\n    return helper()\n",
        encoding="utf-8")

    monkeypatch.setattr(cw, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cw, "SCAN_ROOTS", ("src",))
    monkeypatch.setattr(cw, "CALLER_EXTRA_GLOBS", ())
    monkeypatch.setattr(cw, "BASELINE_PATH", tmp_path / "baseline.txt")

    report = cw.Report()
    cw.gate_wiring(report)
    assert not any("helper" in f.subject for f in report.failures)


def test_version_gate_detects_mismatch(monkeypatch):
    """门 2 必须能发现版本号不一致。"""
    import scripts.check_wiring as cw

    monkeypatch.setattr(cw, "collect_versions", lambda: {
        "a": "1.0.0", "b": "2.0.0", "c": "1.0.0", "d": "1.0.0", "e": "1.0.0",
    })
    report = cw.Report()
    cw.gate_version(report)
    assert any("version" == f.gate for f in report.failures)


def test_version_gate_passes_when_all_equal(monkeypatch):
    import scripts.check_wiring as cw

    monkeypatch.setattr(cw, "collect_versions", lambda: {
        "a": "9.9.9", "b": "9.9.9", "c": "9.9.9", "d": "9.9.9", "e": "9.9.9",
    })
    report = cw.Report()
    cw.gate_version(report)
    assert not report.failures


def test_docs_gate_detects_missing_path(tmp_path, monkeypatch):
    """门 3 必须能发现「文档提到的路径不存在」。"""
    import scripts.check_wiring as cw

    (tmp_path / "README.md").write_text("# x", encoding="utf-8")
    monkeypatch.setattr(cw, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cw, "DOC_PATH_CONTRACTS", (("README.md", "does/not/exist"),))
    monkeypatch.setattr(cw, "DOC_TODO_ROOTS", ())

    report = cw.Report()
    cw.gate_docs(report)
    assert any("does/not/exist" in f.subject for f in report.failures)


def test_docs_gate_detects_todo(tmp_path, monkeypatch):
    """门 3 必须能发现文档里的 `TODO:` 残留。"""
    import scripts.check_wiring as cw

    (tmp_path / "d.md").write_text("> TODO: 之后再做\n", encoding="utf-8")
    monkeypatch.setattr(cw, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cw, "DOC_PATH_CONTRACTS", ())
    monkeypatch.setattr(cw, "DOC_TODO_ROOTS", ("d.md",))

    report = cw.Report()
    cw.gate_docs(report)
    assert any("TODO" in f.detail for f in report.failures)


# ---------------------------------------------------------------- CI 接线


def test_ci_runs_wiring_gate():
    """CI 必须真的挂上这道门（否则它只是本地脚本）。"""
    ci = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/check_wiring.py" in ci
    assert "--check wiring,version,docs" in ci


def test_measure_startup_script_exists():
    """启动基准脚本必须存在且可被 CI/人工复跑。"""
    assert (PROJECT_ROOT / "scripts" / "measure_startup.py").is_file()


@pytest.mark.parametrize("gate_arg", ["wiring", "version", "docs"])
def test_gate_json_output_is_valid(gate_arg: str):
    """--json 输出必须可解析（供 CI 消费）。"""
    import json

    proc = _run("--check", gate_arg, "--json")
    assert proc.returncode == 0
    payload = proc.stdout.split("== TingFeng Hermes 检查门 ==", 1)[-1]
    # 取最后一段 JSON（脚本先打印人读版，再打印 JSON）
    start = payload.rfind("{")
    end = payload.rfind("}") + 1
    parsed = json.loads(payload[start:end])
    assert "failures" in parsed and "warnings" in parsed
