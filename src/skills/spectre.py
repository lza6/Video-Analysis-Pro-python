"""Skill security scanner (spectre) — a rules-based variant of the SkillSpector idea.

Scans a single SKILL.md (frontmatter + body) with four classes of rule checks,
pure stdlib, no LLM:
  1. Prompt injection: spoofed system/fake prefixes (``ignore previous``,
     ``system:``, ``developer:``).
  2. Dangerous commands: ``rm -rf``, ``os.system``, bare ``subprocess``,
     ``eval(``/``exec(``, shell sugar.
  3. Secret patterns: ``sk-`` prefix, hardcoded ``api_key``/``apikey``/
     ``secret``/``token``.
  4. External URL whitelist: only allow trusted domains, flag unknown URLs.

Returns ScanReport (clean + findings). Known-dangerous constructs are flagged
regardless of whitelisting. Loader maps findings to security_warning on Skill.

Pure stdlib + rules-based; no paid LLM calls (hard red line).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 危险命令模式（命中即高优先级，不因白名单豁免）
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS: list[tuple[str, str, str]] = [
    (r"\brm\s+-rf\b", "high", "dangerous_rm_rf"),
    (r"\bos\.system\s*\(", "critical", "os_system_call"),
    (r"\bPopen\s*\(", "high", "subprocess_popen"),
    (r"\beval\s*\(", "critical", "python_eval"),
    (r"\bexec\s*\(", "critical", "python_exec"),
    (r"\bchmod\s+\d{3,4}\b", "medium", "chmod_permission"),
    (r"base64\s+-d\s*[\"'`]", "high", "base64_decode_pipe"),
    (r"curl\s+[^\n|`]+\s*\|\s*(ba)?sh", "high", "curl_pipe_shell"),
    (r"wget\s+[^\n|`]+\s*\|\s*(ba)?sh", "high", "wget_pipe_shell"),
    (r"(?:\s|^)subprocess\s*\.\s*run\s*\(", "high", "subprocess_run"),
]

# 危险命令的行内黑名单（出现即告警；整行含两步安装命令的正常行为不算误报）
_DANGEROUS_KEYWORDS = (
    "os.system",
    "subprocess.run",
    "subprocess.Popen",
    "eval(",
    "exec(",
    "pickle.load",
    "yaml.load(",
)

# ---------------------------------------------------------------------------
# 指令注入模式
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)ignore\s+(a|l|all|the|above|previous)\s+(instruction|prompt|text)s?",
     "system_override"),
    (r"(?i)ignore\s+all\s+previous\s+instructions?\s+and", "system_override"),
    (r"(?i)^\s*<\|?system\|?>\s*:?\s*$", "fake_system_tag"),
    (r"(?i)^\s*system\s*:\s*", "fake_system_prefix"),
    (r"(?i)^\s*developer\s*:\s*", "fake_developer_prefix"),
    (r"(?i)^\s*user\s*:\s*", "fake_user_prefix"),
    (r"(?i)(reveal|show|print|output).{0,30}(system\s*prompt|instructions?)",
     "exfil_prompt"),
]

# ---------------------------------------------------------------------------
# 密钥模式
# ---------------------------------------------------------------------------
_SECRET_PATTERNS: list[tuple[str, str]] = [
    (r"\bsk-[A-Za-z0-9_-]{12,}\b", "sk_token"),
    (r"\b[A-Za-z0-9_-]{20,}:\b\s*[A-Za-z0-9+/=_\-]{16,}", "basic_auth_guess"),
]

_SECRET_KEYS = (
    "api_key",
    "apikey",
    "api-key",
    "secret_key",
    "secret",
    "password",
    "token",
    "passwd",
    "client_secret",
    "access_key",
)

_URL_RE = re.compile(r"https?://([^/\s\"'`)]+)")
_URL_SCHEME_ITEMS = (".zip", ".tar.gz", ".whl", ".exe", ".deb", ".apk")

# 默认受信白名单：框架/平台/官方发布域名（可测试内扩展）
_DEFAULT_WHITELIST = (
    "huggingface.co",
    "raw.githubusercontent.com",
    "github.com",
    "gist.githubusercontent.com",
    "civitai.com",
    "pypi.org",
    "docker.io",
    "registry.npmjs.org",
    "files.pythonhosted.org",
    "cdn.jsdelivr.net",
    "unpkg.com",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "googleapis.com",
    "youtube.com",
)

_WHITELIST = _DEFAULT_WHITELIST


@dataclass(frozen=True)
class Finding:
    """一条安全发现。severity: medium / high / critical。"""

    code: str
    severity: str
    line: int
    snippet: str
    reason: str


@dataclass(frozen=True)
class ScanReport:
    """一次扫描结果。clean=True 表示无 medium 及以上发现。"""

    skill_name: str
    clean: bool
    findings: tuple[Finding, ...]

    @property
    def critical(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "critical")

    @property
    def high(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "high")


def set_url_whitelist(domains: tuple[str, ...]) -> None:
    """覆盖外部 URL 白名单（测试隔离用）。不受信任域名直接挡。"""
    global _WHITELIST
    _WHITELIST = tuple(domains)


def reset_url_whitelist() -> None:
    """恢复默认白名单。"""
    global _WHITELIST
    _WHITELIST = _DEFAULT_WHITELIST


def _match_domain(url: str) -> bool:
    """url 是否落在白名单域名下（含子域匹配）。"""
    host = url.lower()
    for d in _WHITELIST:
        d = d.lower()
        if host == d or host.endswith("." + d):
            return True
    return False


def _classify_code(severity: str) -> str:
    return severity


def _severity_rank(sev: str) -> int:
    return {"medium": 1, "high": 2, "critical": 3}.get(sev, 1)


def _scan_content(name: str, content: str) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    lines = content.splitlines()
    for lineno, line in enumerate(lines, start=1):
        snippet = line.strip()
        if not snippet:
            continue

        # 1. 危险命令（highest priority，白名单不豁免）
        for pat, sev, code in _DANGEROUS_PATTERNS:
            m = re.search(pat, line)
            if m:
                findings.append(Finding(
                    code=code, severity=sev, line=lineno,
                    snippet=snippet[:80], reason=f"危险命令模式: {code}"))
        # 危险关键词辅助（整行多个命中只记首个，避免刷屏）
        hit_kw = next((k for k in _DANGEROUS_KEYWORDS if k in line.lower()), None)
        if hit_kw is not None and not any(
                f.code.startswith("dangerous_") or f.code.startswith("os_")
                or f.code.startswith("python_") for f in findings):
            findings.append(Finding(
                code="dangerous_keyword", severity="high", line=lineno,
                snippet=snippet[:80], reason=f"危险关键词: {hit_kw}"))

        # 2. 指令注入
        for pat, code in _INJECTION_PATTERNS:
            if re.search(pat, line):
                findings.append(Finding(
                    code=code, severity="critical", line=lineno,
                    snippet=snippet[:80], reason=f"指令注入: {code}"))

        # 3. 密钥模式
        for pat, code in _SECRET_PATTERNS:
            m = re.search(pat, line)
            if m:
                findings.append(Finding(
                    code=code, severity="medium", line=lineno,
                    snippet=snippet[:80], reason=f"疑似密钥: {code}"))
        lower = line.lower()
        for key in _SECRET_KEYS:
            if key in lower:
                findings.append(Finding(
                    code="secret_keyword", severity="medium", line=lineno,
                    snippet=snippet[:80], reason=f"疑似敏感字段: {key}"))

    # 4. 外部 URL 白名单（整数行去重）
    seen_url_findings: list[Finding] = []
    for lineno, line in enumerate(lines, start=1):
        for host in _URL_RE.findall(line):
            if not _match_domain(host):
                seen_url_findings.append(Finding(
                    code="untrusted_url", severity="low", line=lineno,
                    snippet=host[:80], reason="外部 URL 不落在受信白名单"))

    # URL finding 去重（同 host 同原因只留第一行）
    seen: set[tuple[str, int]] = set()
    for f in seen_url_findings:
        key = (f.snippet, f.line)
        if key not in seen:
            seen.add(key)
            if not any(x.code == "untrusted_url" and x.line == f.line
                       and x.snippet == f.snippet for x in findings):
                findings.append(f)

    # 确定性排序：severity 降序 → line 升序
    findings.sort(key=lambda f: (-_severity_rank(f.severity), f.line))
    return tuple(findings)


def scan_skill(root: Path) -> ScanReport:
    """扫描 <root>/SKILL.md，返回 ScanReport。

    文件缺失/读取失败按「无法读取」记一条 high finding（非静默通过）。
    """
    root = Path(root)
    md_path = root / "SKILL.md"
    name = root.name
    if not md_path.exists():
        return ScanReport(
            skill_name=name, clean=False,
            findings=(Finding(code="missing_skill_md", severity="high",
                              line=0, snippet="", reason="缺少 SKILL.md 文件"),))
    try:
        content = md_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return ScanReport(
            skill_name=name, clean=False,
            findings=(Finding(code="read_error", severity="high", line=0,
                              snippet=str(exc)[:80], reason="读取 SKILL.md 失败"),))
    findings = _scan_content(name, content)
    # clean：无 medium 及以上 finding（low 的 untrusted_url 不算脏）
    clean = not any(f.severity in ("medium", "high", "critical") for f in findings)
    return ScanReport(skill_name=name, clean=clean, findings=findings)