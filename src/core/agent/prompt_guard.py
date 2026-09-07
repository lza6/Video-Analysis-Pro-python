"""提示注入守卫 — 过滤/转义工具输出,标记不可信内容。

ReactLoopAgent 的 tool_result 来源于工具执行结果,可能包含恶意内容
(例如 search_kb 返回的网页/日志里嵌入 "ignore previous instructions")。
LLM 在下一轮消费 tool message 时,若不隔离,会把恶意指令当成系统指令执行。

本模块实现:
  - `PromptGuard`: 守卫器(可配置模式列表/标签名)。
  - `sanitize_tool_output(text)`: 把工具输出包进 <tool_output> 标签 +
    对命中注入模式的片段加 [UNTRUSTED:...] 前缀转义,让 LLM 明确这是不可信数据。
  - `detect_injection(text)`: 返回命中的注入模式列表(带类型/证据/位置)。
  - `guard_messages(messages)`: 对 messages 列表里 role==tool 的 content 跑 sanitize,
    返回新列表(不改原对象,遵循不可变)。

纯标准库(re + base64 启发式),不引新依赖。
"""
from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern


class InjectionKind(str, Enum):
    """注入模式分类。"""

    IGNORE_INSTRUCTIONS = "ignore_instructions"
    ROLE_HIJACK = "role_hijack"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    BASE64_EVASION = "base64_evasion"
    HIDDEN_UNICODE = "hidden_unicode"
    SYSTEM_SPOOF = "system_spoof"


@dataclass(frozen=True)
class InjectionPattern:
    """单条命中注入模式的描述。

    Attributes:
        kind: 注入分类。
        match: 命中的原文片段(证据)。
        start/end: 在原文中的字符偏移。
        rule: 命中的规则名(便于调试)。
    """

    kind: InjectionKind
    match: str
    start: int
    end: int
    rule: str


@dataclass(frozen=True)
class _Rule:
    """单条检测规则。

    Attributes:
        name: 规则名。
        kind: 命中后归入的注入分类。
        pattern: 编译好的正则(IGNORECASE | DOTALL)。
    """

    name: str
    kind: InjectionKind
    pattern: Pattern[str]


def _build_rules() -> List[_Rule]:
    """构建内置检测规则集。

    所有规则大小写不敏感。短语边界用 \\b 避免误伤单词内子串。
    """
    specs: List[tuple[str, InjectionKind, str]] = [
        # 忽略/无视先前指令
        ("ignore_previous", InjectionKind.IGNORE_INSTRUCTIONS,
         r"\bignore\s+(?:all\s+|the\s+)?previous\s+instructions?\b"),
        ("disregard_above", InjectionKind.IGNORE_INSTRUCTIONS,
         r"\bdisregard\s+(?:all\s+|the\s+)?(?:above|previous)\b"),
        ("forget_previous", InjectionKind.IGNORE_INSTRUCTIONS,
         r"\bforget\s+(?:everything|all\s+previous|prior)\b"),
        ("ignore_above", InjectionKind.IGNORE_INSTRUCTIONS,
         r"\bignore\s+(?:all\s+|the\s+)?(?:above|prior)\s+(?:instructions?|rules?)\b"),
        ("new_instructions", InjectionKind.IGNORE_INSTRUCTIONS,
         r"\b(?:new|updated|real)\s+instructions?\s*[:：]\b"),
        # 角色劫持
        ("you_are_now", InjectionKind.ROLE_HIJACK,
         r"\byou\s+are\s+now\s+\w+"),
        ("act_as", InjectionKind.ROLE_HIJACK,
         r"\bact\s+as\s+\w+"),
        ("pretend_to_be", InjectionKind.ROLE_HIJACK,
         r"\bpretend\s+(?:to\s+be|you\s+are)\b"),
        ("from_now_on", InjectionKind.ROLE_HIJACK,
         r"\bfrom\s+now\s+on\b.{0,40}\b(?:you|act|pretend|mode)\b"),
        ("enter_mode", InjectionKind.ROLE_HIJACK,
         r"\benter\s+(?:developer|debug|admin|root|sudo|dan)\s+mode\b"),
        # 越权 / 泄露系统提示
        ("reveal_system_prompt", InjectionKind.PRIVILEGE_ESCALATION,
         r"\b(?:reveal|show|display|print|output)\s+(?:me\s+)?(?:your\s+)?"
         r"(?:system\s+prompt|instructions?|initial\s+(?:message|prompt))\b"),
        ("show_instructions", InjectionKind.PRIVILEGE_ESCALATION,
         r"\b(?:show|tell|give)\s+me\s+your\s+(?:rules?|guidelines?|config)\b"),
        ("override_rules", InjectionKind.PRIVILEGE_ESCALATION,
         r"\boverride\s+(?:your|the|all)\s+(?:rules?|restrictions?|constraints?)\b"),
        ("no_restrictions", InjectionKind.PRIVILEGE_ESCALATION,
         r"\b(?:you\s+have\s+no|without\s+any)\s+restrictions?\b"),
        # 数据外泄
        ("send_to", InjectionKind.DATA_EXFILTRATION,
         r"\bsend\s+\S+(?:\s+\S+){0,5}\s+to\b"),
        ("exfiltrate", InjectionKind.DATA_EXFILTRATION,
         r"\bexfiltrate\b"),
        ("post_to_url", InjectionKind.DATA_EXFILTRATION,
         r"\bpost\s+\S+(?:\s+\S+){0,5}\s+(?:to|at)\s+https?://\S+"),
        ("upload_to", InjectionKind.DATA_EXFILTRATION,
         r"\bupload\s+\S+(?:\s+\S+){0,5}\s+to\s+https?://\S+"),
        ("curl_pipe", InjectionKind.DATA_EXFILTRATION,
         r"\bcurl\s+https?://\S+\s*\|\s*(?:sh|bash)\b"),
    ]
    return [
        _Rule(name=n, kind=k, pattern=re.compile(p, re.IGNORECASE | re.DOTALL))
        for n, k, p in specs
    ]


# base64 启发式:长度 >= 32 的 [A-Za-z0-9+/=] 串,解码后含可疑关键词。
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{32,}={0,2}")
_BASE64_SUSPICIOUS = re.compile(
    r"(?:ignore|previous|instructions?|system|prompt|exfiltrate|"
    r"act\s+as|pretend|override|reveal)",
    re.IGNORECASE,
)


def _detect_base64_evasion(text: str) -> List[InjectionPattern]:
    """检测 base64 编码的注入指令。

    启发式:
      1. 找到长度 >= 32 的候选 base64 串。
      2. 尝试 base64 解码(容错,失败跳过)。
      3. 解码后若含可疑关键词 → 命中。
    """
    hits: List[InjectionPattern] = []
    for m in _BASE64_RE.finditer(text):
        candidate = m.group(0)
        # 长度必须是 4 的倍数才能合法 base64;补齐对齐也试一次
        padded = candidate + "=" * (-len(candidate) % 4)
        try:
            decoded = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            continue
        try:
            decoded_text = decoded.decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        if _BASE64_SUSPICIOUS.search(decoded_text):
            hits.append(InjectionPattern(
                kind=InjectionKind.BASE64_EVASION,
                match=candidate,
                start=m.start(),
                end=m.end(),
                rule="base64_suspicious_payload",
            ))
    return hits


# 零宽字符集(U+200B..U+200D 零宽空格/ZWSP/ZWNJ/ZWJ, U+2060 词连接符,
# U+FEFF BOM/零宽不换行空格)。攻击者可用其藏指令绕过肉眼审查。
# 用 \u 转义避免源文件本身混入不可见字符。
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")
# system: / assistant: 等角色伪冒前缀(行首或空白后),企图让 LLM 把
# 工具输出当成系统/助手消息执行。
_SYSTEM_SPOOF_RE = re.compile(
    r"(?:^|\n|\s)(?:system|assistant|developer)\s*[:：]\s",
    re.IGNORECASE,
)


def _detect_hidden_unicode(text: str) -> List[InjectionPattern]:
    """检测零宽字符隐藏内容。

    零宽字符本身不可见,可用于在正常文本中藏指令(如同形字注入)。
    命中即标记,提示人工审查;不删字符(保留证据),由 sanitize 转义。
    """
    hits: List[InjectionPattern] = []
    for m in _ZERO_WIDTH_RE.finditer(text):
        hits.append(InjectionPattern(
            kind=InjectionKind.HIDDEN_UNICODE,
            match=repr(m.group(0)),
            start=m.start(),
            end=m.end(),
            rule="zero_width_char",
        ))
    return hits


def _detect_system_spoof(text: str) -> List[InjectionPattern]:
    """检测 system:/assistant:/developer: 角色伪冒前缀。

    工具输出里出现这些前缀,是企图让 LLM 把后续内容当成系统/助手指令
    执行(典型提示注入绕过)。命中即标记。
    """
    hits: List[InjectionPattern] = []
    for m in _SYSTEM_SPOOF_RE.finditer(text):
        hits.append(InjectionPattern(
            kind=InjectionKind.SYSTEM_SPOOF,
            match=m.group(0).strip(),
            start=m.start(),
            end=m.end(),
            rule="role_prefix_spoof",
        ))
    return hits


class PromptGuard:
    """工具输出注入守卫。

    用法:
        guard = PromptGuard()
        safe = guard.sanitize_tool_output(tool_text)
        msgs = guard.guard_messages(raw_messages)
    """

    TOOL_OPEN = "<tool_output>"
    TOOL_CLOSE = "</tool_output>"
    UNTRUSTED_PREFIX = "[UNTRUSTED:"

    def __init__(
        self,
        rules: Optional[List[_Rule]] = None,
        *,
        max_length: int = 200_000,
    ) -> None:
        """初始化守卫器。

        Args:
            rules: 自定义规则集(默认用 _build_rules())。
            max_length: 单次处理文本上限(防超长 DoS),超出截断。
        """
        self._rules: List[_Rule] = rules if rules is not None else _build_rules()
        self._max_length = max_length

    def detect_injection(self, text: str) -> List[InjectionPattern]:
        """检测文本里的注入模式。

        Args:
            text: 待检测文本。

        Returns:
            命中模式列表(按 start 排序)。空列表 = 未命中。
        """
        if not text:
            return []
        hits: List[InjectionPattern] = []
        for rule in self._rules:
            for m in rule.pattern.finditer(text):
                hits.append(InjectionPattern(
                    kind=rule.kind,
                    match=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    rule=rule.name,
                ))
        hits.extend(_detect_base64_evasion(text))
        hits.extend(_detect_hidden_unicode(text))
        hits.extend(_detect_system_spoof(text))
        hits.sort(key=lambda h: (h.start, h.end))
        return hits

    def sanitize_tool_output(self, text: str) -> str:
        """转义工具输出,标记为不可信。

        流程:
          1. 空串直接返回空串。
          2. 超长截断 + 加截断标记。
          3. 先把已有的 [UNTRUSTED:...] 区段用占位符遮蔽,避免 detect 在
             已标记区段内重复命中导致标记膨胀(幂等)。
          4. 把命中的注入片段替换成 [UNTRUSTED:<原片段>](保留原文作证据,
             但用标记声明不可信,LLM 应将其视作数据)。
          5. 还原占位符。
          6. 若文本已形如 <tool_output>...</tool_output>(被二次喂回),
             不再二次包裹;否则整体包进 <tool_output>...</tool_output>。

        Args:
            text: 工具原始输出。

        Returns:
            转义后的安全文本。
        """
        if not text:
            return ""
        truncated = False
        if len(text) > self._max_length:
            text = text[: self._max_length]
            truncated = True
        already_wrapped = (
            text.lstrip().startswith(self.TOOL_OPEN)
            and text.rstrip().endswith(self.TOOL_CLOSE)
        )
        # 遮蔽已有 [UNTRUSTED:...] 区段,防止重复命中
        placeholders: List[str] = []

        def _mask(m: re.Match[str]) -> str:
            placeholders.append(m.group(0))
            return f"\x00U{len(placeholders) - 1}\x00"

        masked = re.sub(
            re.escape(self.UNTRUSTED_PREFIX) + r"[^\]]*\]", _mask, text)
        hits = self.detect_injection(masked)
        out = masked
        for h in reversed(hits):
            replacement = f"{self.UNTRUSTED_PREFIX}{h.match}]"
            out = out[: h.start] + replacement + out[h.end:]
        for i, p in enumerate(placeholders):
            out = out.replace(f"\x00U{i}\x00", p)
        if truncated:
            out += "\n[TRUNCATED: tool output exceeded length limit]"
        if already_wrapped:
            return out
        return f"{self.TOOL_OPEN}\n{out}\n{self.TOOL_CLOSE}"

    def guard_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """对消息列表里 role==tool 的 content 跑 sanitize。

        返回新列表(不改原对象);非 tool 消息原样复制。

        Args:
            messages: LLM 消息列表。

        Returns:
            新的、tool content 已转义的消息列表。
        """
        out: List[Dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "tool":
                content = m.get("content", "")
                if isinstance(content, str):
                    sanitized = self.sanitize_tool_output(content)
                else:
                    # 非字符串 content(罕见):转字符串再守卫
                    sanitized = self.sanitize_tool_output(
                        str(content) if content is not None else "")
                new_m: Dict[str, Any] = dict(m)
                new_m["content"] = sanitized
                out.append(new_m)
            else:
                out.append(dict(m))
        return out


# 模块级便捷函数(单例守卫器,复用规则编译)
_DEFAULT_GUARD = PromptGuard()


def sanitize_tool_output(text: str) -> str:
    """模块级便捷方法:用默认守卫器转义工具输出。"""
    return _DEFAULT_GUARD.sanitize_tool_output(text)


def detect_injection(text: str) -> List[InjectionPattern]:
    """模块级便捷方法:用默认守卫器检测注入。"""
    return _DEFAULT_GUARD.detect_injection(text)


def guard_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """模块级便捷方法:用默认守卫器守卫消息列表。"""
    return _DEFAULT_GUARD.guard_messages(messages)


__all__ = [
    "PromptGuard",
    "InjectionKind",
    "InjectionPattern",
    "sanitize_tool_output",
    "detect_injection",
    "guard_messages",
]
