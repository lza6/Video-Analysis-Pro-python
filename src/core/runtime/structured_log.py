"""结构化日志(JSON 单行) + trace_id/span_id 上下文传播。

为 v9 Agent 框架 / FastAPI 后端提供可关联的链路追踪能力:
- `JSONFormatter`: 把 `logging.LogRecord` 格式化成单行 JSON,
  字段 `{ts, level, logger, msg, trace_id, span_id, fields}`。
  trace_id/span_id 从 `contextvars` 读(缺失则 None),
  record 的非标准属性收进 `fields`。
- `trace_context`: 上下文管理器, 进入设 trace_id(+span_id), 退出 reset。
  trace_id 缺省优先继承当前上下文(便于嵌套子 span), 否则生成 `uuid4().hex`。
- `install_json_logging`: 配置 root logger 走 stdout + JSONFormatter(幂等)。

仅依赖标准库(logging / json / contextvars / uuid / time / sys / contextlib),
不引新依赖,与项目现有 logging 风格一致。本模块只提供能力, 不改动 app.py /
logs.py / loop.py 的接入(后续主控接入再做)。
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

# trace 链路上下文(异步安全: contextvars 自动随 asyncio task 传播)
TRACE: ContextVar[str | None] = ContextVar("trace_id", default=None)
SPAN: ContextVar[str | None] = ContextVar("span_id", default=None)

# LogRecord 标准属性白名单(除此外的 __dict__ 键都算 extra, 进 fields)
_STD_RECORD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
    "trace_id", "span_id",  # 防止调用方经 extra 传入时与 contextvars 字段重复
})

logger = logging.getLogger(__name__)


class JSONFormatter(logging.Formatter):
    """把 LogRecord 格式化成单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
        ts = f"{ts}.{int(record.msecs):03d}Z"
        fields: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in _STD_RECORD_ATTRS:
                fields[key] = value
        if record.exc_info:
            fields["exception"] = self.formatException(record.exc_info)
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": TRACE.get(),
            "span_id": SPAN.get(),
            "fields": fields,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_trace_id() -> str | None:
    """当前 contextvars 中的 trace_id(无则 None)。"""
    return TRACE.get()


@contextmanager
def trace_context(trace_id: str | None = None) -> Iterator[str]:
    """进入 trace 上下文: 设 trace_id + 生成 span_id, 退出 reset。

    Args:
        trace_id: 指定 trace_id; 缺省优先继承当前上下文的 trace_id
            (便于嵌套子 span 共享同一 trace_id), 否则自动生成 `uuid4().hex`。

    Yields:
        生效的 trace_id(供调用方记录/传播)。
    """
    tid = trace_id or TRACE.get() or uuid.uuid4().hex
    sid = uuid.uuid4().hex[:16]
    t_tok = TRACE.set(tid)
    s_tok = SPAN.set(sid)
    try:
        yield tid
    finally:
        TRACE.reset(t_tok)
        SPAN.reset(s_tok)


def install_json_logging(level: int = logging.INFO) -> logging.StreamHandler:
    """配置 root logger 走 stdout + JSONFormatter(幂等可重入)。

    Returns:
        已挂载到 root 的 JSON handler。若 root 已有 JSONFormatter handler,
        返回那个已存在的 handler(不重复添加),保证多次调用返回同一实例。
    """
    root = logging.getLogger()
    root.setLevel(level)
    # 幂等: 已有 JSONFormatter handler 则返回它, 不新建
    for h in root.handlers:
        if isinstance(h.formatter, JSONFormatter):
            return h
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)
    return handler


__all__ = [
    "JSONFormatter",
    "TRACE",
    "SPAN",
    "trace_context",
    "install_json_logging",
    "get_trace_id",
]
