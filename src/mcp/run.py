"""`python -m src.mcp.run` —— MCP stdio server 启动入口。

用法：
    echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
        venv/Scripts/python.exe -m src.mcp.run

或直接与 MCP 客户端对接：
    venv/Scripts/python.exe -m src.mcp.run
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保仓库根在 sys.path（从任意 cwd 启动都能 import src.*）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.mcp.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())