"""创建 v10.3.1 GitHub Release + 上传 Setup.exe（python 直传，替代卡死的 PS1）。

用法: ./venv/Scripts/python.exe scripts/create_release_v1031.py
凭据: GITHUB_TOKEN 环境变量，或 git credential fill。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = "lza6/Video-Analysis-Pro-python"
TAG = "v10.3.1"
EXE = Path(__file__).resolve().parents[1] / "desktop" / "dist" / \
    "TingFeng Hermes Setup 10.3.1.exe"
ASSET_NAME = "TingFeng.Hermes.Setup.10.3.1.exe"

BODY = """## v10.3.1 交付闭环 — 把 v10.3.0 已建成的能力真正送到默认用户手上

### 修了什么
- **默认路径接通**: Agent 引擎默认切 react —— 监督层/写审批/分层记忆/skills roster 此前只挂在灰度路径，默认用户完全拿不到
- **审批 SSE 契约修复**: react 模式审批事件此前到不了前端（写工具静默挂 60s 后拒绝），现 /chat→/run_stream 契约打通
- **工具面统一**: 媒体生成(4) + 浏览器自动化(12) 此前只在 MCP 可用，现统一进 Web agent（16→32 工具）
- **skills 正文注入**: 此前只注入技能名字，SKILL.md 的模板/决策表从未到达模型 —— 现在 skills 真的能用
- **工具风险分类修正**: 字幕/短视频/配音（真实写盘）与 cdp_evaluate（任意 JS）此前被静默放行，现强制审批
- **沙箱防自杀**: Windows Job Object 移除 KILL_ON_JOB_CLOSE（exit 会杀掉后端自身）
- **压缩反增修复**: 上下文压缩结果此前被丢弃，摘要反而增加 token
- **工具调用解析**: 兼容 GLM 系真实输出 function=NAME 变体（此前静默丢失工具调用）
- **安装包修正**: 此前包内 0 个 skill（SKILL.md 被 filter 排除），现 12 个全部进包
- **大白话解释器 eli5 接线**: 每次工具调用产出一句话人话摘要

### 验证
- 全量测试 1336 passed + 2 skipped / 覆盖率 80%
- Playwright E2E 23 passed（真实 chromium）
- 真实后端审批链路 E2E: 允许/拒绝两路全通
- 独立 Critic 审查: CONDITIONAL PASS → 修复 BLOCKER 后 PASS
- 安装包: 697MB, resources/config/skills 实测 12 目录
"""


def get_token() -> str:
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        return tok
    cred = subprocess.run(["git", "credential", "fill"], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          input="protocol=https\nhost=github.com\n\n").stdout
    for line in cred.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("ERR: no token (GITHUB_TOKEN or git credential)")


def api(method: str, url: str, token: str, data: dict | bytes | None = None,
        content_type: str = "application/json") -> dict:
    body = None
    headers = {"Authorization": f"token {token}",
               "User-Agent": "vap-release-script"}
    if data is not None:
        if isinstance(data, (dict,)):
            body = json.dumps(data).encode("utf-8")
        else:
            body = data
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def main() -> None:
    token = get_token()
    # 1. 若已存在则复用
    try:
        rel = api("GET", f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}",
                  token)
        print(f"release 已存在 id={rel['id']}, 复用")
    except Exception:
        rel = api("POST", f"https://api.github.com/repos/{REPO}/releases", token, {
            "tag_name": TAG,
            "name": "v10.3.1 交付闭环（Deliver, not Build）",
            "body": BODY,
            "draft": False,
            "prerelease": False,
        })
        print(f"CREATED {rel['html_url']} id={rel['id']}")

    # 2. 已有同名 asset → 跳过上传
    existing = [a for a in rel.get("assets", []) if a["name"] == ASSET_NAME]
    if existing:
        a = existing[0]
        print(f"asset 已存在 state={a['state']} url={a['browser_download_url']}")
        if a["state"] == "uploaded":
            print("DONE")
            return
        # starter/未完成 → 删除重传
        urllib.request.urlopen(urllib.request.Request(
            f"https://api.github.com/repos/{REPO}/releases/assets/{a['id']}",
            headers={"Authorization": f"token {token}",
                     "User-Agent": "vap-release-script"},
            method="DELETE"), timeout=60)
        print("删除未完成 asset，重新上传")

    # 3. 上传（curl 分块 + 长超时;python urllib 直传 697MB 会 write timeout）
    size = EXE.stat().st_size
    print(f"上传 {EXE.name} ({size:,} bytes) via curl ...")
    import shutil
    curl = shutil.which("curl")
    if not curl:
        raise SystemExit("ERR: curl 不可用")
    up_url = (f"https://uploads.github.com/repos/{REPO}/releases/"
              f"{rel['id']}/assets?name={ASSET_NAME}")
    proc = subprocess.run(
        [curl, "-sS", "-X", "POST", up_url,
         "-H", f"Authorization: token {token}",
         "-H", "Content-Type: application/octet-stream",
         "--upload-file", str(EXE),
         "--connect-timeout", "60",
         "--max-time", "3600",
         "--ssl-no-revoke"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"ERR: curl 上传失败 rc={proc.returncode}: {proc.stderr[:500]}")
    up = json.loads(proc.stdout)
    print(f"ASSET state={up.get('state')} url={up.get('browser_download_url')}")
    print("DONE")


if __name__ == "__main__":
    main()
