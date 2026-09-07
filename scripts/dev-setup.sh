#!/usr/bin/env bash
# 开发者 onboarding 脚本（macOS / Linux bash）
# TingFeng Hermes v9 — 一键配齐开发环境
# 用法：bash scripts/dev-setup.sh

set -e

cd "$(dirname "$0")/.."

cyan='\033[36m'; yellow='\033[33m'; green='\033[32m'; red='\033[31m'
warn='\033[1;33m'; reset='\033[0m'

echo -e "${cyan}=== TingFeng Hermes 开发环境配置 ===${reset}"

# 1. Python 3.10+
echo -e "${yellow}[1/8] 检测 Python 3.10+ ...${reset}"
if ! command -v python3 >/dev/null 2>&1; then
  echo -e "${red}未检测到 python3，请装 3.10+：https://www.python.org/downloads/${reset}"
  exit 1
fi
PY_VER=$(python3 --version 2>&1)
PY_MAJOR=$(echo "$PY_VER" | sed -n 's/Python \([0-9]*\)\.\([0-9]*\).*/\1/p')
PY_MINOR=$(echo "$PY_VER" | sed -n 's/Python \([0-9]*\)\.\([0-9]*\).*/\2/p')
if [ -z "$PY_MAJOR" ] || [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo -e "${red}Python $PY_VER < 3.10，请升级${reset}"
  exit 1
fi
echo -e "${green}  OK: python3 -> $PY_VER${reset}"

# 2. Node 18+
echo -e "${yellow}[2/8] 检测 Node 18+ ...${reset}"
if ! command -v node >/dev/null 2>&1; then
  echo -e "${red}未检测到 node，请装 18+：https://nodejs.org/${reset}"
  exit 1
fi
NV=$(node --version)
NM=$(echo "$NV" | sed -n 's/v\([0-9]*\).*/\1/p')
if [ -z "$NM" ] || [ "$NM" -lt 18 ]; then
  echo -e "${red}Node $NV < 18，请升级${reset}"
  exit 1
fi
echo -e "${green}  OK: node -> $NV${reset}"

# 3. FFmpeg（缺失只 warning，不退出）
echo -e "${yellow}[3/8] 检测 FFmpeg ...${reset}"
if command -v ffmpeg >/dev/null 2>&1; then
  echo -e "${green}  OK: ffmpeg 已安装${reset}"
else
  echo -e "${warn}  FFmpeg 未安装（不阻断。可 pip 装 imageio-ffmpeg 自带）${reset}"
fi

# 4. 建 venv（已存在跳过）
echo -e "${yellow}[4/8] 建 venv ...${reset}"
if [ -f "venv/bin/python" ]; then
  echo -e "${green}  venv 已存在，跳过${reset}"
else
  python3 -m venv venv
fi

# 5. pip install
echo -e "${yellow}[5/8] pip install -r requirements.txt ...${reset}"
source venv/bin/activate
pip install -r requirements.txt

# 6. npm install (webapp + desktop)
echo -e "${yellow}[6/8] npm install (webapp + desktop) ...${reset}"
( cd webapp && npm install )
( cd desktop && npm install )

# 7. .env.example -> .env（已存在跳过）
echo -e "${yellow}[7/8] 配置 .env ...${reset}"
if [ -f ".env" ]; then
  echo -e "${green}  .env 已存在，跳过${reset}"
else
  cp -n .env.example .env
  echo -e "${yellow}  已创建 .env，请填 API key（VAP_HEADLESS_TOKEN / VAP_LLM_* 等）${reset}"
fi

# 8. smoke test（失败只 warning，不阻断——可能依赖未配 key）
echo -e "${yellow}[8/8] smoke test: tests/test_web_api.py ...${reset}"
set +e
python -m pytest tests/test_web_api.py -q --no-header
SMOKE_OK=$?
set -e
if [ "$SMOKE_OK" -eq 0 ]; then
  echo -e "${green}  smoke test 通过${reset}"
else
  echo -e "${warn}  smoke test 未全绿（可能依赖未配 key），不阻断 onboarding${reset}"
fi

echo ""
echo -e "${green}✅ 开发环境就绪：cd desktop && npm start 启动（macOS/Linux）${reset}"
echo -e "${green}   或单跑后端：python -m src.web.serve${reset}"
