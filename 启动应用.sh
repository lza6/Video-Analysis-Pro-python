#!/usr/bin/env bash
# Video Analysis Pro — Linux / macOS 启动脚本
# 用法: bash 启动应用.sh   (或 chmod +x 后 ./启动应用.sh)
cd "$(dirname "$0")"
mkdir -p logs

echo "========================================"
echo "  Video Analysis Pro (视频分析专业版)"
echo "========================================"

if [ -x "venv/bin/python" ]; then
    echo "[INFO] Using venv python."
    exec "venv/bin/python" launcher.py 2>&1 | tee logs/startup.log
    exit $?
fi

# 探测 python3.10-3.12
FOUND=""
for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver=$("$cand" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null)
        case "$ver" in
            310|311|312) FOUND="$cand"; break ;;
        esac
    fi
done

if [ -z "$FOUND" ]; then
    echo "[ERROR] 未找到 Python 3.10-3.12。请安装后重试:"
    echo "    Ubuntu/Debian: sudo apt install python3.10 python3.10-venv"
    echo "    macOS: brew install python@3.10"
    exit 1
fi

echo "[INFO] Using $FOUND"
exec "$FOUND" launcher.py 2>&1 | tee logs/startup.log
