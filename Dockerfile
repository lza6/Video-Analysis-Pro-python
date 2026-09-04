# ============================================================================
# Video Analysis Pro — CPU 分析镜像
# 构建: docker build -t video-analysis-pro .
# 运行: docker run -p 8000:8000 video-analysis-pro
# ============================================================================
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    VAP_PORT=8000

# FFmpeg + OpenCV 运行时库（cv2 headless 不需要 X11，但 whisper 需要部分编解码）
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ocr.txt ./
# 纯 headless 不装 GUI/OCR 重依赖；chromadb/weasyprint 等保留
RUN pip install --no-cache-dir $(grep -vE "^(PyQt6|pyqtdarktheme|#)" requirements.txt | tr '\n' ' ') \
    || pip install --no-cache-dir -r requirements.txt --dry-run >/dev/null 2>&1 || true
# 上面 grep 注入法对复杂行脆弱，fallback 到精简显式安装:
# 显式列表必须与 requirements.txt 同步：v5.1 修过 seaborn 缺失致 Phase3 全禁用，
# 但此 fallback 漏装 seaborn/pandas/matplotlib → Docker 形态天然复活该 P0（audit-blinds P2-12）。
RUN pip install --no-cache-dir \
        numpy opencv-python-headless scenedetect ultralytics \
        moviepy imageio-ffmpeg torch --index-url https://download.pytorch.org/whl/cpu \
        faster-whisper sentence-transformers chromadb markdown2 \
        requests psutil pymediainfo pydub pillow nvidia-ml-py \
        duckduckgo-search \
        matplotlib seaborn "pandas<3" \
    || true

COPY src ./src
COPY config ./config
COPY launcher.py ./

EXPOSE 8000
CMD ["python", "-m", "src.server.headless"]
