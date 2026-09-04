"""Video Analysis Pro — Headless 分析服务 (Docker / 无 GUI 环境)

复用 src/core 的全部分析能力，通过 HTTP 暴露:
  GET  /healthz          → 能力矩阵 + 磁盘余量
  POST /analyze          → multipart 上传视频，返回帧 + 报告 JSON

启动: python -m src.server.headless [--port 8000]
"""
import argparse
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import requests

from src.core.logic import (VideoProcessor, AudioProcessor, VideoAnalyzer,
                            OllamaClient, PromptLoader,
                            CLIP_AVAILABLE, NVIDIA_GPU_AVAILABLE,
                            ADVANCED_FEATURES_AVAILABLE, FFMPEG_AVAILABLE)

logger = logging.getLogger("headless")

TMP_ROOT = Path(tempfile.gettempdir()) / "vap_headless"


def _capability_matrix() -> dict:
    return {
        "clip_semantic": CLIP_AVAILABLE,
        "nvidia_gpu": NVIDIA_GPU_AVAILABLE,
        "advanced_media": ADVANCED_FEATURES_AVAILABLE,
        "ffmpeg": FFMPEG_AVAILABLE,
        "ocr": _module_available("paddleocr"),
        # Ollama 探测留给 /analyze 真正调用时；/healthz 必须永远 <50ms 返回。
        "llm_backend": "unknown",
    }


def _module_available(name: str) -> bool:
    import importlib
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _ollama_alive() -> bool:
    # trust_env=False 绕开系统代理：本机 localhost 探测不应走外部代理，
    # 否则代理会拦截并长时间挂起，导致 /healthz 在客户端超时后才返回。
    s = requests.Session()
    s.trust_env = False
    try:
        return s.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def run_analysis(video_bytes: bytes, filename: str, model: str = "qwen2.5:3b") -> dict:
    """对上传的视频执行完整三阶段分析（LLM 走 Ollama；Ollama 不可用时仅返回结构化数据）。"""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    workdir = TMP_ROOT / job_id
    video_path = workdir / filename
    workdir.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(video_bytes)

    try:
        # Phase 1
        processor = VideoProcessor(video_path, workdir / "frames")
        frames = processor.extract_keyframes(density=0.3)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total / fps if fps > 0 else 0.0
        cap.release()

        transcript = ""
        audio_proc = AudioProcessor()
        audio_path = audio_proc.extract_audio(video_path, workdir)
        if audio_path:
            tr = audio_proc.transcribe(audio_path)
            transcript = tr.text if tr else ""

        # Phase 2 (LLM 可选: Ollama 不可达时优雅降级)
        report = None
        try:
            client = OllamaClient()
            analyzer = VideoAnalyzer(client, model, PromptLoader(),
                                     use_yolo=False, use_ocr=False)
            chunks = []
            for chunk in analyzer.analyze_video(frames, transcript, None):
                if chunk.startswith("__"):
                    continue
                chunks.append(chunk)
            report = "".join(chunks)
        except Exception as e:
            report = f"[LLM unavailable: {e}]"

        return {
            "job_id": job_id,
            "duration": duration,
            "frame_count": len(frames),
            "frames": [{"timestamp": round(f.timestamp, 2),
                        "metrics": {k: round(v, 2) for k, v in f.metrics.items()}}
                       for f in frames],
            "transcript": transcript[:5000],
            "report": report,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info(f"{self.address_string()} {fmt % args}")

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            import shutil as _sh
            usage = _sh.disk_usage("/")
            self._json(200, {
                "status": "ok",
                "capabilities": _capability_matrix(),
                "disk_free_gb": round(usage.free / 1024**3, 1),
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/analyze":
            return self._json(404, {"error": "not found"})

        # T5 DoS 加固: 先校验 Content-Length 再读 body。
        # 历史风险: 恶意客户端声明 10GB 并读入内存 → OOM。
        max_mb = int(os.environ.get("VAP_MAX_UPLOAD_MB", "512"))
        max_bytes = max_mb * 1024 * 1024
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            return self._json(400, {"error": "invalid Content-Length"})
        if length <= 0:
            return self._json(400, {"error": "empty upload"})
        if length > max_bytes:
            # 不读 body，直接拒绝并要求客户端中止
            self.close_connection = True
            return self._json(413, {"error": f"upload exceeds {max_mb} MB"})

        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            # 简化: 也接受原始字节 + X-Filename 头
            data = self.rfile.read(length)
            filename = self.headers.get("X-Filename", "upload.mp4")
        else:
            boundary = ctype.split("boundary=")[-1].encode()
            body = self.rfile.read(length)
            data, filename = self._parse_multipart(body, boundary)

        if not data:
            return self._json(400, {"error": "empty upload"})

        try:
            result = run_analysis(data, filename)
            self._json(200, result)
        except Exception as e:
            logger.exception("analysis failed")
            self._json(500, {"error": str(e)})

    def _parse_multipart(self, body: bytes, boundary: bytes):
        """极简 multipart 解析（单文件字段）。"""
        parts = body.split(b"--" + boundary)
        for part in parts:
            if b"filename=" in part:
                header, _, payload = part.partition(b"\r\n\r\n")
                payload = payload.rsplit(b"\r\n", 1)[0]
                fn_line = [l for l in header.split(b"\r\n") if b"filename=" in l]
                filename = "upload.mp4"
                if fn_line:
                    fn = fn_line[0].split(b"filename=")[-1].strip(b'" ')
                    filename = fn.decode("utf-8", errors="replace") or filename
                return payload, filename
        return b"", "upload.mp4"


def main():
    parser = argparse.ArgumentParser(description="Video Analysis Pro headless")
    parser.add_argument("--port", type=int, default=int(os.environ.get("VAP_PORT", "8000")))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("VideoAnalyzerCore").setLevel(logging.INFO)

    logger.info(f"能力矩阵: {json.dumps(_capability_matrix())}")
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    logger.info(f"Headless 服务已启动: http://0.0.0.0:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
