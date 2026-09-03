"""Headless 服务真实 E2E：启动服务器子进程 → /healthz → /analyze 上传合成视频。

网络层注意：本机环境可能配置系统代理（HTTP_PROXY/HTTPS_PROXY）且 Windows
系统代理会间歇性劫持 127.0.0.1 请求。所有客户端连接统一使用
requests.Session(trust_env=False) 完全绕开代理层。
"""
import cv2
import json
import time

import numpy as np
import pytest
import requests

from pathlib import Path

ROOT = Path(__file__).parent.parent


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False  # 关键: 绕开系统/环境代理，直连 127.0.0.1
    return s


def _wait_health(port, timeout=180):
    deadline = time.time() + timeout
    s = _session()
    while time.time() < deadline:
        if proc_alive():
            try:
                r = s.get(f"http://127.0.0.1:{port}/healthz", timeout=3)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        else:
            return None  # 子进程已退出（端口被占等）
        time.sleep(2)
    return None


def proc_alive():
    global PROC
    return PROC is not None and PROC.poll() is None


PROC = None


def test_headless_health_and_analyze(tmp_path):
    global PROC
    port = 8397  # 避开常用端口段，降低与其它服务冲突概率
    PROC = subprocess_popen(port)
    try:
        health = _wait_health(port)
        assert health is not None, "headless 服务未能在 180s 内就绪（或子进程提前退出）"
        assert health["status"] == "ok"
        assert "capabilities" in health

        # 上传合成视频
        video = tmp_path / "upload_test.mp4"
        w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
        for i in range(20):
            color = (255, 0, 0) if i < 10 else (0, 255, 0)
            w.write(np.full((64, 64, 3), color, dtype=np.uint8))
        w.release()

        s = _session()
        r = s.post(
            f"http://127.0.0.1:{port}/analyze",
            data=video.read_bytes(),
            headers={"X-Filename": "upload_test.mp4",
                     "Content-Type": "application/octet-stream"},
            timeout=300,
        )
        assert r.status_code == 200, r.text[:300]
        result = r.json()

        assert result["job_id"]
        assert result["frame_count"] >= 5
        assert abs(result["duration"] - 2.0) < 0.5
        assert all("timestamp" in f for f in result["frames"])
        # Ollama 不在本机运行 → report 为 None 是预期（能力自动降级）
        assert result["report"] is None or isinstance(result["report"], str)
    finally:
        if PROC is not None:
            PROC.kill()
            PROC.wait(timeout=15)


def subprocess_popen(port):
    import subprocess
    import sys
    return subprocess.Popen(
        [sys.executable, "-m", "src.server.headless", "--port", str(port)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
