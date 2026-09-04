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


class TestParseMultipart:
    """multipart 解析器单测（Critical #2/#3 回归）。"""

    def _parse(self, body, boundary):
        from src.server.headless import Handler
        h = Handler.__new__(Handler)
        return Handler._parse_multipart.__get__(h)(body, boundary)

    def test_normal_with_closing(self):
        body = (b"--X\r\n"
                b'Content-Disposition: form-data; filename="v.mp4"\r\n\r\n'
                b"\x00\x01\x02VIDEO\r\n"
                b"--X--\r\n")
        data, fn = self._parse(body, b"X")
        assert data == b"\x00\x01\x02VIDEO"
        assert fn == "v.mp4"

    def test_binary_with_bare_boundary_seq_not_split(self):
        """payload 含裸 --X（无 \r\n 前导）不得截断（旧实现 bug）。"""
        body = (b"--X\r\n"
                b'Content-Disposition: form-data; filename="v.mp4"\r\n\r\n'
                b"\x00\x01--XTAIL_BINARY\x00\xff\r\n"
                b"--X--\r\n")
        data, _ = self._parse(body, b"X")
        assert data == b"\x00\x01--XTAIL_BINARY\x00\xff"

    def test_no_closing_tolerated(self):
        body = (b"--X\r\n"
                b'Content-Disposition: form-data; filename="v.mp4"\r\n\r\n'
                b"\x00\x01\x02DATA\r\n"
                b"--X\r\n")
        data, _ = self._parse(body, b"X")
        assert data == b"\x00\x01\x02DATA"

    def test_quoted_boundary_in_ctype(self):
        """quoted boundary（RFC 合法）不得解析失败——在 do_POST 层 strip 引号，
        这里验证 _parse 接受去引号后的值。"""
        body = (b"--Y\r\n"
                b'Content-Disposition: form-data; filename="v.mp4"\r\n\r\n'
                b"DATA\r\n"
                b"--Y--\r\n")
        data, _ = self._parse(body, b"Y")
        assert data == b"DATA"


class TestHeadlessAuth:
    """可选 Bearer Token 鉴权（VAP_HEADLESS_TOKEN）。

    用 monkeypatch 隔离环境变量；不依赖子进程启动。
    """

    def _make_handler(self):
        from src.server.headless import Handler
        h = Handler.__new__(Handler)
        # 模拟 BaseHTTPRequestHandler 的必要属性
        h.close_connection = False
        h.headers = {}
        h.request_version = "HTTP/1.1"
        h.protocol_version = "HTTP/1.1"
        h.responses = {}
        h.requestline = "POST /analyze HTTP/1.1"
        h.client_address = ("127.0.0.1", 12345)
        h.wfile = _FakeWfile()
        return h

    def test_analyze_no_token_when_required_returns_401(self, monkeypatch):
        """设 VAP_HEADLESS_TOKEN=x，无 Authorization 头 → 401。"""
        monkeypatch.setenv("VAP_HEADLESS_TOKEN", "secret123")
        h = self._make_handler()
        h.headers = {}  # 无 Authorization
        ok = h._check_auth()
        assert ok is False
        out = h.wfile.getvalue()
        # 状态行 + 头 + 体全写入 wfile
        assert b"401" in out
        assert b'"unauthorized"' in out
        assert b"Connection: close" in out
        assert h.close_connection is True

    def test_analyze_wrong_token_returns_401(self, monkeypatch):
        """Bearer wrong → 401。"""
        monkeypatch.setenv("VAP_HEADLESS_TOKEN", "secret123")
        h = self._make_handler()
        h.headers = {"Authorization": "Bearer wrongpass"}
        ok = h._check_auth()
        assert ok is False
        out = h.wfile.getvalue()
        assert b"401" in out
        assert b'"unauthorized"' in out

    def test_analyze_correct_token_proceeds(self, monkeypatch):
        """Bearer secret123 → 不 401，返回 True（继续到 multipart 解析）。"""
        monkeypatch.setenv("VAP_HEADLESS_TOKEN", "secret123")
        h = self._make_handler()
        h.headers = {"Authorization": "Bearer secret123"}
        ok = h._check_auth()
        assert ok is True
        # 不应写任何响应体
        assert h.wfile.getvalue() == b""

    def test_analyze_auth_disabled_when_token_empty(self, monkeypatch):
        """VAP_HEADLESS_TOKEN 为空 → 鉴权禁用，直接放行。"""
        monkeypatch.delenv("VAP_HEADLESS_TOKEN", raising=False)
        h = self._make_handler()
        h.headers = {}  # 无 Authorization
        ok = h._check_auth()
        assert ok is True

    def test_healthz_no_auth_required(self, monkeypatch):
        """设 token 后 /healthz 仍 200（健康探针永不鉴权）。

        验证方式：do_GET 走 /healthz 分支不调用 _check_auth（通过路径覆盖）。
        """
        monkeypatch.setenv("VAP_HEADLESS_TOKEN", "secret123")
        h = self._make_handler()
        h.path = "/healthz"
        h.headers = {}  # 无 Authorization
        # do_GET 不应因缺鉴权失败；mock disk_usage 避免真实调用
        import src.server.headless as mod
        monkeypatch.setattr(mod.shutil, "disk_usage",
                            lambda *_: type("U", (), {"free": 10 * 1024**3})())
        h.do_GET()
        out = h.wfile.getvalue()
        assert b"200" in out
        assert b'"status"' in out and b'"ok"' in out


class _FakeWfile:
    """模拟 wfile，记录写入内容。"""
    def __init__(self):
        self._buf = bytearray()
    def write(self, data):
        self._buf.extend(data)
    def getvalue(self):
        return bytes(self._buf)
