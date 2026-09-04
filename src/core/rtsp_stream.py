"""RTSP 实时流分析 (VibeNVR, v5.0)

支持接入 RTSP 摄像头做实时抽帧 + VLM 事件检测：
  - RtspFrameGrabber: 后台线程持续拉流抽帧（断线自动重连）
  - MotionEventDetector: 简易帧差运动检测（触发 VLM 的预筛，省调用）
  - RtspMonitor: 组合两者，检出运动 → VLM 判断 → 命中回调/剪辑

用法:
    monitor = RtspMonitor("rtsp://user:pass@cam/stream", backend, key_item_image)
    monitor.start(on_hit=lambda hit: ...)
    ...
    monitor.stop()
"""
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("VideoAnalyzerCore")


def _sanitize_rtsp_url(url: str) -> str:
    """脱敏 RTSP URL 中的凭据（rtsp://user:pass@host → rtsp://user:***@host）。

    T2 安全修复：RTSP URL 常内嵌摄像头密码，直接入日志会泄漏凭据。
    """
    import re
    return re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\g<1>***\g<2>", url)


@dataclass
class StreamEvent:
    """一次实时事件（运动/命中）。"""
    timestamp: float
    kind: str          # motion / hit
    frame_path: str = ""
    detail: str = ""
    confidence: float = 0.0


class RtspFrameGrabber(threading.Thread):
    """后台持续拉 RTSP 流抽帧。断线自动重连，帧回调线程安全。"""

    def __init__(self, rtsp_url: str, on_frame: Callable[[float, np.ndarray], None],
                 fps: float = 1.0, reconnect_delay: float = 5.0):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.on_frame = on_frame
        self.interval = 1.0 / fps if fps > 0 else 1.0
        self.reconnect_delay = reconnect_delay
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                logger.warning(f"RTSP 连接失败，{self.reconnect_delay}s 后重试: {_sanitize_rtsp_url(self.rtsp_url)}")
                self._stop.wait(self.reconnect_delay)
                continue
            last = 0.0
            try:
                while not self._stop.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("RTSP 流中断，重连...")
                        break
                    now = time.time()
                    if now - last >= self.interval:
                        last = now
                        try:
                            self.on_frame(now, frame)
                        except Exception as e:
                            logger.error(f"on_frame 回调异常: {e}")
                    time.sleep(0.05)
            finally:
                cap.release()
            if not self._stop.is_set():
                self._stop.wait(self.reconnect_delay)

    def stop(self):
        self._stop.set()


class MotionEventDetector:
    """帧差运动检测：相邻帧差异超阈值视为运动。

    用途：RTSP 恒有画面，VLM 每帧调用成本高 → 只有画面变化才值得问 VLM。
    """

    def __init__(self, threshold: float = 25.0, min_area: float = 500,
                 cooldown: float = 10.0):
        self.threshold = threshold
        self.min_area = min_area
        self.cooldown = cooldown
        self._prev_gray: Optional[np.ndarray] = None
        self._last_trigger: Optional[float] = None

    def detect(self, frame: np.ndarray, now: float) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return False
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        moved = int((diff > self.threshold).sum())
        if moved < self.min_area:
            return False
        if self._last_trigger is not None and now - self._last_trigger < self.cooldown:
            return False
        self._last_trigger = now
        return True


class RtspMonitor:
    """RTSP 实时监控：拉流 → 运动检测 → VLM 判断 → 命中回调 + 可选剪辑。"""

    def __init__(self, rtsp_url: str, backend, key_item_image: str = "",
                 item_description: str = "", work_dir: str = "cache/rtsp",
                 motion_threshold: float = 25.0, vlm_cooldown: float = 30.0):
        self.rtsp_url = rtsp_url
        self.backend = backend
        self.key_item_image = key_item_image
        self.item_description = item_description or "关键物品"
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.vlm_cooldown = vlm_cooldown
        self._last_vlm = 0.0
        self.events: List[StreamEvent] = []
        self._grabber: Optional[RtspFrameGrabber] = None
        self._detector = MotionEventDetector(threshold=motion_threshold)
        self._lock = threading.Lock()

    def _on_frame(self, ts: float, frame: np.ndarray):
        if not self._detector.detect(frame, ts):
            return
        fp = self.work_dir / f"motion_{ts:.0f}.jpg"
        cv2.imwrite(str(fp), frame)
        with self._lock:
            self.events.append(StreamEvent(timestamp=ts, kind="motion",
                                           frame_path=str(fp)))
            # 防内存泄漏: 事件与帧文件上限（保留最近 500 条）
            if len(self.events) > 500:
                old = self.events.pop(0)
                try:
                    Path(old.frame_path).unlink(missing_ok=True)
                except Exception:
                    pass
        logger.info(f"[RTSP] 运动检测 @ {time.strftime('%H:%M:%S', time.localtime(ts))}")

        # VLM 确认（冷却期内跳过）—— 投递到独立工作线程，绝不能阻塞读流线程
        # （VLM 调用可达 30-600s，阻塞会导致 RTSP 缓冲堆积→流中断→重连循环）
        if self.key_item_image and ts - self._last_vlm >= self.vlm_cooldown:
            self._last_vlm = ts
            frame_path = str(fp)
            t = threading.Thread(target=self._vlm_worker, args=(ts, frame_path), daemon=True)
            t.start()

    def _vlm_worker(self, ts: float, frame_path: str):
        """独立线程执行 VLM 判断（结果回写 events）。"""
        hit = self._vlm_check(frame_path)
        if hit:
            with self._lock:
                ev = StreamEvent(timestamp=ts, kind="hit", frame_path=frame_path,
                                 detail=hit.get("reason", ""),
                                 confidence=float(hit.get("confidence", 0)))
                self.events.append(ev)
            logger.info(f"[RTSP] ★ 命中: {hit.get('reason', '')[:60]}")

    def _vlm_check(self, frame_path: str) -> Optional[dict]:
        import re, json
        try:
            prompt = (f"监控画面。是否出现「{self.item_description}」？"
                      '只回JSON: {"match": true/false, "confidence": 0.0-1.0, "reason": ""}')
            raw = "".join(self.backend.chat_stream(
                messages=[{"role": "user", "content": prompt}],
                image_paths=[self.key_item_image, frame_path] if self.key_item_image else [frame_path],
                temperature=0.1))
            text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            m = re.search(r'\{[^{}]*"match"[^{}]*\}', text, re.DOTALL)
            if m and json.loads(m.group(0)).get("match"):
                return json.loads(m.group(0))
        except Exception as e:
            logger.warning(f"[RTSP] VLM 判断失败: {e}")
        return None

    def start(self, fps: float = 1.0):
        self._grabber = RtspFrameGrabber(self.rtsp_url, self._on_frame, fps=fps)
        self._grabber.start()
        logger.info(f"[RTSP] 监控启动: {_sanitize_rtsp_url(self.rtsp_url)}")

    def stop(self):
        if self._grabber:
            self._grabber.stop()
            self._grabber.join(timeout=10)
        logger.info(f"[RTSP] 监控停止, 事件数: {len(self.events)}")
