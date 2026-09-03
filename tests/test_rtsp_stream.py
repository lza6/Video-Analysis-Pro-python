"""RTSP 模块单测：运动检测 + 事件结构（不依赖真实摄像头）。"""
import numpy as np
import pytest

from src.core.rtsp_stream import MotionEventDetector, StreamEvent


class TestMotionDetector:
    def test_static_frames_no_motion(self):
        det = MotionEventDetector(threshold=25.0, min_area=100, cooldown=0.0)
        frame = np.full((120, 160, 3), 50, dtype=np.uint8)
        assert det.detect(frame, 0.0) is False  # 首帧只建立基线
        assert det.detect(frame, 1.0) is False  # 相同帧无运动

    def test_moving_object_triggers(self):
        det = MotionEventDetector(threshold=25.0, min_area=100, cooldown=0.0)
        base = np.full((120, 160, 3), 50, dtype=np.uint8)
        det.detect(base, 0.0)
        moving = base.copy()
        moving[40:80, 60:110] = 220  # 大块亮区（运动物体）
        assert det.detect(moving, 1.0) is True

    def test_cooldown_suppresses(self):
        det = MotionEventDetector(threshold=25.0, min_area=100, cooldown=100.0)
        base = np.full((120, 160, 3), 50, dtype=np.uint8)
        det.detect(base, 0.0)
        moving = base.copy()
        moving[40:80, 60:110] = 220
        assert det.detect(moving, 1.0) is True
        moving2 = base.copy()
        moving2[10:50, 20:70] = 220
        assert det.detect(moving2, 2.0) is False  # 冷却期内


class TestStreamEvent:
    def test_event_fields(self):
        ev = StreamEvent(timestamp=123.0, kind="motion", frame_path="a.jpg")
        assert ev.kind == "motion"
        assert ev.confidence == 0.0
