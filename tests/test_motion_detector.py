"""MotionDetector 变化检测引擎单测（M1）。

AAA 模式：Arrange 合成视频 / Act 调 detect / Assert 验证时段数。

覆盖：
  1. 合成空视频（全程同一帧，0 变化）→ 0 段
  2. 合成有人视频（后半帧突变）→ 1+ 段
  3. MotionConfig 字段默认值校验
  4. _frame_diff_score 单元：相同帧→0，不同帧→>0
  5. _clamp_segment：超 max_segment_sec 再切多段
"""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from src.core.motion_detector import (
    MotionConfig,
    MotionDetector,
    MotionSegment,
    NVIDIA_MAX_SEGMENT_SEC,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def _make_video(path: Path, duration_sec: float = 6.0, fps: float = 10,
                w: int = 64, h: int = 48, pattern: str = "static",
                static_value: int = 40) -> Path:
    """合成 mp4 用于测试。

    pattern:
      - 'static'  : 全程同一灰度值（0 变化）
      - '突变'    : 前半 static_value，后半 static_value+100（明显变化）
      - 'gradient': 每帧递增 1（缓变）
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    n = int(duration_sec * fps)
    half = n // 2
    for i in range(n):
        if pattern == "static":
            v = static_value
        elif pattern == "突变":
            v = static_value if i < half else static_value + 100
        elif pattern == "gradient":
            v = (static_value + i) % 255
        else:
            v = static_value
        frame = np.full((h, w, 3), v, dtype=np.uint8)
        vw.write(frame)
    vw.release()
    return path


def _make_detector(**kw) -> MotionDetector:
    """构造 MotionDetector，默认用宽松阈值让小合成视频也能触发。"""
    cfg = MotionConfig(
        sample_fps=1.0,
        min_scene_len=2,      # 测试视频短，用小 min_scene_len
        day_threshold=5.0,
        night_threshold=2.0,
        context_padding=1.0,
        max_segment_sec=NVIDIA_MAX_SEGMENT_SEC,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return MotionDetector(cfg)


# ----------------------------------------------------------------------
# 1. 空视频：0 变化 → 0 段
# ----------------------------------------------------------------------
class TestEmptyVideo:
    def test_static_video_returns_zero_segments(self, tmp_path):
        # Arrange：6s 全同帧视频
        video = _make_video(tmp_path / "empty.mp4", duration_sec=6.0,
                            pattern="static", static_value=40)
        detector = _make_detector()

        # Act
        segments = detector.detect(video)

        # Assert：全程无变化，0 段（省 AI 调用）
        assert segments == [], f"空视频应返回 0 段，得到 {segments}"


# ----------------------------------------------------------------------
# 2. 有人视频：后半突变 → 1+ 段
# ----------------------------------------------------------------------
class TestMotionVideo:
    def test_half突变视频_returns_at_least_one_segment(self, tmp_path):
        # Arrange：8s 视频，前 4s 灰度 40，后 4s 灰度 140（明显变化）
        video = _make_video(tmp_path / "motion.mp4", duration_sec=8.0,
                            pattern="突变", static_value=40)
        detector = _make_detector()

        # Act
        segments = detector.detect(video)

        # Assert：至少 1 段变化
        assert len(segments) >= 1, f"突变视频应返回 1+ 段，得到 {segments}"
        # 变化时段应覆盖后半段（突变点 ~4s 附近）
        first = segments[0]
        assert first.end_sec > first.start_sec
        assert first.duration > 0
        # 变化点应在 3-5s 之间（前 4s 静止，第 4s 突变）
        assert any(3.0 <= ts <= 5.0 for ts in first.change_points), \
            f"变化点应在 4s 附近，得到 {first.change_points}"
        # brightness 字段合法
        assert first.brightness in ("day", "night", "mixed")
        # diff_score > 0（有变化）
        assert first.diff_score > 0, "变化时段 diff_score 应 > 0"

    def test_突变视频段长受_max_segment_sec_限制(self, tmp_path):
        # Arrange：长视频 + 突变，但 max_segment_sec 设小强迫再切
        video = _make_video(tmp_path / "long.mp4", duration_sec=10.0,
                            pattern="突变", static_value=40)
        detector = _make_detector(max_segment_sec=3)

        # Act
        segments = detector.detect(video)

        # Assert：每段 duration ≤ 3s（clamp 生效）
        assert len(segments) >= 1
        for s in segments:
            assert s.duration <= 3.0 + 0.01, \
                f"段长应 ≤ max_segment_sec=3，得到 {s.duration}"


# ----------------------------------------------------------------------
# 3. MotionConfig 默认值
# ----------------------------------------------------------------------
class TestMotionConfig:
    def test_default_values(self):
        cfg = MotionConfig()
        assert cfg.sample_fps == 1.0
        assert cfg.min_scene_len == 15
        assert cfg.day_threshold == 15.0
        assert cfg.night_threshold == 6.0
        assert cfg.context_padding == 10.0
        assert cfg.max_segment_sec == NVIDIA_MAX_SEGMENT_SEC == 120


# ----------------------------------------------------------------------
# 4. _frame_diff_score 单元
# ----------------------------------------------------------------------
class TestFrameDiffScore:
    def test_same_frame_returns_zero(self, tmp_path):
        # Arrange：两张相同灰度的 jpg
        a = tmp_path / "a.jpg"
        b = tmp_path / "b.jpg"
        img = np.full((48, 64), 40, dtype=np.uint8)
        cv2.imwrite(str(a), img)
        cv2.imwrite(str(b), img)
        detector = _make_detector()

        # Act
        score = detector._frame_diff_score(str(a), str(b))

        # Assert
        assert score == 0.0, f"相同帧 diff 应为 0，得到 {score}"

    def test_different_frame_returns_positive(self, tmp_path):
        # Arrange：两张差异大的 jpg
        a = tmp_path / "a.jpg"
        b = tmp_path / "b.jpg"
        cv2.imwrite(str(a), np.full((48, 64), 10, dtype=np.uint8))
        cv2.imwrite(str(b), np.full((48, 64), 200, dtype=np.uint8))
        detector = _make_detector()

        # Act
        score = detector._frame_diff_score(str(a), str(b))

        # Assert
        assert score > 100, f"差异大的帧 diff 应 >100，得到 {score}"


# ----------------------------------------------------------------------
# 5. _clamp_segment 单元
# ----------------------------------------------------------------------
class TestClampSegment:
    def test_short_segment_passthrough(self):
        detector = _make_detector(max_segment_sec=120)
        seg = MotionSegment(
            start_sec=0.0, end_sec=30.0, duration=30.0,
            brightness="day", diff_score=10.0, scene_count=1,
            change_points=[5.0],
        )
        out = detector._clamp_segment(seg)
        assert len(out) == 1
        assert out[0] is seg

    def test_long_segment_split(self):
        detector = _make_detector(max_segment_sec=30)
        seg = MotionSegment(
            start_sec=0.0, end_sec=90.0, duration=90.0,
            brightness="day", diff_score=10.0, scene_count=1,
            change_points=[5.0],
        )
        out = detector._clamp_segment(seg)
        assert len(out) == 3, f"90s/30s 应切 3 段，得到 {len(out)}"
        for s in out:
            assert s.duration <= 30.0 + 0.01
        # 首段保留 change_points + diff_score
        assert out[0].change_points == [5.0]
        assert out[0].diff_score == 10.0
        # 后续段 change_points 为空
        assert out[1].change_points == []
        assert out[2].change_points == []
        # 段连续
        assert out[0].start_sec == 0.0
        assert abs(out[0].end_sec - out[1].start_sec) < 0.01
        assert abs(out[1].end_sec - out[2].start_sec) < 0.01
        assert abs(out[2].end_sec - 90.0) < 0.01
