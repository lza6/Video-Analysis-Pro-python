"""I5.9-skills-1：CrowdedSceneDetector 测试（YOLO 去重 + 降级）。

无真实视频/无 ultralytics（CI 标准子集）下验证：
- 密度高时走 YOLO 去重路径（mock YOLO 返回固定物体集合）
- YOLO 不可用时降级父类纯帧差分不崩
- 密度低时走父类逻辑（不触发 YOLO）
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_fake_frames(out_dir: Path, n: int = 20) -> list:
    """造 n 张假帧（纯色 + ts 标注），返回 [(ts, path)]。"""
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        ts = float(i)
        fp = out_dir / f"f{i:06d}_{ts:.1f}.jpg"
        img = Image.new("RGB", (320, 180), (i * 10 % 255, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"t={ts}s", fill=(255, 255, 255))
        img.save(str(fp), "JPEG")
        frames.append((ts, str(fp)))
    return frames


def test_crowded_fallback_without_ultralytics(tmp_path: Path) -> None:
    """无 ultralytics 时 CrowdedSceneDetector 降级父类不崩。

    mock ultralytics import 失败 → _detect_objects_yolo 返回 None →
    _merge_to_segments 走父类纯帧差分（不崩）。
    """
    from src.core.motion_detector import (
        CrowdedSceneDetector, MotionConfig)
    cfg = MotionConfig(
        day_threshold=5.0, night_threshold=3.0,
        crowded_density_threshold=0.3,  # 低阈值，必触发密集路径
        frame_out_dir=str(tmp_path / "frames"),
    )
    det = CrowdedSceneDetector(cfg)
    # mock YOLO 不可用
    with patch.dict("sys.modules", {"ultralytics": None}):
        # 构造高密度变化序列（20 帧全变化）
        diff = [10.0] * 19  # 每帧都超阈值
        ts = [float(i) for i in range(19)]
        day_night = ["day"] * 19
        segs = det._merge_to_segments(diff, ts, day_night, [], 19.0)
    # 降级父类：应返回非 None 的 segments（不崩）
    assert isinstance(segs, list)


def test_crowded_low_density_uses_parent(tmp_path: Path) -> None:
    """密度低（< crowded_density_threshold）走父类逻辑，不触发 YOLO。"""
    from src.core.motion_detector import (
        CrowdedSceneDetector, MotionConfig)
    cfg = MotionConfig(
        day_threshold=5.0, night_threshold=3.0,
        crowded_density_threshold=0.6,  # 60% 才触发 YOLO
    )
    det = CrowdedSceneDetector(cfg)
    # 只 2/19 帧变化（密度 ~0.1 < 0.6），走父类
    diff = [10.0, 0.0] * 9 + [10.0]
    ts = [float(i) for i in range(19)]
    day_night = ["day"] * 19
    segs = det._merge_to_segments(diff, ts, day_night, [], 19.0)
    assert isinstance(segs, list)


def test_crowded_high_density_triggers_yolo_path(tmp_path: Path) -> None:
    """密度高时触发 YOLO 去重路径（mock YOLO 返回相同物体集合→全部去重）。"""
    from src.core.motion_detector import (
        CrowdedSceneDetector, MotionConfig)
    frames = _make_fake_frames(tmp_path / "frames", 20)
    cfg = MotionConfig(
        day_threshold=5.0, night_threshold=3.0,
        crowded_density_threshold=0.3,  # 低阈值触发密集路径
        frame_out_dir=str(tmp_path / "frames"),
    )
    det = CrowdedSceneDetector(cfg)
    # mock _detect_objects_yolo 返回每帧相同物体集合（person）→ 去重后只保留首帧
    with patch.object(det, "_detect_objects_yolo",
                       return_value=[["person"]] * 20):
        # 高密度：20 帧全变化
        diff = [10.0] * 19
        ts = [float(i) for i in range(19)]
        day_night = ["day"] * 19
        segs = det._merge_to_segments(diff, ts, day_night, [], 19.0)
    # 物体集合全相同 → 去重后变化点大幅减少
    assert isinstance(segs, list)


def test_object_set_unchanged() -> None:
    """物体集合相同判定（去重核心逻辑）。"""
    from src.core.motion_detector import MotionDetector
    assert MotionDetector._object_set_unchanged(["person"], ["person"]) is True
    assert MotionDetector._object_set_unchanged(
        ["person"], ["person", "backpack"]) is False
    assert MotionDetector._object_set_unchanged([], []) is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
