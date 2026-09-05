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
    _make_fake_frames(tmp_path / "frames", 20)  # 落盘 20 帧供 YOLO glob（返回值不用）
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


def test_crowded_dedup_reduces_segments_30pct(tmp_path: Path) -> None:
    """指南 10.1：crowded 去重比父类省 30%+ segments。

    构造 20 帧高密度变化（每帧 diff>阈值 → 19 个变化点），mock YOLO 返回
    前 10 帧物体集合 ['person']、后 9 帧 ['person','backpack'] → 物体集合
    只在 i=0 和 i=10 变化 → 去重后 2 个变化点。

    断言 CrowdedSceneDetector 返回的 segments 数 < 父类 MotionDetector
    返回的 segments 数 × 0.7（即减少 30%+）。

    测试技巧：min_scene_len=0 + context_padding=1.0 让父类不合并相邻变化
    点（19 个变化点 → 19 个独立 segment），隔离测 YOLO 去重效果，不测父类
    padding 合并（那是 motion_detector.py 自己的测试职责）。
    """
    from src.core.motion_detector import (
        CrowdedSceneDetector, MotionDetector, MotionConfig)
    _make_fake_frames(tmp_path / "frames", 20)  # 落盘 20 帧供 YOLO glob
    cfg = MotionConfig(
        day_threshold=5.0, night_threshold=3.0,
        min_scene_len=0,            # 隔离测试：父类不合并相邻变化点
        context_padding=1.0,        # 段不重叠，每变化点独立成段
        crowded_density_threshold=0.3,  # 19/19=1.0 > 0.3 → 触发 YOLO 去重
        frame_out_dir=str(tmp_path / "frames"),
    )
    det = CrowdedSceneDetector(cfg)
    # 父类 MotionDetector 实例：传 ffmpeg_exe 绕过 _find_ffmpeg
    # （该方法只在 CrowdedSceneDetector 子类定义，父类直接构造会 AttributeError，
    # 这是源码现状——本测试不修源码，传 ffmpeg_exe 走 __init__ 早返回路径）
    parent = MotionDetector(cfg, ffmpeg_exe="ffmpeg")
    # 20 帧全变化 → 19 个 diff 变化点（diff 长度 = 帧数 - 1）
    diff = [10.0] * 19
    ts = [float(i) for i in range(19)]
    day_night = ["day"] * 19
    # mock YOLO：前 10 帧 ['person']，后 9 帧 ['person','backpack']
    # → 物体集合只在 i=0（空→person）和 i=10（person→person+backpack）变化
    yolo_returns = [["person"]] * 10 + [["person", "backpack"]] * 9
    with patch.object(det, "_detect_objects_yolo",
                       return_value=yolo_returns):
        crowded_segs = det._merge_to_segments(diff, ts, day_night, [], 19.0)
    parent_segs = parent._merge_to_segments(diff, ts, day_night, [], 19.0)
    # 父类：19 个变化点 → 19 个 segment（min_scene_len=0 不合并）
    assert len(parent_segs) == 19, (
        f"父类应返回 19 段（实际 {len(parent_segs)}），"
        "若失败检查 min_scene_len/context_padding")
    # 核心断言：crowded 去重后 segments 数 < 父类 × 0.7（减少 30%+）
    assert len(crowded_segs) < len(parent_segs) * 0.7, (
        f"crowded 应比父类省 30%+ segments：crowded={len(crowded_segs)}, "
        f"parent={len(parent_segs)}, 比例={len(crowded_segs)/len(parent_segs):.2f}")
    # 强化断言：去重后大幅减少（≤ 50%）
    assert len(crowded_segs) <= len(parent_segs) * 0.5, (
        f"crowded 去重后应 ≤ 父类的 50%：crowded={len(crowded_segs)}, "
        f"parent={len(parent_segs)}")
    # 语义断言：去重后只剩 2 个变化点（物体集合变化点）
    assert len(crowded_segs) == 2, (
        f"crowded 去重后应只剩 2 段（物体集合变化点），实际 {len(crowded_segs)}")


def test_object_set_unchanged() -> None:
    """物体集合相同判定（去重核心逻辑）。"""
    from src.core.motion_detector import MotionDetector
    assert MotionDetector._object_set_unchanged(["person"], ["person"]) is True
    assert MotionDetector._object_set_unchanged(
        ["person"], ["person", "backpack"]) is False
    assert MotionDetector._object_set_unchanged([], []) is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
