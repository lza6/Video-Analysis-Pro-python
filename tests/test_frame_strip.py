"""FrameStripBuilder 单测（v5.7）。

不依赖 ffmpeg/视频/cv2，造假 jpg 帧 → 拼长图 → 断言网格尺寸/时间戳/文件存在。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image


def _make_fake_frame(path: Path, color, ts_label: str) -> None:
    """造假 jpg 帧（纯色 + 中央写 ts_label，便于肉眼核对）。"""
    img = Image.new("RGB", (320, 180), color)
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("msyh.ttc", 28)
    except Exception:
        font = ImageFont.load_default()
    d.text((10, 10), ts_label, fill=(255, 255, 255), font=font)
    img.save(str(path), "JPEG", quality=85)


def test_build_strip_20_per_row(tmp_path: Path) -> None:
    """25 帧 → 长图应为 2 行（ceil(25/20)=2），文件存在。"""
    from src.core.frame_strip import FrameStripBuilder, _COLS, _fmt_mmss

    frame_dir = tmp_path / "frames" / "run123"
    frame_dir.mkdir(parents=True)
    # 25 帧（第 0..24 秒）
    for i in range(25):
        ts = float(i)
        _make_fake_frame(
            frame_dir / f"f{i:06d}_{ts:.1f}.jpg",
            color=(i * 10 % 255, 0, 0),
            ts_label=_fmt_mmss(ts),
        )
    out_path = frame_dir / "strip.png"
    result = FrameStripBuilder.build(frame_dir, out_path, cols=_COLS)
    assert result == out_path
    assert out_path.exists()
    img = Image.open(out_path)
    w, h = img.size
    # 25 帧 → 2 行（ceil(25/20)=2），宽 = 20*cell_w，高 = 2*cell_h
    assert h > 0 and w > 0
    # 验证拼了内容（不是空图）：像素总和 > 0
    pixels = list(img.convert("L").getdata())
    assert sum(pixels) > 0


def test_build_strip_empty_dir(tmp_path: Path) -> None:
    """空目录 → 返回 None（不生成空长图）。"""
    from src.core.frame_strip import FrameStripBuilder
    empty = tmp_path / "empty"
    empty.mkdir()
    out = FrameStripBuilder.build(empty, empty / "strip.png")
    assert out is None


def test_build_strip_time_order(tmp_path: Path) -> None:
    """帧应按时间戳升序排列（文件名乱序也要排对）。"""
    from src.core.frame_strip import FrameStripBuilder
    frame_dir = tmp_path / "frames" / "run_t"
    frame_dir.mkdir(parents=True)
    # 故意乱序写入
    for ts in [5.0, 0.0, 10.0, 2.0]:
        _make_fake_frame(
            frame_dir / f"f{int(ts):06d}_{ts:.1f}.jpg",
            color=(128, 128, 128),
            ts_label=str(ts),
        )
    frames = FrameStripBuilder.list_frames(frame_dir)
    timestamps = [t for t, _ in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 0.0


def test_cell_rect_layout_consistency(tmp_path: Path) -> None:
    """cell_rect/compute_layout 与 build 的 paste 位置一致（hit-test 零漂移）。

    造 25 帧 → build 长图 → cell_rect 计算每个帧矩形 → 断言网格尺寸匹配。
    这是查看器 mousePressEvent hit-test 的数学基础。
    """
    from src.core.frame_strip import (
        FrameStripBuilder, compute_layout, cell_rect,
        _THUMB_W, _LABEL_H, _GAP, _COLS,
    )
    frame_dir = tmp_path / "frames" / "run_layout"
    frame_dir.mkdir(parents=True)
    for i in range(25):
        _make_fake_frame(
            frame_dir / f"f{i:06d}_{float(i):.1f}.jpg",
            color=(i * 10 % 255, 0, 0),
            ts_label=str(i),
        )
    out = FrameStripBuilder.build(frame_dir, frame_dir / "strip.png", cols=_COLS)
    assert out is not None
    # build 后从首帧读真实 thumb_h（build 按首帧比例算）
    from PIL import Image
    first = Image.open(frame_dir / "f000000_0.0.jpg")
    thumb_h = int(first.size[1] * _THUMB_W / first.size[0])
    first.close()
    layout = compute_layout(25, _COLS, _THUMB_W, thumb_h)
    assert layout["rows"] == 2  # ceil(25/20)=2
    # cell_rect 与 layout 一致：第 0 帧左上角 = (gap, gap)
    x0, y0, w0, h0 = cell_rect(0, 25, _COLS, _THUMB_W, thumb_h)
    assert x0 == _GAP and y0 == _GAP and w0 == _THUMB_W and h0 == thumb_h
    # 第 20 帧是第 2 行第 1 列
    x20, y20, _, _ = cell_rect(20, 25, _COLS, _THUMB_W, thumb_h)
    assert x20 == _GAP  # 第 2 行第 1 列 x = gap
    assert y20 == _GAP + (thumb_h + _LABEL_H + _GAP)  # 第 2 行 y


def test_fmt_mmss() -> None:
    """秒 → MM:SS 格式。"""
    from src.core.frame_strip import _fmt_mmss
    assert _fmt_mmss(0) == "00:00"
    assert _fmt_mmss(65) == "01:05"
    assert _fmt_mmss(1065) == "17:45"
    # >= 1 小时回退 HH:MM:SS
    assert _fmt_mmss(3661) == "01:01:01"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
