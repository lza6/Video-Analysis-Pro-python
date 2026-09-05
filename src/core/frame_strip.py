"""帧长图证据拼接器（v5.7）。

监控批量分析里"无变化视频"零证据是核心痛点——motion_detector 判定无变化后
直接跳过 AI，用户拿到"0 命中"但看不到任何画面，无法核对算法是否漏判、
无法定位"这一刻到底有没有人经过"。本模块把 motion_detector 已落盘的 1fps
帧（frames/<run_id>/f000000_0.0.jpg ...）拼成一张带时间戳标注的长图，
按 20 张/行横向铺满换行（行1=帧0..19，行2=帧20..39…），供可放大查看器展示。

纯 Pillow 拼接（毫秒级，零 AI 调用），依赖项目已声明的 pillow>=11.3。
不依赖 torch / cv2 / NVIDIA API。

用法：
    FrameStripBuilder.build(
        frame_dir=Path("E2E实测结果/batch_63_v2/frames/<run_id>"),
        out_path=Path(".../frames/<run_id>/strip.png"),
    )
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("VideoAnalyzerCore")

# 长图网格参数
_COLS = 20                 # 每行 20 张（按用户确认的"20张/行 横向铺满换行"）
_THUMB_W = 160             # 缩略图宽（保持原比例）
_LABEL_H = 22              # 每帧下方时间戳标签条高度
_BG = (30, 30, 30)         # 深色底（匹配 PyQt6 深色主题）
_LABEL_BG = (0, 0, 0)      # 时间戳条底色（黑）
_LABEL_FG = (255, 255, 255)  # 时间戳文字色（白）
_GAP = 2                  # 帧间距

# 文件名时间戳解析：f000000_12.3.jpg → 12.3 秒
_TS_RE = re.compile(r"f\d{6}_(\d+(?:\.\d+)?)\.jpg$", re.IGNORECASE)


def _fmt_mmss(sec: float) -> str:
    """秒 → MM:SS（几分几秒）。≥1 小时回退到 HH:MM:SS。"""
    sec = int(round(sec))
    if sec >= 3600:
        return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _scan_frames(frame_dir: Path) -> List[Tuple[float, Path]]:
    """扫帧目录，按时间戳升序返回 [(ts, path)]。

    复用 motion_detector 的命名 f{idx:06d}_{ts:.1f}.jpg；解析不到 ts 的按
    文件名排序兜底（ts=0）。
    """
    if not frame_dir.exists():
        return []
    out: List[Tuple[float, Path]] = []
    for p in sorted(frame_dir.glob("f*.jpg")):
        m = _TS_RE.match(p.name)
        ts = float(m.group(1)) if m else 0.0
        out.append((ts, p))
    out.sort(key=lambda x: x[0])
    return out


def compute_layout(n: int, cols: int, thumb_w: int, thumb_h: int,
                   label_h: int = _LABEL_H, gap: int = _GAP) -> dict:
    """计算长图网格布局（供查看器 hit-test 单帧复用，与 build 保持一致）。

    Returns:
        {rows, cell_w, cell_h, canvas_w, canvas_h, gap}
        cell_w/cell_h/canvas_w/canvas_h 含 gap 间距
    """
    rows = (n + cols - 1) // cols if n > 0 else 0
    cell_w = thumb_w
    cell_h = thumb_h + label_h
    canvas_w = cols * cell_w + (cols + 1) * gap
    canvas_h = rows * cell_h + (rows + 1) * gap
    return {
        "rows": rows, "cell_w": cell_w, "cell_h": cell_h,
        "canvas_w": canvas_w, "canvas_h": canvas_h, "gap": gap,
    }


def cell_rect(idx: int, n: int, cols: int, thumb_w: int, thumb_h: int,
              label_h: int = _LABEL_H, gap: int = _GAP) -> tuple:
    """第 idx 帧在长图里的 (x, y, w, h) 像素矩形（左上角 + 宽高，含帧本身不含标签）。

    供查看器 hit-test 单帧用，与 build 的 paste 位置完全一致。
    """
    layout = compute_layout(n, cols, thumb_w, thumb_h, label_h, gap)
    r, c = divmod(idx, cols)
    x = gap + c * (layout["cell_w"] + gap)
    y = gap + r * (layout["cell_h"] + gap)
    return (x, y, thumb_w, thumb_h)


class FrameStripBuilder:
    """帧长图拼接器。纯静态方法，无状态。"""

    @staticmethod
    def build(frame_dir: Path, out_path: Path,
              cols: int = _COLS) -> Optional[Path]:
        """把 frame_dir 下的帧拼成长图写到 out_path。

        Args:
            frame_dir: motion_detector 落盘的帧目录（f*.jpg）
            out_path:  输出 PNG 路径（通常 frame_dir/strip.png）
            cols:      每行帧数（默认 20）

        Returns:
            out_path（成功）/ None（无帧或 Pillow 不可用）
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.warning("[strip] Pillow 不可用，跳过长图生成")
            return None

        frames = _scan_frames(frame_dir)
        if not frames:
            logger.info(f"[strip] {frame_dir} 无帧，跳过长图")
            return None

        # 先定缩略图尺寸（取首帧比例，统一缩放避免参差）
        first = Image.open(frames[0][1])
        orig_w, orig_h = first.size
        first.close()
        if orig_w <= 0 or orig_h <= 0:
            logger.warning(f"[strip] 帧尺寸异常 {frames[0][1]}")
            return None
        thumb_w = _THUMB_W
        thumb_h = max(1, int(orig_h * thumb_w / orig_w))

        n = len(frames)
        layout = compute_layout(n, cols, thumb_w, thumb_h)
        canvas_w = layout["canvas_w"]
        canvas_h = layout["canvas_h"]
        canvas = Image.new("RGB", (canvas_w, canvas_h), _BG)

        # 字体：优先系统等线/微软雅黑，失败用 PIL 默认位图字体
        font = None
        for cand in ("msyh.ttc", "simhei.ttf", "DejaVuSans.ttf",
                     "Arial.ttf"):
            try:
                font = ImageFont.truetype(cand, 14)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(canvas)

        for i, (ts, fp) in enumerate(frames):
            r, c = divmod(i, cols)
            x, y, _fw, _fh = cell_rect(i, n, cols, thumb_w, thumb_h)
            try:
                img = Image.open(fp).convert("RGB")
                img.thumbnail((thumb_w, thumb_h))
                canvas.paste(img, (x, y))
                img.close()
            except Exception as e:
                logger.debug(f"[strip] 帧读取失败 {fp}: {e}")
                # 占位灰块（防整张长图因单帧损坏失败）
                draw.rectangle([x, y, x + thumb_w, y + thumb_h],
                                fill=(60, 60, 60))
            # 时间戳标签条
            draw.rectangle(
                [x, y + thumb_h, x + thumb_w, y + thumb_h + _LABEL_H],
                fill=_LABEL_BG,
            )
            label = _fmt_mmss(ts)
            # 文字居中（粗略：textbbox 取宽高）
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = 30, 14
            tx = x + (thumb_w - tw) // 2
            ty = y + thumb_h + (_LABEL_H - th) // 2
            draw.text((tx, ty), label, fill=_LABEL_FG, font=font)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(out_path), "PNG")
        canvas.close()
        logger.info(
            f"[strip] 长图已生成 {out_path}（{n} 帧，{cols}×{layout['rows']} 网格）")
        return out_path

    @staticmethod
    def list_frames(frame_dir: Path) -> List[Tuple[float, Path]]:
        """供查看器/AI 查询用：按时间序返回 [(ts, path)]。"""
        return _scan_frames(frame_dir)
