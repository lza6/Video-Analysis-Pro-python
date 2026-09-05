# 帧长图证据设计（v5.7.0）

> 日期：2026-09-05 ｜ 版本：v5.7.0 ｜ 状态：已实现并 E2E 验证

## 问题根因

监控批量分析里 motion_detector 判定"无变化"的视频直接跳过 AI（`batch_runner.py:274-282`），用户拿到 0 命中但**看不到任何画面**——无法核对算法是否漏判、无法定位"这一刻到底有没有人经过"。这是伪证据盲区：motion_detector 的 1fps 抽帧（`_sample_frames`）落到 `tempfile.mkdtemp` 临时目录，`detect()` 的 `finally` 立即 `shutil.rmtree` 删掉（`motion_detector.py:180-181`），帧从未持久化。

## 设计

### 数据流（改后）

```
batch_runner._run_single_video
  ├─ _segment_video(video, run_id, duration)
  │   ├─ MotionDetector(frame_out_dir=frames/<run_id>/).detect(video)
  │   │    ├─ _sample_frames → 帧落盘 frames/<run_id>/f000000_0.0.jpg ...（保留）
  │   │    └─ 返回 MotionSegment[]（可能为空）
  │   └─ 切变化时段分片到 segments/<run_id>/
  ├─ _build_filmstrip(run_id, video)  ← v5.7 新增
  │   └─ FrameStripBuilder.build(frames/<run_id>/, strip.png) → 写 run_store.strip_path
  ├─ [无变化] 标 done，跳过 AI ✓（保留帧+长图）
  ├─ [有变化] 逐片判 AI → 命中裁剪到 clips/<run_id>/
  └─ clean_segments 只清 segments/，不清 frames/ ✓
```

### 组件

| 组件 | 文件 | 职责 |
|---|---|---|
| MotionConfig.frame_out_dir | `motion_detector.py` | 帧持久化目录；非空时 `_sample_frames` 落盘不删，空时走旧临时目录（零回归） |
| FrameStripBuilder | `frame_strip.py`（新） | 纯 Pillow 拼接：扫帧→缩略 160px→底部画 MM:SS→20 列网格 PNG |
| BatchRunner._build_filmstrip | `batch_runner.py` | 全视频调 FrameStripBuilder，写 run_store.strip_path |
| RunStore.runs.strip_path | `run_store.py` | 新列，`_ensure_column` 旧库自动补 |
| FrameStripDialog | `frame_strip_dialog.py`（新） | QGraphicsView+Pixmap，滚轮缩放+拖动+AI查询+跳转播放器 |
| _RunDetailDialog 帧长图按钮 | `batch_tab.py` | 读 run.strip_path 弹 FrameStripDialog |
| _on_strip_seek_request | `main_window.py` | 长图跳转→打开/复用 VideoPlayerDialog 定位 ts |
| append_thoughts | `agent_dialog.py` | 补齐累积式思考链（v5.6 误调用但只存在 set_thoughts，启动崩溃修复） |

### 网格布局（用户确认）

20 张/行横向铺满换行：行1=帧0..19，行2=帧20..39…，按时间升序左到右。

### AI 查询时间点（解决伪证据）

长图查看器"💬 询问 AI 这一帧"按钮：取当前帧时间戳 → 复制查询消息到剪贴板 + 跳转播放器定位到该秒。用户粘贴到 Agent 对话框即可让 AI 描述这一刻画面。

## 验证

- **pyflakes**：7 个改/新文件零告警
- **单测** `tests/test_frame_strip.py`：4 passed（网格尺寸/空目录/时间排序/MM:SS 格式）
- **回归** `tests/test_agent_tools.py + test_core_pipeline.py`：17 passed
- **真实 E2E**（`_376.mp4` 无变化视频 17.7min）：detect 190s → 1065 帧落盘 → strip.png 3242×6158（54 行×20 列）生成成功

## 不做（YAGNI）

- 不给有变化视频单独做"变化时段长图"（全覆盖即可）
- 不改 motion_detector 算法本身（只改帧落盘接缝）
- 不加视频内帧时间戳 OCR（Pillow 画文字即可）
