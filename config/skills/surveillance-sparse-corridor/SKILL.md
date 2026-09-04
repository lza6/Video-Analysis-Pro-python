---
name: surveillance-sparse-corridor
description: 稀疏走廊/楼梯口监控分析（长时间无人），1fps抽帧+场景检测+帧差分+昼夜自适应，只送变化时段给AI，省90%调用
---

# 稀疏走廊监控分析 Skill

## 适用场景
- 长时间没人的走廊/楼梯口/电梯厅监控
- 特征：画面大部分时间静止，偶尔有人/物变化

## 算法
1. 1fps 抽帧（ffmpeg）
2. scenedetect AdaptiveDetector 找场景变化点
3. 帧差分（cv2.absdiff）计算变化分
4. 昼夜判断（亮度均值）→ 自适应阈值（白天高/夜间低防噪点）
5. 合并变化点为时段，加 ±10s 上下文
6. 只对变化时段送 NVIDIA Nemotron Omni（video_url）

## 参数
- sample_fps=1.0, min_scene_len=15, context_padding=10
- day_threshold=8.0, night_threshold=3.0

## 何时用
- 用户说"分析监控找X" + 视频是走廊/楼梯口

## 何时不该用
- 人流密集（商场/路口）→ 用 surveillance-crowded-scene skill
