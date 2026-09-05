---
name: surveillance-parking-lpr
description: 停车场车牌识别监控，车牌检测+OCR，追踪车辆进出
triggers: 停车场,车牌,lpr,停车,车辆
---

# surveillance-parking-lpr

## 适用场景
用户说停车场车牌识别监控等场景

## 算法
YOLO 车牌检测 + PaddleOCR 识别 + 时序进出统计

## 参数
（沿用 surveillance-sparse-corridor 默认：sample_fps=1.0, day_threshold=15, night_threshold=6）

## 何时用
用户说停车场车牌识别监控等场景

## 何时不该用
稀疏走廊/楼梯口（长时间无人）→ 用 surveillance-sparse-corridor

## 降级行为
无对应依赖时降级为纯帧差分，不崩
