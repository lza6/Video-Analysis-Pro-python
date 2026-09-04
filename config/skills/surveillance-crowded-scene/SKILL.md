---
name: surveillance-crowded-scene
description: 人多密集场景监控分析（商场/路口），YOLO目标追踪，后续实现
---

# 人多密集场景监控分析 Skill（占位）

## 适用场景
- 商场、路口、车站等高人流密度监控
- 特征：画面持续有人，需要追踪个体轨迹而非"有无变化"

## 算法（待实现）
1. YOLO 目标检测 + DeepSORT/ByteTrack 多目标追踪
2. 轨迹聚类识别主流向/异常停留
3. 密度热力图统计
4. 不走帧差分（持续变化导致帧差分失效）

## 何时用
- 用户说"商场人流分析""路口密度统计""拥挤检测"

## 何时不该用
- 稀疏走廊/楼梯口（长时间无人）→ 用 surveillance-sparse-corridor skill
