---
name: surveillance-night-adaptive
description: 夜间/红外/低光照监控分析，降低帧差分阈值防噪点放大误判，专攻夜间监控找物场景
---

# 夜间自适应监控分析 Skill

## 适用场景
- 夜间、红外补光、低光照环境监控
- 特征：画面整体偏暗，红外补光导致噪点放大，纯帧差分容易把噪点误判为变化
- 走廊夜间找物、仓库过夜监控、红外摄像头

## 算法
1. 1fps 抽帧（ffmpeg fast seek，与 sparse-corridor 同款）
2. scenedetect AdaptiveDetector 找场景变化点（补强缓变）
3. 帧差分（cv2.absdiff）计算变化分
4. **昼夜判断**（亮度均值 < 50 判夜间）→ 夜间用**更低阈值**（night_threshold=3）
   - 白天帧差分噪点 <2，有人经过 >10，阈值 15 合适
   - 夜间红外补光画面偏灰偏暗，噪点放大到 4-8，若仍用白天阈值 15 会漏判；
     夜间阈值降到 3 更敏感，能捕捉红外画面里的人/物移动
5. 合并变化点为时段，加 ±10s 上下文
6. 只对变化时段送 NVIDIA Nemotron Omni（video_url）

## 参数
- `night_threshold: 3.0`（比 sparse-corridor 的 6.0 更低，更敏感）
- `day_threshold: 15.0`（与 sparse-corridor 一致，白天不变）
- `brightness_threshold: 50`（灰度均值 <50 判夜间）
- `sample_fps: 1.0`（1fps 抽帧，监控场景足够）
- `context_padding: 10.0`（变化时段前后加 10s 上下文）

## 何时用
- 用户说"夜间监控找包""红外摄像头找人""过夜监控""低光照"等夜间场景
- 监控视频画面整体偏暗（夜间/红外模式）

## 何时不该用
- 白天正常光照监控 → 用 surveillance-sparse-corridor（阈值更高防噪点）
- 人多密集场景（商场/路口）→ 用 surveillance-crowded-scene（YOLO 去重）
- 夜间但人流密集 → 仍用 crowded-scene（密集优先，夜晚只是阈值调整）

## 降级行为
- 无 ultralytics/scenedetect 时降级为纯帧差分（仍可工作，只是场景切分粗一些）
- 无 ffmpeg 时 detect() 返回 []，调用方跳过

## 与其他 skill 的关系
- 本 skill 是 sparse-corridor 的"夜间增强版"，只调低 night_threshold
- 实际可通过 BatchConfig.frame_change_pct 档位映射间接覆盖（5% 档 → day=5/night=3，
  与本 skill 默认值一致），但独立 skill 让 agent 语义化匹配"夜间"场景更直观
