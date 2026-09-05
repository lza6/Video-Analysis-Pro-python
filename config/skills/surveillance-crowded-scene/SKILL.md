---
name: surveillance-crowded-scene
description: 人多密集场景监控分析（商场/路口），YOLO目标追踪+物体类别聚类去重，过滤人来回走动的重复变化，只送新物体出现的时段给AI
---

# 人多密集场景监控分析 Skill

## 适用场景
- 商场、路口、车站等高人流密度监控
- 特征：画面持续有人，纯帧差分会把"人来回走动"误判为无数变化点，送 AI 分片爆炸

## 算法
1. 1fps 抽帧（ffmpeg fast seek，与 sparse-corridor 同款）
2. 帧差分（cv2.absdiff）+ scenedetect 找变化点
3. **密度判定**：变化帧数 / 总帧数 > `crowded_density_threshold`（默认 0.6）才走密集路径
4. **YOLO 物体类别去重**（核心）：
   - 用 `ultralytics.YOLO`（yolov8n.pt，CPU 可跑）检测每帧物体类别
   - 按"物体类别集合"聚类：同类别集合不变的变化点视为重复（人来回走动），过滤掉
   - 只保留"新物体类别出现"的变化点（如新人进场、物品出现）
5. 去重后的变化点合并为时段，加 ±10s 上下文
6. 只对这些时段送 NVIDIA Nemotron Omni（video_url）

## 参数
- `crowded_density_threshold: 0.6`（变化点密度阈值，超过才启用 YOLO 去重）
- `day_threshold: 15.0` / `night_threshold: 6.0`（与 sparse-corridor 一致）
- `sample_fps: 1.0`（1fps 抽帧）
- `context_padding: 10.0`（变化时段前后加 10s 上下文）
- YOLO 模型：`yolov8n.pt`（最小，CPU 也能跑，~6MB）

## 何时用
- 用户说"商场人流分析""路口密度统计""拥挤检测""车站人流"
- 监控画面持续有人（非稀疏走廊）

## 何时不该用
- 稀疏走廊/楼梯口（长时间无人）→ 用 surveillance-sparse-corridor（帧差分够用）
- 夜间/红外/低光照 → 用 surveillance-night-adaptive（降阈值更敏感）
- 夜间但人流密集 → 仍用本 skill（密集优先级 > 夜间）

## 降级行为（关键）
- **无 ultralytics**（CI 标准子集不装）：CrowdedSceneDetector._detect_objects_yolo
  返回 None，_merge_to_segments 降级调父类纯帧差分，不崩。
- YOLO 推理失败（模型缺失/CPU 不足）：同上降级。
- 密度低于阈值：直接走父类（不触发 YOLO，省推理）。

## 与其他 skill 的关系
- 本 skill 是 MotionDetector 的密集场景增强版（子类 CrowdedSceneDetector）
- 密度低时自动退化为 sparse-corridor 行为（零回归）
- agent_prompt.match_skills 优先级：密集 > 夜间 > 稀疏
