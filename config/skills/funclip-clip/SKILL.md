---
name: funclip-clip
description: 按转录文本或画面描述定位时间戳并剪辑视频片段（FunClip 工作流改编），支持多段选择与集锦导出
triggers: 剪辑,切片,截取,说话人,按文本剪
---

# 按文本/说话人定向剪辑

## 适用场景

当用户要求"剪出这段话""把这句话截出来""按文本切片""剪辑某人说的话"时激活。

## 工作流

1. **确认转录可用**：Phase 1 已完成后读取 `transcript.segments`（faster-whisper 输出，每条含 start/end/text 时间戳）。若视频静音（`transcript.language == "und"` 或文本为"[未检测到有效语音/视频静音]"），改走视觉路线：用 `search_visual(画面描述)` 按关键帧语义定位时间点。
2. **文本匹配定位**：把用户给的目标文本做标点/空白归一化后，在 `transcript.segments` 中逐段模糊匹配（子串+字符重合率），得到每段目标的 start/end 秒。多段目标按 `#` 或换行拆开分别匹配，合并为时间区间表。用户给偏移量时（如"往前多留 1 秒"）直接加到区间上。
3. **交叉验证**：对匹配到的每个区间中点调 `get_frame_details(seconds)`，确认画面内容与目标文本语义一致；不一致则回退 `search_visual` 重新定位。
4. **粗剪出片**：调 `create_highlights(description)`，description 填目标文本关键词。注意其真实行为是"对每帧 vision_content 与 description 做词频粗匹配取 top-3 命中帧，各取 ±2 秒拼接"，输出 `output_dir/highlights.mp4`——结果可能不完全等于第 2 步的精确区间。
5. **精确裁剪（需手动，主项目暂无精确文本→subclip 的自动链路）**：用第 2 步的精确区间跑 MoviePy 一行命令：

   ```bash
   ./venv/Scripts/python -c "from moviepy import VideoFileClip, concatenate_videoclips; v=VideoFileClip('video.mp4'); clips=[v.subclipped(s,e) for s,e in [(12.3,18.7),(45.0,52.1)]]; concatenate_videoclips(clips).write_videofile('clip.mp4', codec='libx264')"
   ```

6. **说话人裁剪限制**：主项目说话人分离未接入（`transcribe(diarize=True)` 被忽略，`transcript.speakers` 恒为空），"剪出某说话人的所有段落"暂不支持，需后续版本接入说话人模型；当前回退为按文本内容定位。

## 输出格式

- 片段文件（`highlights.mp4` 或手动 `clip.mp4`）
- 时间区间表：每行 `start - end | 对应转录文本 | 匹配方式（文本/视觉）`
- 未命中的目标文本需明确列出并说明原因
