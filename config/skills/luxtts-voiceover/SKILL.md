---
name: luxtts-voiceover
description: 为视频生成配音/旁白（LuxTTS 轻量本地 TTS 思路，含声音克隆），主项目暂无 TTS 能力，给出手动工作流
triggers: 配音,旁白,朗读,TTS,语音合成
---

# 本地 TTS 配音生成

## 适用场景

当用户要求"给这段视频配音""生成旁白""把总结读出来""换个人声朗读"时激活。

## 工作流

1. **产出文案**：基于 Phase 2 总结报告或用户给的文本整理旁白稿。按 `transcript.segments` 的静默间隙切成段落，每段独立成句组，方便后续对齐原视频时间轴。标注每段的目标起始时间。
2. **确认能力边界（主项目暂无 TTS，需外部工具）**：主项目当前只有 ASR（faster-whisper）没有语音合成，本 skill 全程手动执行。推荐 LuxTTS（zipvoice 蒸馏 4 步、48kHz、<1GB 显存、CPU 也可实时）：`pip install -r requirements.txt`（克隆 LuxTTS 仓库）。
3. **准备参考音色**：克隆声音需 ≥3 秒干净人声参考（wav/mp3）。可从目标视频 `audio.mp3` 里挑一段无背景音乐的区间截取；或用户自备。参考音频质量直接决定克隆效果。
4. **逐段合成**：每段文案独立推理，参数起点——`rms=0.01`（响度）、`num_steps=4`（质量/速度平衡）、`speed=1.0`；听到金属音则 `return_smooth=True`；发音错则调低 `t_shift`。输出按段落命名 `seg_001.wav` 起。

   ```python
   from zipvoice.luxvoice import LuxTTS
   lux = LuxTTS('YatharthS/LuxTTS', device='cuda')  # 无 GPU 用 device='cpu', threads=2
   prompt = lux.encode_prompt('ref.wav', rms=0.01)
   wav = lux.generate_speech('第一段旁白文本', prompt, num_steps=4)
   wav.numpy().squeeze()  # soundfile 写盘，48000 Hz
   ```

5. **对齐与混音**：用 MoviePy 按第 1 步的时间戳把各段 wav 摆到音轨（原视频人声可先降音量或静音区间替换），导出混音后成片。段长超出原间隙时宁可压缩文案重合成，不做变速（变声风险）。
6. **验收**：通听全片检查——音色一致、无金属杂音、与画面/字幕同步、响度均匀（各段 rms 一致但实际听感仍需抽查）。

## 输出格式

- 旁白稿：每段 `目标时间 | 文本 | 音频文件名`
- 音频文件（`seg_*.wav`）与混音成片（如 `voiced.mp4`）
- 参数记录：参考音频来源、rms/num_steps/t_shift/speed 取值
- 未合成段落及原因（如文本过长、参考音不足 3 秒）
