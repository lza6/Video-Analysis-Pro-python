# -*- coding: utf-8 -*-
"""探测脚本：拿第一个监控视频的元数据 + Key 连通性 + 抽帧覆盖率（只 1 次 API 调用）。
跑完即可决策：Key 活着吗、抽多少帧、AI 覆盖视频百分之几、预计 VLM 调用几次。
临时文件，正式 E2E 完成后删除。
"""
import os
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

# 加载 .env
env_file = Path(".env")
for line in env_file.read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import cv2  # noqa: E402

VIDEO = "D:/监控/36#2单元地下二层梯口_20260801-000000_20260807-115959_375.mp4"
KEY_ITEM = os.environ.get("VAP_KEY_ITEM_IMAGE", "D:/监控/关键物品.jpg")

# ---- 1. 视频元数据 ----
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
cap.release()
duration = total / fps if fps > 0 else 0
print("=" * 60)
print(f"[元数据] {Path(VIDEO).name}")
print(f"  分辨率 {w}x{h} | fps {fps:.2f} | 总帧 {total} | 时长 {duration:.1f}s ({duration/60:.1f}min)")
print("=" * 60)

# ---- 2. Key 连通性（1 次真实 API 调用，glm 强制思考链 max_tokens≥1000）----
from src.core.llm_gateway import AnthropicBackend  # noqa: E402

b = AnthropicBackend(
    api_key=os.environ["VAP_LLM_API_KEY"],
    base_url=os.environ.get("VAP_LLM_BASE_URL", "https://api.yjs.im/v1"),
    model=os.environ.get("VAP_LLM_MODEL", "glm-5.3-flash"),
    max_tokens=1000,
)
t0 = time.time()
try:
    chunks = list(b.chat_stream(
        messages=[{"role": "user", "content": "只回复两个字: PONG"}],
        temperature=0.1,
    ))
    raw = "".join(chunks)
    print(f"[连通性] OK  耗时 {time.time()-t0:.1f}s")
    print(f"  响应前 200 字: {raw[:200]!r}")
except Exception as e:
    print(f"[连通性] FAIL  {time.time()-t0:.1f}s  错误: {e}")
    sys.exit(1)

# ---- 3. 抽帧 + CLIP 粗筛（不调 VLM）----
from src.core.surveillance_agent import SurveillanceAgent  # noqa: E402

agent = SurveillanceAgent(
    backend=b, key_item_image=KEY_ITEM,
    item_description="白色手提袋（上下开合，表面有商标图案）",
    fps=1.0, max_frames_per_video=600, clip_duration=20,
)
tmp = Path(tempfile.mkdtemp(prefix="vap_probe_"))
frames = agent._extract_frames(Path(VIDEO), tmp / "frames")
covered_secs = len(frames)  # 1fps → 帧数≈覆盖秒数
print(f"[抽帧] {len(frames)} 帧 @ 1fps")
print(f"  覆盖前 {covered_secs}s / 视频 {duration:.0f}s = {covered_secs/duration*100:.1f}%")
if len(frames) < duration:
    print(f"  ⚠ max_frames_per_video=600 截断，未覆盖后 {duration - covered_secs:.0f}s ({(duration-covered_secs)/duration*100:.1f}%)")

cand = agent._clip_prefilter(frames)
print(f"[CLIP粗筛] {len(frames)} -> {len(cand)} 候选帧送 VLM")
print(f"[预计VLM] {len(cand)} 次 | 串行 ~{len(cand)*20}s | 并发8 ~{len(cand)*20/8:.0f}s")
print("=" * 60)
print("探测完成，Key 活着、链路通，可进入正式 E2E")
