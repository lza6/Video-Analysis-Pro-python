"""全链路 E2E：合成视频从抽帧到知识库索引到 LLM 报告到媒体生成（LLM 用 mock，不产生真实 API 费用）。"""
import json
import subprocess
import sys
import os

from pathlib import Path

PIPELINE_SCRIPT = r'''
import sys, os, json
sys.path.insert(0, '.')
os.environ["QT_QPA_PLATFORM"] = "offscreen"
try:
    import torch
except OSError:
    torch = None

import numpy as np
import cv2
from pathlib import Path

from src.core.logic import VideoProcessor, AudioProcessor, VideoAnalyzer, OllamaClient, PromptLoader, ModelContextManager
from src.core.history_manager import HistoryManager
import tempfile

results = {}
def step(name, ok, detail=""):
    results[name] = {"ok": bool(ok), "detail": str(detail)[:200]}

tmp = Path(tempfile.mkdtemp(prefix="vap_e2e_"))

# ---------- Step 1: 合成视频（2s，两场景，带音轨省略——转录用 mock）----------
video = tmp / "e2e_sample.mp4"
w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
for i in range(20):
    color = (255, 0, 0) if i < 10 else (0, 255, 0)
    w.write(np.full((64, 64, 3), color, dtype=np.uint8))
w.release()
step("make_video", video.exists() and video.stat().st_size > 0)

# ---------- Step 2: Phase 1 抽帧 ----------
out_dir = tmp / "cache" / "e2e_sample"
proc = VideoProcessor(video, out_dir)
frames = proc.extract_keyframes(density=0.5)
step("extract_frames", len(frames) >= 5, f"{len(frames)} frames")
step("frame_metrics", all(f.metrics for f in frames))

# duration 读取（ExtractionWorker 同款逻辑）
cap = cv2.VideoCapture(str(video))
fps = cap.get(cv2.CAP_PROP_FPS) or 0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
duration = total / fps if fps > 0 and total > 0 else 0.0
cap.release()
step("duration_extracted", abs(duration - 2.0) < 0.5, f"{duration:.2f}s")

# ---------- Step 3: 历史记录 + 知识库索引 ----------
hm = HistoryManager(str(tmp / "cfg"))
sid = hm.add_session(video, out_dir)
step("session_added", sid and len(sid) == 32)

from src.core.kb_indexer import index_frames, get_embedder
emb = get_embedder()
if emb is not None:
    n = index_frames(hm, sid, "e2e_sample.mp4", str(video), frames)
    step("kb_indexed", n == len(frames), f"{n}/{len(frames)}")
    hits = hm.search_kb(emb.encode("绿色画面", convert_to_tensor=False), top_k=3, min_score=0.0)
    step("kb_search", len(hits) > 0, f"{len(hits)} hits, best={hits[0]['score'] if hits else 0}")
else:
    step("kb_indexed", True, "SKIPPED (no CLIP weights)")
    step("kb_search", True, "SKIPPED")

# ---------- Step 4: Phase 2 mock LLM ----------
class MockClient(OllamaClient):
    """不发网络请求，产出固定流式报告。"""
    def chat_stream(self, model, prompt, image_paths=None, temperature=0.2, timeout=600):
        # 验证 prompt 里包含帧信息与中文指令
        assert "frame" in prompt.lower() or "关键帧" in prompt or "{frame_info}" in prompt or "s:" in prompt
        yield "## E2E 报告标题\n"
        yield f"视频时长 {duration:.1f} 秒，共 {len(frames)} 帧。"
        yield "\n分析完成。"

analyzer = VideoAnalyzer(MockClient(), "mock-model", PromptLoader(), use_yolo=False, use_ocr=False)
chunks = list(analyzer.analyze_video(frames, "", custom_template=None))
report = "".join(chunks)
step("llm_report", "E2E 报告标题" in report, report[:80])
step("no_dirty_marker", "__FULL_RESPONSE_END__" not in report)
step("no_json_fragment", '"message"' not in report)

# ---------- Step 5: Phase 3 媒体生成（真实 moviepy）----------
from src.core.logic import create_summary_media_artifacts
clips, sel, summary_video, gif = create_summary_media_artifacts(
    str(video), duration, frames, out_dir, "e2e_sample",
    num_clips=2, clip_duration_around_keyframe=1.0,
    make_video=True, make_gif=False
)
step("media_clips", bool(clips) and all(Path(c).exists() for c in (clips or [])), f"{len(clips or [])} clips")
step("summary_video", summary_video and Path(summary_video).exists(), str(summary_video))

# ---------- Step 6: 会话删除 → 知识库同步清理 ----------
if emb is not None and hm.kb_count() > 0:
    before = hm.kb_count()
    hm.delete_session(sid)
    step("kb_cleanup", hm.kb_count() == 0, f"{before} -> {hm.kb_count()}")

print("PIPELINE_RESULT:", json.dumps(results, ensure_ascii=False, indent=1))
failed = [k for k, v in results.items() if not v["ok"]]
print(f"PIPELINE_SUMMARY: {len(results)-len(failed)}/{len(results)}")
sys.exit(1 if failed else 0)

'''


def test_full_pipeline(qapp):
    child = subprocess.run(
        [sys.executable, "-c", PIPELINE_SCRIPT],
        capture_output=True, timeout=600,
        cwd=str(Path(__file__).parent.parent),
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONIOENCODING": "utf-8"},
    )
    stdout = (child.stdout or b"").decode("utf-8", errors="replace")
    stderr = (child.stderr or b"").decode("utf-8", errors="replace")
    tail = "\n".join(stdout.splitlines()[-25:])
    print(tail)
    assert child.returncode == 0, f"全链路失败:\n{tail}\nSTDERR:{stderr[-1000:]}"
    summary = stdout.split("PIPELINE_SUMMARY:")[-1].split()[0]
    assert summary.startswith("13/13") or summary.startswith("11/11") or "/13" in summary or "/11" in summary
