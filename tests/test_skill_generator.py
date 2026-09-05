"""v7.0 指南 4.3 skill 自动生成 + 4.5 RTSP 流式测试。

- 4.3：skill_generator 规则版（无 LLM 真实调用，付费 API 红线）
- 4.5：rtsp_stream 流式版（已有骨架，测 agent 触发接口）
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---- 4.3 skill 自动生成 ----

def test_detect_scene_parking() -> None:
    """停车场场景识别。"""
    from src.core.skill_generator import detect_scene
    assert detect_scene("停车场车牌识别") == "parking"
    assert detect_scene("parking lot LPR") == "parking"


def test_detect_scene_face() -> None:
    """人脸场景识别。"""
    from src.core.skill_generator import detect_scene
    assert detect_scene("人脸识别找人员") == "face"
    assert detect_scene("face recognition") == "face"


def test_detect_scene_fire() -> None:
    """火灾场景识别。"""
    from src.core.skill_generator import detect_scene
    assert detect_scene("火焰烟雾检测") == "fire"
    assert detect_scene("着火 fire") == "fire"


def test_detect_scene_unknown() -> None:
    """未知场景返回 None。"""
    from src.core.skill_generator import detect_scene
    assert detect_scene("分析视频找猫") is None
    assert detect_scene("") is None


def test_draft_skill_from_scene() -> None:
    """场景模板生成草稿。"""
    from src.core.skill_generator import draft_skill_from_scene
    draft = draft_skill_from_scene("parking")
    assert draft is not None
    assert draft.name == "surveillance-parking-lpr"
    assert "车牌" in draft.description
    assert "停车场" in draft.triggers


def test_render_skill_md_has_frontmatter() -> None:
    """渲染的 SKILL.md 含 frontmatter + 正文。"""
    from src.core.skill_generator import render_skill_md, SkillDraft
    draft = SkillDraft(name="test-skill", description="测试",
                       triggers="测试,test", algorithm="纯规则")
    md = render_skill_md(draft)
    assert md.startswith("---\nname: test-skill")
    assert "## 适用场景" in md
    assert "## 算法" in md
    assert "## 降级行为" in md


def test_save_skill_creates_file(tmp_path: Path) -> None:
    """save_skill 在 skills_dir/<name>/SKILL.md 创建文件。"""
    from src.core.skill_generator import save_skill, SkillDraft
    draft = SkillDraft(name="test-gen", description="测试",
                       triggers="test", algorithm="rule")
    md_path = save_skill(draft, tmp_path)
    assert md_path.exists()
    assert md_path.name == "SKILL.md"
    content = md_path.read_text(encoding="utf-8")
    assert "name: test-gen" in content


def test_save_skill_no_overwrite_raises(tmp_path: Path) -> None:
    """目录已存在且 overwrite=False 抛 FileExistsError。"""
    from src.core.skill_generator import save_skill, SkillDraft
    draft = SkillDraft(name="test-gen", description="测试",
                       triggers="test", algorithm="rule")
    save_skill(draft, tmp_path)  # 第一次成功
    with pytest.raises(FileExistsError):
        save_skill(draft, tmp_path)  # 第二次拒绝


def test_save_skill_overwrite(tmp_path: Path) -> None:
    """overwrite=True 覆盖已存在 skill。"""
    from src.core.skill_generator import save_skill, SkillDraft
    draft = SkillDraft(name="test-gen", description="v1",
                       triggers="test", algorithm="rule")
    save_skill(draft, tmp_path)
    draft2 = SkillDraft(name="test-gen", description="v2",
                        triggers="test", algorithm="rule2")
    md_path = save_skill(draft2, tmp_path, overwrite=True)
    assert "v2" in md_path.read_text(encoding="utf-8")


def test_generate_skill_end_to_end(tmp_path: Path) -> None:
    """generate_skill 端到端：文本 → 存盘。"""
    from src.core.skill_generator import generate_skill
    result = generate_skill("停车场找车牌", tmp_path)
    assert result is not None
    assert result["ok"] is True
    assert result["scene"] == "parking"
    assert Path(result["path"]).exists()


def test_generate_skill_unknown_scene_returns_none(tmp_path: Path) -> None:
    """未知场景 generate_skill 返回 None。"""
    from src.core.skill_generator import generate_skill
    assert generate_skill("分析视频找猫", tmp_path) is None


def test_generate_skill_generated_loads(tmp_path: Path) -> None:
    """生成的 skill 能被 load_skills 加载（frontmatter 格式正确）。"""
    from src.core.skill_generator import generate_skill
    from src.skills.loader import load_skills
    generate_skill("停车场找车牌", tmp_path)
    skills = load_skills(root=tmp_path)
    names = [s.name for s in skills]
    assert "surveillance-parking-lpr" in names


# ---- 4.5 RTSP 流式（已有骨架，测接口不真实拉流）----

def test_rtsp_monitor_constructs() -> None:
    """RtspMonitor 可构造（不真实拉流）。"""
    from src.core.rtsp_stream import RtspMonitor
    backend = MagicMock()
    monitor = RtspMonitor("rtsp://test:554/stream", backend,
                          key_item_image="", item_description="包")
    assert monitor.rtsp_url == "rtsp://test:554/stream"
    assert monitor.item_description == "包"
    assert monitor.events == []


def test_rtsp_motion_event_detector_threshold() -> None:
    """MotionEventDetector 帧差超阈值触发。"""
    import numpy as np
    from src.core.rtsp_stream import MotionEventDetector
    det = MotionEventDetector(threshold=10.0, min_area=10, cooldown=0)
    # 首帧不触发（无前帧）
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    assert det.detect(frame1, 0.0) is False
    # 相同帧不触发
    assert det.detect(frame1, 1.0) is False
    # 差异大的帧触发
    frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert det.detect(frame2, 2.0) is True


def test_rtsp_sanitize_url() -> None:
    """RTSP URL 脱敏（密码替换为 ***）。"""
    from src.core.rtsp_stream import _sanitize_rtsp_url
    url = "rtsp://admin:secret123@192.168.1.1:554/stream"
    masked = _sanitize_rtsp_url(url)
    assert "secret123" not in masked
    assert "***" in masked
    assert "admin" in masked


def test_stream_event_dataclass() -> None:
    """StreamEvent 数据类字段。"""
    from src.core.rtsp_stream import StreamEvent
    ev = StreamEvent(timestamp=100.0, kind="motion")
    assert ev.kind == "motion"
    assert ev.confidence == 0.0
    assert ev.frame_path == ""


def test_rtsp_monitor_events_thread_safe() -> None:
    """RtspMonitor.events 列表线程安全（_lock 保护）。"""
    from src.core.rtsp_stream import RtspMonitor, StreamEvent
    backend = MagicMock()
    monitor = RtspMonitor("rtsp://test/stream", backend)
    # 模拟并发 append
    import threading
    def add_events():
        for i in range(100):
            with monitor._lock:
                monitor.events.append(
                    StreamEvent(timestamp=float(i), kind="motion"))
    threads = [threading.Thread(target=add_events) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(monitor.events) == 500


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
