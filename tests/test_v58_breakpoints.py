"""v5.8 断点闭环测试（B1/B2/B6/B7）。

- B7（P0 生产阻断）：_collect_config 返回 dict → BatchConfig 转换不崩
- B6：frame_change_pct 百分比 → 阈值映射
- B1：per-model 分片配置（get_video_config 接入）
- B2：on_segment_judged 回调（stop/deep_dive/continue）
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_b7_collect_config_dict_to_batchconfig_no_crash() -> None:
    """B7：_collect_config 返回的 dict 能直接 BatchConfig(**dict) 不崩。

    之前直接传 dict 给 BatchRunner 立即 AttributeError，
    被 _on_start try/except 吞成"错误：dict object"——生产阻断。
    """
    from src.core.batch_runner import BatchConfig
    # 模拟 batch_tab._collect_config 返回的 dict
    cfg_dict = {
        "video_dir": "D:/监控",
        "key_item_image": "D:/监控/关键物品.jpg",
        "item_description": "黑色旅行袋",
        "segment_sec": 120,
        "fps_sample": 1.0,
        "clip_padding": 10.0,
        "concurrency_per_key": 1,
        "max_tokens": 65536,
        "reasoning_budget": 8192,
        "clean_segments": True,
        "keep_frames": "auto",
        "resume": True,
        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "enable_thinking": True,
        "temperature": 0.2,
        "request_timeout": 120,
        "out_dir": "out/batch",
        "video_concurrency": 4,
        "confidence_threshold": 0.7,
        "enable_verification": True,
        "frame_change_pct": 10,  # B6 新增字段
    }
    cfg = BatchConfig(**cfg_dict)
    assert cfg.frame_change_pct == 10
    assert cfg.key_item_image == "D:/监控/关键物品.jpg"
    # 旧 dict（无 frame_change_pct）也能构造，用默认 20
    cfg2 = BatchConfig(video_dir="x", key_item_image="",
                      clean_segments=False, keep_frames="never")
    assert cfg2.frame_change_pct == 20  # 默认
    assert cfg2.keep_frames == "never"


def test_b6_frame_change_pct_to_thresholds() -> None:
    """B6：百分比 → (day, night) 阈值映射，默认 20 保持 v5.7 行为。"""
    from src.core.batch_runner import frame_change_pct_to_thresholds
    assert frame_change_pct_to_thresholds(5) == (5.0, 3.0)
    assert frame_change_pct_to_thresholds(10) == (10.0, 5.0)
    assert frame_change_pct_to_thresholds(20) == (15.0, 6.0)  # v5.7 默认
    assert frame_change_pct_to_thresholds(30) == (25.0, 10.0)
    assert frame_change_pct_to_thresholds(50) == (40.0, 20.0)
    # 未知档位回退 20% 默认
    assert frame_change_pct_to_thresholds(99) == (15.0, 6.0)
    assert frame_change_pct_to_thresholds(7) == (15.0, 6.0)


def test_b1_per_model_video_config_loaded() -> None:
    """B1：BatchRunner.__init__ 调 get_video_config 不再硬编码 120/720/2/256。

    用真实 omni model id，断言 _video_cfg 来自 nvidia_models 注册表。
    """
    from src.core.batch_runner import BatchRunner, BatchConfig
    cfg = BatchConfig(video_dir="x", key_item_image="",
                      model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    # mock run_store / router 避免真实依赖
    rs = MagicMock()
    rs.update_run.return_value = True
    router = MagicMock()
    runner = BatchRunner(config=cfg, run_store=rs, router=router)
    # _video_cfg 应来自 get_video_config，含 omni 的 720p/2fps/256帧
    assert runner._video_cfg["target_height"] == 720
    assert runner._video_cfg["target_fps"] == 2
    assert runner._video_cfg["max_frames"] == 256
    # 未知模型回退默认
    cfg2 = BatchConfig(video_dir="x", key_item_image="",
                       model="nvidia/unknown-model")
    runner2 = BatchRunner(config=cfg2, run_store=rs, router=router)
    assert runner2._video_cfg["target_height"] == 720  # 默认兜底
    assert runner2._video_cfg["max_segment_sec"] == 120


def test_b6_motion_config_uses_frame_change_pct() -> None:
    """B6：BatchRunner 构造 MotionConfig 时用档位映射的 day/night_threshold。"""
    from src.core.batch_runner import BatchRunner, BatchConfig
    cfg = BatchConfig(video_dir="x", key_item_image="",
                      frame_change_pct=5)  # 敏感档
    rs = MagicMock()
    rs.update_run.return_value = True
    runner = BatchRunner(config=cfg, run_store=rs, router=MagicMock())
    assert runner._motion_config.day_threshold == 5.0
    assert runner._motion_config.night_threshold == 3.0
    # 50% 宽松档
    cfg2 = BatchConfig(video_dir="x", key_item_image="", frame_change_pct=50)
    runner2 = BatchRunner(config=cfg2, run_store=rs, router=MagicMock())
    assert runner2._motion_config.day_threshold == 40.0
    assert runner2._motion_config.night_threshold == 20.0


def test_b2_on_segment_judged_stop_sets_cancel() -> None:
    """B2：_agent_decide_segment 规则：连续命中≥2 且 conf>0.8 → stop。

    直接测规则逻辑（纯函数，不依赖 Qt widget 状态），用未绑定方法调用。
    """
    from src.ui.batch_tab import BatchTab
    # stop：命中 2 片 + conf 0.9
    assert BatchTab._agent_decide_segment(
        None,  # self 不参与计算，传 None
        {"hits_so_far": 2, "confidence": 0.9, "match": True,
         "seg_idx": 3}) == "stop"
    # continue：未命中
    assert BatchTab._agent_decide_segment(
        None,
        {"hits_so_far": 0, "confidence": 0.0, "match": False,
         "seg_idx": 0}) == "continue"
    # deep_dive：灰色地带 conf 0.65 未命中
    assert BatchTab._agent_decide_segment(
        None,
        {"hits_so_far": 0, "confidence": 0.65, "match": False,
         "seg_idx": 1}) == "deep_dive"
    # continue：命中但只 1 片（不够 stop 阈值）
    assert BatchTab._agent_decide_segment(
        None,
        {"hits_so_far": 1, "confidence": 0.9, "match": True,
         "seg_idx": 2}) == "continue"


def test_b2_batch_runner_accepts_on_segment_judged() -> None:
    """B2：BatchRunner.__init__ 接受 on_segment_judged 参数并存。"""
    from src.core.batch_runner import BatchRunner, BatchConfig
    cfg = BatchConfig(video_dir="x", key_item_image="")
    calls = []

    def cb(payload):
        calls.append(payload)
        return "continue"

    runner = BatchRunner(config=cfg, run_store=MagicMock(),
                         router=MagicMock(), on_segment_judged=cb)
    assert runner._on_segment_judged is cb


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
