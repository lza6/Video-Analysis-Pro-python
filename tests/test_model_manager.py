"""ModelManager: 模型路径解析 / 类型探测（tmp_path 隔离）。"""
from pathlib import Path

from src.core.logic import ModelManager


def test_get_model_path_maps_yolo(tmp_path):
    m = ModelManager(models_dir=tmp_path)
    (tmp_path / "yolo11n.pt").write_bytes(b"x")
    assert m.get_model_path("yolo_v11n") == tmp_path / "yolo11n.pt"


def test_get_model_path_missing_returns_none(tmp_path):
    m = ModelManager(models_dir=tmp_path)
    assert m.get_model_path("yolo_v11n") is None


def test_list_local_models_scans_dir(tmp_path):
    m = ModelManager(models_dir=tmp_path)
    (tmp_path / "custom.gguf").write_bytes(b"x")
    (tmp_path / "another.pt").write_bytes(b"x")
    models = m.list_local_models()
    assert "custom.gguf" in models
    assert "another.pt" in models


def test_detect_model_type_vision_keywords(tmp_path):
    m = ModelManager(models_dir=tmp_path)
    assert m.detect_model_type("qwen-vl-chat.gguf") == "Vision-Language (VL)"
    assert m.detect_model_type("llava-13b.gguf") == "Vision-Language (VL)"


def test_detect_model_type_text_only(tmp_path):
    m = ModelManager(models_dir=tmp_path)
    assert m.detect_model_type("llama3-8b.gguf") == "Text-only LLM"
