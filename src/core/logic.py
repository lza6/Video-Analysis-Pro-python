import os
import json
import logging
import shutil
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
import requests
import cv2
import threading
import numpy as np
import torch
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import base64
import subprocess
import hashlib

# Optional dependency probes — 使用 importlib 只探测不绑定，
# 实际使用处（filter_frames_semantically / extract_smart_keyframes 等）再做局部 import。
import importlib.util as _ilu


def _probe(module: str) -> bool:
    return _ilu.find_spec(module) is not None


SCENEDETECT_AVAILABLE = _probe("scenedetect")
try:
    from PIL import Image  # noqa: F401  (used by semantic dedup / KB indexer)
    from sentence_transformers import SentenceTransformer, util  # noqa: F401
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

DECORD_AVAILABLE = _probe("decord")
try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_GPU_AVAILABLE = True
except ImportError:
    NVIDIA_GPU_AVAILABLE = False
except Exception: # Handle pynvml.NVMLError or other init errors safely (pynvml might be loaded but init failed)
    NVIDIA_GPU_AVAILABLE = False

def check_cuda_health() -> bool:
    """Checks if CUDA and cuDNN are actually functional."""
    if not NVIDIA_GPU_AVAILABLE:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        # Test a small operation
        x = torch.tensor([1.0]).cuda()
        del x
        # Check cuDNN specifically
        return torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled
    except Exception as e:
        logger.warning(f"CUDA/cuDNN initialization health check failed: {e}")
        return False
def _detect_advanced_features() -> bool:
    """Probe the lazy-loaded Phase-3 dependencies for real availability.

    Historical bug: this used to be `try: pass; FLAG=True except ...`, so the
    flag was always True and Phase 3 crashed at runtime instead of being
    disabled up front.
    """
    import importlib
    for module in ("moviepy", "matplotlib.pyplot", "seaborn"):
        try:
            importlib.import_module(module)
        except Exception:
            return False
    return True

ADVANCED_FEATURES_AVAILABLE = _detect_advanced_features()

MEDIAINFO_AVAILABLE = _probe("pymediainfo")

GPU_LOCK = threading.Lock()


def imwrite_unicode(path: str, frame) -> bool:
    """cv2.imwrite 对含 Unicode 的 Windows 路径不稳定，改用 imencode + tofile。
    保持调用方签名与 cv2.imwrite 一致，便于全局替换。
    """
    ext = ".jpg"
    try:
        ok, buf = cv2.imencode(ext, frame)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception as e:
        logging.debug(f"imwrite_unicode failed for {path}: {e}")
        return False


def videocapture_unicode(path) -> "cv2.VideoCapture":
    """cv2.VideoCapture 对含 Unicode 的 Windows 路径可能失败，先尝试转义。
    若非 Windows 或路径为 ASCII，直接传 str 即可。
    """
    try:
        return cv2.VideoCapture(str(path))
    except Exception:
        # 兜底：编码后重试（部分 OpenCV 构建支持 bytes 路径）
        return cv2.VideoCapture(str(path).encode("utf-8"))

# Generic Logger (Connected to UI later)
logger = logging.getLogger("VideoAnalyzerCore")

def check_ffmpeg():
    if shutil.which("ffmpeg"):
        return True
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

FFMPEG_AVAILABLE = check_ffmpeg()

# --- Classes ---

@dataclass
class Frame:
    path: Path
    timestamp: float
    metrics: Dict[str, float] = field(default_factory=dict)
    vision_content: str = ""
    ocr_text: str = ""

class ModelContextManager:
    """
    确保同一时间只有一个重型模型（YOLO, Whisper, LLM）在 GPU 上运行。
    """
    def __init__(self):
        self.active_models = {} # name -> model_instance

    def request_vram(self, requesting_model: str):
        """当某个模型需要显存时，可能需要关闭其他模型。"""
        # 加锁：AnalysisWorker / ModelLoadWorker 多 QThread 并发访问 active_models
        with GPU_LOCK:
            logger.info(f"显存管理：模型 {requesting_model} 请求加载。")
            # 如果是加载 LLM，强烈建议清理本地视觉和语言模型
            if requesting_model in ["LLM", "Whisper", "YOLO"]:
                to_unload = [m for m in list(self.active_models) if m != requesting_model]
                for m in to_unload:
                    self.unload(m)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def register(self, name: str, instance: Any):
        with GPU_LOCK:
            self.active_models[name] = instance

    def unload(self, name: str):
        with GPU_LOCK:
            if name in self.active_models:
                logger.info(f"显存管理：正在卸载模型 {name}")
                del self.active_models[name]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

class VideoProcessor:
    def __init__(self, video_path: Path, output_dir: Path):
        self.video_path = video_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_keyframes(self, density: float, max_frames: int = 10000) -> List[Frame]:
        """
        Extract frames based on density (0.0 - 1.0).
        1.0 = High density (~1 FPS), 0.1 = Low density.
        Ensures minimum 5 frames even for short videos.
        """
        if density <= 0: density = 0.1
        if density > 1.0: density = 1.0
        
        cap = videocapture_unicode(self.video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {self.video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release() 
       
        # Calculate Target Frames
        # Strategy: Max Density = 1 FPS. Min Frames = 5.
        base_min_frames = 5
        max_density_fps = 1.0 
        
        raw_target = int(duration * max_density_fps * density)
        target_frames = max(base_min_frames, raw_target)
        
        # Clamp to actual total frames and max_frames
        target_frames = min(target_frames, total_frames, max_frames)
        
        if target_frames <= 0: return []
            
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_indices = np.unique(frame_indices) # Ensure uniqueness
        
        # Parallel Processing
        max_workers = min(4, os.cpu_count() or 1) 
        extracted_frames = []
        
        # Split indices for workers
        chunks = np.array_split(frame_indices, max_workers)
        
        logger.info(f"Extracting {len(frame_indices)} frames (Density: {density:.2%})...")


        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._extract_chunk, chunk) for chunk in chunks if len(chunk) > 0]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    extracted_frames.extend(result)
                except Exception as e:
                    logger.error(f"Chunk extraction failed: {e}")

        # Sort by timestamp as threads might finish out of order
        extracted_frames.sort(key=lambda x: x.timestamp)
        return extracted_frames

    def _extract_chunk(self, indices: np.ndarray) -> List[Frame]:
        """Worker method to extract specific frames."""
        local_frames = []
        # Re-open capture in this thread
        cap = videocapture_unicode(self.video_path)
        if not cap.isOpened(): return []
        
        try:
            for frame_index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame_data = cap.read()
                if ret:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    # Use unique filename with timestamp to avoid thread collision (though unlikely with different indices)
                    frame_filename = self.output_dir / f"frame_{int(frame_index)}_{timestamp:.2f}s.jpg"
                    imwrite_unicode(str(frame_filename), frame_data)
                    frame_metrics = get_frame_metrics(frame_data)
                    local_frames.append(Frame(path=frame_filename, timestamp=timestamp, metrics=frame_metrics))
        finally:
            cap.release()
            
        return local_frames

    def extract_smart_keyframes(self, min_scene_len: int = 15) -> List[Frame]:
        if not SCENEDETECT_AVAILABLE:
            return self.extract_keyframes(0.2, max_frames=200)
        from scenedetect import detect, AdaptiveDetector

        scene_list = detect(str(self.video_path), AdaptiveDetector(min_scene_len=min_scene_len))
        extracted_frames = []
        # Parallel Smart Extraction
        max_workers = min(4, os.cpu_count() or 1)

        
        # Split scenes into chunks
        chunks = np.array_split(scene_list, max_workers)
        
        logger.info(f"Using {max_workers} threads for smart extraction of {len(scene_list)} scenes...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_scenes_chunk, chunk, i * len(chunks[0])) for i, chunk in enumerate(chunks) if len(chunk) > 0]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    extracted_frames.extend(result)
                except Exception as e:
                    logger.error(f"Smart chunk extraction failed: {e}")

        # Sort by timestamp
        extracted_frames.sort(key=lambda x: x.timestamp)
        


        if not extracted_frames: return self.extract_keyframes(0.2, max_frames=3600)
        return extracted_frames

    def _process_scenes_chunk(self, scenes, start_index_offset):
        local_frames = []
        cap = videocapture_unicode(self.video_path)
        if not cap.isOpened(): return []
        fps = cap.get(cv2.CAP_PROP_FPS)

        try:
            for i, scene in enumerate(scenes):
                global_idx = start_index_offset + i
                start_frame, end_frame = scene
                middle_frame_idx = (start_frame.get_frames() + end_frame.get_frames()) // 2
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
                ret, frame_data = cap.read()
                
                if ret:
                    # Blur detection logic
                    if self._is_blurry(frame_data):
                        # Try next 5 frames
                        for _ in range(5):
                            ret_next, frame_next = cap.read()
                            if ret_next and not self._is_blurry(frame_next):
                                frame_data = frame_next
                                middle_frame_idx += (_ + 1)
                                break
                    
                    timestamp = middle_frame_idx / fps if fps > 0 else 0
                    frame_filename = self.output_dir / f"scene_{global_idx:03d}_{timestamp:.2f}s.jpg"
                    imwrite_unicode(str(frame_filename), frame_data)
                    frame_metrics = get_frame_metrics(frame_data)
                    local_frames.append(Frame(path=frame_filename, timestamp=timestamp, metrics=frame_metrics))
        finally:
            cap.release()
        return local_frames

    def _is_blurry(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

    def filter_frames_semantically(self, frames: List[Frame], threshold: float = 0.85) -> List[Frame]:
        """使用 CLIP 模型对相似帧进行语义去重，只保留内容差异大的精华帧。"""
        if not CLIP_AVAILABLE or not frames:
            return frames
        
        logger.info(f"正在进行语义级帧去重 (CLIP)，初始帧数: {len(frames)}...")
        try:
            # Using a very light model for speed
            model = SentenceTransformer('clip-ViT-B-32')
            
            # Encode images
            image_paths = [str(f.path) for f in frames]
            images = [Image.open(p) for p in image_paths]
            embeddings = model.encode(images, convert_to_tensor=True)
            
            keep_indices = [0] # Keep the first frame
            for i in range(1, len(frames)):
                is_duplicate = False
                for j in keep_indices:
                    sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                    if sim > threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    keep_indices.append(i)
                    
            logger.info(f"语义去重完成: {len(frames)} -> {len(keep_indices)}")
            return [frames[i] for i in keep_indices]
        except Exception as e:
            logger.error(f"CLIP 语义去重失败: {e}")
            return frames

@dataclass
class AudioTranscript:
    text: str
    segments: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "zh"
    speakers: List[Dict[str, Any]] = field(default_factory=list)
    waveform: Optional[np.ndarray] = None # Normalized waveform for UI

class ModelManager:
    """Handles downloading and managing local model files."""
    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.download_threads: Dict[str, Any] = {}

    def get_model_path(self, model_id: str) -> Optional[Path]:
        # Mapping for specific internal components
        mapping = {
            "yolo_v11n": "yolo11n.pt",
            "whisper_base": "whisper_base.pt",
            "st_minilm": "all-MiniLM-L6-v2",
            "ffmpeg": "ffmpeg.exe"
        }
        filename = mapping.get(model_id)
        if filename:
            path = self.models_dir / filename
            if path.exists(): return path
            root_path = Path(filename)
            if root_path.exists(): return root_path
            
        # Generic scanning for local models (GGUF, PT, BIN)
        for ext in ["*.gguf", "*.pt", "*.bin"]:
            for f in self.models_dir.glob(ext):
                if model_id in f.name:
                    return f
        return None

    def list_local_models(self) -> List[str]:
        """Returns a list of all model files in the models directory."""
        models = []
        for ext in ["*.gguf", "*.pt", "*.bin"]:
            for f in self.models_dir.glob(ext):
                models.append(f.name)
        return models

    def detect_model_type(self, model_filename: str) -> str:
        """Detects if a model is 'Vision-Language (VL)' or 'Text-only'."""
        vl_keywords = ["llava", "qwen-vl", "vision", "vl", "moondream", "cogvlm", "internvl", "minicpm-v", "qwen3"]
        name_lower = model_filename.lower()
        
        # Check for vision keywords in the filename
        for kw in vl_keywords:
            if kw in name_lower:
                return "Vision-Language (VL)"
        
        # For GGUF files, we could ideally probe metadata but for now heuristic is safer/faster
        if model_filename.endswith(".pt") and "yolo" not in name_lower and "whisper" not in name_lower:
            return "Torch Model (Unknown)"
            
        return "Text-only LLM"

    # 供应链安全：已知的预期 SHA256（下载完成后校验，防止 MITM 投毒）。
    # None 表示该来源未提供稳定 hash，跳过校验（仅体积/可用性检查）。
    EXPECTED_SHA256 = {
        "yolo_v11n": None,  # ultralytics 偶尔变更 release 资产 hash
        "whisper_base": "ed3a0b6b1c0fd3c4b11f62cee8e285d8d052d0fa55b850d5337f71b16e864ee0",
    }

    def verify_model_integrity(self, model_id: str) -> bool:
        """下载后/启动时校验模型文件 SHA256，防篡改。"""
        expected = self.EXPECTED_SHA256.get(model_id)
        path = self.get_model_path(model_id)
        if not path or not path.exists():
            return False
        if expected is None:
            return True  # 未约束 hash → 仅存在性校验
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual = h.hexdigest()
        ok = actual == expected
        if not ok:
            logger.error(
                f"模型 {model_id} 完整性校验失败！预期 {expected[:16]}… 实际 {actual[:16]}… "
                f"文件可能被篡改/损坏，建议删除后重新下载。"
            )
        return ok

    def download_model(self, model_id: str, progress_callback=None):
        """Downloads model with resume support + SHA256 verification."""
        urls = {
            "yolo_v11n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
            "whisper_base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0fd3c4b11f62cee8e285d8d052d0fa55b850d5337f71b16e864ee0/base.pt",
            "st_minilm": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin", # Partial for health check or full
            "ffmpeg": "https://github.com/imageio/imageio-binaries/raw/master/ffmpeg/ffmpeg-win64-v4.2.2.exe"
        }
        url = urls.get(model_id)
        if not url:
            logger.error(f"Unknown model_id for download: {model_id}")
            return False

        filename = url.split('/')[-1]
        # logic.py mapping fix
        if model_id == "whisper_base": filename = "whisper_base.pt"
        if model_id == "st_minilm": filename = "st_minilm_model.bin" # simplified
        if model_id == "ffmpeg": filename = "ffmpeg.exe"

        dest_path = self.models_dir / filename

        try:
            logger.info(f"Downloading {model_id} from {url} to {dest_path}")
            headers = {}
            mode = 'wb'
            current_size = 0

            if dest_path.exists():
                current_size = dest_path.stat().st_size
                headers['Range'] = f'bytes={current_size}-'
                mode = 'ab'

            response = requests.get(url, headers=headers, stream=True, timeout=30)

            if response.status_code == 416:
                logger.info(f"Model {model_id} already fully downloaded.")
            else:
                # Case where server doesn't support range
                if response.status_code == 200 and 'Range' in headers:
                    logger.warning("Server does not support Range requests. Restarting download.")
                    current_size = 0
                    mode = 'wb'

                total_size = int(response.headers.get('content-length', 0)) + current_size

                with open(dest_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            current_size += len(chunk)
                            if progress_callback and total_size > 0:
                                progress_callback(int(current_size / total_size * 100))

            # 下载完成后校验完整性（供应链安全）
            if not self.verify_model_integrity(model_id):
                logger.error(f"模型 {model_id} 校验失败，请删除 models/{filename} 后重试。")
                return False
            return True
        except Exception as e:
            logger.error(f"Download {model_id} failed with exception: {e}")
            return False

class AudioProcessor:
    def __init__(self, vram_manager: Optional[ModelContextManager] = None):
        self.model = None
        self.vram_manager = vram_manager

    def extract_audio(self, video_path: Path, output_dir: Path) -> Optional[Path]:
        try:
            # moviepy 2.x removed the moviepy.editor shim; the top-level
            # import works on both 1.x and 2.x.
            from moviepy import VideoFileClip
            audio_path = output_dir / "audio.mp3"
            if audio_path.exists(): return audio_path
            
            with VideoFileClip(str(video_path)) as video:
                if video.audio:
                    video.audio.write_audiofile(str(audio_path), logger=None)
                    return audio_path
            return None
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None

    def transcribe(self, audio_path: Path, diarize: bool = False) -> Optional[AudioTranscript]:
        if diarize:
            # 说话人分离 (pyannote) 未接入：v4.5 保持接口占位，显式声明避免静默忽略
            logger.info("diarize=True 请求被忽略: 说话人分离将在后续版本接入。")
        try:
            from faster_whisper import WhisperModel
            
            # Check for silence using pydub first
            try:
                import pydub
                audio = pydub.AudioSegment.from_file(str(audio_path))
                # dBFS is a measure of loudness in decibels relative to full scale.
                # Silence is typically below -60 dBFS.
                dbfs = audio.dBFS
                if dbfs < -60:
                    logger.info(f"检测到静音音频 (dBFS: {dbfs:.2f}), 跳过 Whisper 转录。")
                    return AudioTranscript(
                        text="[未检测到有效语音/视频静音]",
                        segments=[],
                        language="und",
                        speakers=[],
                        waveform=None
                    )
            except Exception as e:
                logger.warning(f"静音检测失败: {e}")

            if self.vram_manager:
                self.vram_manager.request_vram("Whisper")
            
            
            # Perform a deep health check for CUDA/cuDNN before attempting to use it
            cuda_is_healthy = check_cuda_health()
            
            model_size = "medium" if cuda_is_healthy else "small"
            device = "cuda" if cuda_is_healthy else "cpu"
            compute_type = "float16" if cuda_is_healthy else "int8"
            
            try:
                logger.info(f"加载 Faster-Whisper ({model_size}) on {device}...")
                # download_root 指向 models/ 目录，复用本地已缓存的模型，离线也可用
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type,
                                          download_root=str(self.models_dir))
                assert self.model is not None  # 刚赋值，帮助类型检查器收窄

                # Test transcription to catch lazy-load DLL errors
                segments, info = self.model.transcribe(str(audio_path), beam_size=5)
                segments_list = list(segments) 
            except Exception as e:
                if device == "cuda":
                    # If CUDA was healthy but transcription crashed (likely a missing DLL only loaded on demand),
                    # we must catch it and fallback.
                    logger.warning(f"CUDA transcription failed (Error: {e}). Switching to CPU fallback...")
                    try:
                        # Ensure old model is cleared
                        if hasattr(self, 'model') and self.model:
                            del self.model
                            self.model = None
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        device = "cpu"
                        compute_type = "int8"
                        logger.info(f"重新加载 Faster-Whisper ({model_size}) on CPU...")
                        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
                        assert self.model is not None  # 刚赋值，帮助类型检查器收窄
                        segments, info = self.model.transcribe(str(audio_path), beam_size=5)
                        segments_list = list(segments)
                    except Exception as e2:
                        logger.error(f"CPU 回退也失败了: {e2}")
                        raise e2
                else:
                    raise e

            if self.vram_manager:
                self.vram_manager.register("Whisper", self.model)
            
            full_text = "".join([s.text for s in segments_list])
            
            # Generate Waveform (Downsampled for UI)
            try:
                import pydub
                audio = pydub.AudioSegment.from_file(str(audio_path))
                samples = np.array(audio.get_array_of_samples()).astype(np.float32)
                # Normalize and downsample
                samples = samples / np.max(np.abs(samples))
                step = len(samples) // 1000 if len(samples) > 1000 else 1
                waveform = samples[::step]
            except Exception:
                waveform = None

            if self.vram_manager:
                self.vram_manager.unload("Whisper")
            else:
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            return AudioTranscript(
                text=full_text,
                segments=[s._asdict() for s in segments_list],
                language=info.language,
                speakers=[], # speakers not implemented yet
                waveform=waveform
            )
        except Exception as e:
            logger.error(f"Faster-Whisper 转录失败: {e}")
            return None

class PromptLoader:
    def __init__(self, prompt_dir: Optional[str] = None):
        self.prompts: Dict[str, str] = {}
        # The rich, curated prompt templates live under
        # config/prompts/frame_analysis/ (video_summary.txt etc). The old
        # default pointed at config/prompts/, where no files exist, so the
        # fallback one-liner in code was always used.
        self.prompt_dir = Path(prompt_dir) if prompt_dir else Path("config") / "prompts" / "frame_analysis"
        # Define defaults in code for robustness
        self.defaults = {
            "Video Summary": "Analyze this video based on these keyframes: {frame_info}. Audio: {audio_transcript}. User Request: {user_prompt}",
        }

    def get_prompt(self, name: str) -> Optional[str]:
        # Try to read from file first
        safe_name = name.lower().replace(" ", "_")
        p_path = self.prompt_dir / f"{safe_name}.txt"
        if p_path.exists():
            return p_path.read_text(encoding="utf-8")
        return self.defaults.get(name, "")

class BaseAPIClient:
    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    def chat_stream(self, *args, **kwargs): raise NotImplementedError



class OllamaClient(BaseAPIClient):
    def __init__(self, base_url: str = "http://localhost:11434",
                 num_ctx: int = 0):
        self.base_url = base_url.rstrip('/')
        self.api_generate = f"{self.base_url}/api/generate"
        self.api_chat = f"{self.base_url}/api/chat"
        # P2-3: 上下文窗口可配置（0=默认 4096；长视频/长转录场景调大，
        # 避免转录被静默截断）。也可用环境变量 VAP_OLLAMA_NUM_CTX 配置。
        if num_ctx <= 0:
            num_ctx = int(os.environ.get("VAP_OLLAMA_NUM_CTX", "4096") or 4096)
        self.num_ctx = num_ctx
        # 本机 Ollama 永远不应走外部系统代理：默认 localhost 会被代理拦截挂起。
        self._session = requests.Session()
        self._session.trust_env = False

    def chat_stream(self, model: str, prompt: str, image_paths: Optional[List[str]] = None, temperature: float = 0.2, timeout: int = 600) -> Iterator[str]:
        # Logic to support images in Ollama
        images_base64 = []
        if image_paths:
            for p in image_paths:
                images_base64.append(self._encode_image_to_base64(p))

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": images_base64}],
            "stream": True,
            "temperature": temperature,
            "options": {"num_ctx": self.num_ctx}
        }

        # Historical bug: this method yielded raw SSE JSON lines, which the
        # ChatWorker forwarded verbatim to the chat UI (users saw
        # '{"message":{"content":...' fragments). Parse the protocol here so
        # every consumer receives plain-text deltas, matching the
        # APIGatewayClient contract. Reasoning content is wrapped in <think>
        # tags that AgentPanel already knows how to render/collapse.
        try:
            with self._session.post(self.api_chat, json=payload, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        line_str = line_str[6:]
                    if line_str.strip() == '[DONE]':
                        continue
                    try:
                        chunk = json.loads(line_str)
                    except json.JSONDecodeError:
                        # Not JSON — surface as-is rather than swallowing it.
                        yield line_str
                        continue
                    message = chunk.get('message') or {}
                    reasoning = message.get('thinking') or chunk.get('thinking') or ""
                    if reasoning:
                        yield f"<think>{reasoning}</think>"
                    content = message.get('content') or chunk.get('response') or ""
                    if content:
                        yield content
                    if chunk.get('error'):
                        yield f"Ollama Error: {chunk['error']}"
        except Exception as e:
            yield f"Ollama Error: {str(e)}"

    def get_status(self) -> Dict[str, Any]:
        """Returns status including running models and VRAM usage."""
        status = {"models": [], "vram_used": 0, "vram_total": 0, "gpu_util": 0}
        try:
            # Get running models (ps)
            resp = self._session.get(f"{self.base_url}/api/ps", timeout=3)
            if resp.status_code == 200:
                models = resp.json().get('models', [])
                status['models'] = models
            
            # Get System GPU stats via pynvml if available
            if NVIDIA_GPU_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    status['vram_used'] = mem.used / 1024**3 # GB
                    status['vram_total'] = mem.total / 1024**3
                    status['gpu_util'] = util.gpu
                except Exception as e:
                    logger.warning(f"Failed to get GPU stats: {e}")

        except Exception as e:
            logger.debug(f"Failed to check Ollama status: {e}")
            
        return status
        
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        try:
            # Sending specific request to unload (keep_alive=0)
            payload = {"model": model_name, "keep_alive": 0}
            resp = self._session.post(f"{self.api_chat}", json=payload, timeout=5)
            # Note: Ollama API might return 200 even if model wasn't loaded, which is fine.
            # Using /api/chat or /api/generate with keep_alive=0 works for unloading.
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False

class APIGatewayClient(BaseAPIClient):
    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        # Smart URL Parsing
        self.base_url, self.chat_endpoint, self.models_endpoint = self.parse_endpoint(api_url)

    @staticmethod
    def parse_endpoint(url: str) -> Tuple[str, str, str]:
        url = url.strip()
        if url.endswith('#'):
            # Force raw mode (User explicit override)
            raw = url.rstrip('#').rstrip('/')
            return raw, f"{raw}/chat/completions", f"{raw}/models"
        
        url = url.rstrip('/')
        # Detailed heuristics
        if url.endswith("/chat/completions"):
            base = url.replace("/chat/completions", "")
            return base, url, f"{base}/models"
        
        if url.endswith("/v1"):
             return url, f"{url}/chat/completions", f"{url}/models"
             
        # Default: Assume host root, append /v1
        return f"{url}/v1", f"{url}/v1/chat/completions", f"{url}/v1/models"

    def list_models(self) -> List[str]:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(self.models_endpoint, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                # Standard OpenAI: {'data': [{'id': 'foo'}, ...]}
                if 'data' in data and isinstance(data['data'], list):
                    return sorted([item['id'] for item in data['data'] if 'id' in item])
                # Some compatible APIs might return list direct?
                return []
        except Exception as e:
            logger.warning(f"Failed to list models from {self.models_endpoint}: {e}")
            return []
        return []

    def chat_stream(self, model: str, prompt: str, image_paths: Optional[List[str]] = None, temperature: float = 0.2, timeout: int = 600) -> Iterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        # Split prompt into System and User if system tags are present.
        # content 可以是 str 或 OpenAI 多模态 parts 列表。
        messages: List[Dict[str, Any]] = []
        if "--- System Context ---" in prompt:
            parts = prompt.split("--- System Context ---")
            # Try to extract the block
            if len(parts) > 1:
                sub_parts = parts[1].split("--------------------")
                system_content = sub_parts[0].strip()
                user_content = (parts[0] + (sub_parts[1] if len(sub_parts) > 1 else "")).strip()
                messages.append({"role": "system", "content": f"系统上下文与视频信息：\n{system_content}"})
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": prompt})

        # Inject images into the LAST message if it's a user message
        if image_paths and messages[-1]["role"] == "user":
            content_parts = [{"type": "text", "text": messages[-1]["content"]}]
            for image_path in image_paths:
                base64_image = self._encode_image_to_base64(image_path)
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            messages[-1]["content"] = content_parts

        # Try streaming first, but be ready for non-streaming
        payload = {"model": model, "messages": messages, "temperature": temperature, "stream": True}
        
        try:
            logger.info(f"发送 API 请求: {self.chat_endpoint} stream=True")
            with requests.post(self.chat_endpoint, headers=headers, json=payload, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                
                content_type = response.headers.get("Content-Type", "").lower()
                logger.info(f"收到响应: Status={response.status_code}, Type={content_type}")
                
                # Case 1: Standard Streaming (SSE)
                if "text/event-stream" in content_type:
                    line_count = 0
                    for line in response.iter_lines():
                        if not line: continue
                        
                                    # Debug logic: Log first 5 lines to understand format
                        line_count += 1
                        # if line_count <= 5: logger.debug(f"Stream Line {line_count}: {line}")
                            
                        # Robust check: startswith b'data:' (optional space)
                        if line.startswith(b'data:'):
                            # Remove prefix, robustly
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                decoded = line_str[6:]
                            elif line_str.startswith("data:"):
                                decoded = line_str[5:]
                            else:
                                decoded = line_str # Should not happen given if check

                            if decoded.strip() == '[DONE]': continue
                            try:
                                chunk = json.loads(decoded)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    
                                    # Support 'reasoning_content'
                                    reasoning = delta.get('reasoning_content', "")
                                    if reasoning:
                                        yield f"<think>{reasoning}</think>"
                                        
                                    content = delta.get('content', "")
                                    if content:
                                        yield content
                                    elif not reasoning:
                                        pass
                            except Exception as e:
                                logger.warning(f"JSON Parse Error on line: {decoded[:50]}... -> {e}")
                        else:
                            pass # skip non-data lines
                            
                # Case 2: Non-streaming (JSON) - API ignored stream=True or doesn't support it
                elif "application/json" in content_type.lower():
                    data = response.json()
                    # Robust search for content in standard and common custom formats
                    target_data = data
                    
                    # 1. Check if 'body' contains a nested OpenAI-like object or is the direct content
                    if 'body' in data and data['body'] is not None:
                        if isinstance(data['body'], dict):
                            target_data = data['body']
                        elif isinstance(data['body'], str) and 'choices' not in data:
                            yield data['body']
                            return

                    # 2. Heuristic: If 'choices' not found at root/body level, check any other dict-type value
                    if 'choices' not in target_data:
                        for val in data.values():
                            if isinstance(val, dict) and 'choices' in val:
                                target_data = val
                                break
                
                    # 3. Handle standard 'choices' format if found
                    if 'choices' in target_data and isinstance(target_data['choices'], list) and len(target_data['choices']) > 0:
                         message = target_data['choices'][0].get('message', {})
                         reasoning = message.get('reasoning_content', "") or message.get('reasoning', "")
                         if reasoning: yield f"<think>{reasoning}</think>"
                         content = message.get('content', "")
                         if content: yield content
                    else:
                        # 4. Final Fallback: Log and search for any content-bearing keys
                        logger.warning(f"Non-standard API response keys: {list(data.keys())}. Value of 'body': {type(data.get('body'))}")
                        
                        # Use 'msg' if 'body' is None/missing, as some APIs return errors in 'msg'
                        fallback = (data.get('body') if isinstance(data.get('body'), str) else None) or \
                                   data.get('result') or data.get('response') or data.get('text') or \
                                   data.get('output') or data.get('content') or data.get('msg')
                        
                        if fallback:
                            if isinstance(fallback, str):
                                yield fallback
                            elif isinstance(fallback, dict):
                                # Deep search inside nested fallback dict
                                deep_val = fallback.get('content') or fallback.get('text')
                                if deep_val: yield str(deep_val)
                            else:
                                yield f"[API 输出]: {str(fallback)}"
                        elif data.get('status') and data.get('status') != 200:
                            yield f"⚠️ API 返回了错误状态 ({data.get('status')}): {data.get('msg', '未知错误')}"
                        else:
                            yield "⚠️ API 返回了空响应或非标准格式，且无法自动提取内容。请检查模型配置或 API URL。"



                # Case 3: Fallback (yield raw text if generic)
                else:
                    text = response.text
                    logger.warning(f"未知 Content-Type, 返回原始文本: {text[:100]}...")
                    yield text
                    
        except Exception as e:
            logger.error(f"API 请求失败: {e}")
            yield f'Error: {str(e)}'

class LMStudioClient(APIGatewayClient):
    """LM Studio typically uses an OpenAI-compatible /v1 endpoint."""
    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio"):
        # 基类签名是 (api_key, api_url)，按位置传入，否则两者对调导致请求崩
        super().__init__(api_key, base_url)

class LocalModelClient(BaseAPIClient):
    """Client for local model files (GGUF etc). Placeholder for local inference engine."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        
    def chat_stream(self, model: str, prompt: str, image_paths=None, **kwargs):
        yield json.dumps({"message": {"content": f"本地模型模式 ({self.model_path}) 目前仅用于展示识别。要进行实际推理，请配置 Ollama 或 LM Studio。"}})

class VideoAnalyzer:
    def __init__(self, client: BaseAPIClient, model: str, prompt_loader: PromptLoader, 
                 temperature: float = 0.2, request_timeout: int = 600, 
                 use_yolo: bool = True, use_ocr: bool = False,
                 vram_manager: Optional[ModelContextManager] = None):
        self.client = client
        self.model = model
        self.prompt_loader = prompt_loader
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.user_prompt = ""
        self.context_length = 4096
        self.use_yolo = use_yolo
        self.use_ocr = use_ocr
        self.vram_manager = vram_manager
        self.yolo_model = None
        self.ocr_reader = None
        
        # Determine local model path
        base_dir = Path(__file__).parent.parent.parent
        self.model_dir = base_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        cuda_is_healthy = check_cuda_health()
        
        if use_yolo:
            if self.vram_manager:
                self.vram_manager.request_vram("YOLO")
            
            logging.info("Loading YOLO11...")
            try:
                from ultralytics import YOLO
                yolo_path = self.model_dir / "yolo11n.pt"
                # Remove old v8 if exists
                old_yolo = self.model_dir / "yolov8n.pt"
                if old_yolo.exists():
                    try: os.remove(old_yolo)
                    except Exception: pass
                
                # YOLO typically auto-detects, but we can be explicit or just let it fail safely
                self.yolo_model = YOLO(str(yolo_path))
                if self.vram_manager:
                    self.vram_manager.register("YOLO", self.yolo_model)
            except Exception as e:
                logging.error(f"YOLO loading failed: {e}")
                self.yolo_model = None

        if use_ocr:
            logging.info("Initializing PaddleOCR...")
            try:
                from paddleocr import PaddleOCR
                self.ocr_reader = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=cuda_is_healthy, show_log=False)
            except Exception as e:
                logging.error(f"OCR loading failed: {e}")
                self.ocr_reader = None
        # 注：self.embedder 死代码已移除。CLIP 共享实例统一走 kb_indexer.get_embedder()，
        # 避免 5 处各自重复加载 SentenceTransformer('clip-ViT-B-32') 浪费 VRAM。

    def unload_models(self):
        """释放 YOLO 和 OCR 模型以减少显存占用。"""
        if self.yolo_model:
            del self.yolo_model
            self.yolo_model = None
        if self.ocr_reader:
            del self.ocr_reader
            self.ocr_reader = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logging.info("本地处理模型已卸载，释放显存给 LLM 使用。")

    def extract_text_from_frame(self, frame_path: str) -> str:
        if not self.ocr_reader: return ""
        try:
            result = self.ocr_reader.ocr(frame_path, cls=True)
            if not result or not result[0]: return ""
            texts = [line[1][0] for line in result[0]]
            return " ".join(texts)
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return ""

    def detect_objects_in_frame(self, frame_path: str) -> str:
        if not self.yolo_model: return "Detection disabled"
        try:
            results = self.yolo_model(frame_path, verbose=False)
            detections = [self.yolo_model.names[int(box.cls[0])] for r in results for box in r.boxes if float(box.conf[0]) > 0.5]
            if not detections: return "None"
            return ", ".join([f"{count} {name}" for name, count in Counter(detections).items()])
        except Exception: return "Error"

    def _process_stream(self, stream_iterator: Iterator[str]) -> Iterator[str]:
        full_response_text = ""
        for chunk_data in stream_iterator:
            try:
                # If the client already yielded a string (e.g. from chat_stream), use it directly
                if isinstance(chunk_data, str):
                    delta = chunk_data
                else:
                    # Otherwise, try to parse as JSON if it's a raw line
                    chunk_str = chunk_data
                    if isinstance(chunk_str, bytes):
                        chunk_str = chunk_str.decode('utf-8')

                    if chunk_str.startswith('data: '): chunk_str = chunk_str[6:]
                    if chunk_str.strip() == '[DONE]': break

                    try:
                        chunk = json.loads(chunk_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") or chunk.get("message", {}).get("content", "")
                    except Exception:
                        # Fallback for non-JSON strings
                        delta = chunk_str

                if delta:
                    full_response_text += delta
                    yield delta
            except Exception: continue
        # Historical bug: an internal sentinel "__FULL_RESPONSE_END__<text>"
        # was appended to the stream and leaked verbatim into the rendered
        # report and Agent transcript. The UI layer now accumulates the
        # chunks itself, so the stream ends silently instead.
        return

    def analyze_video(self, frames: List[Frame], transcript: Any, custom_template: Optional[str] = None) -> Iterator[str]:
        if custom_template:
            prompt_template = custom_template
        else:
            prompt_template = self.prompt_loader.get_prompt("Video Summary") or "Please summarize this video based on these keyframes: {frame_info}. Audio: {audio_transcript}. User Request: {user_prompt}"
        
        prompt_template = prompt_template + "\n\n请使用中文进行总结和回答。"
        
        frame_info_list = []
        for f in frames:
            objects = self.detect_objects_in_frame(str(f.path))
            ocr_text = self.extract_text_from_frame(str(f.path)) if self.use_ocr else ""
            info = f"- {f.timestamp:.2f}s: 物体: {objects}"
            if ocr_text:
                info += f", OCR 文字: {ocr_text}"
            frame_info_list.append(info)
        
        frame_info = "\n".join(frame_info_list)
        
        # Unload models before calling LLM
        self.unload_models()

        # Safe transcript handling
        transcript_text = "无音频"
        if transcript:
            if hasattr(transcript, 'text'):
                transcript_text = transcript.text
            else:
                transcript_text = str(transcript)
                
        prompt = prompt_template.format(user_prompt=self.user_prompt, audio_transcript=transcript_text, frame_info=frame_info)
        
        # Check if model supports vision. If not, don't send images.
        model_is_vision = False
        vision_keywords = ["vl", "vision", "llava", "qwen-vl", "moondream", "internvl", "minicpm-v", "qwen3"]
        if any(kw in self.model.lower() for kw in vision_keywords):
            model_is_vision = True
        
        frame_paths = [str(f.path) for f in frames] if model_is_vision else None
        
        if not model_is_vision and frames:
            logger.info(f"Model {self.model} detected as text-only. Skipping image payloads to prevent API errors.")

        return self._process_stream(self.client.chat_stream(self.model, prompt, frame_paths, self.temperature, self.request_timeout))

# --- Utils ---
def get_frame_metrics(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return {
        "brightness": np.mean(gray),
        "contrast": np.std(gray),
        "saturation": np.mean(hsv[:, :, 1]),
        "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var()
    }

def get_advanced_video_metrics(video_path: str, num_frames_to_sample=100):
    # Lazy imports for plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if not ADVANCED_FEATURES_AVAILABLE: 
        logger.warning("高级视频指标不可用：MoviePy 或 Matplotlib 未加载。")
        return {}, None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        logger.error(f"无法打开视频用于高级指标分析: {video_path}")
        return {}, None
   
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames == 0 or fps == 0:
        cap.release()
        return {}, None
    sample_indices = np.linspace(0, total_frames - 1, min(num_frames_to_sample, total_frames), dtype=int)
   
    metrics_over_time: Dict[str, List[float]] = {'timestamps': [], 'brightness': [], 'saturation': [], 'sharpness': []}
    frame_durations: List[float] = []
    last_timestamp = 0.0
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: 
            continue
       
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        metrics_over_time['timestamps'].append(float(timestamp))
       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       
        metrics_over_time['brightness'].append(float(np.mean(gray)))
        metrics_over_time['saturation'].append(float(np.mean(hsv[:, :, 1])))
        metrics_over_time['sharpness'].append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
       
        if last_timestamp > 0:
            duration = timestamp - last_timestamp
            if duration > 0: frame_durations.append(1.0 / duration)
        last_timestamp = timestamp
    cap.release()
   
    avg_metrics = {
        "平均亮度 (0-255)": np.mean(metrics_over_time['brightness']) if metrics_over_time['brightness'] else 0,
        "平均饱和度 (0-255)": np.mean(metrics_over_time['saturation']) if metrics_over_time['saturation'] else 0,
        "平均清晰度 (拉普拉斯方差)": np.mean(metrics_over_time['sharpness']) if metrics_over_time['sharpness'] else 0
    }
    
    # Create Figure for Qt
    sns.set_style("darkgrid")
    
    # --- Fix Chinese Font Rendering ---
    # Try multiple common Chinese fonts to ensure compatibility
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False # Fix negative sign
    
    # Force apply to current figure if needed, though rcParams should handle new ones
    plt.rc('font', family=['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'sans-serif'])
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    # fig.tight_layout(pad=3.0) 
    
    # Brightness
    sns.lineplot(x='timestamps', y='brightness', data=metrics_over_time, ax=axes[0, 0], color='skyblue', label=f"Avg: {avg_metrics['平均亮度 (0-255)']:.1f}")
    axes[0, 0].set_title('Brightness')
    
    # Saturation
    sns.lineplot(x='timestamps', y='saturation', data=metrics_over_time, ax=axes[0, 1], color='salmon', label=f"Avg: {avg_metrics['平均饱和度 (0-255)']:.1f}")
    axes[0, 1].set_title('Saturation')
    
    # Sharpness
    sns.lineplot(x='timestamps', y='sharpness', data=metrics_over_time, ax=axes[1, 0], color='lightgreen', label=f"Avg: {avg_metrics['平均清晰度 (拉普拉斯方差)']:.1f}")
    axes[1, 0].set_title('Sharpness')
    
    # FPS / Stability
    if frame_durations:
        mean_fps = np.mean(frame_durations)
        sns.histplot(frame_durations, ax=axes[1, 1], color='orchid', bins=20, kde=True)
        axes[1, 1].set_title(f'FPS Stability (Avg: {mean_fps:.1f})')
        axes[1, 1].axvline(mean_fps, color='r', linestyle='--')
    else:
        axes[1, 1].text(0.5, 0.5, 'N/A', ha='center')
        
    plt.close(fig) # Prevent showing in non-GUI thread if any
    return avg_metrics, fig

def get_unique_filepath(output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base, ext = os.path.splitext(filename)
    filepath = output_dir / filename
    if filepath.exists():
        timestamp = datetime.now().strftime("_%Y%m%d%H%M%S")
        filepath = output_dir / f"{base}{timestamp}{ext}"
    return filepath

def create_summary_media_artifacts(
    original_video_path: str,
    video_duration: float,
    frames: List[Frame],
    output_dir: Path,
    video_stem: str,
    num_clips: int = 10,
    clip_duration_around_keyframe: float = 5.0,
    make_video: bool = True,
    make_gif: bool = False,
    gif_resolution: str = "中"
) -> Tuple[Optional[List[str]], Optional[List[Frame]], Optional[str], Optional[str]]:
    
    if not ADVANCED_FEATURES_AVAILABLE or not (make_video or make_gif):
        logger.warning(f"跳过摘要媒体创建 (依赖可用: {ADVANCED_FEATURES_AVAILABLE})")
        return None, None, None, None
   
    if not frames:
        return None, None, None, None
    
    # Sort by sharpness
    frames_sorted = sorted(frames, key=lambda x: x.metrics.get('sharpness', 0), reverse=True)
    # Pick top N then re-sort by time
    num_clips = min(num_clips, len(frames))
    selected_frames = sorted(frames_sorted[:num_clips], key=lambda x: x.timestamp)
    
    individual_clip_paths = []
    
    try:
        from moviepy import VideoFileClip, concatenate_videoclips
        
        with VideoFileClip(original_video_path) as video:
            for i, frame_obj in enumerate(selected_frames):
                try:
                    start_time = max(0, frame_obj.timestamp - clip_duration_around_keyframe / 2)
                    end_time = min(video.duration, frame_obj.timestamp + clip_duration_around_keyframe / 2)
                    if end_time <= start_time: continue
                    
                    sub_clip = video.subclipped(start_time, end_time)
                    clip_path = get_unique_filepath(output_dir, f"{video_stem}_clip_{i:02d}.mp4")
                    sub_clip.write_videofile(str(clip_path), codec='libx264', audio_codec='aac', logger=None, threads=4) 
                    individual_clip_paths.append(str(clip_path))
                    sub_clip.close()
                except Exception as e:
                    logger.error(f"Error creating clip {i}: {e}")
                    continue

        if not individual_clip_paths: return None, None, None, None

        concatenated_video_path = None
        gif_path = None
        reloaded_clips = []
        try:
            reloaded_clips = [VideoFileClip(p) for p in individual_clip_paths]
            final_clip = concatenate_videoclips(reloaded_clips, method="compose")
            
            if make_video:
                concatenated_video_path = get_unique_filepath(output_dir, f"{video_stem}_summary.mp4")
                final_clip.write_videofile(str(concatenated_video_path), fps=24, codec='libx264', audio_codec='aac', logger=None, threads=4)
            
            if make_gif:
                gif_path = get_unique_filepath(output_dir, f"{video_stem}_summary.gif")
                res_map = {"低": 0.3, "中": 0.5, "高": 0.8}
                resize_factor = res_map.get(gif_resolution, 0.5)
                resized = final_clip.resized(resize_factor)
                resized.write_gif(str(gif_path), fps=10)
                resized.close()
                
            final_clip.close()
            
        except Exception as e:
            logger.error(f"Concatenation/GIF error: {e}")
        finally:
            for c in reloaded_clips: c.close()
            
        return individual_clip_paths, selected_frames, str(concatenated_video_path) if concatenated_video_path else None, str(gif_path) if gif_path else None

    except Exception as e:
        logger.error(f"MoviePy error: {e}")
        return None, None, None, None

def export_report_as_pdf(report_md, output_dir):
    try:
        import markdown2
        from weasyprint import HTML
        
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        output_path = Path(output_dir) / f"report_{timestamp}.pdf"
        
        html_content = markdown2.markdown(report_md, extras=["tables"])
        styled_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: "Microsoft YaHei", sans-serif; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; }}
                pre {{ background-color: #f5f5f5; padding: 10px; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        HTML(string=styled_html).write_pdf(str(output_path))
        logger.info(f"PDF Report exported to {output_path}")
        return str(output_path)
    except ImportError:
        logger.error("Export failed: markdown2 or weasyprint not installed.")
        return None
    except Exception as e:
        logger.error(f"PDF Export failed: {e}")
        return None
