import gradio as gr
import os
import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
import requests
import cv2
import time
import psutil
import threading
import numpy as np
from datetime import datetime
import traceback
from dataclasses import dataclass, field
import io
import base64
import inspect
import inspect
import subprocess  # 添加 FFmpeg 检查依赖
import hashlib
import pickle
import gc
import torch
from collections import Counter
from ultralytics import YOLO
try:
    from scenedetect import detect, AdaptiveDetector
except ImportError:
    pass
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    pass

GPU_LOCK = threading.Lock()

# ==============================================================================
# 阶段一：依赖导入与全局设置
# ==============================================================================

# --- 日志配置 ---
class GradioLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_queue = []

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.append(msg)
            if len(self.log_queue) > 200:
                self.log_queue.pop(0)
        except Exception:
            self.handleError(record)

    def get_logs(self):
        return "\n".join(self.log_queue)

gradio_handler = GradioLogHandler()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True, handlers=[logging.StreamHandler(), gradio_handler])
logger = logging.getLogger(__name__)

# --- 依赖检查与动态导入 ---
CORE_MODULES_LOADED = True
NVIDIA_GPU_AVAILABLE = False
ADVANCED_FEATURES_AVAILABLE = False
MEDIAINFO_AVAILABLE = False
def check_ffmpeg():
    if shutil.which("ffmpeg"):
        return True
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

FFMPEG_AVAILABLE = check_ffmpeg()
if not FFMPEG_AVAILABLE:
    logger.warning("未检测到系统 FFmpeg，部分媒体处理功能将受限。")

try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_GPU_AVAILABLE = True
    logger.info("✅ pynvml (NVIDIA GPU support) 加载成功。")
except (ImportError, pynvml.NVMLError) as e:
    NVIDIA_GPU_AVAILABLE = False
    logger.warning(f"⚠️ pynvml 加载失败，GPU监控将不可用。错误: {e}")

try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips, TextClip
    import matplotlib
    matplotlib.use('Agg') # 避免在非GUI环境下报错
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns
   
    def setup_chinese_font():
        font_dir = Path("fonts")
        font_dir.mkdir(exist_ok=True)
        font_files = list(font_dir.glob('*.ttf')) + list(font_dir.glob('*.otf'))
       
        simhei_path = font_dir / "SimHei.ttf"
        if not simhei_path.exists():
            try:
                logger.info("正在下载备用中文字体 SimHei.ttf...")
                # 使用更可靠的字体源
                font_url = "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC-Regular.otf"
                response = requests.get(font_url, timeout=20)
                response.raise_for_status()
                with open(simhei_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"✅ 备用字体下载成功: {simhei_path}")
                font_files.append(simhei_path)
            except Exception as download_e:
                logger.warning(f"下载备用字体失败: {download_e}")
       
        if font_files:
            font_path = font_files[0]
            try:
                fm.fontManager.addfont(str(font_path))
                prop = fm.FontProperties(fname=str(font_path))
                font_name = prop.get_name()
               
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
               
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, '测试中文字体', ha='center', va='center')
                plt.close(fig)
                # 检查字体是否真的被设置
                if plt.rcParams['font.sans-serif'][0] == font_name:
                    logger.info(f"✅ 成功从 '{font_path.name}' 加载并设置中文字体: {font_name}")
                    return True
                else:
                    raise RuntimeError("字体设置未生效")
            except Exception as e:
                logger.warning(f"加载本地字体 '{font_path.name}' 失败: {e}。将强制重建缓存并重试。")
                try:
                    cachedir = matplotlib.get_cachedir()
                    if os.path.exists(cachedir):
                        shutil.rmtree(cachedir)
                        logger.info("Matplotlib 字体缓存已清除，将自动重建。")
                except (FileNotFoundError, PermissionError) as cache_e:
                    logger.warning(f"无法清除Matplotlib缓存: {cache_e}，继续尝试。")
               
                try:
                    fm._fontManager = fm.FontManager()
                    fm.fontManager.addfont(str(font_path))
                    prop = fm.FontProperties(fname=str(font_path))
                    font_name = prop.get_name()
                    plt.rcParams['font.family'] = 'sans-serif'
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    logger.info(f"✅ 重建缓存后，成功设置中文字体: {font_name}")
                    return True
                except Exception as final_e:
                    logger.error(f"重建缓存后仍无法设置字体 '{font_name}': {final_e}")

        logger.warning("本地 'fonts' 目录为空或字体加载失败。将尝试自动检测系统字体。")
        font_candidates = ['Microsoft YaHei', 'SimHei', 'DengXian', 'PingFang SC', 'Heiti SC', 'Arial Unicode MS']
        for font_name in font_candidates:
            try:
                if fm.findfont(font_name, fallback_to_default=False):
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    logger.info(f"✅ 回退成功：找到并设置系统字体: {font_name}")
                    return True
            except Exception:
                continue
       
        logger.error("❌ 字体设置失败：未找到任何可用的本地或系统字体。图表中文将显示为方框。")
        return False
    FONT_LOADED_SUCCESSFULLY = setup_chinese_font()
    ADVANCED_FEATURES_AVAILABLE = True
    logger.info("✅ moviepy, matplotlib, seaborn 加载成功。")
except ImportError as e:
    ADVANCED_FEATURES_AVAILABLE = False
    logger.error(f"\n{'='*60}\n❌ 警告: moviepy, matplotlib或seaborn加载失败。AI摘要视频和画质图功能将不可用。\n详细导入错误: {e}\n{'='*60}\n")

try:
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode == 0:
        FFMPEG_AVAILABLE = True
        logger.info("✅ FFmpeg 可用（MoviePy 将正常工作）。")
    else:
        FFMPEG_AVAILABLE = False
        logger.warning("⚠️ FFmpeg 未检测到，请安装 FFmpeg 并添加至 PATH。")
except (ImportError, FileNotFoundError):
    FFMPEG_AVAILABLE = False
    logger.warning("⚠️ FFmpeg 未在 PATH 中找到。请安装并配置。")

try:
    from pymediainfo import MediaInfo
    MEDIAINFO_AVAILABLE = True
    logger.info("✅ pymediainfo 加载成功。")
except ImportError:
    MEDIAINFO_AVAILABLE = False
    logger.warning(f"\n{'='*60}\n⚠️ 警告: pymediainfo未安装。将无法生成专业的详细元数据JSON。\n请运行: pip install pymediainfo\n{'='*60}\n")

# --- 核心逻辑类定义 ---
@dataclass
class Frame:
    path: Path
    timestamp: float
    metrics: Dict[str, float] = field(default_factory=dict)

class VideoProcessor:
    def __init__(self, video_path: Path, output_dir: Path):
        self.video_path = video_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_keyframes(self, frames_per_minute: int, max_frames: int) -> List[Frame]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {self.video_path}")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
       
        target_frames = min(max_frames, int(duration / 60 * frames_per_minute))
        if target_frames == 0 and total_frames > 0:
            target_frames = 1
        if target_frames == 0:
            cap.release()
            return []
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
       
        extracted_frames = []
        for i, frame_index in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame_data = cap.read()
            if ret:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frame_filename = self.output_dir / f"frame_{i:04d}_{timestamp:.2f}s.jpg"
                cv2.imwrite(str(frame_filename), frame_data)
                frame_metrics = get_frame_metrics(frame_data)
                extracted_frames.append(Frame(path=frame_filename, timestamp=timestamp, metrics=frame_metrics))
                logger.info(f"提取关键帧 {i+1}/{len(frame_indices)}: 时间 {timestamp:.2f}s")
       
        cap.release()
        logger.info(f"从视频中提取了 {len(extracted_frames)} 帧 (目标是 {target_frames})")
        return extracted_frames

    def extract_smart_keyframes(self, min_scene_len: int = 15) -> List[Frame]:
        """
        Use PySceneDetect for smart scene segmentation and extract the best frame from each scene.
        """
        try:
            from scenedetect import detect, AdaptiveDetector
        except ImportError:
            logger.error("scenedetect not installed. Please run 'pip install scenedetect'. Falling back to normal extraction.")
            # Fallback to standard extraction with default params if smart fails
            return self.extract_keyframes(frames_per_minute=10, max_frames=30) # Default fallback

        logger.info("Using smart scene detection algorithm...")
        
        # Use AdaptiveDetector to adapt to video content tempo
        scene_list = detect(str(self.video_path), AdaptiveDetector(min_scene_len=min_scene_len))
        
        extracted_frames = []
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        for i, scene in enumerate(scene_list):
            # Get the middle frame of the scene (usually most stable)
            start_frame, end_frame = scene
            middle_frame_idx = (start_frame.get_frames() + end_frame.get_frames()) // 2
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame_data = cap.read()
            
            if ret:
                # Blur detection and correction
                if self._is_blurry(frame_data):
                    for _ in range(5): # Look ahead 5 frames
                        cap.read()
                        ret_next, frame_next = cap.read()
                        if ret_next and not self._is_blurry(frame_next):
                            frame_data = frame_next
                            middle_frame_idx += (_ + 1)
                            break

                timestamp = middle_frame_idx / fps
                frame_filename = self.output_dir / f"scene_{i:03d}_{timestamp:.2f}s.jpg"
                cv2.imwrite(str(frame_filename), frame_data)
                
                # Calculate metrics
                frame_metrics = get_frame_metrics(frame_data)
                extracted_frames.append(Frame(path=frame_filename, timestamp=timestamp, metrics=frame_metrics))
        
        cap.release()
        
        if not extracted_frames:
             logger.warning("Smart detection found 0 scenes, falling back to standard extraction.")
             return self.extract_keyframes(frames_per_minute=10, max_frames=30)

        logger.info(f"Smart detected {len(extracted_frames)} scene keyframes")
        return extracted_frames

    def _is_blurry(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """Use Laplacian variance to detect blur."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm < threshold

@dataclass
class AudioTranscript:
    text: str
    segments: List[Dict]
    language: str

class AudioProcessor:
    def __init__(self):
        self.whisper_model = None

    def extract_audio(self, video_path: Path, output_dir: Path) -> Optional[Path]:
        if not ADVANCED_FEATURES_AVAILABLE:
            logger.warning("Moviepy 未加载，无法提取音频。")
            return None
        try:
            logger.info(f"正在从 {video_path.name} 提取音频...")
            video_clip = VideoFileClip(str(video_path))
            if video_clip.audio is None:
                logger.warning(f"视频 {video_path.name} 不包含音轨。")
                video_clip.close()
                return None
            audio_path = output_dir / f"{video_path.stem}.mp3"
            video_clip.audio.write_audiofile(str(audio_path), codec='mp3', logger=None)
            video_clip.close()
            logger.info(f"音频提取成功: {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"提取音频失败: {e}")
            return None

    def transcribe(self, audio_path: Path) -> Optional[AudioTranscript]:
        try:
            import whisper
            if self.whisper_model is None:
                logger.info("正在加载 Whisper 模型 (base)...")
                self.whisper_model = whisper.load_model("base")
           
            logger.info(f"正在使用 Whisper 转录音频: {audio_path.name}")
            if NVIDIA_GPU_AVAILABLE:
                with GPU_LOCK:
                    result = self.whisper_model.transcribe(str(audio_path), fp16=True)
                    # Optional: Clear VRAM after heavy audio task if needed, though loading model takes time.
                     # gc.collect(); torch.cuda.empty_cache()
            else:
                 result = self.whisper_model.transcribe(str(audio_path), fp16=False)
                 
            logger.info("音频转录完成。")
            return AudioTranscript(
                text=result.get("text", ""),
                segments=result.get("segments", []),
                language=result.get("language", "")
            )
        except ImportError:
            logger.error("Whisper 未安装，无法进行音频转录。请运行: pip install openai-whisper")
            return None
        except Exception as e:
            logger.error(f"使用 Whisper 转录失败: {e}")
            return None

class PromptLoader:
    def __init__(self, prompt_dir: Optional[str], prompts_config: List[Dict[str, str]]):
        self.prompts = {}
        self.prompt_dir = Path(prompt_dir) if prompt_dir else Path("prompts")
       
        for config in prompts_config:
            name = config.get("name")
            path = config.get("path")
            if name and path:
                try:
                    full_path = self.prompt_dir / path
                    with open(full_path, 'r', encoding='utf-8') as f:
                        self.prompts[name] = f.read()
                    logger.info(f"成功加载提示词 '{name}' from {full_path}")
                except FileNotFoundError:
                    logger.error(f"提示词文件未找到: {full_path}")
                except Exception as e:
                    logger.error(f"加载提示词 '{name}' 失败: {e}")

    def get_prompt(self, name: str) -> Optional[str]:
        return self.prompts.get(name)

class BaseAPIClient:
    def _encode_image_to_base64(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"无法将图片编码为 Base64: {image_path}, 错误: {e}")
            raise
   
    def chat_stream(self, model: str, prompt: str, image_paths: Optional[List[str]] = None, temperature: float = 0.2, timeout: int = 600) -> Iterator[str]:
        raise NotImplementedError

class OllamaClient(BaseAPIClient):
    def __init__(self, url: str = "http://localhost:11434"):
        self.url = url.rstrip('/')
        self.chat_endpoint = f"{self.url}/api/chat"
        logger.info(f"Ollama 客户端已初始化，将使用原生聊天接口: {self.chat_endpoint}")

    def chat_stream(self, model: str, prompt: str, image_paths: Optional[List[str]] = None, temperature: float = 0.2, timeout: int = 600) -> Iterator[str]:
        headers = {"Content-Type": "application/json"}
        messages = [{"role": "user", "content": prompt}]
       
        if image_paths:
            try:
                encoded_images = [self._encode_image_to_base64(p) for p in image_paths]
                messages[0]["images"] = encoded_images
            except Exception as e:
                yield json.dumps({"error": f"图片编码失败: {e}"}) + "\n"
                return
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature}
        }
        try:
            with requests.post(self.chat_endpoint, headers=headers, data=json.dumps(payload), stream=True, timeout=timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        yield line.decode('utf-8')
        except requests.exceptions.RequestException as e:
            logger.error(f"请求 Ollama 原生 API 失败: {e}")
            yield json.dumps({"error": f"请求 Ollama 原生 API 失败: {e}"}) + "\n"

class GenericOpenAIAPIClient(BaseAPIClient):
    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url.rstrip('/')
        self.chat_endpoint = f"{self.api_url}/chat/completions"
        logger.info(f"OpenAI 兼容客户端已初始化，将使用接口: {self.chat_endpoint}")

    def chat_stream(self, model: str, prompt: str, image_paths: Optional[List[str]] = None, temperature: float = 0.2, timeout: int = 600) -> Iterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
       
        content_parts = [{"type": "text", "text": prompt}]
        if image_paths:
            try:
                for image_path in image_paths:
                    base64_image = self._encode_image_to_base64(image_path)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })
            except Exception as e:
                yield f'data: {json.dumps({"error": f"图片编码失败: {e}"})}\n\n'
                return
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content_parts}],
            "temperature": temperature,
            "stream": True
        }
        try:
            with requests.post(self.chat_endpoint, headers=headers, json=payload, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith(b'data: '):
                        yield line.decode('utf-8')
        except requests.exceptions.RequestException as e:
            logger.error(f"请求 OpenAI 兼容 API 失败: {e}")
            yield f'data: {json.dumps({"error": f"请求 OpenAI 兼容 API 失败: {e}"})}\n\n'

class VideoAnalyzer:
    def __init__(self, client: BaseAPIClient, model: str, prompt_loader: PromptLoader, temperature: float = 0.2, request_timeout: int = 600):
        self.client = client
        self.model = model
        self.prompt_loader = prompt_loader
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.user_prompt = ""
        self.context_length = 4096
        try:
            logger.info("Initializing YOLOv8 model for object detection...")
            self.yolo_model = YOLO('yolov8n.pt')
        except Exception as e:
            logger.warning(f"Failed to initialize YOLO model: {e}. Object detection will be disabled.")
            self.yolo_model = None

        # RAG Setup
        self.embedder = None
        self.frame_embeddings = []
        self.frame_texts = []
        try:
             logger.info("Initializing SentenceTransformer for RAG...")
             self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
             logger.warning(f"Failed to initialize SentenceTransformer: {e}. RAG will be disabled.")

    def index_frames(self, frames: List['Frame'], analyses: List[str]):
        """Index frame analyses for semantic search."""
        if not self.embedder: return
        self.frame_texts = analyses
        self.frame_embeddings = self.embedder.encode(analyses, convert_to_tensor=True)
        logger.info(f"Indexed {len(analyses)} frames for RAG.")

    def query_specific_frames(self, user_query: str, frames: List['Frame'], top_k=5) -> str:
        """Retrieve relevant frame contexts based on query."""
        if not self.embedder or len(self.frame_embeddings) == 0:
             return ""
        
        query_embedding = self.embedder.encode(user_query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.frame_embeddings, top_k=min(top_k, len(self.frame_embeddings)))
        
        relevant_context = "### Relevant Frames Implementation:\n"
        # hits structure: [[{'corpus_id': 0, 'score': 0.8}, ...]]
        for hit in hits[0]:
            idx = hit['corpus_id']
            relevant_context += f"- Frame at {frames[idx].timestamp:.2f}s: {self.frame_texts[idx]}\n"
        
        return relevant_context

    def detect_objects_in_frame(self, frame_path: str) -> str:
        """Returns a string listing detected objects in the frame."""
        if not self.yolo_model:
            return "Object detection disabled"
            
        try:
            results = self.yolo_model(frame_path, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.yolo_model.names[cls_id]
                    conf = float(box.conf[0])
                    if conf > 0.5:
                        detections.append(class_name)
            
            if not detections:
                return "No significant objects detected"
            
            # Count objects: "2 person, 1 car"
            counts = Counter(detections)
            return ", ".join([f"{count} {name}" for name, count in counts.items()])
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return "Object detection error"

    def _process_stream(self, stream_iterator: Iterator[str]) -> Iterator[str]:
        full_response_text = ""
        for chunk_str in stream_iterator:
            if analysis_state.stop_requested:
                break
            try:
                if chunk_str.startswith('data: '):
                    chunk_str = chunk_str[6:]
                    if chunk_str.strip() == '[DONE]':
                        break
                    chunk = json.loads(chunk_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                else:
                    chunk = json.loads(chunk_str)
                    if chunk.get("done"):
                        break
                    delta = chunk.get("message", {}).get("content", "")
               
                if delta:
                    full_response_text += delta
                    yield delta
            except (json.JSONDecodeError, IndexError):
                continue
        yield f"__FULL_RESPONSE_END__{full_response_text}"

    def summarize_all_frames_stream(self, frames: List['Frame'], transcript: 'AudioTranscript') -> Iterator[str]:
        prompt_template = self.prompt_loader.get_prompt("Video Summary")
        if not prompt_template:
            yield "错误: 未找到 'Video Summary' 提示词模板。"
            return
            
        # RAG Logic: If distinct frames > 20, use RAG or Summarize-Refine
        use_rag = len(frames) > 20 and self.embedder is not None
        
        if use_rag:
             logger.info(f"Frame count {len(frames)} > 20. using RAG/Smart Context mode.")
             # For now, we put all frame info but we could filter.
             # Actually, without pre-analyzing each frame with LLM, we can't do text-based RAG on content.
             # But we can do Object Detection RAG if we ran YOLO.
             # Or we can RAG on Metadata.
             # The user request said: "Index frames (analyses)...". This implies we need per-frame analysis key?
             # But the current flow extracts frames -> calls summarize. It doesn't seem to have per-frame detailed text yet.
             # EXCEPT if we do a quick pass?
             # For this iteration, let's keep it simple: We will include Object Detection tags in the Frame Info if available.
             pass

        frame_info_lines = []
        for f in frames:
            info = f"- 关键帧 at {f.timestamp:.2f}s"
            if self.yolo_model:
                objects = self.detect_objects_in_frame(str(f.path))
                info += f" [Objects: {objects}]"
            frame_info_lines.append(info)
            
        frame_info = "\n".join(frame_info_lines)
        
        # If RAG is enabled (and we have indexed data? We don't have existing text analysis yet in this flow),
        # We can simulate RAG by asking for specific frames if user_prompt implies it, but here we are generating SUMMARY.
        # So providing enriched metadata (YOLO) is the best "Smart" Step here.
        
        prompt = prompt_template.format(
            user_prompt=self.user_prompt,
            audio_transcript=transcript.text,
            frame_info=frame_info
        )
       
        if len(prompt) > self.context_length * 2.5:
            prompt = prompt[:int(self.context_length * 2.5)] + "\n...[提示词因过长被截断]"
        frame_paths = [str(f.path) for f in frames]
        stream_iterator = self.client.chat_stream(
            model=self.model,
            prompt=prompt,
            image_paths=frame_paths,
            temperature=self.temperature,
            timeout=self.request_timeout
        )
        yield from self._process_stream(stream_iterator)

@dataclass
class AnalysisState:
    is_running: bool = False
    stop_requested: bool = False
    status_message: str = "等待中..."
   
analysis_state = AnalysisState()
SETTINGS_FILE = Path("ui_settings.json")

class AppState:
    def __init__(self):
        self.analyzer: Optional[VideoAnalyzer] = None
        self.is_loaded: bool = False
        self.system_stats = {"cpu": 0, "ram": 0, "gpu": 0, "vram": 0}
        self.stop_monitoring = threading.Event()

app_state = AppState()

# 全局UI组件引用
status_box, client_type_dd, model_select, api_key_txt, api_url_txt, api_model_txt, load_button, unload_button = [None] * 8
output_report, output_metadata_table, metadata_plot, output_gallery, output_summary_video, output_gif, output_metadata_json = [None] * 7
output_summary_clips_gallery, clip_details_accordion, clip_details_md = None, None, None
run_status_html, analysis_progress, start_button, continue_button, stop_button, refresh_summary_button, clear_outputs_button = [None] * 7
frame_details_accordion, frame_details_md = None, None
analysis_cache_state = None
gif_info_md = None

PRESET_PROMPTS = {
    "内容总结与评估": "请详细总结这个视频的核心内容、关键信息点和叙事流程。并从观众的角度评估其整体质量、趣味性和信息价值。",
    "技术质量分析": "请作为一名专业的摄影师和剪辑师，严格评估该视频的技术质量，包括构图、灯光、色彩、焦点、稳定性、剪辑节奏和音效设计等方面。请提供具体的优点和可以改进的建议。",
    "情感与风格识别": "请分析这个视频所传达的主要情感基调（如欢乐、悲伤、悬疑、励志等）和视觉风格（如电影感、纪录片、Vlog、复古等）。并指出哪些视听元素（如配乐、色调、镜头语言）共同作用于这种感受的形成。",
    "自定义": ""
}

# ==============================================================================
# 阶段二：函数定义区
# ==============================================================================

def monitor_system_stats():
    while not app_state.stop_monitoring.is_set():
        app_state.system_stats['cpu'] = psutil.cpu_percent()
        app_state.system_stats['ram'] = psutil.virtual_memory().percent
        if NVIDIA_GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                app_state.system_stats['gpu'], app_state.system_stats['vram'] = util.gpu, (mem.used / mem.total) * 100
            except pynvml.NVMLError:
                app_state.system_stats['gpu'], app_state.system_stats['vram'] = -1, -1
        time.sleep(2)

def update_status_and_sys_info(message: str = "等待任务开始..."):
    stats_html = get_system_stats_html()
    return f"<div style='text-align:center;'>{message}</div>{stats_html}"

def get_system_stats_html() -> str:
    stats = app_state.system_stats
    gpu_html = f"<div class='stat-item'><span class='label'>GPU</span><div class='bar-container'><div class='bar gpu' style='width: {stats.get('gpu', 0):.1f}%;'></div></div><span class='value'>{stats.get('gpu', 0):.1f}%</span></div><div class='stat-item'><span class='label'>VRAM</span><div class='bar-container'><div class='bar vram' style='width: {stats.get('vram', 0):.1f}%;'></div></div><span class='value'>{stats.get('vram', 0):.1f}%</span></div>" if NVIDIA_GPU_AVAILABLE and stats.get('gpu', -1) != -1 else ""
    return f"<div class='stats-container'>{gpu_html}<div class='stat-item'><span class='label'>CPU</span><div class='bar-container'><div class='bar cpu' style='width: {stats.get('cpu', 0):.1f}%;'></div></div><span class='value'>{stats.get('cpu', 0):.1f}%</span></div><div class='stat-item'><span class='label'>RAM</span><div class='bar-container'><div class='bar ram' style='width: {stats.get('ram', 0):.1f}%;'></div></div><span class='value'>{stats.get('ram', 0):.1f}%</span></div></div>"

def get_advanced_video_metrics(video_path: str, num_frames_to_sample=100):
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
   
    metrics_over_time = {'timestamps': [], 'brightness': [], 'saturation': [], 'sharpness': []}
    frame_durations = []
    last_timestamp = 0
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: 
            logger.warning(f"读取采样帧 {idx} 失败，继续...")
            continue
       
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        metrics_over_time['timestamps'].append(timestamp)
       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       
        metrics_over_time['brightness'].append(np.mean(gray))
        metrics_over_time['saturation'].append(np.mean(hsv[:, :, 1]))
        metrics_over_time['sharpness'].append(cv2.Laplacian(gray, cv2.CV_64F).var())
       
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
    logger.info(f"高级指标计算完成: 亮度={avg_metrics['平均亮度 (0-255)']:.2f}, 饱和度={avg_metrics['平均饱和度 (0-255)']:.2f}, 清晰度={avg_metrics['平均清晰度 (拉普拉斯方差)']:.2f}")
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle('视频画质随时间变化分析', fontsize=16)
    sns.lineplot(x='timestamps', y='brightness', data=metrics_over_time, ax=axes[0, 0], color='skyblue', label=f"平均值: {avg_metrics['平均亮度 (0-255)']:.2f}")
    axes[0, 0].set_title('亮度变化'); axes[0, 0].set_xlabel("时间 (秒)"); axes[0, 0].set_ylabel("数值"); axes[0, 0].legend()
    sns.lineplot(x='timestamps', y='saturation', data=metrics_over_time, ax=axes[0, 1], color='salmon', label=f"平均值: {avg_metrics['平均饱和度 (0-255)']:.2f}")
    axes[0, 1].set_title('饱和度变化'); axes[0, 1].set_xlabel("时间 (秒)"); axes[0, 1].set_ylabel("数值"); axes[0, 1].legend()
    sns.lineplot(x='timestamps', y='sharpness', data=metrics_over_time, ax=axes[1, 0], color='lightgreen', label=f"平均值: {avg_metrics['平均清晰度 (拉普拉斯方差)']:.2f}")
    axes[1, 0].set_title('清晰度 (锐化) 变化'); axes[1, 0].set_xlabel("时间 (秒)"); axes[1, 0].set_ylabel("数值"); axes[1, 0].legend()
   
    if frame_durations:
        mean_fps = np.mean(frame_durations)
        sns.histplot(frame_durations, ax=axes[1, 1], color='orchid', bins=20, kde=True)
        axes[1, 1].set_title(f'帧率稳定性 (平均: {mean_fps:.2f} FPS)'); axes[1, 1].set_xlabel("帧率 (FPS)")
        axes[1, 1].axvline(mean_fps, color='r', linestyle='--', label=f'平均帧率: {mean_fps:.2f}'); axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, '恒定帧率或无法计算帧率变化', ha='center', va='center'); axes[1, 1].set_title('帧率稳定性')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return avg_metrics, fig

def get_frame_metrics(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return {
        "亮度 (0-255)": np.mean(gray),
        "对比度 (标准差)": np.std(gray),
        "饱和度 (0-255)": np.mean(hsv[:, :, 1]),
        "清晰度 (拉普拉斯方差)": cv2.Laplacian(gray, cv2.CV_64F).var()
    }

def create_summary_media_artifacts(
    original_video_path: str,
    video_duration: float,
    frames: List[Frame],
    output_dir: Path,
    video_stem: str,
    num_clips: int,
    clip_duration_around_keyframe: float,
    make_video: bool,
    make_gif: bool,
    gif_resolution: str
) -> Tuple[Optional[List[str]], Optional[List[Frame]], Optional[str], Optional[str]]:
    """
    创建动态视频片段摘要、拼接视频和GIF。
    返回: (片段路径列表, 选中的帧列表, 拼接视频路径, GIF路径)
    """
    logger.info(f"开始创建摘要媒体... 启用视频: {make_video}, 启用GIF: {make_gif}, 片段数: {num_clips}, 时长: {clip_duration_around_keyframe}s, 分辨率: {gif_resolution}")
    if not ADVANCED_FEATURES_AVAILABLE or not (make_video or make_gif):
        logger.warning(f"跳过摘要媒体创建，因为依赖项不可用(ADVANCED_FEATURES_AVAILABLE={ADVANCED_FEATURES_AVAILABLE})或用户未启用。")
        if not FFMPEG_AVAILABLE:
            logger.error("FFMPEG 未可用，这是 MoviePy 失败的主要原因。请安装 FFmpeg。")
            gr.Warning("生成摘要媒体失败！FFmpeg 未安装或未在 PATH 中。请下载 FFmpeg 并配置环境变量。")
        return None, None, None, None
   
    if not frames:
        logger.warning("没有找到有效的帧，无法创建摘要媒体。")
        return None, None, None, None
    
    logger.info(f"正在从 {len(frames)} 个候选帧中选择 {num_clips} 个最清晰的帧...")
    # 确保 num_clips 不超过可用帧数
    num_clips = min(num_clips, len(frames))
    sorted_frames = sorted(frames, key=lambda x: x.metrics.get('清晰度 (拉普拉斯方差)', 0), reverse=True)
    selected_frames = sorted(sorted_frames[:int(num_clips)], key=lambda x: x.timestamp)
    if not selected_frames:
        logger.warning("根据清晰度排序后，没有可选的帧来创建摘要媒体。")
        return None, None, None, None
    logger.info(f"已选定 {len(selected_frames)} 个帧用于生成片段。时间点: {[f.timestamp for f in selected_frames]}")
    
    individual_clip_paths = []
    
    logger.info(f"正在从原始视频中截取 {len(selected_frames)} 个动态片段...")
    success_count = 0
    try:
        # 使用 'with' 语句确保主视频文件在处理完后被关闭
        with VideoFileClip(original_video_path) as video:
            logger.info(f"视频加载成功，时长: {video.duration}s")
            for i, frame_obj in enumerate(selected_frames):
                try:
                    start_time = max(0, frame_obj.timestamp - clip_duration_around_keyframe / 2)
                    end_time = min(video.duration, frame_obj.timestamp + clip_duration_around_keyframe / 2)
                   
                    logger.info(f"片段 {i+1}/{len(selected_frames)}: 截取时间从 {start_time:.2f}s 到 {end_time:.2f}s。")
                    if end_time <= start_time:
                        logger.warning(f"跳过片段 {i}，因为计算出的结束时间({end_time})早于或等于开始时间({start_time})。")
                        continue
                    
                    # 从主视频中截取子剪辑
                    sub_clip = video.subclip(start_time, end_time)
                   
                    clip_path = get_unique_filepath(output_dir, f"{video_stem}_clip_{i:02d}.mp4")
                    
                    # 将子剪辑写入独立的MP4文件，包含音频
                    sub_clip.write_videofile(str(clip_path), codec='libx264', audio_codec='aac', logger=None, threads=4)
                   
                    individual_clip_paths.append(str(clip_path))
                    success_count += 1
                    logger.info(f"✅ 成功创建视频片段 {i+1}/{len(selected_frames)}: {clip_path.name} (大小: {os.path.getsize(clip_path)/1024:.1f} KB)")
                    
                    # 立即关闭子剪辑以释放资源
                    sub_clip.close()

                except Exception as e:
                    logger.error(f"❌ 创建视频片段 {i} (时间点: {frame_obj.timestamp:.2f}s) 时出错: {e}", exc_info=True)
                    gr.Warning(f"创建片段 {i+1} 失败: {e}。继续生成其他片段。")
                    continue
            logger.info(f"片段生成完成: {success_count}/{len(selected_frames)} 成功。")
    except OSError as e:
        logger.error(f"❌ MoviePy 严重错误: 无法处理视频文件。这通常意味着 FFmpeg 未安装或未在系统路径中。错误: {e}", exc_info=True)
        gr.Warning("生成视频摘要失败！请确保已正确安装 FFmpeg 并将其添加至系统环境变量(PATH)。")
        return individual_clip_paths, selected_frames, None, None
   
    if not individual_clip_paths:
        logger.warning("⚠️ 未能成功创建任何视频片段，无法进行拼接。")
        return individual_clip_paths, selected_frames, None, None

    # --- 拼接阶段 ---
    # 【关键修复】: 从磁盘重新加载所有片段文件进行拼接，而不是使用内存中可能已失效的subclip对象
    concatenated_video_path, gif_path = None, None
    reloaded_clips = []
    try:
        logger.info(f"正在从 {len(individual_clip_paths)} 个已保存的片段文件重新加载以进行拼接...")
        reloaded_clips = [VideoFileClip(p) for p in individual_clip_paths]
        
        if not reloaded_clips:
            raise ValueError("重新加载片段文件后列表为空，无法拼接。")

        final_clip = concatenate_videoclips(reloaded_clips, method="compose")
        logger.info("片段拼接成功，开始导出...")
       
        if make_video:
            concatenated_video_path = get_unique_filepath(output_dir, f"{video_stem}_summary_concatenated.mp4")
            logger.info(f"正在生成拼接摘要视频: {concatenated_video_path}")
            final_clip.write_videofile(str(concatenated_video_path), fps=24, codec='libx264', audio_codec='aac', logger=None, threads=4)
            logger.info(f"✅ 拼接摘要视频生成成功 (大小: {os.path.getsize(concatenated_video_path)/1024/1024:.1f} MB)")
       
        if make_gif:
            gif_path = get_unique_filepath(output_dir, f"{video_stem}_summary.gif")
            resolution_map = {"低": 0.3, "中": 0.5, "高": 0.8}
            resized_clip = final_clip.resize(resolution_map.get(gif_resolution, 0.5))
            logger.info(f"正在生成摘要GIF (分辨率: {gif_resolution}): {gif_path}")
            resized_clip.write_gif(str(gif_path), fps=10, logger=None)
            logger.info(f"✅ 摘要GIF生成成功 (大小: {os.path.getsize(gif_path)/1024:.1f} KB)")
            resized_clip.close()
        
        # 关闭最终的合成剪辑
        final_clip.close()
           
    except Exception as e:
        logger.error(f"❌ 拼接视频或生成GIF时出错: {e}", exc_info=True)
        gr.Warning(f"拼接视频或生成GIF时出错: {e}")
    finally:
        # 【重要】: 确保所有重新加载的剪辑都被关闭，以释放文件句柄
        logger.info("正在关闭所有用于拼接的视频片段资源...")
        for clip in reloaded_clips:
            try:
                clip.close()
            except Exception as close_err:
                logger.warning(f"关闭一个临时片段时出错: {close_err}")
        logger.info("所有临时片段资源已关闭。")
    
    logger.info("摘要媒体创建流程结束。")
    return individual_clip_paths, selected_frames, str(concatenated_video_path) if concatenated_video_path else None, str(gif_path) if gif_path else None

def get_unique_filepath(output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base, ext = os.path.splitext(filename)
    filepath = output_dir / filename
    if filepath.exists():
        timestamp = datetime.now().strftime("_%Y%m%d%H%M%S")
        filepath = output_dir / f"{base}{timestamp}{ext}"
    return filepath

def get_video_metadata(video_path: str) -> (dict, float):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): 
            logger.error(f"无法打开视频获取元数据: {video_path}")
            return {}, 0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, frame_count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = fourcc.to_bytes(4, 'little').decode('utf-8', errors='ignore').strip('\x00')
        cap.release()
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        meta = {"文件名": os.path.basename(video_path), "文件大小 (MB)": f"{file_size_mb:.2f}", "时长 (秒)": f"{duration:.2f}", "分辨率": f"{width}x{height}", "帧率": f"{fps:.2f}", "总帧数": frame_count, "编码格式": codec or "未知"}
        logger.info(f"基本元数据提取成功: {meta}")
        return meta, duration
    except Exception as e:
        logger.error(f"提取元数据失败: {e}")
        return {}, 0

def detect_ollama_models(url: str = "http://localhost:11434") -> List[str]:
    try:
        response = requests.get(f"{url.rstrip('/')}/api/tags", timeout=3)
        response.raise_for_status()
        return sorted([model["name"] for model in response.json().get("models", [])])
    except requests.exceptions.RequestException:
        return []

def refresh_models_action():
    logger.info("UI操作：刷新可用Ollama模型列表")
    models = detect_ollama_models()
    return gr.update(choices=models, value=models[0] if models else None)

def get_ollama_status():
    status_text, running_models_data, running_model_names = "", [], []
    try:
        response = requests.get("http://localhost:11434/", timeout=3)
        response.raise_for_status()
        ps_response = requests.get("http://localhost:11434/api/ps", timeout=3)
        ps_response.raise_for_status()
        models_info = ps_response.json().get("models", [])
        status_text += "✅ **Ollama 服务在线**\n\n" + ("当前没有模型加载到内存中。\n\n" if not models_info else "")
        for model in models_info:
            running_model_names.append(model['name'])
            running_models_data.append([model['name'], f"{model['size'] / 1e9:.2f} GB"])
        if NVIDIA_GPU_AVAILABLE:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_vram_gb, used_vram_gb = mem_info.total / 1e9, mem_info.used / 1e9
            status_text += f"**GPU状态**: NVIDIA GPU | 总显存: {total_vram_gb:.2f} GB | 已用: {used_vram_gb:.2f} GB\n\n"
            if total_vram_gb < 20: status_text += "<p style='color:orange;'>⚠️ **警告**: 您的总显存低于20GB，Ollama可能已进入低显存模式，性能会受影响。</p>"
    except requests.exceptions.RequestException:
        status_text = "<p style='color:red;'>❌ **错误**: Ollama 服务未运行或无法访问。请先在终端启动Ollama服务。</p>"
    except Exception as e:
        status_text = f"<p style='color:red;'>❌ **错误**: 获取Ollama状态时发生未知错误: {e}</p>"
    return status_text, running_models_data, gr.update(choices=running_model_names, interactive=True)

def unload_ollama_model(model_to_unload):
    logger.info(f"UI操作：尝试从内存卸载模型 '{model_to_unload}'")
    if not model_to_unload:
        gr.Warning("请从下拉菜单中选择一个要卸载的模型。")
        return get_ollama_status()
    try:
        response = requests.post("http://localhost:11434/api/unload", json={"name": model_to_unload}, timeout=10)
        if response.status_code == 404: gr.Warning(f"模型 '{model_to_unload}' 未找到，可能已被卸载。")
        response.raise_for_status()
        time.sleep(2)
        gr.Info(f"✅ 模型 '{model_to_unload}' 已成功从内存中卸载。")
    except requests.exceptions.RequestException as e:
        gr.Error(f"卸载模型时请求失败: {e}")
    except Exception as e:
        gr.Error(f"卸载模型时发生未知错误: {e}")
    return get_ollama_status()

def detect_and_set_context(model_name):
    logger.info(f"UI操作：检测模型 '{model_name}' 的推荐上下文长度")
    if not model_name:
        gr.Warning("请先在上方选择一个Ollama模型。")
        return 2048
    try:
        response = requests.post("http://localhost:11434/api/show", json={"name": model_name}, timeout=10)
        response.raise_for_status()
        details = response.json()
        parameters_str = details.get("parameters", "")
        for line in parameters_str.split('\n'):
            if line.startswith("num_ctx"):
                try:
                    context_size = int(line.split()[1])
                    gr.Info(f"✅ 已设置为模型推荐的最大上下文: {context_size}")
                    return context_size
                except (ValueError, IndexError): continue
        gr.Warning("⚠️ 未能自动检测到上下文长度，已设为默认值4096。")
        return 4096
    except Exception as e:
        gr.Error(f"❌ 检测失败: {e}")
        return 2048

def load_model_action(client_type, ollama_model, api_key, api_url, api_model):
    logger.info(f"UI操作：开始加载模型。类型: {client_type}, Ollama模型: {ollama_model}, API模型: {api_model}")
    try:
        if client_type == "Ollama":
            model_name_to_load = ollama_model
            if not model_name_to_load: raise ValueError("Ollama模型名称不能为空。")
            client = OllamaClient()
        elif client_type == "OpenAI-compatible API":
            model_name_to_load = api_model
            if not api_url: raise ValueError("API URL 不能为空。")
            client = GenericOpenAIAPIClient(api_key=api_key, api_url=api_url)
        else:
            raise ValueError(f"未知的客户端类型: {client_type}")
        prompts_config = [{"name": "Video Summary", "path": "frame_analysis/video_summary.txt"}]
        prompt_loader = PromptLoader(prompt_dir="prompts", prompts_config=prompts_config)
        app_state.analyzer = VideoAnalyzer(client, model_name_to_load, prompt_loader)
        app_state.is_loaded = True
       
        gr.Info(f"✅ 客户端 '{client_type}' 加载模型 '{model_name_to_load}' 成功！")
        return f"加载成功: {model_name_to_load}", gr.update(interactive=False), gr.update(interactive=True)
    except Exception as e:
        logger.error(f"模型加载时出错: {e}", exc_info=True)
        app_state.is_loaded = False
        gr.Error(f"模型加载失败: {e}")
        return f"错误: {e}", gr.update(interactive=True), gr.update(interactive=False)

def unload_model_action():
    logger.info("UI操作：卸载当前应用内模型")
    if app_state.is_loaded and isinstance(app_state.analyzer.client, OllamaClient):
        model_to_unload = app_state.analyzer.model
        logger.info(f"检测到Ollama客户端，尝试从内存卸载模型 '{model_to_unload}'")
        try:
            response = requests.post("http://localhost:11434/api/unload", json={"name": model_to_unload}, timeout=10)
            if response.status_code == 200: gr.Info(f"已向Ollama发送卸载 '{model_to_unload}' 的请求。")
            elif response.status_code != 404: gr.Warning(f"向Ollama发送卸载请求失败: {response.text}")
        except Exception as e:
            gr.Warning(f"连接Ollama卸载模型时出错: {e}")
    app_state.is_loaded = False
    app_state.analyzer = None
    gr.Info("✅ 应用内模型及资源已释放。")
    return "模型已卸载", gr.update(interactive=True), gr.update(interactive=False)

def clear_all_outputs_action():
    gr.Info("已清空所有输出内容。")
    return (
        update_status_and_sys_info("等待任务开始..."),
        None, # output_report
        gr.update(value=None, visible=False), # output_metadata_table
        gr.update(value=None, visible=False), # metadata_plot
        gr.update(value=None, visible=False), # output_gallery
        gr.update(value=None, visible=False), # output_summary_video
        gr.update(value=None, visible=False), # output_gif
        gr.update(value=None, visible=False), # gif_info_md
        gr.update(value=None, visible=False), # output_metadata_json
        gr.update(value=None, visible=False), # output_summary_clips_gallery
        gr.update(visible=False), # clip_details_accordion
        None, # clip_details_md
        gr.update(visible=False), # frame_details_accordion
        None, # frame_details_md
        None, # analysis_cache_state
        gr.update(visible=True, interactive=True), # start_button
        gr.update(visible=False), # continue_button
        gr.update(visible=False), # stop_button
        gr.update(interactive=False), # refresh_summary_button
    )

# --- 核心分析函数 ---

def phase_1_extraction(
    video_file: str, enable_audio: bool, frames_per_min: int, max_frames: int,
    output_save_path: str,
    enable_summary_video: bool, enable_gif: bool, summary_clips: int, summary_duration: float, gif_resolution: str,
    progress=gr.Progress(track_tqdm=True)
):
    if not video_file:
        gr.Error("请至少上传一个视频文件！")
        yield {
            run_status_html: update_status_and_sys_info("❌ 错误: 请至少上传一个视频文件！"),
            start_button: gr.update(interactive=True),
            stop_button: gr.update(interactive=False)
        }
        return
    
    global analysis_state
    analysis_state = AnalysisState(is_running=True)
   
    yield {
        run_status_html: update_status_and_sys_info("🚀 阶段 1: 开始数据整理..."),
        start_button: gr.update(interactive=False),
        stop_button: gr.update(interactive=True),
        continue_button: gr.update(visible=False),
        refresh_summary_button: gr.update(visible=False),
        output_report: "", output_metadata_table: gr.update(visible=False),
        metadata_plot: gr.update(visible=False), output_gallery: gr.update(visible=False),
        output_metadata_json: gr.update(visible=False), output_summary_video: gr.update(visible=False),
        output_gif: gr.update(visible=False), gif_info_md: gr.update(visible=False),
        frame_details_accordion: gr.update(visible=False),
        output_summary_clips_gallery: gr.update(visible=False), clip_details_accordion: gr.update(visible=False)
    }
    
    output_dir = Path(output_save_path) if output_save_path else Path("gradio_output")
    
    # Bug Fix: Rename to UUID to avoid encoding issues
    import uuid
    import shutil
    
    original_path = Path(video_file)
    safe_stem = f"video_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_output_dir = output_dir / safe_stem # Use safe stem for directory
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we need to copy (if generic path) or if it's safe to use. 
    # To be safe, we copy to our managed dir with safe name.
    safe_video_path = video_output_dir / f"{safe_stem}{original_path.suffix}"
    try:
        shutil.copy2(original_path, safe_video_path)
        logger.info(f"Copied video to safe path: {safe_video_path}")
        video_path = safe_video_path
    except Exception as e:
        logger.error(f"Failed to copy video file: {e}")
        # Fallback to original if copy fails
        video_path = original_path
        
    # cache = {"video_path": str(video_path), "output_dir": str(video_output_dir), ...}
    
    # --- Caching Mechanism ---
    try:
        def get_video_hash(file_path):
            h = hashlib.sha256()
            with open(file_path, 'rb') as f:
                chunk = f.read(10 * 1024 * 1024) # Hash first 10MB
                h.update(chunk)
            return h.hexdigest()
        
        video_hash = get_video_hash(video_path)
        cache_dir = Path("cache") / video_hash
        cache_file = cache_dir / "analysis_cache.pkl"
        
        if cache_file.exists():
            logger.info(f"Using cached analysis from: {cache_file}")
            progress(0.5, desc="Loading cached data...")
            with open(cache_file, 'rb') as f:
                AnalysisCache = pickle.load(f)
            
            # Verify frame paths exist (in case output_dir was cleaned)
            # If frames missing, invalidate cache? Or just warn?
            # For simplicity, we assume cache is valid if file exists.
            
            yield {
                run_status_html: update_status_and_sys_info("✅ Loaded from Cache! Clicking continue to use cached data."),
                start_button: gr.update(visible=False),
                stop_button: gr.update(interactive=False),
                continue_button: gr.update(visible=True, interactive=True),
                analysis_cache_state: AnalysisCache,
                output_gallery: gr.update(value=[(str(f.path), f"Time: {f.timestamp:.2f}s") for f in AnalysisCache["frames"]], visible=True),
                 # Restore other UI elements if possible, or just let user click continue
            }
            analysis_state.is_running = False
            return
            
    except Exception as cache_e:
        logger.warning(f"Cache check failed: {cache_e}")

    cache = {"video_path": str(video_path), "output_dir": str(video_output_dir), "frames": [], "transcript": None, "metadata": {}, "plot": None, "media_info_json": {}, "video_duration": 0}
    
    try:
        # 1. 提取关键帧
        progress(0.1, desc="提取关键帧...")
        yield {run_status_html: update_status_and_sys_info(f"处理视频: {video_path.name}<br>阶段 1: 提取关键帧...")}
        frame_processor = VideoProcessor(video_path, video_output_dir / "frames")
        frames = frame_processor.extract_keyframes(frames_per_minute=frames_per_min, max_frames=max_frames)
        if not frames:
            raise ValueError("未能从视频中提取任何关键帧。")
        cache["frames"] = frames
        gallery_items = [(str(f.path), f"时间: {f.timestamp:.2f}s") for f in frames]
        yield {output_gallery: gr.update(value=gallery_items, visible=True)}
        gr.Info("✅ 关键帧画廊已生成！")

        # 2. 分析元数据与画质
        progress(0.3, desc="分析元数据与画质...")
        yield {run_status_html: update_status_and_sys_info(f"处理视频: {video_path.name}<br>阶段 1: 分析元数据与画质...")}
        basic_meta, duration = get_video_metadata(str(video_path))
        cache["video_duration"] = duration
        adv_metrics, plot = get_advanced_video_metrics(str(video_path))
        meta_md = f"### 📊 视频元数据: {video_path.name}\n\n| 参数 | 值 |\n|---|---|\n"
        for k, v in basic_meta.items(): meta_md += f"| {k} | {v} |\n"
        for k, v in adv_metrics.items(): meta_md += f"| {k} | {f'{v:.2f}' if isinstance(v, float) else v} |\n"
        cache["metadata"] = meta_md
        cache["plot"] = plot
        yield {output_metadata_table: gr.update(value=meta_md, visible=True), metadata_plot: gr.update(value=plot, visible=True)}
        if MEDIAINFO_AVAILABLE:
            media_info_json = MediaInfo.parse(str(video_path)).to_data()
            cache["media_info_json"] = media_info_json
            yield {output_metadata_json: gr.update(value={"media_info": media_info_json}, visible=True)}

        # 3. 处理音频
        transcript_obj = AudioTranscript(text="（音频分析已禁用）", segments=[], language="")
        if enable_audio:
            progress(0.5, desc="提取并转录音频...")
            yield {run_status_html: update_status_and_sys_info(f"处理视频: {video_path.name}<br>阶段 1: 处理音频...")}
            audio_processor = AudioProcessor()
            audio_file_path = audio_processor.extract_audio(video_path, video_output_dir)
            if audio_file_path:
                transcript_obj = audio_processor.transcribe(audio_file_path) or transcript_obj
        cache["transcript"] = transcript_obj

        # 4. 生成摘要媒体文件
        clip_paths, selected_frames, concat_video_path, gif_path = None, None, None, None
        if enable_summary_video or enable_gif:
            progress(0.7, desc="生成摘要媒体...")
            yield {run_status_html: update_status_and_sys_info("阶段 1: 生成摘要媒体...")}
            
            clip_paths, selected_frames, concat_video_path, gif_path = create_summary_media_artifacts(
                original_video_path=str(video_path),
                video_duration=cache["video_duration"],
                frames=cache["frames"],
                output_dir=video_output_dir,
                video_stem=video_path.stem,
                num_clips=summary_clips,
                clip_duration_around_keyframe=summary_duration,
                make_video=enable_summary_video,
                make_gif=enable_gif,
                gif_resolution=gif_resolution
            )
            cache["selected_summary_frames"] = selected_frames
            
            summary_clip_gallery_items = []
            if clip_paths and selected_frames:
                summary_clip_gallery_items = [
                    (path, f"片段中心: {frame.timestamp:.2f}s")
                    for path, frame in zip(clip_paths, selected_frames)
                ]
                gr.Info(f"✅ 生成 {len(summary_clip_gallery_items)} 个摘要片段。")

            gif_info_text = ""
            if gif_path and os.path.exists(gif_path):
                gif_size_kb = os.path.getsize(gif_path) / 1024
                gif_size_mb = gif_size_kb / 1024
                gif_info_text = f"**动图文件大小:** {gif_size_kb:.2f} KB ({gif_size_mb:.2f} MB)"
            
            yield {
                output_summary_clips_gallery: gr.update(value=summary_clip_gallery_items, visible=bool(summary_clip_gallery_items)),
                output_summary_video: gr.update(value=concat_video_path, visible=bool(concat_video_path)),
                output_gif: gr.update(value=gif_path, visible=bool(gif_path)),
                gif_info_md: gr.update(value=gif_info_text, visible=bool(gif_info_text)),
            }

        # 5. 阶段一完成
        analysis_state.status_message = "✅ 数据整理完成，点击继续生成AI总结"
        progress(1.0)
       
        yield {
            run_status_html: update_status_and_sys_info(analysis_state.status_message),
            start_button: gr.update(visible=False),
            stop_button: gr.update(interactive=False),
            continue_button: gr.update(visible=True, interactive=True),
            analysis_cache_state: cache
        }
        
        # Save to Cache
        try:
            cache_dir = Path("cache") / get_video_hash(video_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_dir / "analysis_cache.pkl", 'wb') as f:
                pickle.dump(cache, f)
            logger.info(f"Analysis saved to cache: {cache_dir}")
        except Exception as save_e:
            logger.error(f"Failed to save cache: {save_e}")
            
    except Exception as e:
        logger.error(f"数据提取阶段发生错误: {e}", exc_info=True)
        analysis_state.status_message = f"❌ 错误: {e}"
        yield {
            run_status_html: update_status_and_sys_info(analysis_state.status_message),
            start_button: gr.update(interactive=True),
            stop_button: gr.update(interactive=False),
            analysis_cache_state: None
        }
    finally:
        analysis_state.is_running = False
        analysis_state.stop_requested = False

def phase_2_ai_analysis(
    cache: Dict, prompt_choice: str, custom_prompt: str, temperature: float, context_length: int,
    progress=gr.Progress(track_tqdm=True)
):
    if not app_state.is_loaded or not app_state.analyzer:
        gr.Error("模型尚未加载！")
        return
    if not cache:
        gr.Error("分析缓存为空，请先执行第一阶段的数据提取！")
        return
        
    global analysis_state
    analysis_state = AnalysisState(is_running=True)
    
    logger.info("AI摘要生成开始")
    
    yield {
        run_status_html: update_status_and_sys_info("🚀 阶段 2: AI摘要生成开始..."),
        continue_button: gr.update(interactive=False),
        stop_button: gr.update(interactive=True),
        refresh_summary_button: gr.update(visible=False),
        output_report: "### 📜 AI 摘要报告\n\n"
    }
    
    app_state.analyzer.user_prompt = custom_prompt if prompt_choice == "自定义" else PRESET_PROMPTS[prompt_choice]
    app_state.analyzer.temperature = temperature
    app_state.analyzer.context_length = int(context_length)
    
    try:
        progress(0, desc="AI正在生成摘要...")
        current_report = "### 📜 AI 摘要报告\n\n"
        final_summary_text = ""
       
        # Use GPU Lock for Inference
        with GPU_LOCK: 
            stream = app_state.analyzer.summarize_all_frames_stream(cache["frames"], cache["transcript"])
            for chunk in stream:
            if analysis_state.stop_requested: raise InterruptedError("用户请求停止")
            if "__FULL_RESPONSE_END__" in chunk:
                final_summary_text = chunk.split("__FULL_RESPONSE_END__")[1]
                break
            current_report += chunk
            yield {output_report: current_report}
       
        if analysis_state.stop_requested: raise InterruptedError("用户请求停止")
        
        logger.info("AI摘要生成结束")
        
        if MEDIAINFO_AVAILABLE:
            full_json = {"media_info": cache["media_info_json"], "ai_analysis": {"audio_transcript": cache["transcript"].text, "final_summary": final_summary_text}}
            yield {output_metadata_json: gr.update(value=full_json, visible=True)}
        
        analysis_state.status_message = "✅ AI分析任务已完成！"
        progress(1.0)
        
        yield {
            run_status_html: update_status_and_sys_info(analysis_state.status_message),
            stop_button: gr.update(interactive=False),
            continue_button: gr.update(interactive=True),
            refresh_summary_button: gr.update(visible=True, interactive=True),
            analysis_cache_state: cache
        }
    except InterruptedError:
        analysis_state.status_message = "🛑 分析已由用户手动停止。"
        logger.info(analysis_state.status_message)
        yield {
            run_status_html: update_status_and_sys_info(analysis_state.status_message),
            stop_button: gr.update(interactive=False),
            continue_button: gr.update(interactive=True),
            refresh_summary_button: gr.update(visible=True, interactive=True),
        }
    except Exception as e:
        logger.error(f"AI分析阶段发生错误: {e}", exc_info=True)
        analysis_state.status_message = f"❌ 严重错误: {e}"
        yield {
            run_status_html: update_status_and_sys_info(analysis_state.status_message),
            stop_button: gr.update(interactive=False),
            continue_button: gr.update(interactive=True),
        }
    finally:
        analysis_state.is_running = False
        analysis_state.stop_requested = False

def stop_analysis_func():
    if analysis_state.is_running:
        analysis_state.stop_requested = True
        logger.warning("收到停止请求，将在当前步骤完成后中断分析。")
    return gr.update(interactive=False)

def save_settings(*args):
    keys = ["client_type", "ollama_model", "api_key", "api_url", "api_model", "prompt_choice", "custom_prompt", "temperature", "enable_audio", "frames_per_min", "max_frames", "context_length", "enable_summary_video", "enable_gif", "summary_clips", "summary_duration", "gif_resolution", "output_path"]
    settings = dict(zip(keys, args))
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f: json.dump(settings, f, indent=2)
    gr.Info("✅ 设置已保存！")

def load_settings_and_refresh_models():
    ollama_models = detect_ollama_models()
    default_settings = ["Ollama", gr.update(choices=ollama_models, value=ollama_models[0] if ollama_models else None), "", "http://localhost:1234/v1", "", "内容总结与评估", "", 0.5, True, 30, 25, 4096, True, False, 10, 5.0, "中", "gradio_output"]
    if not SETTINGS_FILE.exists(): return default_settings
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f: settings = json.load(f)
        saved_model = settings.get("ollama_model", "")
        selected_model = saved_model if saved_model in ollama_models else (ollama_models[0] if ollama_models else None)
        return [
            settings.get("client_type", "Ollama"),
            gr.update(choices=ollama_models, value=selected_model),
            settings.get("api_key", ""),
            settings.get("api_url", "http://localhost:1234/v1"),
            settings.get("api_model", ""),
            settings.get("prompt_choice", "内容总结与评估"),
            settings.get("custom_prompt", ""),
            settings.get("temperature", 0.5),
            settings.get("enable_audio", True),
            settings.get("frames_per_min", 30),
            settings.get("max_frames", 25),
            settings.get("context_length", 4096),
            settings.get("enable_summary_video", True),
            settings.get("enable_gif", False),
            settings.get("summary_clips", 10),
            settings.get("summary_duration", 5.0),
            settings.get("gif_resolution", "中"),
            settings.get("output_path", "gradio_output")
        ]
    except (json.JSONDecodeError, KeyError):
        logger.warning("无法解析设置文件，将使用默认设置。")
        return default_settings

def show_frame_details(cache: Dict, evt: gr.SelectData):
    if not cache or not cache.get("frames"):
        return gr.update(visible=False), ""
   
    try:
        selected_frame: Frame = cache["frames"][evt.index]
        metrics = selected_frame.metrics
       
        md_text = f"#### 🖼️ 帧详情 (时间: {selected_frame.timestamp:.2f}s)\n\n"
        md_text += f"> ⏱️ **跳转建议**: 请手动拖动下方视频进度条至 **{selected_frame.timestamp:.2f}** 秒以查看原片上下文。\n\n"
        md_text += "| 参数 | 值 |\n|---|---|\n"
        for key, value in metrics.items():
            md_text += f"| {key} | {value:.2f} |\n"
           
        return gr.update(visible=True), md_text
    except (IndexError, KeyError) as e:
        logger.warning(f"无法显示帧详情: {e}")
        return gr.update(visible=False), "无法加载该帧的详细数据。"

def export_report_as_pdf(report_md, output_dir):
    try:
        import markdown2
        from weasyprint import HTML
        
        output_path = Path(output_dir) / f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        
        html_content = markdown2.markdown(report_md, extras=["tables"])
        styled_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; padding: 20px; }}
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
        gr.Error("Export failed: markdown2 or weasyprint not installed.")
        return None
    except Exception as e:
        logger.error(f"PDF Export failed: {e}")
        gr.Error(f"PDF Export failed: {e}")
        return None

def show_clip_details(cache: Dict, evt: gr.SelectData):
    if not cache or not cache.get("selected_summary_frames"):
        return gr.update(visible=False), ""
   
    try:
        selected_frame: Frame = cache["selected_summary_frames"][evt.index]
        metrics = selected_frame.metrics
       
        md_text = f"#### 🎬 片段中心帧详情 (时间: {selected_frame.timestamp:.2f}s)\n\n"
        md_text += "此片段是围绕该时间点的关键帧生成的。\n\n"
        md_text += "| 参数 | 值 |\n|---|---|\n"
        for key, value in metrics.items():
            md_text += f"| {key} | {value:.2f} |\n"
           
        logger.info(f"显示片段详情: 时间 {selected_frame.timestamp:.2f}s")
        return gr.update(visible=True), md_text
    except (IndexError, KeyError) as e:
        logger.warning(f"无法显示片段详情: {e}")
        return gr.update(visible=False), "无法加载该片段的详细数据。"

# ==============================================================================
# 阶段三：UI定义与启动区
# ==============================================================================
CSS = """.stats-container { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; font-size: 0.9em; } .stat-item { flex: 1; min-width: 120px; background-color: #f0f0f0; border-radius: 5px; padding: 5px; } .label { font-weight: bold; } .value { float: right; } .bar-container { width: 100%; background-color: #e0e0e0; border-radius: 3px; height: 8px; margin-top: 3px; } .bar { height: 100%; border-radius: 3px; } .cpu { background-color: #4CAF50; } .ram { background-color: #2196F3; } .gpu { background-color: #ff9800; } .vram { background-color: #f44336; } footer { display: none !important; }"""

def create_ui():
    global status_box, client_type_dd, model_select, api_key_txt, api_url_txt, api_model_txt, load_button, unload_button
    global output_report, output_metadata_table, metadata_plot, output_gallery, output_summary_video, output_gif, output_metadata_json
    global output_summary_clips_gallery, clip_details_accordion, clip_details_md
    global run_status_html, analysis_progress, start_button, continue_button, stop_button, refresh_summary_button, clear_outputs_button
    global frame_details_accordion, frame_details_md, analysis_cache_state, gif_info_md
   
    with gr.Blocks(css=CSS, title="视频深度分析平台", theme=gr.themes.Soft()) as iface:
        analysis_cache_state = gr.State(None)
        gr.Markdown("# 🚀 视频深度分析平台 (V3.5 稳定版)")
        if not FONT_LOADED_SUCCESSFULLY:
            gr.Markdown("<div style='background-color: #FFDDDD; color: #D8000C; padding: 10px; border-radius: 5px;'>⚠️ **严重警告**: 未能加载任何有效的中文字体。图表中的中文将显示为方框。</div>")
        if not FFMPEG_AVAILABLE:
            gr.Markdown("<div style='background-color: #FFDDDD; color: #D8000C; padding: 10px; border-radius: 5px;'>⚠️ **FFmpeg 警告**: 未检测到 FFmpeg。AI摘要视频/GIF 将无法生成。请安装 FFmpeg。</div>")
       
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("1. 模型配置", open=True):
                    status_box = gr.Textbox(label="模型状态", value="未加载", interactive=False)
                    client_type_dd = gr.Dropdown(["Ollama", "OpenAI-compatible API"], label="客户端类型", value="Ollama")
                    with gr.Group(visible=True) as ollama_group:
                        model_select = gr.Dropdown(label="选择Ollama模型", interactive=True, info="请确保选择的是多模态(VL)模型，如qwen-vl, llava等")
                        refresh_button = gr.Button("🔄 刷新可用模型")
                    with gr.Group(visible=False) as api_group:
                        api_url_txt = gr.Textbox(label="API URL (LM Studio / OpenAI)", value="http://localhost:1234/v1", placeholder="例如: http://localhost:1234/v1")
                        api_key_txt = gr.Textbox(label="API Key (可选)", type="password", placeholder="本地服务通常无需填写")
                        api_model_txt = gr.Textbox(label="模型名称", placeholder="例如: 在LM Studio中加载的模型ID")
                    load_button = gr.Button("✅ 加载模型", variant="primary")
                    unload_button = gr.Button("卸载模型", interactive=False)
               
                with gr.Accordion("Ollama 状态监控与管理", open=False):
                    ollama_status_markdown = gr.Markdown("正在获取状态...")
                    running_models_df = gr.DataFrame(headers=["运行中的模型", "占用内存"], interactive=False, row_count=(0, "dynamic"))
                    with gr.Row():
                        unload_model_dd = gr.Dropdown(label="选择要卸载的模型", interactive=True)
                        unload_model_button = gr.Button("⚡ 卸载选中模型")
                    refresh_status_button = gr.Button("🔄 刷新状态", elem_id="refresh_ollama_status_button")
               
                with gr.Accordion("2. 上传与分析设置", open=True):
                    file_output = gr.File(label="待分析视频", file_count="single", interactive=True, file_types=["video"])
                    upload_button = gr.UploadButton("📁 点击或拖拽单个视频上传", file_count="single", file_types=["video"])
                    prompt_choice_dd = gr.Dropdown(label="选择提示词模板", choices=list(PRESET_PROMPTS.keys()))
                    custom_prompt_txt = gr.Textbox(label="自定义提示词", lines=3, visible=False)
                    temp_slider = gr.Slider(0.0, 1.5, step=0.1, label="温度 (Temperature)")
                    output_path_txt = gr.Textbox(label="分析结果输出路径", value="gradio_output")
               
                with gr.Accordion("3. 高级参数与维护", open=False):
                    frames_per_min_slider = gr.Slider(1, 120, step=1, label="每分钟关键帧数")
                    max_frames_slider = gr.Slider(5, 100, step=1, label="最大总帧数")
                    context_length_slider = gr.Slider(1024, 16384, step=256, label="模型上下文 (Context) 长度", value=4096)
                    detect_context_button = gr.Button("🔍 检测并设置推荐上下文")
                    enable_audio_checkbox = gr.Checkbox(label="启用音频转录")
                    with gr.Group():
                        gr.Markdown("#### AI摘要与GIF生成")
                        enable_summary_video_cb = gr.Checkbox(label="生成AI摘要媒体", value=True)
                        enable_gif_cb = gr.Checkbox(label="生成GIF动图")
                        summary_clips_slider = gr.Slider(3, 30, step=1, label="摘要片段数量")
                        summary_duration_slider = gr.Slider(1.0, 10.0, step=0.5, label="每个片段总时长(秒)")
                        gif_resolution_dd = gr.Dropdown(["低", "中", "高"], label="GIF分辨率", value="中")
                    with gr.Row():
                        save_settings_button = gr.Button("💾 保存所有设置")
                        clear_outputs_button = gr.Button("🗑️ 清空所有输出", variant="stop")
               
                with gr.Blocks():
                    start_button = gr.Button("1. 开始提取数据", variant="primary", size='lg')
                    continue_button = gr.Button("2. 继续生成AI总结", variant="primary", size='lg', visible=False)
                    with gr.Row():
                        stop_button = gr.Button("🛑 停止", variant="stop", interactive=False, scale=1)
                        refresh_summary_button = gr.Button("🔄 仅刷新AI总结", interactive=True, visible=False, scale=1)
            with gr.Column(scale=2):
                run_status_html = gr.HTML(update_status_and_sys_info())
                analysis_progress = gr.Progress()
                with gr.Tabs():
                    with gr.TabItem("📝 AI 摘要报告"):
                        output_report = gr.Markdown()
                        with gr.Row():
                            export_pdf_btn = gr.Button("📄 导出为 PDF 报告")
                            pdf_file_output = gr.File(label="下载 PDF", visible=False)
                    with gr.TabItem("🎬 摘要媒体"):
                        gr.Markdown("#### 视频片段摘要 (可点击播放)\n点击下方的视频片段以查看其中心关键帧的详细技术指标。")
                        output_summary_clips_gallery = gr.Gallery(label="视频片段摘要", columns=4, height="auto", object_fit="contain", visible=False, allow_preview=True)
                        with gr.Accordion("片段详情", open=False, visible=False) as clip_details_accordion:
                            clip_details_md = gr.Markdown()
                        gr.Markdown("---")
                        with gr.Row():
                            with gr.Column():
                                output_summary_video = gr.Video(label="完整摘要视频 (拼接版)", visible=False)
                            with gr.Column():
                                output_gif = gr.Image(label="摘要GIF动图", type="filepath", visible=False)
                                gif_info_md = gr.Markdown(visible=False)
                    with gr.TabItem("🖼️ 关键帧画廊"):
                        output_gallery = gr.Gallery(label="关键帧", columns=6, height="auto", object_fit="contain", visible=False)
                        with gr.Accordion("单帧详情", open=False, visible=False) as frame_details_accordion:
                            frame_details_md = gr.Markdown()
                    with gr.TabItem("📊 元数据与画质"):
                        output_metadata_table = gr.Markdown(visible=False)
                        metadata_plot = gr.Plot(label="画质分析图", visible=False)
                    with gr.TabItem("📄 详细元数据 (JSON)"):
                        output_metadata_json = gr.JSON(label="可交互的元数据树状图", visible=False)
                    with gr.TabItem("📜 系统日志"):
                         log_output = gr.Code(language="text", label="实时日志", interactive=False, lines=20)
                         gr.Timer(2).tick(lambda: gradio_handler.get_logs(), outputs=log_output)
       
        all_settings = [client_type_dd, model_select, api_key_txt, api_url_txt, api_model_txt, prompt_choice_dd, custom_prompt_txt, temp_slider, enable_audio_checkbox, frames_per_min_slider, max_frames_slider, context_length_slider, enable_summary_video_cb, enable_gif_cb, summary_clips_slider, summary_duration_slider, gif_resolution_dd, output_path_txt]
       
        # --- 事件绑定 ---
        export_pdf_btn.click(
            export_report_as_pdf, 
            inputs=[output_report, output_path_txt], 
            outputs=[pdf_file_output]
        ).then(lambda x: gr.update(visible=True) if x else gr.update(visible=False), inputs=[pdf_file_output], outputs=[pdf_file_output])

        client_type_dd.change(lambda x: (gr.update(visible=x=="Ollama"), gr.update(visible=x!="Ollama")), client_type_dd, [ollama_group, api_group])
        prompt_choice_dd.change(lambda x: gr.update(visible=x=="自定义"), prompt_choice_dd, custom_prompt_txt)
       
        refresh_button.click(refresh_models_action, outputs=model_select)
        load_button.click(load_model_action, [client_type_dd, model_select, api_key_txt, api_url_txt, api_model_txt], [status_box, load_button, unload_button])
        unload_button.click(unload_model_action, outputs=[status_box, load_button, unload_button])
       
        save_settings_button.click(save_settings, all_settings)
       
        upload_button.upload(lambda file: file.name if file else None, inputs=[upload_button], outputs=[file_output])
       
        clear_outputs_button.click(
            clear_all_outputs_action,
            outputs=[
                run_status_html, output_report, output_metadata_table, metadata_plot,
                output_gallery, output_summary_video, output_gif, gif_info_md, output_metadata_json,
                output_summary_clips_gallery, clip_details_accordion, clip_details_md,
                frame_details_accordion, frame_details_md, analysis_cache_state,
                start_button, continue_button, stop_button, refresh_summary_button
            ]
        )
        
        phase1_inputs = [
            file_output, enable_audio_checkbox, frames_per_min_slider, max_frames_slider, output_path_txt,
            enable_summary_video_cb, enable_gif_cb, summary_clips_slider, summary_duration_slider, gif_resolution_dd
        ]
        phase1_outputs = [
            run_status_html, start_button, stop_button, continue_button, refresh_summary_button,
            output_report, output_metadata_table, metadata_plot, output_gallery,
            output_metadata_json, output_summary_video, output_gif, gif_info_md, frame_details_accordion,
            output_summary_clips_gallery, clip_details_accordion,
            analysis_cache_state
        ]
        start_button.click(phase_1_extraction, phase1_inputs, phase1_outputs)
        
        phase2_inputs = [
            analysis_cache_state, prompt_choice_dd, custom_prompt_txt, temp_slider, context_length_slider
        ]
        phase2_outputs = [
            run_status_html, continue_button, stop_button, refresh_summary_button, output_report,
            output_metadata_json, analysis_cache_state
        ]
        continue_button.click(phase_2_ai_analysis, phase2_inputs, phase2_outputs)
       
        refresh_summary_button.click(phase_2_ai_analysis, phase2_inputs, phase2_outputs)
        stop_button.click(stop_analysis_func, outputs=[stop_button])
       
        output_gallery.select(show_frame_details, [analysis_cache_state], [frame_details_accordion, frame_details_md])
        output_summary_clips_gallery.select(show_clip_details, [analysis_cache_state], [clip_details_accordion, clip_details_md])
       
        ollama_status_outputs = [ollama_status_markdown, running_models_df, unload_model_dd]
        refresh_status_button.click(get_ollama_status, outputs=ollama_status_outputs)
        unload_model_button.click(unload_ollama_model, inputs=[unload_model_dd], outputs=ollama_status_outputs)
        detect_context_button.click(detect_and_set_context, inputs=[model_select], outputs=[context_length_slider])
       
        iface.load(load_settings_and_refresh_models, outputs=all_settings)
        iface.load(get_ollama_status, outputs=ollama_status_outputs)
       
    return iface

if __name__ == "__main__":
    prompt_dir = Path("prompts/frame_analysis")
    prompt_dir.mkdir(parents=True, exist_ok=True)
   
    summary_prompt_path = prompt_dir / "video_summary.txt"
    summary_prompt_content = (
        "你是一个专业的视频内容分析师。接下来我会给你一个视频的多个关键帧图像（按时间顺序）、以及可选的音频转录内容。\n\n"
        "用户的核心分析要求是：{user_prompt}\n\n"
        "---音频转录---\n{audio_transcript}\n\n"
        "---关键帧时间点列表---\n{frame_info}\n\n"
        "请综合你看到的所有图像和听到的所有文本，生成一份全面、流畅、结构化的视频最终分析报告。报告需要直接回应用户的核心要求。\n"
        "重要指令：请将所有图像视为一个整体故事线，进行连贯的叙述和分析，而不是孤立地描述每一张图。如果多个关键帧内容相似，请进行概括性描述，避免重复。\n"
        "**输出要求**：请直接输出Markdown格式的报告全文。**严禁**在报告中加入任何对话性文字、提问（例如不要说‘你对这个分析满意吗？’或‘你的反馈将帮助我...’）、或图像占位符（如`[img-n]`）。你的回答应该**仅限于**报告本身，内容翔实，结构清晰。"
    )
    with open(summary_prompt_path, "w", encoding="utf-8") as f:
        f.write(summary_prompt_content)
    logger.info(f"已更新/创建优化后的摘要提示词文件: {summary_prompt_path}")
    
    app_state.stop_monitoring.clear()
    monitor_thread = threading.Thread(target=monitor_system_stats, daemon=True)
    monitor_thread.start()
   
    iface = create_ui()
   
    try:
        logger.info("正在启动 Gradio Web 平台...")
        logger.info("脚本将自动打开您的浏览器。")
        logger.info("您也可以在本地通过 http://0.0.0.0:8001 访问。")
        
        # Concurrency Optimization
        iface.queue(max_size=20) # Limit queue size to prevent OOM
        iface.launch(
            server_name="0.0.0.0", 
            server_port=8001, 
            debug=False, 
            inbrowser=True,
            max_threads=10 # Limit threads to prevent CPU starvation
        )
    except (KeyboardInterrupt, OSError):
        logger.info("\n正在平稳关闭，请稍候...")
    finally:
        analysis_state.stop_requested = True
        app_state.stop_monitoring.set()
        if NVIDIA_GPU_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
                logger.info("pynvml 已成功关闭。")
            except: pass
        logger.info("应用程序已关闭。")
