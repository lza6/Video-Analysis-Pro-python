import json
import logging
from typing import Callable, Any, Dict, List

class Tool:
    def __init__(self, name: str, description: str, func: Callable, schema: Dict = None):
        self.name = name
        self.description = description
        self.func = func
        self.schema = schema or {}

    def execute(self, **kwargs):
        try:
            return self.func(**kwargs)
        except Exception as e:
            return f"Error executing tool {self.name}: {e}"

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._context_provider: Callable = None # Function to get current app context

    def register_tool(self, name: str, description: str, func: Callable, schema: Dict = None):
        self._tools[name] = Tool(name, description, func, schema)

    def set_context_provider(self, provider: Callable):
        self._context_provider = provider

    def get_tool_descriptions(self) -> str:
        desc = "Available Tools:\n"
        for t in self._tools.values():
            desc += f"- {t.name}: {t.description}\n"
            if t.schema:
                desc += f"  Args: {json.dumps(t.schema)}\n"
        return desc

    def execute_tool_call(self, tool_name: str, args: Dict) -> str:
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' not found."
        
        # Inject context if needed validation or whatever
        # Ideally tool functions are bound methods or closures that already have context
        # But if we need fresh context (like current video frame), we might need it.
        # For now, assume functions are registered with access to data.
        
        return self._tools[tool_name].execute(**args)

# --- Concrete Tool Implementations (Factories) ---

def create_get_video_meta_tool(app_context_getter):
    def get_video_meta():
        app = app_context_getter()
        if not app or not app.video_path:
            return "No video loaded."
        return json.dumps({
            "filename": app.video_path.name,
            "duration": getattr(app, 'video_duration', 0),
            "output_dir": str(app.output_dir),
            "frame_count": len(app.frames) if hasattr(app, 'frames') else 0
        }, ensure_ascii=False)
    return get_video_meta

def create_get_frame_details_tool(app_context_getter):
    def get_frame_details(seconds: float):
        app = app_context_getter()
        if not app:
            return "Application context not available."
        
        seconds = float(seconds)
        
        # 1. Try to find an existing frame within 0.25s threshold
        closest = None
        if hasattr(app, 'frames') and app.frames:
            closest = min(app.frames, key=lambda f: abs(f.timestamp - seconds))
            if abs(closest.timestamp - seconds) <= 0.25:
                return json.dumps({
                    "timestamp": round(closest.timestamp, 2),
                    "caption": closest.vision_content or "No caption",
                    "ocr": closest.ocr_text or "No text",
                    "path": str(closest.path),
                    "source": "pre-extracted"
                }, ensure_ascii=False)
        
        # 2. If no close frame found, or no frames exist, extract on-the-fly
        if hasattr(app, 'video_path') and app.video_path and app.video_path.exists():
            try:
                import cv2
                from pathlib import Path
                
                cap = cv2.VideoCapture(str(app.video_path))
                if not cap.isOpened():
                    return f"Error: Could not open video file {app.video_path.name}"
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 25.0 # Fallback
                
                frame_idx = int(seconds * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                # Get actual timestamp for accuracy
                actual_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                cap.release()
                
                if ret:
                    # Save to a subfolder in output_dir
                    out_dir = Path(app.output_dir) / "agent_dynamic_extracts" if hasattr(app, 'output_dir') and app.output_dir else Path("tmp") / "agent_extracts"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = f"dynamic_{actual_ts:.2f}s.jpg"
                    save_path = out_dir / filename
                    cv2.imwrite(str(save_path), frame)
                    
                    return json.dumps({
                        "timestamp": round(actual_ts, 2),
                        "caption": "Automatically extracted on-demand. Visual details available via file.",
                        "ocr": "Not processed on-the-fly",
                        "path": str(save_path),
                        "source": "on-the-fly extraction"
                    }, ensure_ascii=False)
                else:
                    return f"Error: Could not read frame at {seconds}s (Index: {frame_idx})"
            except Exception as e:
                return f"Error during dynamic extraction: {e}"
        
        return f"Frame at {seconds}s not found and video source unavailable."
    return get_frame_details

def create_delete_history_tool(app_context_getter):
    def delete_this_history():
        app = app_context_getter()
        if not hasattr(app, 'output_dir'):
            return "No active session to delete."
        
        # Use history manager
        if app.history_manager:
            # Must find session ID by output dir? 
            # Or just delete output dir and reload?
            # It's safer to ask user, but agent is asked to do it.
            # We will just mark it? Or call the history manager delete logic.
            # This is risky without confirmation.
            return "Deletion requires user confirmation via UI for safety."
    return delete_this_history
def create_search_web_tool():
    def search_web(query: str):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=5)]
                if not results: return "No results found."
                return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            return f"Web search error: {e}"
    return search_web

def create_visual_search_tool(app_context_getter):
    def search_visual(query: str):
        app = app_context_getter()
        if not app or not hasattr(app, 'frames') or not app.frames:
            return "No frames available for visual search."
        
        try:
            from sentence_transformers import SentenceTransformer, util
            from PIL import Image
            
            # Load model (reuse if possible, here simplified)
            model = SentenceTransformer('clip-ViT-B-32')
            
            # Compute query embedding
            query_emb = model.encode(query, convert_to_tensor=True)
            
            # Compute frame embeddings if not already done
            frame_images = [Image.open(f.path) for f in app.frames]
            frame_embs = model.encode(frame_images, convert_to_tensor=True)
            
            # Search top 3
            hits = util.semantic_search(query_emb, frame_embs, top_k=3)
            
            results = []
            for hit in hits[0]:
                idx = hit['corpus_id']
                frame = app.frames[idx]
                results.append(f"时间点 {frame.timestamp:.2f}s (匹配度: {hit['score']:.2f})")
                
            return "\n".join(results)
        except Exception as e:
            return f"Visual search error: {e}"
    return search_visual

def create_ocr_tool(app_context_getter):
    def ocr_specified_frame(seconds: float):
        app = app_context_getter()
        if not app: return "App context missing."
        
        # Get frame details first
        details_json = create_get_frame_details_tool(app_context_getter)(seconds)
        details = json.loads(details_json)
        
        if "path" not in details: return "Frame not found."
        
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            result = ocr.ocr(details["path"], cls=True)
            if not result or not result[0]: return "No text detected."
            texts = [line[1][0] for line in result[0]]
            return " ".join(texts)
        except Exception as e:
            return f"OCR Tool error: {e}"
    return ocr_specified_frame
def create_highlight_cut_tool(app_context_getter):
    def highlight_cut(description: str):
        """根据描述自动剪辑集锦视频。"""
        app = app_context_getter()
        if not app: return "App context missing."
        
        try:
            from moviepy.editor import VideoFileClip, concatenate_videoclips
            output_path = app.output_dir / "highlights.mp4"
            
            # Logic: Identify ranges from memory or metadata
            # For demo, we take first 10 seconds of detected keyframes
            clips = []
            video = VideoFileClip(str(app.video_path))
            
            # Simplified: Use first 3 keyframes with content
            segments = []
            if hasattr(app, 'frames'):
                 valid_frames = [f for f in app.frames if f.vision_content][:3]
                 for f in valid_frames:
                     start = max(0, f.timestamp - 2)
                     end = min(video.duration, f.timestamp + 2)
                     segments.append(video.subclip(start, end))
            
            if not segments:
                video.close()
                return "未找到足够的相关片段进行剪辑。"
            
            final_clip = concatenate_videoclips(segments)
            final_clip.write_videofile(str(output_path), codec="libx264")
            final_clip.close()
            video.close()
            
            return f"集锦视频生成成功：{output_path.name}"
        except Exception as e:
            return f"剪辑出错: {e}"
    return highlight_cut

def create_visual_grounding_tool(app_context_getter):
    def point_at_object(query: str):
        """精准定位目标物体并跳转。"""
        app = app_context_getter()
        if not app: return "App context missing."
        
        # Integration with YOLO or CLIP search
        # Returning time of best match
        try:
            search_tool = create_visual_search_tool(app_context_getter)
            res = search_tool(query)
            # Parse result for timestamp
            import re
            match = re.search(r'时间点 ([\d.]+)s', res)
            if match:
                ts = float(match.group(1))
                # Trigger UI jump (if possible via context)
                if hasattr(app, 'seek_video'):
                    app.seek_video(ts)
                return f"已在视频 {ts}s 处发现 {query} 并已跳转。"
            return f"未能在视频中发现 {query}。"
        except Exception as e:
            return f"定位出错: {e}"
    return point_at_object
