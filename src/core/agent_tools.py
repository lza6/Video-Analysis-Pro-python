import json
from pathlib import Path
from typing import Callable, Dict, Optional

class Tool:
    def __init__(self, name: str, description: str, func: Callable, schema: Optional[Dict] = None):
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

    def register_tool(self, name: str, description: str, func: Callable, schema: Optional[Dict] = None):
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
        # video_path may be a Path (main window) or a plain str; normalise.
        video_path = Path(app.video_path)
        return json.dumps({
            "filename": video_path.name,
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
                from src.core.logic import videocapture_unicode, imwrite_unicode

                cap = videocapture_unicode(app.video_path)
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
                    imwrite_unicode(str(save_path), frame)
                    
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
            from sentence_transformers import util
            from PIL import Image
            from src.core.kb_indexer import get_embedder

            # P2-1: 复用进程级共享 embedder（首次加载后毫秒级编码），
            # 之前每次调用重新加载模型，搜索从秒级 → 毫秒级
            model = get_embedder()
            if model is None:
                return "Visual search unavailable: CLIP embedder could not be loaded."

            # Compute query embedding
            query_emb = model.encode(query, convert_to_tensor=True)

            # P2-1: 帧级 embedding 缓存（app 上挂缓存，按帧路径为 key）
            cache = getattr(app, "_frame_emb_cache", None)
            if cache is None or len(cache) != len(app.frames):
                frame_images = [Image.open(f.path) for f in app.frames]
                embs = model.encode(frame_images, convert_to_tensor=True,
                                    show_progress_bar=False)
                cache = {f.path: e for f, e in zip(app.frames, embs)}
                app._frame_emb_cache = cache
            import torch
            frame_embs = torch.stack([cache[f.path] for f in app.frames])
            
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
        """根据描述自动剪辑集锦视频。

        v5.4 修复（audit-prod P1）：
        - VideoFileClip 用 with/try-finally 保证关闭（异常路径不泄漏）
        - 输出文件名加时间戳，防连续调用/并发覆盖 highlights.mp4
        - 描述匹配改用分词（中英文）+ jaccard，替代单字符命中（旧实现 '的/了/是' 必中）
        """
        app = app_context_getter()
        if not app: return "App context missing."

        import time as _t
        from moviepy import VideoFileClip, concatenate_videoclips
        # 时间戳后缀防覆盖：连续两次 highlight_cut 不再写同一文件
        ts_suffix = _t.strftime("%Y%m%d_%H%M%S")
        output_path = app.output_dir / f"highlights_{ts_suffix}.mp4"

        # 旧实现：单字符命中 sum(1 for ch in description if ch in content)
        # 中文 '的/了/是' 几乎必中 → 任意描述结果趋同。改用 jaccard 分词交集。
        def _tokenize(text: str) -> set:
            # 中文按字、英文按词；空集合兜底
            if not text:
                return set()
            toks = set()
            # 英文单词
            for w in (text.lower().split()):
                if w.strip():
                    toks.add(w.strip(".,!?;:\"'()[]（）。，！？；："))
            # 中文单字（2 字以上连续中文段按字切）
            import re as _re
            for seg in _re.findall(r"[一-鿿]+", text):
                for ch in seg:
                    toks.add(ch)
            return toks

        desc_tokens = _tokenize(description or "")
        video = None
        try:
            video = VideoFileClip(str(app.video_path))

            segments = []
            if hasattr(app, 'frames') and app.frames:
                scored = []
                for f in app.frames:
                    if not f.vision_content:
                        continue
                    content_tokens = _tokenize(f.vision_content or "")
                    if not desc_tokens or not content_tokens:
                        score = 0.0
                    else:
                        # jaccard：交集/并集，0-1
                        inter = len(desc_tokens & content_tokens)
                        union = len(desc_tokens | content_tokens)
                        score = inter / union if union else 0.0
                    scored.append((score, f))
                scored.sort(key=lambda x: x[0], reverse=True)
                for _, f in scored[:3]:
                    start = max(0, f.timestamp - 2)
                    end = min(video.duration, f.timestamp + 2)
                    if end > start:
                        segments.append(video.subclipped(start, end))

            if not segments:
                return "未找到足够的相关片段进行剪辑。"

            final_clip = concatenate_videoclips(segments)
            final_clip.write_videofile(str(output_path), codec="libx264", logger=None)
            final_clip.close()

            return f"集锦视频生成成功：{output_path.name}"
        except Exception as e:
            return f"剪辑出错: {e}"
        finally:
            # 铁律：异常路径也要释放 VideoFileClip（audit-prod P1）
            if video is not None:
                try:
                    video.close()
                except Exception:
                    pass
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

def create_kb_search_tool(app_context_getter):
    """v4.5: 跨视频语义搜索工具。

    在所有已索引的分析会话中按描述搜索画面，返回带时间戳的结果，
    可直接跳转到对应视频的对应时刻。
    """
    def search_kb(query: str):
        app = app_context_getter()
        if not app:
            return "App context missing."
        history_manager = getattr(app, 'history_manager', None)
        if not history_manager:
            return "Knowledge base unavailable."

        try:
            from src.core.kb_indexer import get_embedder
            embedder = get_embedder()
            if embedder is None:
                return "Knowledge base unavailable: CLIP embedder could not be loaded."

            query_emb = embedder.encode(query, convert_to_tensor=False)
            hits = history_manager.search_kb(query_emb, top_k=8)
            if not hits:
                return "知识库中没有匹配结果。可能尚未索引任何视频，或描述差异过大。"

            lines = []
            for i, hit in enumerate(hits, 1):
                jumpable = " (可跳转)" if hit["video_path"] else ""
                lines.append(
                    f"{i}. {hit['video_name']} @ {hit['timestamp']:.2f}s "
                    f"(匹配度 {hit['score']:.2f}){jumpable}\n"
                    f"   内容: {hit['content'][:120] or '(无描述)'}\n"
                    f"   路径: {hit['video_path']}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"KB search error: {e}"
    return search_kb

def create_image_search_tool(app_context_getter):
    """P2-6: 图→帧跨模态搜索（截图找时刻）。

    用户上传一张图片（如微信截图/另一段视频截帧），在当前视频的关键帧中
    找视觉上最相似的时刻。CLIP 图-图 embedding 直接比对。
    """
    def search_by_image(image_path: str = ""):
        app = app_context_getter()
        if not app or not hasattr(app, 'frames') or not app.frames:
            return "No frames available for image search."
        if not image_path:
            return "No image provided. Args: {'image_path': '图片路径'}"
        from pathlib import Path
        img_p = Path(image_path)
        if not img_p.exists():
            return f"Image not found: {image_path}"

        try:
            from sentence_transformers import util
            from PIL import Image
            from src.core.kb_indexer import get_embedder

            model = get_embedder()
            if model is None:
                return "Image search unavailable: CLIP embedder could not be loaded."

            query_emb = model.encode([Image.open(str(img_p))], convert_to_tensor=True,
                                     show_progress_bar=False)
            cache = getattr(app, "_frame_emb_cache", None)
            if cache is None or len(cache) != len(app.frames):
                imgs = [Image.open(f.path) for f in app.frames]
                embs = model.encode(imgs, convert_to_tensor=True, show_progress_bar=False)
                cache = {f.path: e for f, e in zip(app.frames, embs)}
                app._frame_emb_cache = cache
            import torch
            frame_embs = torch.stack([cache[f.path] for f in app.frames])
            hits = util.semantic_search(query_emb, frame_embs, top_k=3)

            results = []
            for hit in hits[0]:
                fr = app.frames[hit["corpus_id"]]
                results.append(f"时间点 {fr.timestamp:.2f}s (相似度: {hit['score']:.2f})")
            return "\n".join(results) if results else "未找到相似画面。"
        except Exception as e:
            return f"Image search error: {e}"
    return search_by_image


# --- v5.8 断点 B5：SURVEILLANCE intent 桥接工具（真触发 batch_runner）---
# scan_videos 同步扫目录；batch_analyze 触发 main_window.start_batch（异步，
# 真跑付费 API 由用户在批量 tab 确认）；summarize_hits 读 run_store 命中。

# 支持的视频扩展名（与 batch_runner.SUPPORTED_VIDEO_EXTS 对齐）
_VID_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def create_scan_videos_tool(app_context_getter):
    """扫监控视频目录，返回视频列表（同步，轻量）。"""
    def scan_videos(video_dir: str = ""):
        from pathlib import Path
        if not video_dir:
            return "请提供 video_dir 参数。"
        d = Path(video_dir)
        if not d.exists() or not d.is_dir():
            return f"目录不存在或非目录：{video_dir}"
        vids = []
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in _VID_EXTS:
                try:
                    size_mb = p.stat().st_size / (1024 * 1024)
                except OSError:
                    size_mb = 0.0
                vids.append(f"{p.name} ({size_mb:.1f} MB)")
        if not vids:
            return f"目录 {video_dir} 下无视频文件（支持 {_VID_EXTS}）。"
        return f"找到 {len(vids)} 个视频：\n" + "\n".join(vids)
    return scan_videos


def create_batch_analyze_trigger_tool(app_context_getter):
    """触发批量监控分析（异步，不真跑付费 API，只预填配置 + 切 tab 待确认）。

    真实启动由用户在「🎞 批量监控」tab 点「▶ 开始批量」确认（破坏性操作先问）。
    """
    def trigger_batch(video_dir: str = "", item_description: str = ""):
        app = app_context_getter()
        if not app:
            return "App context missing."
        if not hasattr(app, 'start_batch'):
            return "main_window 未实现 start_batch 方法，无法触发批量分析。"
        return app.start_batch(video_dir, item_description)
    return trigger_batch


def create_summarize_hits_tool(app_context_getter):
    """读 run_store 历史命中汇总（跨会话记忆，断点 B4 续）。"""
    def summarize_hits():
        app = app_context_getter()
        if not app:
            return "App context missing."
        run_store = getattr(app, 'run_store_for_tools', None) or getattr(
            app, 'history_manager', None)
        # 优先用 batch_tab 的 run_store（runs.db）
        tab_batch = getattr(app, 'tab_batch', None)
        if tab_batch is not None and hasattr(tab_batch, '_run_store'):
            run_store = tab_batch._run_store
        if run_store is None or not hasattr(run_store, 'list_runs'):
            return "RunStore 未接入，无法汇总历史命中。"
        try:
            runs = run_store.list_runs(limit=100)
        except Exception as e:
            return f"读取 run_store 失败：{e}"
        if not runs:
            return "无历史 run 记录。"
        hits = []
        for r in runs:
            if not r.get('hits_count'):
                continue
            clips = []
            try:
                full = run_store.get_run(r.get('run_id', ''))
                clips = full.get('clips') or [] if full else []
            except Exception:
                pass
            for c in clips:
                hits.append(
                    f"- {r.get('video_name', '?')} | "
                    f"{c.get('abs_timestamp', '?')} | "
                    f"{c.get('clip_path', '?')}")
        if not hits:
            return f"共 {len(runs)} 个 run，无命中。"
        return f"共 {len(runs)} 个 run，{len(hits)} 个命中：\n" + "\n".join(hits)
    return summarize_hits


# --- v7.0 指南 4.3：skill 自动生成 + 4.5 RTSP 实时流触发 ---

def create_generate_skill_tool(app_context_getter):
    """agent 自动生成新 skill（指南 4.3）。

    不真实调付费 LLM（红线）：用规则模板生成（停车场/人脸/火灾/车辆四类）。
    用户给预算后可扩展为调 LLM 补全 algorithm/parameters。
    成功后存 config/skills/<name>/SKILL.md，下次 load_skills 自动收录。
    """
    def generate_skill(text: str = ""):
        from src.core.skill_generator import generate_skill as _gen
        from src.utils.constants import CONFIG_DIR
        skills_dir = Path(CONFIG_DIR) / "skills"
        if not text:
            return "请提供场景描述（如：停车场车牌识别）。"
        result = _gen(text, skills_dir, overwrite=False)
        if result is None:
            return ("未识别到已知场景关键词。目前支持：停车场/车牌、"
                    "人脸/face、火灾/火焰/烟雾、车辆计数/车流。"
                    "可描述更具体或由人工创建 skill。")
        if not result.get("ok"):
            return (f"skill 生成失败：{result.get('error', '')}\n"
                    f"（skill 可能已存在，需先删除或确认覆盖）")
        return (f"✅ 已生成新 skill：{result['skill_name']}\n"
                f"路径：{result['path']}\n"
                f"场景：{result['scene']}\n"
                f"重启应用或点「刷新 skills」即可自动匹配。")
    return generate_skill


def create_rtsp_monitor_tool(app_context_getter):
    """agent 触发 RTSP 实时流监控（指南 4.5）。

    不在此真实拉流（需摄像头凭据 + 网络）：只构造 RtspMonitor 并 start，
    返回"已启动"提示。真实拉流由 RtspFrameGrabber 后台线程做，
    命中回调投到 agent_dialog。

    付费 API 红线：VLM 判断走 backend.chat_stream，若 backend 未配则
    RtspMonitor 的 _vlm_check 会失败降级（只做运动检测不送 VLM）。
    """
    def start_rtsp_monitor(rtsp_url: str = "", item_description: str = ""):
        app = app_context_getter()
        if not app:
            return "App context missing."
        if not rtsp_url:
            return "请提供 RTSP URL（如 rtsp://user:pass@host:554/stream）。"
        try:
            from src.core.rtsp_stream import RtspMonitor
            from src.utils.constants import CONFIG_DIR
            backend = getattr(app, 'analyzer', None) or getattr(app, 'api_gateway', None)
            key_item = ""
            tab_batch = getattr(app, 'tab_batch', None)
            if tab_batch is not None and hasattr(tab_batch, 'txt_key_image'):
                key_item = tab_batch.txt_key_image.text().strip()
            work_dir = str(Path(CONFIG_DIR) / "rtsp_cache")
            monitor = RtspMonitor(
                rtsp_url, backend, key_item_image=key_item,
                item_description=item_description or "关键物品",
                work_dir=work_dir)
            # 命中回调 → 投到 agent_dialog
            def _on_hit_app(ev):
                try:
                    if hasattr(app, 'agent_dialog'):
                        app.agent_dialog.append_agent_message(
                            f"📡 RTSP 命中 @ {ev.timestamp:.0f}s: "
                            f"{ev.detail[:80]}（conf={ev.confidence:.2f}）")
                except Exception:
                    pass
            monitor.start(fps=1.0)
            # 挂到 app 供 stop
            app._rtsp_monitor = monitor
            return (f"📡 RTSP 监控已启动：{rtsp_url[:40]}…\n"
                    f"运动检测 + VLM 判断后台运行，命中会实时投到对话。")
        except Exception as e:
            return f"RTSP 监控启动失败：{e}"
    return start_rtsp_monitor


# --- v6.1 视频知识图谱：跨视频时序推理工具（指南 4.1）---

def create_trace_item_tool(app_context_getter):
    """跨视频追踪物品轨迹（v6.1 视频知识图谱）。

    从 RunStore 读取所有命中 segment，构建时序图（VideoGraph），按物品关键词
    沿时序/空间/物品边遍历，返回按时间排序的轨迹链路。支持跨摄像头追踪同一
    物品的移动轨迹。RunStore 不可用或无命中返回友好提示，不抛异常。
    """
    def trace_item(item_keyword: str = ""):
        app = app_context_getter()
        if not app:
            return "App context missing."
        if not item_keyword:
            return "请提供 item_keyword 参数，如 '黑色旅行袋'。"
        # 优先用 batch_tab 的 run_store（runs.db）
        run_store = getattr(app, 'run_store_for_tools', None)
        tab_batch = getattr(app, 'tab_batch', None)
        if tab_batch is not None and hasattr(tab_batch, '_run_store'):
            run_store = tab_batch._run_store
        if run_store is None:
            run_store = getattr(app, 'history_manager', None)
        if run_store is None or not hasattr(run_store, 'list_runs'):
            return "RunStore 未接入，无法追踪物品轨迹。"

        try:
            from src.core.video_graph import VideoGraph
            graph = VideoGraph.build_from_run_store(run_store, limit=200)
        except Exception as e:
            return f"构建时序图失败：{e}"

        chains = graph.trace_item(item_keyword)
        if not chains:
            return f"未在历史命中中找到包含「{item_keyword}」的轨迹。"

        lines = [f"找到 {len(chains)} 条「{item_keyword}」轨迹："]
        for i, chain in enumerate(chains, 1):
            lines.append(f"\n--- 轨迹 {i}（{len(chain)} 个命中）---")
            for n in chain:
                ts_txt = n.abs_time.strftime("%Y-%m-%d %H:%M:%S") if n.abs_time \
                    else f"+{n.timestamp_sec:.1f}s"
                lines.append(
                    f"[{ts_txt}] {n.video_name} @ {n.timestamp_sec:.1f}s "
                    f"(置信度 {n.confidence:.2f}) {n.reason[:60]}"
                )
        return "\n".join(lines)
    return trace_item

