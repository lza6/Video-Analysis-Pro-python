# 📐 Video Analysis Pro — 技术文档（Technical Documentation）

> **版本**：v4.5.0
> **维护方**：听风公司（Tingfeng）
> **文档日期**：2026-09-03
> **配套文档**：产品规格见 `docs/PRD.md`；改造路线与证据链见 `参考的结果计划指南.md`；变更日志见 `CHANGELOG.md`。

本文档描述 v4.5 已交付的技术架构、模块边界、数据流、API 接口与安全合规。Phase B 增量改造的技术方案见 `参考的结果计划指南.md` §11，本文档不重复。

---

## 0. 文档分层说明

| 层 | 范围 | 状态 | 章节 |
|----|------|------|------|
| L1 | v4.5 已交付技术架构 | ✅ 已发布 | §1–§9 |
| L2 | Phase B 增量技术方案 | ⏳ 待审批 | 见 `参考的结果计划指南.md` §11（本文档不重复，避免双源） |

---

## 1. 架构总览（Architecture Overview）

### 1.1 架构风格

采用 **MVC 变体 + 三阶段流水线 + ReAct Agent** 混合架构：

```
┌─────────────────────────────────────────────────────────────┐
│                     UI 层 (src/ui/)                         │
│   main_window(DesktopApp) + agent_panel + 10 个 Worker 类    │
│   信号槽驱动，PyQt6 QThread 保证界面不卡顿                    │
└───────────────────────┬─────────────────────────────────────┘
                        │ 信号/槽 + QThread
┌───────────────────────▼─────────────────────────────────────┐
│              核心逻辑层 (src/core/logic.py)                  │
│  VideoProcessor / AudioProcessor / VideoAnalyzer            │
│  ModelManager / ModelContextManager / PromptLoader          │
│  OllamaClient / APIGatewayClient / LMStudioClient           │
└──────────┬──────────────┬───────────────┬──────────────────┘
           │              │               │
   ┌───────▼──────┐ ┌─────▼─────┐ ┌──────▼──────────┐
   │ 工具层        │ │ 数据层    │ │ 服务层           │
   │ agent_tools   │ │ history_  │ │ server/headless │
   │ ToolRegistry  │ │ manager   │ │ (HTTP /analyze)  │
   │ 9 工具        │ │ kb_indexer│                  │
   └───────────────┘ └───────────┘ └─────────────────┘
           │              │
   ┌───────▼──────┐ ┌─────▼──────────────┐
   │ 配置/凭据    │ │ 向量库/SQLite       │
   │ config_      │ │ ChromaDB(kb_frames  │
   │ manager      │ │  + session_{id})    │
   │ keyring      │ │ SQLite(sessions/    │
   └──────────────┘ │  checkpoints)       │
                    └─────────────────────┘
```

### 1.2 分层职责

| 层 | 目录 | 职责 | 关键文件 |
|----|------|------|---------|
| UI 层 | `src/ui/` | PyQt6 界面 + QThread 异步 | `main_window.py`（2355 行，含 DesktopApp + 10 Worker）、`agent_panel.py`（516 行） |
| 核心逻辑层 | `src/core/` | 视频/音频/LLM 处理 + 模型管理 | `logic.py`（1338 行）、`agent_tools.py`（310 行）、`history_manager.py`（264 行）、`kb_indexer.py`（82 行） |
| 服务层 | `src/server/` | Headless HTTP 服务 | `headless.py`（211 行） |
| 工具层 | `src/utils/` | 常量 + 配置 + 凭据 + 安装向导 | `constants.py`、`config_manager.py`（137 行）、`ui_components.py` |
| 配置 | `config/` | INI + JSON + 提示词模板 | `app_config.ini`、`prompts.json`、`prompts/frame_analysis/` |

> 注：`main_window.py` 2355 行存在 God Object 倾向（DesktopApp 含 ~74 方法 + 10 Worker 类），Phase B P2-7 计划拆分，详见 `参考的结果计划指南.md` §8.2。

---

## 2. 核心流程图（Data Flow）

### 2.1 三阶段流水线主流程

```
用户拖拽视频
      │
      ▼
┌──────────────────────────────────────────────────────┐
│ Phase 1: 数据提取 (ExtractionWorker QThread)          │
│                                                       │
│  ┌─ VideoProcessor.extract_keyframes(density)        │
│  │    OpenCV 按 density 均匀抽帧 → List[Frame]        │
│  │    或 extract_smart_keyframes() (PySceneDetect)    │
│  │                                                     │
│  ├─ AudioProcessor.extract_audio() (ffmpeg)           │
│  │    → transcribe() (faster-whisper) → AudioTranscript│
│  │                                                     │
│  ├─ VideoAnalyzer.detect_objects_in_frame() (YOLO11) │
│  │    → frame.vision_content                            │
│  │                                                     │
│  └─ ExtractionWorker 返回 duration（v4.5 修复恒为 0）  │
│                                                       │
│  完成后: KBIndexWorker 后台把关键帧写入                │
│  ChromaDB kb_frames 全局 collection（v4.5 新增）      │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│ Phase 2: AI 分析 (AnalysisWorker QThread)             │
│                                                       │
│  ┌─ PromptLoader.get_prompt(template)                │
│  │    从 config/prompts/frame_analysis/ 加载          │
│  │                                                     │
│  ├─ VideoAnalyzer.analyze_video(frames, transcript)   │
│  │    逐帧构建提示词（帧信息 + 字幕）                   │
│  │    → client.chat_stream() 流式 token                │
│  │    → _process_stream() 过滤哨兵 __FULL_RESPONSE_END__│
│  │    → Markdown 报告                                  │
│  │                                                     │
│  └─ plot_metrics() 亮度/清晰度/饱和度趋势图           │
│      （注: v4.5 为基础折线，见 PRD NFR-3）            │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│ Phase 3: 媒体生成 (MediaWorker QThread)               │
│                                                       │
│  ┌─ create_highlight_cut_tool(description)            │
│  │    MoviePy 拼接近距离片段 → highlights.mp4         │
│  │    （注: v4.5 硬编码前 3 帧，见 PRD P1-3）         │
│  │                                                     │
│  └─ GIF 摘要生成（moviepy 2.x）                       │
└──────────────────────────────────────────────────────┘
```

### 2.2 Agent ReAct 循环

```
用户在 agent_panel 输入
      │
      ▼
ChatWorker.run (QThread)
      │
      ├─ inject_agent_system_context 拼系统提示词
      │   （注: v4.5 薄弱，仅拼 "--- System Context ---" + 元数据，见 PRD P1-1）
      │
      ▼
循环 (max_turns 上限):
      ├─ client.chat_stream(current_prompt) 流式
      ├─ 解析 LLM 输出 → 工具调用 or 最终回答
      ├─ 工具参数解析（注: v4.5 靠猜 key 名，见 PRD P1-2）
      ├─ ToolRegistry.execute_tool_call(name, args)
      │   → 9 工具之一:
      │     - get_video_meta / get_frame_details / delete_history
      │     - search_web (DuckDuckGo)
      │     - search_visual (CLIP 文→帧)
      │     - run_ocr (PaddleOCR)
      │     - create_highlights (MoviePy)
      │     - point_and_jump (定位 + seek_video 跳转)
      │     - search_kb (跨视频 ChromaDB)
      ├─ 工具结果回灌 current_prompt（注: v4.5 累积拼接，见 PRD P1-2）
      └─ 信号回 agent_panel ThinkingWidget + ChatBubble
```

### 2.3 跨视频知识库数据流

```
Phase 1 完成
      │
      ▼
KBIndexWorker (main_window.py:306)
      │
      ▼
kb_indexer.index_frames(history_manager, session_id, video_name, video_path, frames)
      │
      ├─ get_embedder() 进程内共享 SentenceTransformer('clip-ViT-B-32')
      │   （首次加载，后续复用，避免重复加载）
      │
      ├─ 分批 encode（batch_size=64，控内存峰值）
      │   [str(f.path) for f in chunk]  # Path 必须转 str，否则 encode 抛 WindowsPath 错
      │
      └─ history_manager.add_frame_to_kb(session_id, video_name, video_path,
                                          timestamp, content, embedding, ocr_text)
          │
          ├─ ChromaDB kb_frames collection.upsert(
          │     ids=[f"{session_id}_{timestamp:.3f}"],
          │     embeddings, metadatas, documents)
          │
          └─ metadata: session_id / video_name / video_path / timestamp / content / ocr_text

用户跨视频搜索
      │
      ▼
search_kb 工具 (agent_tools.py:273)
      │
      ├─ get_embedder().encode(query)
      ├─ history_manager.search_kb(query_emb, top_k=8, min_score=0.25)
      │   ├─ ChromaDB kb_frames.query(n_results=top_k)
      │   ├─ L2 距离 → 0-1 相似度 score = max(0, 1 - dist/2)
      │   └─ 过滤 min_score，返回 [{video_name, video_path, timestamp, score, content}]
      │
      └─ 返回带时间戳结果 → 可 seek_video 跳转
```

---

## 3. 技术规格（Technical Specifications）

### 3.1 技术栈

| 组件 | 版本/要求 | 用途 | 来源 |
|------|----------|------|------|
| Python | 3.10+ | 运行时 | `launcher.py` 版本门禁 |
| PyQt6 | >=6.6.0 | GUI 框架 | 信号槽异步更新 |
| QThread | - | 多线程 | 耗时分析在后台，防界面假死 |
| OpenCV | headless >=4.8.0 | 视频抽帧 | `opencv-python-headless` |
| ultralytics | >=8.0.0 | YOLO11 目标检测 | `detect_objects_in_frame` |
| faster-whisper | >=1.0.0 | ASR（CTranslate2 加速） | `AudioProcessor.transcribe` |
| sentence-transformers | >=2.3.0 | CLIP 语义编码 | `clip-ViT-B-32`，KB 索引 + search_visual |
| chromadb | - | 向量库 | 跨视频 KB + 会话级帧记忆 |
| moviepy | 2.x | 视频剪辑/集锦/GIF | `subclipped`/`resized`（v4.5 迁移） |
| scenedetect | >=0.6.2 | 智能场景切分 | `extract_smart_keyframes` |
| torch | - | YOLO/CLIP 推理 | CUDA/CPU 自动 |
| pynvml | - | NVIDIA 显存监控 | `NVIDIA_GPU_AVAILABLE` |
| keyring | - | OS 密钥环 | API Key（DPAPI/Keychain/SecretService） |
| paddleocr | 可选 | OCR | `requirements-ocr.txt`（CPU extras） |
| requests | - | Ollama/OpenAI API | `OllamaClient`/`APIGatewayClient` |
| psutil | - | 进程资源监控 | `status_console.py` |

### 3.2 能力探测（Capability Probing）

`logic.py:21-96` 在启动时探测可选依赖，失败优雅降级：

| 探测标志 | 探测方式 | 影响 |
|----------|---------|------|
| `CLIP_AVAILABLE` | `try: import sentence_transformers` | False 则跳过 KB 索引 + search_visual |
| `NVIDIA_GPU_AVAILABLE` | `try: pynvml.nvmlInit()` | False 则 CPU 回退 |
| `ADVANCED_FEATURES_AVAILABLE` | `_detect_advanced_features()` 探测 moviepy/matplotlib/seaborn | False 则跳过 Phase 3 |
| `FFMPEG_AVAILABLE` | `check_ffmpeg()` | False 则音频提取失败 |
| `SCENEDETECT_AVAILABLE` | `_probe("scenedetect")` | False 则智能关键帧不可用 |
| `DECORD_AVAILABLE` | `_probe("decord")` | v4.5 未用 |
| `MEDIAINFO_AVAILABLE` | `_probe("pymediainfo")` | - |

> v4.5 修复：`ADVANCED_FEATURES_AVAILABLE` 从恒真（`try:pass;FLAG=True`）改为真实探测，Phase 3 不再运行时崩溃。

### 3.3 显存互斥（ModelContextManager）

`logic.py:108-138` 保证同一时间只有一个重型模型在 GPU 上：

```python
class ModelContextManager:
    def request_vram(self, requesting_model: str):
        # LLM/Whisper/YOLO 加载时，卸载其他模型
        if requesting_model in ["LLM", "Whisper", "YOLO"]:
            to_unload = [m for m in self.active_models if m != requesting_model]
            for m in to_unload:
                self.unload(m)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
```

CUDA 健康检查 `check_cuda_health()`（`logic.py:48`）：探测 `torch.cuda.is_available()` + cuDNN，做一次小张量运算验证，避免"驱动在但 CUDA 实际不可用"的假阳性。

---

## 4. API 与接口（APIs and Interfaces）

### 4.1 Headless HTTP 服务接口

`src/server/headless.py`，启动：`python -m src.server.headless [--port 8000]`

#### 4.1.1 `GET /healthz`

健康检查 + 能力矩阵，**必须 <50ms 返回**（Ollama 探测留给 `/analyze`，`/healthz` 不阻塞）。

**响应**：
```json
{
  "status": "ok",
  "capabilities": {
    "clip_semantic": true,
    "nvidia_gpu": false,
    "advanced_media": true,
    "ffmpeg": true,
    "ocr": false,
    "llm_backend": "unknown"
  },
  "disk_free_gb": 125.3
}
```

> `_ollama_alive()` 用 `trust_env=False` 绕开系统代理：本机 localhost 探测不应走外部代理，否则代理拦截导致 `/healthz` 在客户端超时后才返回（`headless.py:54-62`）。

#### 4.1.2 `POST /analyze`

上传视频，执行完整三阶段分析，返回帧 + 报告 JSON。

**请求**：
- `multipart/form-data`（单文件字段）或
- 原始字节 + `X-Filename` 头

**限制**：`VAP_MAX_UPLOAD_MB` 环境变量（默认 512MB）

**响应**：
```json
{
  "job_id": "a1b2c3d4e5f6",
  "duration": 3600.0,
  "frame_count": 120,
  "frames": [
    {"timestamp": 0.0, "metrics": {"brightness": 0.5, "sharpness": 0.8}}
  ],
  "transcript": "...(截断至 5000 字)",
  "report": "Markdown 报告或 [LLM unavailable: ...]"
}
```

> LLM 走 Ollama；Ollama 不可达时优雅降级（`report` 标记 `[LLM unavailable]`，仍返回结构化数据）。

### 4.2 LLM 客户端接口（BaseAPIClient）

`logic.py:635-940`，统一抽象 `chat_stream`：

| 客户端 | 类 | 默认端点 | 说明 |
|--------|-----|---------|------|
| Ollama | `OllamaClient` `logic.py:643` | `http://localhost:11434` | 客户端层解析 SSE，产出纯文本 delta（v4.5 修复） |
| OpenAI 格式 API | `APIGatewayClient` `logic.py:742` | 用户配置 | `parse_endpoint(url)` 解析 URL（注: 用字符串分隔符解析 system 消息，见 PRD P2-2） |
| LMStudio | `LMStudioClient` `logic.py:928` | `http://localhost:1234/v1` | 继承 APIGatewayClient |
| 本地 GGUF | `LocalModelClient` `logic.py:933` | - | 本地 llama.cpp |

```python
class BaseAPIClient:
    def _encode_image_to_base64(self, image_path: str) -> str: ...
    def chat_stream(self, *args, **kwargs) -> Iterator[str]: raise NotImplementedError
```

### 4.3 Agent 工具接口（ToolRegistry）

`agent_tools.py`，9 个工具通过工厂函数创建（注入 `app_context_getter` 闭包）：

| 工具 | 工厂 | schema | 说明 |
|------|------|--------|------|
| `get_video_meta` | `create_get_video_meta_tool` | - | 返回 filename/duration/output_dir/frame_count |
| `get_frame_details` | `create_get_frame_details_tool` | `{"seconds": float}` | 先查预抽帧（±0.25s），无则 OpenCV 动态抽帧 |
| `delete_history` | `create_delete_history_tool` | - | 安全限制：要求用户 UI 确认 |
| `search_web` | `create_search_web_tool` | `{"query": str}` | DuckDuckGo，top 5 |
| `search_visual` | `create_visual_search_tool` | `{"query": str}` | CLIP 文→帧，top 3（注: 无缓存，见 PRD P2-1） |
| `run_ocr` | `create_ocr_tool` | `{"seconds": float}` | PaddleOCR，先 get_frame_details 再 OCR |
| `create_highlights` | `create_highlight_cut_tool` | `{"description": str}` | MoviePy 拼接（注: 硬编码前 3 帧，见 PRD P1-3） |
| `point_and_jump` | `create_visual_grounding_tool` | `{"query": str}` | search_visual + seek_video 跳转 |
| `search_kb` | `create_kb_search_tool` | `{"query": str}` | 跨视频 ChromaDB 搜索，top 8 |

```python
class ToolRegistry:
    def register_tool(self, name, description, func, schema): ...
    def get_tool_descriptions(self) -> str: ...  # 拼成 Agent 系统提示词
    def execute_tool_call(self, tool_name, args) -> str: ...
```

### 4.4 配置接口

`config_manager.py`，`ConfigurationManager`：

| 方法 | 用途 |
|------|------|
| `load_main_config()` | 加载 `config/app_config.ini`，缺则建默认 |
| `update_config(section, key, value)` | 更新并保存 |
| `load_api_presets()` / `save_api_presets()` | `config/api_presets.json` |
| `load_prompts()` / `save_prompts()` | `config/prompts.json`（3 内置模板） |
| `_secure_set(key, value)` / `_secure_get(key)` | OS keyring 存储 API Key（`config_manager.py:19-39`） |

---

## 5. 数据层设计（Data Layer）

### 5.1 持久化存储

| 存储 | 位置 | 用途 | schema |
|------|------|------|--------|
| SQLite | `{config_dir}/history.db` | 会话 + 检查点 | `sessions` / `checkpoints` 两表 |
| ChromaDB | `{config_dir}/chroma_db` | 向量库 | `kb_frames`（全局）+ `session_{id}`（会话级） |
| INI | `config/app_config.ini` | 应用配置 | Application/Environment/LastUsed 三段 |
| JSON | `config/api_presets.json` | API 预设 | - |
| JSON | `config/prompts.json` | 提示词模板 | name + content |
| OS keyring | - | API Key | service=`VideoAnalysisPro` |

### 5.2 SQLite Schema

```sql
-- history_manager.py:32-51
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,           -- uuid4 hex（v4.5 修复秒级冲突）
    timestamp TEXT,                -- ISO utcnow
    video_path TEXT,
    video_name TEXT,
    output_dir TEXT,
    summary TEXT,
    status TEXT DEFAULT 'completed'
);

CREATE TABLE IF NOT EXISTS checkpoints (
    session_id TEXT,
    last_processed_second REAL,
    data TEXT,                      -- JSON
    PRIMARY KEY(session_id)
);
```

> v4.5 修复：`session_id` 从 `int(time.time())` 改为 `uuid.uuid4().hex`，同秒创建不再主键冲突。

### 5.3 ChromaDB Schema

#### 5.3.1 全局知识库 `kb_frames`（v4.5 新增）

```python
# history_manager.py:93-121
KB_COLLECTION_NAME = "kb_frames"

collection.upsert(
    ids=[f"{session_id}_{timestamp:.3f}"],   # upsert 防重复分析同视频冲突
    embeddings=[embedding.tolist()],
    metadatas=[{
        "session_id": session_id,
        "video_name": video_name,
        "video_path": video_path,           # 用于跳转
        "timestamp": float(timestamp),
        "content": (content or "")[:500],   # 截断控体积
        "ocr_text": (ocr_text or "")[:500],
    }],
    documents=[(content or "")[:1000]],
)
```

**搜索**（`history_manager.py:123-158`）：
```python
def search_kb(self, query_embedding, top_k=8, min_score=0.25) -> list:
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["metadatas", "distances"],
    )
    # ChromaDB 默认 L2 距离 → 0-1 相似度
    score = max(0.0, 1.0 - float(dist) / 2.0)
    if score < min_score: continue
```

#### 5.3.2 会话级 `session_{id}`

```python
# history_manager.py:72-85
collection = self.chroma_client.get_or_create_collection(name=f"session_{session_id}")
collection.add(
    ids=[f"frame_{timestamp}"],
    embeddings=[embedding.tolist()],
    metadatas=[{"timestamp": timestamp, "content": content}],
    documents=[content]
)
```

### 5.4 数据生命周期

| 操作 | 方法 | 说明 |
|------|------|------|
| 创建会话 | `add_session` | uuid4 主键 |
| 写检查点 | `save_checkpoint` | 断点续传 |
| 读检查点 | `get_checkpoint` | - |
| 会话级帧 | `add_frame_to_memory` | `session_{id}` collection |
| KB 帧写入 | `add_frame_to_kb` | `kb_frames` upsert |
| KB 搜索 | `search_kb` | 跨视频 |
| KB 计数 | `kb_count` | UI 展示 |
| 删会话 | `delete_session` | 删 SQLite + 目录 + session collection + KB 中该 session 条目（v4.5 修复孤儿） |
| 清空 | `clear_all_history` | 遍历删 |
| 过期清理 | `cleanup_old_sessions(retention_days=7)` | - |

> v4.5 修复：`delete_session` 的 Chroma 清理从 `if row:` 内移到无条件执行——此前若 session 有 KB 条目但 sessions 表无 row（或已删），会短路留下孤儿向量。

---

## 6. 关键模块详解（Key Modules）

### 6.1 VideoProcessor（`logic.py:140-330`）

| 方法 | 用途 |
|------|------|
| `extract_keyframes(density, max_frames=10000)` | 按 density 均匀抽帧（density=1.0 约 1FPS） |
| `extract_smart_keyframes(min_scene_len=15)` | PySceneDetect 切场景 + `_is_blurry` 过滤 |
| `_extract_chunk(indices)` | 分块抽帧控内存 |
| `_process_scenes_chunk(scenes, offset)` | 场景分块处理 |
| `_is_blurry(image, threshold=100.0)` | Laplacian 方差判模糊 |
| `filter_frames_semantically(frames, threshold=0.85)` | CLIP 语义去重 |

### 6.2 AudioProcessor（`logic.py:482-613`）

| 方法 | 用途 |
|------|------|
| `extract_audio(video_path, output_dir)` | ffmpeg 提取音频 |
| `transcribe(audio_path, diarize=False)` | faster-whisper 转录 → `AudioTranscript` |

```python
@dataclass
class AudioTranscript:  # logic.py:332
    text: str
    # ... 其他字段
```

> v4.5 修复：transcript 保留完整 `AudioTranscript` 对象（不再降级为字符串），时间轴波形恢复。

### 6.3 VideoAnalyzer（`logic.py:941-1130`）

| 方法 | 用途 |
|------|------|
| `__init__(client, model, prompt_loader, use_yolo, use_ocr)` | 注入 LLM 客户端 + 配置 |
| `analyze_video(frames, transcript, custom_template)` | 逐帧构建提示词 → 流式 LLM → Markdown |
| `extract_text_from_frame(frame_path)` | OCR（可选） |
| `detect_objects_in_frame(frame_path)` | YOLO11 |
| `_process_stream(stream_iterator)` | 过滤 `__FULL_RESPONSE_END__` 哨兵 |
| `unload_models()` | 释放 |

### 6.4 ModelManager（`logic.py:339-481`）

| 方法 | 用途 |
|------|------|
| `get_model_path(model_id)` | 返回 `models/` 下路径 |
| `list_local_models()` | 列本地模型 |
| `detect_model_type(filename)` | .pt→YOLO，.gguf→llama.cpp |
| `verify_model_integrity(model_id)` | SHA256 校验（防 MITM 投毒） |
| `download_model(model_id, progress_callback)` | 下载 + 进度回调 |

### 6.5 ToolRegistry（`agent_tools.py:18-46`）

ReAct Agent 核心：`register_tool` 注册 → `get_tool_descriptions` 拼系统提示词 → `execute_tool_call` 执行。

### 6.6 KBIndexer（`kb_indexer.py`）

| 方法 | 用途 |
|------|------|
| `get_embedder()` | 进程内共享 `SentenceTransformer('clip-ViT-B-32')`（首次加载，后续复用） |
| `index_frames(history_manager, session_id, video_name, video_path, frames, batch_size=64)` | 分批 encode + 写 KB |

> 关键：`[str(f.path) for f in chunk]`——Frame.path 是 Path，encode 只接受 str/PIL.Image/ndarray，不转会抛 `Unsupported input type: WindowsPath`。

---

## 7. UI 层设计（UI Layer）

### 7.1 DesktopApp（`main_window.py:424`）

主窗口，含 ~74 方法 + 10 Worker 类。核心信号槽链路：

```
用户操作 → DesktopApp 槽 → 启动 QThread → Worker 后台处理
                                     │
                                     └─ 信号回 UI（progress/result/error）
```

### 7.2 Worker 类（`main_window.py:50-380`）

| Worker | 行 | 职责 |
|--------|----|------|
| `ExtractionWorker` | 81 | Phase 1 抽帧 + 转录 + YOLO |
| `AnalysisWorker` | 154 | Phase 2 LLM 分析 |
| `ChatWorker` | 175 | Agent ReAct 循环（注: 简陋，见 PRD P1-2） |
| `MediaWorker` | 263 | Phase 3 集锦 + GIF |
| `ModelDownloadWorker` | 293 | 模型下载 |
| `KBIndexWorker` | 306 | v4.5 KB 后台索引 |
| `OllamaRefreshWorker` | 331 | Ollama 模型列表刷新 |
| `ModelLoadWorker` | 347 | 模型加载 |
| `ApiCheckWorker` | 364 | API 可用性检查 |
| `ImageLoader` (QRunnable) | 53 | 帧图异步加载 |

### 7.3 Agent 面板组件（`agent_panel.py`）

| 组件 | 行 | 职责 |
|------|----|------|
| `ThinkingWidget` | 7 | 类 DeepSeek R1 思考过程可视化 |
| `ChatBubble` | 73 | 用户/AI 气泡 |
| `ChatInput` | 234 | 输入框 |
| `AgentPanel` | 266 | 面板容器 |

### 7.4 其他 UI 组件

| 文件 | 职责 |
|------|------|
| `status_console.py` | 底部状态 + 资源监控（RSS/VRAM/模型 MB） |
| `carousel_widget.py` | 帧画廊轮播 |
| `timeline_widget.py` | 视频时间轴（波形 + 帧标记） |
| `video_player_dialog.py` | 专业播放器（AI 叠加层） |
| `model_manager_tab.py` | 模型下载/校验管理 |
| `help_dialog.py` | 使用说明 |
| `api_intro_page.py` | API 获取指南（注: 硬编码 iflow.cn，见 PRD P2-4） |

---

## 8. 部署与运维（Deployment & Operations）

### 8.1 桌面分发

| 方式 | 入口 | 说明 |
|------|------|------|
| Windows 一键 | `启动应用.bat` | `py -3.10/3.11/3.12` 版本探测 + 失败引导下载页 |
| Linux/macOS | `启动应用.sh` | ffmpeg 路径按平台分支 |
| 通用 | `launcher.py` | 版本门禁 + 自动建 venv + 装依赖 |
| 调试 | `debug_launcher.py` | - |
| 打包 | `build_windows.spec` | PyInstaller onedir + 内置 FFmpeg，双击运行 |

### 8.2 Docker 部署

| 镜像 | Dockerfile | 用途 |
|------|-----------|------|
| CPU | `Dockerfile` | 无 GPU 环境 |
| CUDA | `Dockerfile.cuda` | GPU 加速 |

`docker-compose.yml` 含 GPU profile。

### 8.3 Headless 服务

见 §4.1。启动：`python -m src.server.headless [--port 8000]`，环境变量 `VAP_PORT`（默认 8000）、`VAP_MAX_UPLOAD_MB`（默认 512）。

### 8.4 测试

| 类型 | 位置 | 数量 |
|------|------|------|
| 单元 + 集成 + E2E | `tests/` 11 文件 | 59 个 pytest |
| E2E 冒烟 | - | GUI 启动 16 项 |
| 全链路 | - | 13 项 |
| Headless | - | `/healthz` + `/analyze` |

运行：`pytest --cov=src --cov-report=term-missing`

### 8.5 CI

`.github/` 三平台测试矩阵（见 README + CHANGELOG）。

---

## 9. 安全与合规（Security & Compliance）

### 9.1 凭据安全

- **API Key** 优先存 OS 密钥环（Windows DPAPI / macOS Keychain / Linux SecretService），`config_manager.py:19-39`
- keyring 不可用时降级 ini 明文，日志明确警示
- **模型下载** SHA256 完整性校验，防 MITM 投毒，`logic.py:399`

### 9.2 隐私

- **本地优先**：Ollama + 本地模型完全离线，视频不出本地
- 云端 API 为可选项，用户自配

### 9.3 已知安全项（v4.5 现状，非阻塞）

| 项 | 现状 | PRD 编号 |
|----|------|---------|
| 默认 API 端点硬编码 `https://api.iflow.cn/v1` | `config_manager.py:75` | P2-4（Phase B 改中立空） |
| Headless 无鉴权 | `headless.py` 无 auth | -（单机/内网形态，若公网暴露需用户自加反代鉴权） |
| 工具调用无权限护栏 | Agent 工具直接执行 | -（delete_history 已要求 UI 确认） |

### 9.4 合规

- **协议**：GPL-3.0（传染性开源，修改须开源）
- **依赖许可**：见各包（PyQt6 GPL/商业双许可、torch BSD、ultralytics AGPL-3.0 等，用户商业使用须自行核查 ultralytics AGPL 条款）

---

## 10. 工程债与已知限制

| 项 | 现状 | 处理 |
|----|------|------|
| `main_window.py` God Object | 2355 行 + ~74 方法 + 10 Worker | Phase B P2-7 拆分（建议 v5.x） |
| Agent 系统提示词薄弱 | `main_window.py:2257-2286` | Phase B P1-1 重写 |
| ChatWorker ReAct 简陋 | `main_window.py:191-235` 累积拼接 + 猜参数 + 截断 | Phase B P1-2 改造 |
| `create_highlight_cut` 硬编码 | `agent_tools.py:211-247` 前 3 帧 | Phase B P1-3 语义化 |
| `search_visual` 无缓存 | `agent_tools.py:167-174` 每次重载 | Phase B P2-1 复用 embedder |
| `APIGatewayClient` 脆弱协议 | `logic.py:781-789` 字符串分隔符 | Phase B P2-2 结构化 |
| `OllamaClient` num_ctx 硬编码 | `logic.py:664` 4096 | Phase B P2-3 可配置 |
| `plot_metrics` 基础折线 | `main_window.py:2165` | Phase B P2-5 升级 |
| 默认端点商业推荐 | `config_manager.py:75` iflow.cn | Phase B P2-4 中立空 |
| 未提交 lint diff | bare except + 类型注解 + mypy.ini | Phase B 前确认是否纳入（PRD Q3） |

---

## 11. 术语表

| 术语 | 含义 |
|------|------|
| MVC 变体 | UI/核心逻辑/工具/数据四层，非严格 MVC |
| 三阶段流水线 | Phase 1 提取 → Phase 2 AI 分析 → Phase 3 媒体生成 |
| ReAct | Reason + Act 循环 |
| `kb_frames` | ChromaDB 全局跨视频 collection |
| `session_{id}` | ChromaDB 会话级 collection |
| ModelContextManager | 显存互斥管理器 |
| Headless | 无 GUI HTTP 服务模式 |
| ADVANCED_FEATURES_AVAILABLE | Phase 3 依赖探测标志 |
| onedir | PyInstaller 单目录打包模式 |

---

*本技术文档为 v4.5 已交付架构的单一规格源。Phase B 增量技术方案见 `参考的结果计划指南.md` §11。变更日志见 `CHANGELOG.md`。*
