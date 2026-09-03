# Video Analysis Pro (Python Edition) — 项目级指南

> 本文件是本项目（Python 桌面视频分析工具）的工程指南。通用行为准则、安全、测试、
> Git 工作流、复杂度分级遵循用户全局 `~/.claude/CLAUDE.md` 与 `~/.claude/rules/`；
> 此处只记录**本项目特有**的事实、命令、约定与已知坑。冲突时以本文件为准。

- **身份**：Video Analysis Pro，由 听风公司 (Tingfeng) 出品，维护者 `lza6`
- **仓库**：https://github.com/lza6/Video-Analysis-Pro-python
- **版本**：`v4.5.0`（见 `src/utils/constants.py:APP_VERSION`，改版本要同步 `CHANGELOG.md`）
- **License**：GPL-3.0（传染性开源，修改必开源）
- **Python**：3.10+（CI 矩阵测 3.10 / 3.11，本地 venv 子目录名 `venv`）
- **本项目不使用 OpenWolf**（`.wolf/` 不存在），全局 CLAUDE.md 的 OpenWolf 协议段不适用。

---

## 一、技术栈与架构

### MVC 变体分层

| 层 | 目录 | 职责 |
|----|------|------|
| **View (UI)** | `src/ui/` | PyQt6 深色 GUI。`main_window.py` 入口，`agent_panel.py` 思考链侧栏，`status_console.py` 资源监控，`carousel_widget.py` / `timeline_widget.py` / `video_player_dialog.py` 自定义组件，`model_manager_tab.py` 模型下载校验 |
| **Controller/Service** | `src/core/` | `logic.py`（`VideoProcessor` / `AudioProcessor` / `VideoAnalyzer` / `OllamaClient` / `PromptLoader` + 能力标志 `CLIP_AVAILABLE` / `NVIDIA_GPU_AVAILABLE` / `ADVANCED_FEATURES_AVAILABLE` / `FFMPEG_AVAILABLE`）、`agent_tools.py`（`ToolRegistry` + 9 个工具，含 `search_kb`）、`history_manager.py`（SQLite + ChromaDB 跨视频知识库）、`kb_indexer.py`（QThread 索引器） |
| **Headless 服务** | `src/server/headless.py` | `GET /healthz` + `POST /analyze`，Docker 用，无 GUI |
| **Utils** | `src/utils/` | `config_manager.py`（配置 + 密钥环存储）、`constants.py`（版本/路径/常量）、`ui_components.py`（tkinter 安装向导） |

### 关键依赖分层

- `requirements.txt` = core（应用启动最小集，**分层标注**，固定版本区间）
- `requirements-ocr.txt` = 可选 PaddleOCR（~1GB，缺失时应用自动跳过 OCR 功能，**非必需**）
- 全量 = `pip install -r requirements-ocr.txt`（已含其余可选）

核心库：`PyQt6` / `opencv-python-headless` / `ultralytics`(YOLOv11) / `scenedetect` / `faster-whisper` / `sentence-transformers` / `chromadb>=1.5.9` / `moviepy>=2.0`(已迁 2.x API) / `torch>=2.2` / `imageio-ffmpeg`(自带 FFmpeg) / `nvidia-ml-py`(无 GPU 自动跳过)。

### 三阶段处理流程

1. **Phase 1 数据提取**：OpenCV 智能抽帧 + Whisper 音频转录 + YOLO 物体检测 → 写入结构化缓存 + ChromaDB 全局 collection
2. **Phase 2 AI 分析**：选模型（Ollama 本地 / OpenAI 格式 API 云端）+ 提示词模板 → LLM 推理 → Markdown 报告
3. **Phase 3 媒体生成**：MoviePy 智能剪辑高光片段 + GIF 摘要 + 可视化数据图表（亮度/清晰度/饱和度）

---

## 二、运行 / 构建 / 测试命令

### 启动

```bash
# Windows（双击亦可）：版本探测 3.10/3.11/3.12 + 自动建 venv + 装依赖 + 启 GUI
启动应用.bat

# macOS / Linux
python launcher.py

# 调试
python debug_launcher.py

# Headless 服务（Docker / 无 GUI）
python -m src.server.headless --port 8000
```

`launcher.py` 含**版本门禁 + venv 自动创建 + import 验证脚本**，不要绕过它直接 `python -m src.ui.main_window`（会丢失环境自愈）。

### 测试

```bash
# 标准子集（CI 跑这个，不依赖 ultralytics/faster-whisper 等重依赖）
QT_QPA_PLATFORM=offscreen PYTHONIOENCODING=utf-8 \
  python -m pytest tests/ -q \
  --ignore=tests/test_headless_server.py \
  --ignore=tests/test_e2e_smoke.py

# 全量套件（需先 pip install -r requirements.txt 全量）
QT_QPA_PLATFORM=offscreen PYTHONIOENCODING=utf-8 \
  python -m pytest tests/ -q --ignore=tests/test_headless_server.py
```

- 测试框架：`pytest`（`conftest.py` 把项目根插入 `sys.path` 并先 `import torch` 守卫 DLL 顺序）
- 现有 11 个测试文件：`test_agent_tools` / `test_api_clients` / `test_api_gateway_stream` / `test_core_pipeline` / `test_e2e_full_pipeline` / `test_e2e_smoke` / `test_headless_server` / `test_history_manager` / `test_model_manager` / `test_ui_components`
- GUI 测试**必须**设 `QT_QPA_PLATFORM=offscreen`，否则无显示环境崩溃

### 静态检查

```bash
python -m pyflakes src/ launcher.py   # CI 强制，零告警才放行
mypy src/                              # mypy.ini: python 3.10, ignore_missing_imports
```

### 打包发布

```bash
pyinstaller build_windows.spec --noconfirm   # onedir + 内置 FFmpeg
docker build -t video-analysis-pro .           # CPU 镜像
docker build -f Dockerfile.cuda -t video-analysis-pro:cuda .   # GPU 镜像
docker compose up                              # 编排（含 GPU profile）
```

CI 在打 `v*` tag 时触发 `build-windows` job 跑全量测试 + PyInstaller + 上传 artifact。

---

## 三、关键约定与已知坑（改代码前必读）

### 1. torch 必须先于 PyQt6 导入（Windows，P0）

`src/ui/main_window.py` 和 `tests/conftest.py` 顶部都有：

```python
try:
    import torch  # noqa: F401  (DLL load-order fix)
except OSError:
    torch = None
```

**原因**：PyQt6 先注册自己的 `vcruntime140/msvcp140` DLL 目录，torch 的 `c10.dll` 会解析到旧副本 → `WinError 1114` 崩溃。新建任何导入 PyQt6 + torch 的模块都要保持此顺序。

### 2. 凭据存储：密钥环优先（安全）

`src/utils/config_manager.py` 的 `_secure_set` / `_secure_get`：API Key 优先存 OS 密钥环（Windows DPAPI / macOS Keychain / Linux SecretService），不可用时降级到 `app_config.ini` 明文并日志告警。**禁止把真实 Key 写进 ini 或硬编码**，`.env` 已 gitignore。

### 3. 模型下载需 SHA256 校验

`model_manager_tab.py` 的模型下载流程含完整性校验（防 MITM 投毒）。新增下载源必须保留校验逻辑。

### 4. 依赖双写同步

`src/utils/constants.py` 的 `REQUIRED_PACKAGES` 列表与 `requirements.txt` **手动同步**（launcher 用它做 import 验证）。改 `requirements.txt` 时必须同步改 `constants.py`，否则 venv 验证漏装。

### 5. moviepy 2.x API

代码已迁移到 moviepy 2.x：用 `subclipped()` / `resized()` 等 2.x 命名，**不要**回退到 1.x 的 `subclip` / `resize`。

### 6. PyQt6 严格类型

`setSize()` 接受 `QSizeF` 而非 `QSize`；类似严格类型多处存在。改 UI 代码时按 PyQt6 严格签名走，不要套 PySide6/PyQt5 旧写法。

### 7. 已修复的回归坑（不要重新引入）

- `session_id` 用 `uuid4`，**不要**用 `int(time.time())`（同秒冲突）
- `OllamaClient` 必须在客户端层解析 SSE 协议，产出纯文本 delta，**不要**向 UI 泄漏 `{"message":{"content":...}}` JSON 碎片
- `ADVANCED_FEATURES_AVAILABLE` 必须**真实探测** moviepy/matplotlib/seaborn，不要硬编码为 `True`
- 视频播放器 `closeEvent` 引用的是 `self.media_player`，不是 `self.player`
- 内部哨兵 `__FULL_RESPONSE_END__` **不得**泄漏进最终报告 / Agent 输出
- `smart_extraction` 配置键名必须接入 `extract_smart_keyframes()`，不要被 worker 忽略
- `closeEvent` 要主动 `stop()` / `wait()` 运行中的 QThread，否则退出崩溃

### 8. RTSP / 监控功能

`.env` 支持 `VAP_RTSP_URL` / `VAP_MONITOR_DIR` / `VAP_KEY_ITEM_IMAGE`（监控实时流 + 关键物品检测，运行时读取，不入库）。

---

## 四、配置与环境变量

### `.env`（复制 `.env.example`，已 gitignore）

| 变量 | 用途 |
|------|------|
| `VAP_LLM_PROVIDER` | LLM 提供商（`anthropic` / `ollama` / `openai` 格式） |
| `VAP_LLM_BASE_URL` | API 端点（如 `https://api.yjs.im/v1`） |
| `VAP_LLM_MODEL` | 模型名（如 `glm-5.3-flash`） |
| `VAP_LLM_API_KEY` | API Key（**绝不入库**） |
| `VAP_MONITOR_DIR` | 监控目录 |
| `VAP_KEY_ITEM_IMAGE` | 关键物品参考图 |
| `VAP_RTSP_URL` | RTSP 流地址 |

### `config/app_config.ini`（运行时，已 gitignore）

存 theme / version / venv_path / client_type / api_url / api_key / model_name。`api_key` 字段实际由密钥环覆盖，ini 只留标记位。

### 提示词模板（入库）

`config/prompts/frame_analysis/` 下三个 `.txt`：`describe.txt` / `frame_analysis.txt` / `video_summary.txt`。`PromptLoader` 默认指向此目录。新增模板放此处。

---

## 五、禁止提交的文件（见 `.gitignore`）

| 路径 | 类型 |
|------|------|
| `venv/` `.venv/` | 虚拟环境 |
| `__pycache__/` `.pytest_cache/` `.coverage` `htmlcov/` | Python 缓存 |
| `logs/` `软产生的缓存/` | 运行时日志/缓存（`CACHE_DIR = "软产生的缓存"`） |
| `config/chroma_db/` `config/history.db` `config/app_config.ini` | 运行时数据/配置 |
| `.env` | 密钥 |
| `E2E实测结果/` | E2E 产物 |
| `website/` | 独立前端项目，不属于本 Python 仓库 |
| `.claude/` `.codegraph/` `.code-review-graph/` graft/ | AI 工具产物 |

> 注：历史上有过 `chroma.sqlite3` / `history.db` 误入库后被清理的提交（见 `69e5374` / `157e21c`），新增运行时数据文件务必先加 gitignore。

---

## 六、知识工具（已配置）

- **graft 代码图谱**：`.mcp.json` + `.claude/helpers/graft-hooks.cjs`，PostToolUse 自动索引。结构性问题先 `graft ask` / `graft callers`，再回退 Grep/Read。
- **CodeGraph**：`.codegraph/`，MCP 工具 `codegraph_*`。
- **code-review-graph**：`.code-review-graph/`，需先 `code-review-graph build`。
- MCP 服务器可能因网络超时连不上——视为连接失败而非未配置，提示用户重试即可。

---

## 七、Skill 使用

本项目已安装 superpowers-zh 技能框架（见 `.claude/skills/` 与 `skills-lock.json`）。匹配时优先用：

- **brainstorming** — 任何创造性工作前先做需求分析
- **test-driven-development** — 写实现前先写测试
- **systematic-debugging** — 任何 bug/测试失败/异常前先用
- **verification-before-completion** — 声称完成前必须跑验证命令并确认输出
- **receiving-code-review** / **requesting-code-review** — 审查反馈闭环

---

## 八、常见任务速查

| 想做 | 怎么做 |
|------|--------|
| 加一个 Agent 工具 | `src/core/agent_tools.py` 用 `create_*_tool` 工厂 + 注册到 registry，在 `tests/test_agent_tools.py` 补测试 |
| 加一个提示词模板 | `config/prompts/frame_analysis/` 放 `.txt`，UI 模板下拉自动收录 |
| 加一个 UI 组件 | `src/ui/` 新建，遵循 PyQt6 严格类型；若涉及 torch/Qt 共存，顶部复制 import 守卫 |
| 改 LLM 接入 | `src/core/logic.py` 的 `VideoAnalyzer` / `OllamaClient`；API 客户端测试在 `tests/test_api_clients.py` |
| 改版本号 | `src/utils/constants.py:APP_VERSION` + `CHANGELOG.md` 顶部加条目 + `config/app_config.ini` 的 `version`（运行时文件，勿入库） |
| 加运行时数据文件 | 先加 `.gitignore` 再创建，避免误入库 |

---

*最后更新：2026-09-03（v4.5.0）*
