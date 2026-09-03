# Changelog — Video Analysis Pro

## [4.5.0] — 2026-09-03

### 🎉 重大更新
- **跨视频向量知识库 (v4.5 Roadmap 落地)**：Phase 1 完成后自动把关键帧写入
  ChromaDB 全局 collection，支持自然语言跨视频搜索
  （"帮我找找过去一年所有视频里出现过的红色跑车"→返回时间戳+视频名+跳转）。
  复用已有 PersistentClient，不引入新向量库。
- **可分发软件包**：新增 `build_windows.spec`（PyInstaller onedir + 内置 FFmpeg），
  双击产物即可运行，无需手动安装 Python/依赖。

### 🐛 关键修复 (P0/P1)
- **启动链脆弱**：`启动应用.bat` 重写为 py -3.10/3.11/3.12 版本探测 + 失败引导下载页。
- **torch/PyQt6 DLL 冲突**：main_window.py 顶部强制 `import torch` 先于 PyQt6，
  消除 Windows `c10.dll` WinError 1114 崩溃。
- **Agent 面板 Ollama 模式输出损坏**：OllamaClient 现在在客户端层解析 SSE 协议，
  产出纯文本 delta（不再向 UI 泄漏 `{"message":{"content":...}}` JSON 碎片）。
- **未定义名崩溃**：修复 pyflakes 报告的 3 个 undefined name
  (`psutil` / `remaining` / `QDialog`) + 死代码（`__main__` 之后的 init_backend/seek_video）。
- **视频播放器崩溃**：closeEvent 引用不存在的 `self.player` → 改为 `self.media_player`；
  `setSize(QSize)` → `setSize(QSizeF)`（PyQt6 严格类型）。
- **智能关键帧开关失效**：`chk_smart` → `smart_extraction` 配置曾被 ExtractionWorker 忽略，
  现接入 `extract_smart_keyframes()`。
- **视频时长恒为 0**：ExtractionWorker 现返回 `duration`；transcript 保留完整
  AudioTranscript 对象（不再降级为字符串，时间轴波形恢复）。
- **模板选择形同虚设**：选中内置模板现真正传给 analyzer（不再落入英文一句话兜底）；
  PromptLoader 默认目录指向 `config/prompts/frame_analysis/`（此前从不被加载）。
- **报告脏标记**：`__FULL_RESPONSE_END__` 内部哨兵不再泄漏进报告/Agent 输出。
- **ADVANCED_FEATURES_AVAILABLE 恒真**：改为真实探测 moviepy/matplotlib/seaborn。
- **会话主键冲突**：`session_id` 从 `int(time.time())` 改为 uuid4（同一秒内不再冲突）。
- **QThread 退出崩溃**：closeEvent 现主动 stop/wait 运行中的 worker。

### 🔒 安全
- API Key 优先存 OS 密钥环（Windows DPAPI / macOS Keychain / Linux SecretService）。
- 模型下载新增 SHA256 完整性校验（防 MITM 投毒），UI"校验"按钮真正生效。

### 🏗️ 工程改进
- **依赖收敛**：requirements.txt 重写为分层（core / ocr extras），固定版本，
  删除 paddlepaddle-gpu（改 CPU paddlepaddle extras）、pyannote、decord、gradio。
- **moviepy 2.x 迁移**：`moviepy.editor` → `moviepy` 顶层；`subclip`→`subclipped`，
  `resize`→`resized`，移除 `verbose` 参数。
- **跨平台**：新增 `启动应用.sh`（Linux/macOS），ffmpeg 路径按平台分支。
- **Docker**：新增 `Dockerfile`（CPU）/ `Dockerfile.cuda`（GPU）/ `docker-compose.yml`
  + `src/server/headless.py`（HTTP 分析服务，复用 core 层零改动）。
- **资源面板增强**：显示本进程 RSS 内存、模型文件实际磁盘占用（非"预计大小"）。
- **版本单源**：`APP_VERSION` 统一在 `src/utils/constants.py`（4.5.0）。

### 🧪 测试与可观测性
- 新增 59 个 pytest 测试（含 E2E：完整 GUI 启动冒烟 16 项、全链路分析 13 项、
  Headless 服务 /healthz+/analyze），core 模块覆盖率显著提升。
- `.gitignore` 补全 venv/logs/cache/__pycache__。

### 📦 Agent 强化
- 5 个未注册工具补齐（search_web/search_visual/run_ocr/create_highlights/point_and_jump），
  此前因死代码位于 `if __name__` 之后从未生效。
- 新增 `search_kb` 跨视频知识库搜索工具。

### 🗑️ 清理
- 删除 `legacy/` 目录（12 个与 src 重复的旧 Gradio 版文件）。
- 删除 `app.py` 回退路径（Gradio 版入口已不存在）。

## [4.0.0] — 初始 Python 桌面版
- 三阶段流水线（提取→分析→媒体）+ Agent 面板 + 模型管理 + SQLite 历史。
