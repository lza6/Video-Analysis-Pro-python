# Changelog — Video Analysis Pro

## [5.3.0] — 2026-09-04 · 接线收口与发布一致性

### 📝 发布一致性
- **APP_VERSION 4.5.0 → 5.3.0**：tag v5.0/v5.1/v5.2 已发布但版本常量停在 4.5.0（窗口标题/关于页显示错误），本次对齐。
- **CHANGELOG 补 v5.1/v5.2 条目**：这两版实际已发布（git tag）但 CHANGELOG 缺记录，补齐。
- **e2e_smoke 断言升级**：14 项 → 28 项，新增监控/Skills/决策日志三 tab 挂载断言 + agent_prompt 八段增补断言；修复 `FAIL_SAFE` 段名与 `"FAIL" not in` 断言的误判冲突（改为逐行 `: FAIL` 后缀匹配）。
- **.gitignore 补 build//dist/**：PyInstaller 中间产物（build/build_windows/*.toc 1.0MB）不再出现在 git status。

## [5.2.0] — 2026-09-04 · B1/B3/B4 收口（对照 CL4R1T4S）

### ✨ Agent 智能度（P1-1 提示词八段增补）
- agent_prompt.py 六模块 → 十五段：AGENT_LOOP（Manus 六步）/ THOUGHT（Gemini 思考块）/ CLARIFY_GATE（DROID 意图澄清）/ CITATION（Codex 时间戳引用）/ FAIL_SAFE（Devin-Replit 失败3次求助）/ NOTIFY_ASK（Manus 双消息）/ PARALLEL（Claude 并行）/ INTENT_VOICE（Cursor 意图前置+工具名隐藏）；总长 1958 字符（小模型 4K 预算内）。
- **skills 触发注入**（P2-1）：`match_skills` 按 triggers 双向子串匹配（大小写不敏感），命中 → `# SKILLS` 段注入；未命中不占上下文（Progressive Disclosure 轻量版）。
- **用户偏好个性化**（P2-7）：`on_agent_query` 召回 `recall_preferences` top-3 注入 prompt；`inject_agent_system_context` 落盘 `remember_preference`（ADD-only，失败静默不碍主流程）。
- **B0 依赖收尾**：requirements.txt 补 `pandas<3`（seaborn 依赖链与 numpy<2.3 约束收敛）。

### ✅ 验证（v5.2 tag 时点）
pytest 113 passed | mypy src/ 35 files 0 错误 | ruff(SOP 口径) 0 错误 | e2e_smoke / e2e_full_pipeline / surveillance_e2e 真实链路 PASS

## [5.1.0] — 2026-09-04 · 孤岛接线批次（B0+B2）

### ✨ 黑匣子透明化（P1-4，直击"小白易用"）
- **eli5 大白话解释器**（`src/core/eli5.py`，154 行）：把"工具名+参数+结果"翻译成一句用户能懂的人话；纯函数可单测，Exception 分支/已知工具模板/未知工具退化三级，绝不抛异常到 UI。ChatWorker 展示行由 `str(result)[:100]` 截断升级为 eli5 摘要（messages 内部仍 2000 字符防 token 爆炸）。
- **决策日志**（`src/core/decision_log.py` + `src/ui/decision_log_panel.py`）：每次工具调用落 `{step, action, decision, reason, duration_ms, status, risk}` 条目，Qt 跨线程经 `entry_append` 信号投递，"🧭 决策日志" tab 实时展示 Agent 每一步。
- **接线三孤岛 tab**：`🎥 监控分析`（SurveillanceTab，RTSP 拉流+运动检测+VLM 命中）/ `🧩 Skills`（SkillsManagerTab，列表+启用切换+导入）/ `🧭 决策日志` 全部挂载进主窗口（7→10 tab）。

### 🔒 安全（P0）
- **headless Bearer Token 鉴权**：`VAP_HEADLESS_TOKEN` 非空时 `/analyze` 强制校验（`hmac.compare_digest` 防时序攻击）；默认关闭向后兼容；4 个 multipart 回归测试配套。

### 🐛 P0 修复
- **Phase 3 摘要媒体全链路恢复**：seaborn 声明在 requirements.txt 但 venv 缺失 → `ADVANCED_FEATURES_AVAILABLE=False` → MediaWorker 跳过 → 用户"生成摘要媒体"无产出（唯一红测根因）。pip 安装后 e2e_full_pipeline 恢复通过。

### ✅ 验证（v5.1 tag 时点）
pytest 102 passed | py_compile 全过 | website: tsc 0 错误 / next build 6/6 / Playwright E2E 8/8

## [Unreleased] — 2026-09-04 · 终局闭环审计

### 🐛 关键修复（P0 阻塞）
- **生产启动崩溃**：`qdarktheme.setup_theme`/`enable_hi_dpi` 在 pyqtdarktheme 0.1.7 上不存在，
  新增 `src/utils/theme_compat.py` 桥接 0.1.7 与 2.x API；requirements.txt 放宽到 `>=0.1.7,<3`；
  `main_window.py` / `tests/test_e2e_smoke.py` / `capture_desktop_ui.py` 同步引入。E2E 冒烟由红转绿。
- **跨视频知识库语义搜索完全失效**：`kb_indexer.py` 原把路径字符串当文本 embedding（不是图像），
  导致 `search_kb` 100% 返回垃圾结果。改用 `PIL.Image` 真实图像 embedding。

### 🐛 高优先级修复（P1）
- **LMStudioClient 构造参数对调**：`api_key` 与 `api_url` 传反，请求崩。
- **Whisper download_root**：复用本地缓存，离线环境不再误报"whisper_base 已下载"却联网。
- **ModelContextManager 并发竞态**：`request_vram`/`register`/`unload` 全部加 `GPU_LOCK`，
  修复多 QThread 下 `dictionary changed size during iteration`。
- **OllamaRefreshWorker 代理挂起**：localhost 请求走系统代理被拦截，改 `Session(trust_env=False)`。
- **Agent 工具参数解析硬编码**：4 个工具拿到错误参数名（TypeError），改为按 schema 首参数名解析。
- **closeEvent 漏 kb_worker**：关窗时 KB 索引在跑触发 "QThread destroyed while running"，已补。
- **save_current_settings 密钥环迁移不清空 ini**：成功写入 keyring 后清空 ini 旧明文残留。
- **cv2 中文路径静默失败**：新增 `imwrite_unicode`/`videocapture_unicode`，全量替换 logic.py /
  agent_tools.py / surveillance_agent.py 中的 `cv2.imwrite(str(...))` / `VideoCapture(str(...))`。
- **surveillance cut_clip total=0 永不裁剪**：监控视频读不到 frame count 时兜底。
- **highlight_cut 硬编码前 3 帧**：改为按 description 词频打分取 top-3（诚实实现，去伪宣传）。
- **VideoAnalyzer.embedder 死代码**：移除，CLIP 统一走 `kb_indexer.get_embedder()` 共享单例。

### 🔒 安全
- `config_manager.py` 默认 `api_url` 去硬编码商业代理（`api.iflow.cn`），改中立空值。
- `requirements.txt` 新增 `keyring>=24.0.0`（API Key OS 密钥环存储依赖，此前遗漏导致降级明文 ini）。

### 🎨 官网修复
- **reduced-motion 死代码**：Hero/Stats/Download 三处 `prefersReduced=false` 硬编码 →
  全部改 `useReducedMotion()` 真调用，偏好减少动效用户不再被全速动画轰炸。
- **HeroVisual WebGL 假兜底**：`onError` 对 div 不触发 → 改 ErrorBoundary + WebGL 预检。
- **HeroScene pointer 死代码**：移除 `pointerEvents:none` + pointer 耦合，保留时间自转。
- **metadataBase 占位域名**：`.example.com` → `process.env.NEXT_PUBLIC_SITE_URL ?? localhost`；
  JSON-LD url 改真实 GitHub 仓库。
- **Download macOS 空图标**：补 🍎。
- **Stats "7类"→"9类"**：与 agent_tools.py 9 工具对齐。
- **website 解除 gitignore**：官网源码纳入版本控制（仅排除 node_modules/.next）。

### 📐 工程
- 新增 `SPEC.md`（结构化规范）+ `workflow_status.md`（终局审计进度）。
- 新增 `src/utils/theme_compat.py`。
- 新增 `website/src/components/ErrorBoundary.tsx`。
- `.gitignore`：`CACHE_DIR` 由中文"软产生的缓存"改 ASCII "cache"；移除 `website/` 整体排除。

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
