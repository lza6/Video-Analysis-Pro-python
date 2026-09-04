# Changelog — Video Analysis Pro

## [5.6.0] — 2026-09-05 · 监控批量分析 Agent 化（NVIDIA Nemotron + 新算法省 99% 调用 + 二次验证）

### 🎯 监控批量分析全链路（核心新功能）
- **新算法引擎 motion_detector**（`src/core/motion_detector.py`）：1fps 抽帧 + scenedetect 场景切分 + 帧差分 + 昼夜自适应阈值（day 15 / night 6），只对画面有变化的时段送 AI，跳过空走廊。实测 63 个 17.7min 视频只送 15 片 AI（旧方案要 2268 片），**省 99.3% API 调用**。
- **批量引擎 batch_runner**（`src/core/batch_runner.py`）：视频级并发 4（CPU 多核并行 motion_detector）+ 单 key 并发 2 + 断点续跑（resume）+ 内存回收（每视频 gc.collect + 清分片临时目录）+ 命中自动裁剪（ffmpeg ±clip_padding 无损 -c copy）。
- **二次验证防误判**：首次判断 match=true 且 confidence≥0.7 → 自动再送 NVIDIA 用严格 prompt（has_person/has_target_item/description）确认，只有 has_target_item=true 才最终算命中。实测消除 `_375`/`_388` 旧误判。
- **总命中报告**：跑完自动生成 `HITS_REPORT.md`（视频名/命中时间码 HH:MM:SS/片段起止/置信度/片段路径/AI 描述），含二次验证标注。

### 🔌 NVIDIA 多 provider 路由（`src/core/provider_router.py`）
- **多 key 轮换**：11 个 NVIDIA nvapi key 全活（实测），优先级 + LRU 轮换，40 req/min 每 key 滑动窗口令牌桶。
- **503 退避优化**：撞 503/5xx 不立即切 key，先短退避 1.5s 重试同 key 2 次（给 worker 池空槽窗口），仍失败再切下一 key，无限重试语义保留。解决 11 key 雪崩式互相挤占空转。
- **per-key 并发 Semaphore(2)**：11 key × 2 = 22 并发上限，拉满又不互挤。
- **致命错误分类**：仅 401/403/404/422 标 key 失效并 raise（无权访问），429/5xx/网络错误切 key 重试。
- **per-model 分片配置不写死**（`nvidia_models.py`）：NvidiaModel 加 max_segment_sec/max_video_mb/max_frames/target_height/target_fps 字段 + `get_video_config()`，不同模型不同限制。

### 🤖 Agent 对话框主界面（豆包风格）
- **agent_dialog.py**：消息气泡 + 上传视频/照片 + 工具箱侧栏（传统功能/监控分析/批量监控/模型管理）+ agent 回复区（思考链+工具调用+结果）。
- **agent_orchestrator.py**：意图分类 7 类 + skill 匹配 + plan 构建 + run_plan 驱动 + on_task_step_done 每轮介入决策（继续/停/换策略）+ configure_provider_dialog 对话式配 key（测活性）+ download_model_dialog 帮用户下模型（SHA256 校验）。
- **main_window 集成**：主界面改为 AgentDialog，旧 AgentPanel 保留兼容，batch_tab 作为工具箱一项接入。

### 🗄️ 数据库存储层（`src/core/run_store.py`）
- **三表 schema**：runs（视频级）/ segments（分片级，含 first_token_ms）/ clips（命中片段），WAL 并发读写 + 外键级联删除 + status 枚举校验。
- **旧库自动迁移**：`_ensure_column` 探测缺失列 ALTER TABLE 补齐（first_token_ms 向后兼容）。
- **一键清理**：`clear_all(purge_files=True)` 清 DB + 删磁盘 clip 文件。

### 📊 UI 增强（`src/ui/batch_tab.py`）
- **画面变化像素选项带案例**（贴心设计）：5 档下拉每项带实际监控场景案例（"10% - 有人经过门廊"），用户一看就懂选哪档。
- **实时预计完成时间**：基于已跑片平均耗时 × 剩余 / 并发数。
- **任务树两级**：QTreeWidget 顶级=视频，子级=分片懒加载（idx/时间戳/match/confidence/attempts/first_token_ms）。
- **单视频汇总**：总耗时 / API 调用次数 / 平均首字耗时 / 命中数 / 覆盖率%。

### 🧠 skills 蒸馏 + agent 自动匹配
- **surveillance-sparse-corridor skill**（`config/skills/`）：稀疏走廊场景（长时间无人），1fps+场景检测+帧差分+昼夜自适应，只送变化时段给 AI。
- **surveillance-crowded-scene skill**（占位）：人多密集场景用 YOLO 追踪，后续实现。
- **agent_prompt.match_skills 接入**：用户说"分析监控找包" → 自动匹配 sparse-corridor；"商场人流分析" → crowded-scene；密集优先；显式 triggers 优先。

### 🔗 Kilo + 知识库 RAG
- **kilo_provider.py**：Kilo OpenAI 兼容多 key 轮换（401/403/429 切 key），用于 agent/编码通道（Kilo 不支持视频）。
- **kb_rag.py**：复用现有 history_manager.search_kb + kb_indexer.get_embedder（同一 kb_frames collection），Kilo 做 LLM 问答，Kilo 不可用时退化为纯检索。

### ✅ 验证（真实证据）
- **全量回归**：362 passed, 2 deselected, 28 warnings（warnings 全是预存在第三方 deprecation）。
- **pyflakes**：10 个新源文件零告警。
- **真实 E2E（63 视频）**：63/63 done, 0 failed, 1 可靠命中（`_388.mp4` 第536秒，二次验证 has_target_item=true，AI 描述"有人从左走向右拖着黑色旅行袋"）。
- **11 个 NVIDIA key 全活**（实测 PONG）。

## [5.5.0] — 2026-09-04 · 黑匣子透明化 Critic 闭环 + 小白易用弹窗 + 资源/内存安全

### 🧭 黑匣子透明化（Critic 轮1 全 MAJOR/MINOR 修复）
- **决策日志异常分支补齐**（MAJOR-1）：ChatWorker 工具抛异常时也写 `DecisionEntry(status="error")`，黑匣子不再漏掉最有记录价值的失败调用。
- **决策日志 args 字段**（MAJOR-2）：`DecisionEntry` 新增 `args_json`，ChatWorker 工具调用点传入真实参数 JSON；`DecisionLogPanel` 详情区"参数"行从拼凑伪内容改为展示真实工具入参。
- **headless filename 路径消毒**（MAJOR-3）：`run_analysis` 对客户端 multipart filename 取 `Path(filename).name`，防 `..\\x.mp4` / `C:\\evil.mp4` 逃逸 workdir 写任意路径（commit 213cf57 声称已修但实际未落地）。
- **delete_this_history 文案诚实化**（MAJOR-4）：eli5 文案改"⚠️ 危险操作，后端会拦截等待你确认"，与后端实际行为（返回需确认提示不真删）一致。
- **bearer scheme 大小写不敏感**（MINOR-5）：`_check_auth` 按 RFC 7235 scheme 不区分大小写；非 ASCII token `compare_digest` 抛 TypeError 兜底 401 不 500。
- **skills 导入反馈**（MINOR-7）：`SkillsManagerTab` 新增顶部状态行，导入失败（无 SKILL.md / 目标已存在 / 拷贝失败）UI 可见红字提示，不再只 logger.warning。
- **_prompt_to_messages 语义倒置**（MINOR-5 logic）：注释说明真实链路 prompt 形态（六模块 system prompt + 视频上下文分隔符 + user question），P2-2 双路径（APIGatewayClient + _GatewayClientAdapter）都把分隔符解析成 system+user 双消息，未来 main_window 改用 build_system_prompt 后可移除过渡分支。

### 🪟 小白易用 P1（audit-blinds "假按钮"）
- `main_window.py`：无视频/无关键帧/无模型时点按钮静默 return → `QMessageBox` 中文弹窗 + 明确下一步动作（Phase 1 先提取 / 检查抽帧密度 / 加载模型流程）。
- `agent_tools.py create_highlight_cut`：VideoFileClip try-finally 关闭（异常路径不泄漏）；输出文件名加时间戳 `highlights_YYYYMMDD_HHMMSS.mp4` 防连续调用覆盖；描述匹配从单字符命中改 jaccard 分词交集（旧实现中文'的/了/是'几乎必中，任意描述趋同，集锦等于取前3帧）。

### 🛡️ 资源/内存安全（audit-prod）
- `logic.py` kb 帧编码分批 `BATCH=64` + 即时 `Image.close()`，防 10000 帧全量载入内存峰值。
- `history_manager.py clear_all_history` 兜底清理 ChromaDB 孤儿向量（sessions 无行但 kb_frames 残留的条目）。
- `.env.example` 改为只列代码真正读取的变量（grep 实测），移除历史误导项 `VAP_LLM_PROVIDER/BASE_URL/MODEL/API_KEY/MONITOR_DIR`（代码从未消费，监控真实凭据来源是 GUI LastUsed 配置）。
- `Dockerfile` / `Dockerfile.cuda` / `docker-compose.yml` 同步补 seaborn/matplotlib/pandas 与 VAP_HEADLESS_TOKEN 生产鉴权提示（Docker 形态不再天然复活 Phase3 全禁用 + 公网裸奔）。

### 🔬 headless 服务加固
- 并发信号量 `_ANALYZE_SEMAPHORE`（`VAP_ANALYZE_CONCURRENCY` 默认 1）串行化 run_analysis，防并发重载 Whisper/Ollama 致 VRAM OOM；503 `_ServerBusy` 映射。
- 500 响应不回传 `str(e)`（含 workdir/模型路径），只回通用 message + job_id。
- 可选 Bearer Token 鉴权（`VAP_HEADLESS_TOKEN` 空=禁用；`/healthz` 永不鉴权供 Docker healthcheck）。

### ✅ 验证（全部真实执行）
- `pytest tests/`：140 passed（标准 + smoke + headless + gateway + decision_log + eli5 + skills + surveillance_tab + concurrency）
- `ruff check src/ launcher.py --select F`：All checks passed
- `mypy` 新模块 8 文件：Success no issues
- 集成 E2E：ChatWorker→entry_append→DecisionLogPanel 跨线程信号链路真实工作；headless token 鉴权真实 401/200；gateway 工厂路由 4 协议；filename 消毒 `..\\..\\evil.mp4`→`evil.mp4`

---

## [5.4.0] — 2026-09-04 · 官网前端诊断闭环 + spec-kit 审计工作流

### 🌐 官网前端（website/）— 诊断报告全部 P0/P1 改进落地
- **3D Hero → Canvas 2D 产品语义动画**：移除 three/@react-three/fiber/@react-three/drei 三依赖（-55 packages），改绘"检测框扫帧+时间码+命中光晕+时间轴指针"动画，与 AI 视频分析产品语义强相关，LCP 负担大降。
- **App Router 容错位补齐**：新增 `error.tsx`（生产只露 digest 不泄漏 message）/`loading.tsx`（骨架屏）/`not-found.tsx`（自定义 404）。
- **基元层**：`lib/utils.ts`(cn) + `ui/Container`(wide/default/narrow 三档) + `ui/Button`(primary/glass/chip 变体)。
- **设计 token 落地**：oklch 字面量全量收敛进 `text-mute`/`text-accent`/`text-accent-2`/`accent-2-deep` 等 class，19+ 处散落字面量清零，改色一处生效。
- **死抽象消除**：6 个 section 标题编排统一收敛进现成 `SectionHeading`（之前造了不用、各 section 手抄重复）。
- **交互闭环**：Header 移动菜单补 Esc 关闭 + overlay 点外部关闭；`globals.css` 加 `scroll-padding-top: 5rem` 修锚点被 fixed 顶栏遮挡；Download 源码卡深链差异化；修 `Features.tsx` `glass-hover` 重复笔误。
- `website/.gitignore` 补 `E2E实测结果/` 排除 Playwright 截图产物。

### 🔬 spec-kit 审计工作流建立
- `.spec/` 宪法 v1.0 + agent-skills ×10 安装，为 v5.4+ 专项审计提供 spec-driven-development 基建。

### ✅ 验证（全部真实执行）
- **website**：`tsc --noEmit` 0 错误 / `eslint` 0 / `next build` 6/6 静态页通过 / **Playwright E2E 8/8 PASS**（Hero canvas 渲染、锚点不遮挡、FAQ 手风琴、自定义 404、移动菜单 Esc+外部关闭、三断点 320/768/1440 截图）。
- **主项目**：`pytest 102 passed`（含 e2e_full_pipeline 修复后）/ `py_compile` 全过 / `ruff` 0 错误。

---

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
