# SOP — 标准操作手册
## Video Analysis Pro v5.0（听风公司）

> 本文档面向：新接手的开发者、运维人员、调用 headless API 的集成方。
> 原则：任何操作先看本文，再动代码；改代码必同步本文。

---

## 1. 环境准备（新机器从零到运行）

```powershell
# Windows（或 Linux/macOS 用 启动应用.sh）
git clone https://github.com/lza6/Video-Analysis-Pro-python
cd Video-Analysis-Pro-python
# 双击 启动应用.bat → 自动建 venv + 装依赖 + 启动
```

- Python 版本门禁：**3.10–3.12**（3.13+ 无 wheel，launcher 会弹窗拒绝）
- FFmpeg：`imageio-ffmpeg` 自带，无需系统安装；也可放 `models/ffmpeg.exe` 覆盖
- LLM 凭据：复制 `.env.example` → `.env`，填 `VAP_LLM_API_KEY`（**.env 永不入库**）

## 2. 日常开发循环

```powershell
py -3.10 -m pytest tests/ -q --ignore=tests/test_surveillance_e2e.py   # 快速回归 (~2min)
py -3.10 -m pytest tests/test_surveillance_e2e.py -m slow -q           # 真实 VLM E2E (~3-10min，需 .env)
py -3.10 -m ruff check src/ launcher.py --select F --ignore F403,F401  # 静态检查
py -3.10 -m mypy --config-file mypy.ini src/core/*.py src/server/*.py src/utils/config_manager.py  # 类型检查
```

**改代码前先查** `workflow_status.md`（已审计项避免重复劳动）与 `.claude/projects` 记忆（E2E 方法/踩坑记录）。

## 3. Headless API 调用（集成方）

```bash
# 健康/能力探测（永远 <50ms）
curl http://localhost:8000/healthz

# 提交视频分析（上传 ≤512MB，超限 413）
curl -X POST http://localhost:8000/analyze \
  -H "X-Filename: video.mp4" \
  --data-binary @video.mp4
# 返回: {job_id, duration, frame_count, frames[], transcript, report}
```

错误码：400 空上传/非法长度｜413 超限｜500 分析失败（看服务端日志）。
限流：上传上限 `VAP_MAX_UPLOAD_MB`（默认512）；Ollama 不可用时 report 为 `[LLM unavailable: ...]` 而非 500（能力自动降级）。

## 4. 监控视频搜索（Surveillance Agent）

```python
from src.core.llm_gateway import AnthropicBackend
from src.core.surveillance_agent import SurveillanceAgent

backend = AnthropicBackend(api_key, base_url, model, max_tokens=1200)
agent = SurveillanceAgent(backend, key_item_image="物品.jpg",
                          item_description="黑色旅行袋", fps=0.5,
                          max_frames_per_video=600, clip_duration=20)
report = agent.run(video_dir="D:/监控", output_dir="E2E实测结果", max_videos=0)
# 产物: search_report.json / search_report.md / clips/*.mp4
```

**已知模型特性**：glm-5.3-flash 强制思考链 → 单帧判断 10-30s + 429 限流（网关已内置退避重试 2s/4s/8s）。选非思考视觉模型可提速 5-10 倍。

## 5. 故障排查速查

| 症状 | 根因 | 处置 |
|---|---|---|
| 启动即弹"Python 版本不受支持" | 3.13+ | 装 3.10-3.12 |
| WinError 1114 c10.dll | torch/PyQt6 DLL 顺序 | 已在 main_window 顶部固化 import torch，勿删 |
| Agent 面板输出 `{"message":...}` | 旧版 bug | 已修（llm_gateway 客户端层解析）；若复现查 OllamaClient.chat_stream |
| 抽帧报 PPS/h264 错误 | OpenCV 对监控流兼容差 | 已用 ffmpeg 抽帧（surveillance_agent），忽略 OpenCV stderr 噪音 |
| Ollama 调用挂起 30s+ | 系统代理劫持 localhost | OllamaClient 已 trust_env=False；确认代理软件绕过规则 |
| 429 Too Many Requests | 中转站限流 | 网关自动退避重试 3 次；持续 429 换模型/降并发 |
| 历史库锁死 | 多进程写 SQLite | 单实例运行；删 config/history.db 前先备份 |

## 6. 发布清单（Release 前）

1. `pytest` 全绿（含 slow E2E）
2. `ruff` + `mypy` 零错误
3. CI 三平台 6/6 success
4. bump `src/utils/constants.py:APP_VERSION`（唯一版本源）
5. `CHANGELOG.md` 追加条目
6. git tag vXXX → push → CI build-windows 自动产 exe
7. GitHub Release 挂产物 + 校验和

## 7. 演进指南（加新功能/新 API 的标准路径）

**新增 LLM 供应商**：在 `llm_gateway.py` 加 `XxxBackend(ProtocolBackend)` → 注册进 `_PROTOCOL_MAP` → `tests/test_llm_gateway.py` 加协议单测 → 三步完成，UI/Agent 自动获得新供应商。

**新增 Agent 工具**：`agent_tools.py` 写 `create_xxx_tool(app_context_getter)` 工厂 → `main_window.init_backend` 注册（name/description/schema）→ 若需 embedding 走 `kb_indexer.get_embedder()` 共享实例 → 补单测。

**新增 headless 端点**：`headless.py` Handler 加 do_GET/do_POST 分支 → 复用 core 层函数（勿在 handler 写业务）→ tests/test_headless_server.py 加集成测试。

## 8. 测试范围台账（防重复劳动）

| 范围 | 上次审计 | 结论 |
|---|---|---|
| gateway 重试/协议路由 | 2026-09-04 | 9 测试全覆盖，无已知缺口 |
| KB/偏好记忆 | 2026-09-04 | 并发锁已加，round-trip 已测 |
| RTSP 运动检测 | 2026-09-04 | 冷却期 bug 已修 |
| headless 上传 | 2026-09-04 | Content-Length 前置校验已加 |
| UI 线程安全 | 2026-09-04 | 信号槽全走 pyqtSignal |
| DB 慢查询 | N/A | SQLite 单机小库，无索引问题（勿盲目优化） |

> 修改上表任一范围的代码后，删除对应行结论并重新审计。
