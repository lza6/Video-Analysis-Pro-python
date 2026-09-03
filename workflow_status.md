# 终局闭环总审计 · workflow_status.md

> 节点 N1 产物 · 2026-09-04 · 状态：**推演完成，主线程推进中**
> 本文件是主线程+审查线程循环的起点。所有「已闭环」判定必须有证据；无证据一律降级。

---

## A. 需求追踪矩阵

### A.1 显式需求

| # | 需求 | 映射位置 | 状态 | 证据 | 缺口 | 后续 |
|---|------|---------|------|------|------|------|
| E1 | Video Analysis Pro 官网落地页 | `website/src/app/page.tsx` + 7 section | 已闭环 | `npm run build` ✓ · HTTP 200 · Playwright 截图 8 张 | 无 | — |
| E2 | 玻璃拟态 2.0 视觉 | `globals.css` glass/glass-edge + 各组件 | 已闭环 | 截图可见毛玻璃+棱边高光 | 无 | — |
| E3 | Motion 动效 + reduced-motion 降级 | 各 section `useReducedMotion` + CSS `@media` | 部分闭环 | 代码存在 | 未实测 reduced-motion 路径 | N4 |
| E4 | R3F 3D（中度·仅 Hero） | `HeroScene.tsx` + `HeroVisual.tsx` | **部分闭环** | 截图可见晶体+粒子 | WebGL 失败兜底是假的（B-特检1） | N3a |
| E5 | 现代 SEO（meta/JSON-LD/sitemap/robots/OG） | `layout.tsx` + `sitemap.ts` + `robots.txt` + `opengraph-image.tsx` | **部分闭环** | build 产出路由 | metadataBase 用占位域名（B-特检2） | N3d |
| E6 | 软件桌面 UI 截图进 README | `capture_desktop_ui.py` + `docs/screenshots/desktop/` | 已闭环 | 8 张 PNG，README 嵌入 | 无 | — |
| E7 | 截图放仓库 md 展示 | `UI-展示.md` + `README.md` 路径 `docs/screenshots/` | 已闭环 | 16 PNG 存在 | 无 | — |
| E8 | 不需要部署 | — | 不适用 | 用户明确说 | — | — |
| E9 | 真实可上生产（已在生产使用） | 整个项目 | **未闭环** | 见 B 节多处伪闭环 | 多项 | N3 全量 |
| E10 | 生成 HTML 变更报告+测验 | 未创建 | 未闭环 | — | 还没做 | N7 |
| E11 | critical-code-reviewer 严审 | 未执行 | 未闭环 | — | — | N5 |

### A.2 隐式需求

| # | 需求 | 映射 | 状态 | 证据/缺口 |
|---|------|------|------|----------|
| I1 | 官网可被 `git clone` 后获取 | `.gitignore` 含 `website/` | **未闭环** | website 源码被排除，clone 者拿不到源码，只有截图 |
| I2 | 桌面应用真实可启动 | `main_window.py:2313 qdarktheme.setup_theme("dark")` | **未闭环** | Py3.14 装的是 0.1.7 无 setup_theme，真实启动会崩（B-特检4） |
| I3 | 文档与真实行为一致 | README/UI-展示/CLAUDE.md | 部分闭环 | CLAUDE.md 引用 `.wolf/` 但目录不存在 |
| I4 | 无伪实现 | HeroVisual onError | **未闭环** | B-特检1 假兜底 |
| I5 | 链路完整（UI↔core） | PyQt 信号槽 | 不适用 | 本次未改桌面核心逻辑 |
| I6 | 关键路径可验证 | — | 部分闭环 | build+HTTP 已验；axe/Lighthouse/CWV 未跑 |

### A.3 验收导向要求

| # | 要求 | 状态 | 缺口 |
|---|------|------|------|
| V1 | 一次调用跑通 | 官网 ✓；桌面 ✗(Py3.14) | N3b/N4 |
| V2 | UI/按钮真实接通 | 官网静态无后端，链接点通未验证；桌面仅截图未交互测 | N4 |
| V3 | 非付费资源真实验证 | 未跑 axe/Lighthouse | N4 |
| V4 | md/README 主动更新 | README✓ UI-展示✓ CHANGELOG 未更新 | N6 |

### A.4 非功能性要求 + SaaS 基础设施适用性裁定

项目本质 = **单用户 PyQt6 桌面应用 + 纯静态 Next.js 落地页**。无后端服务器、无暴露 API、无多租户、无 auth、无外部 DB（chromadb 本地嵌入 + sqlite history）。

| 基础设施 | 适用 | 原因 |
|---------|------|------|
| Load Balancer | ✗ | 无服务端，静态站 CDN 托管，桌面单进程 |
| Redis/Cache Aside | ✗ | 无服务端缓存层 |
| CDN | △ 仅落地页 | Next.js 静态资源自带 |
| DB Replication | ✗ | 本地 sqlite + chromadb 单机 |
| Sharding | ✗ | 同上 |
| Kafka/RabbitMQ | ✗ | 无跨服务消息；QThread 是进程内异步 |
| Rate Limiting | ✗ | 落地页无端点；桌面调 LLM 受对方限流 |
| Circuit Breaker | △ 适用 | llm_gateway.py 调 Ollama/OpenAI 应有重试/超时/熔断 — 需审查 |
| Health Checks | ✗ | 无长驻服务 |
| Observability | △ 桌面有 | RotatingFileHandler 日志已存；落地页静态无观测 |
| 前后端契约防坑 | ✗ | 无传统后端 |
| 极限施压 | ✗ | 单机桌面应用 |
| 慢查询猎杀 | △ | chromadb 向量检索 + sqlite history 可能慢，非本次范围 |
| SQL 注入审查 | △ | sqlite 参数化查询需复核 history_manager.py |

**诚实结论**：用户列的 SaaS 基础设施 80% 不适用于本项目本质。真正适用：LLM 网关熔断/重试/超时、sqlite 参数化复核、桌面日志可观测性。其余不假装做。

---

## B. 最强自我反驳

**BLUF：中等偏差。** 官网视觉层真实闭环（截图为证），但有 1 个 P0 级伪兜底、1 个 P1 级生产启动断裂、若干 P2 文档/SEO 伪闭环被包装成「已完成」。

| # | 方向 | 具体问题 | 影响 | 风险 | 已修 | 下一步 |
|---|------|---------|------|------|------|--------|
| 1 | 偷换需求 | 用户要「可上生产」，我交付「能 build」 | 生产启动路径未验 | P1 | ✗ | N2/N4 |
| 2 | 浅实现当完整 | HeroVisual WebGL 兜底是死代码（特检1） | 无 GPU 用户黑屏 | P0 | ✗ | N3a |
| 3 | 只修表面 | 截图脚本能跑≠应用能跑（特检4 shim 掩盖崩溃） | 截图误导 | P1 | ✗ | N3b |
| 4 | 漏操作路径 | 官网 Download 链接、导航锚点未点通验证 | 死链风险 | P2 | ✗ | N4 |
| 5 | 漏调用方坑 | metadataBase 占位域名（特检2）误导 SEO 闭环 | 上线即错 | P2 | ✗ | N3d |
| 6 | 破坏旧逻辑 | capture 脚本 monkeypatch qdarktheme，未改生产代码 | 无回归 | — | — | — |
| 7 | 默认环境存在 | Py3.14 装不上 pyqtdarktheme≥2.1.0（特检4） | 真实启动崩 | P1 | ✗ | N3b |
| 8 | 没更新文档 | CHANGELOG 未更新；CLAUDE.md 引用不存在的 `.wolf/` | 误导 | P2 | ✗ | N6 |
| 9 | 无证据宣称完成 | 宣称「WCAG AA」「CWV 达标」从未跑 axe/Lighthouse | 伪闭环 | P1 | ✗ | N4 |
| 10 | 漏用户会遇到 | website 被 gitignore，clone 拿不到源码（特检3） | 交付丢失 | P1 | ✗ | N3c |

### B.特检 诚实逐条核实

**特检1 — HeroVisual onError 是假兜底？** → **P0 伪兜底**
React `onError` 对 `<div>` 不会因子 `<canvas>` 的 WebGL 上下文丢失而触发。WebGL 失败走 canvas `webglcontextlost` 事件 + 控制台 error，不冒泡到父 div 的 error 事件。`webglFailed` 永远不会被这条路径置真。无 GPU / WebGL 被禁用户看到空白 Canvas，不触发 fallback。
修法：R3F `onCreated` 回调捕获错误，或监听 `webglcontextlost` 事件，或用 `detect-gpu` 预检。

**特检2 — metadataBase 占位域名是伪闭环？** → **P2（scope 可忍，上线升 P0）**
`SITE_URL = "https://video-analysis-pro.example.com"` 渗透到 metadataBase / canonical / sitemap / robots / JSON-LD url。`.example.com` 是保留域名，永不解析。用户说「不需要部署」本地可接受，但「SEO 闭环」宣称是伪的。修法：改成 `process.env.NEXT_PUBLIC_SITE_URL` 可配置。

**特检3 — 截图真在 docs/ 下了吗（website 被 gitignore）？** → **截图闭环 ✓，网站源码交付未闭环 P1**
16 PNG 确实存在于 `docs/screenshots/`。但 website 源码被 `.gitignore` 的 `website/` 排除，`git add -A` 不跟踪。clone 者拿不到 Next.js 源码，只拿到截图。修法：要么 commit website/，要么显式说明。

**特检4 — qdarktheme 版本，生产会不会炸？** → **P1 生产启动断裂**
- requirements.txt 写 `pyqtdarktheme>=2.1.0`
- 当前 venv 是 Python 3.14.3
- `pip install "pyqtdarktheme==2.1.0"` → `ERROR: No matching distribution found`（2.x 无 Py3.14 wheel 或已更名）
- 实装 0.1.7，**无 `setup_theme` 属性**
- 生产代码 `main_window.py:1334/1422/2313` 直接调 `qdarktheme.setup_theme(...)`
- 截图脚本的 monkeypatch shim 让截图跑通，**掩盖了真实启动会崩**
- 修法：requirements.txt 放宽到 `pyqtdarktheme>=0.1.7` + main_window.py 加 API 兼容层，或限制 Python<3.13

**特检5 — github.com/lza6/Video-Analysis-Pro-python 真实存在？** → **P2 待验证**
来自用户原始 README.md，我继承引用。从未验证它解析。应标注「待验证」。

---

## C. 工作流节点图

```
N1 (推演·本文件) ──串行──> N2 (落盘 workflow_status.md)
                                    │
                                    ▼
                              N3 (修复批·并行四子节点)
                              ├── N3a 修 WebGL 真兜底
                              ├── N3b 修 qdarktheme 版本兼容
                              ├── N3c website 源码进 git 或显式说明
                              └── N3d metadataBase 可配置化
                                    │
                                    ▼ (全部 done)
                              N4 (真实验证·串行)
                              ├── npm run build + dev
                              ├── Playwright emulate reduced-motion
                              ├── axe-core 无障碍扫描
                              ├── Lighthouse CWV
                              ├── 桌面 launcher.py 真实启动（非 shim）
                              └── 死链扫描（Download/导航锚点）
                                    │
                                    ▼
                              N5 (独立审查线程·只读·6 维)
                              需求完整性/逻辑正确/边界/代码质量/测试覆盖/运行结果
                                    │
                            ┌───────┴───────┐
                            ▼               ▼
                        PASS            返修清单→回 N3
                            │
                            ▼
                              N6 (文档同步·串行)
                              ├── README.md
                              ├── UI-展示.md
                              ├── CHANGELOG.md
                              ├── workflow_status.md (标记 done)
                              └── CLAUDE.md/.wolf 一致性
                                    │
                                    ▼
                              N7 (HTML 变更报告 + 测验)
                                    │
                                    ▼
                              N8 (Skill 化 + 记忆沉淀)
                              ├── 截图+审计流程封装可复用 skill
                              └── 写入记忆：本次验证范围/已优化点，避免重复跑
                                    │
                                    ▼
                               DONE
```

### 节点契约

| 节点 | 类型 | 交付物 | 验收标准 | 依赖 |
|------|------|--------|---------|------|
| N1 | 推演 | 本文件 | 三块齐全，特检5条有结论 | — |
| N2 | 落盘 | workflow_status.md | 文件存在，主线程可读 | N1 |
| N3a | 修 | HeroVisual.tsx | WebGL 失败真触发 fallback | N2 |
| N3b | 修 | requirements.txt + main_window.py | `python launcher.py` 在当前 venv 不崩 | N2 |
| N3c | 修 | .gitignore 或 website 归属说明 | clone 后能拿到源码或明确文档说明 | N2 |
| N3d | 修 | layout.tsx + sitemap.ts + robots.txt | SITE_URL 来自 env，默认值合理 | N2 |
| N4 | 验证 | 验证报告 | build✓ + axe 0 critical + Lighthouse CWV + launcher 不崩 + 无死链 | N3 全 |
| N5 | 审查 | 修复清单 | 6 维逐项 PASS 或返修项 | N4 |
| N6 | 文档 | 4 份 md 更新 | 与真实状态一致，无「已完成」谎言 | N5 PASS |
| N7 | 报告 | HTML 报告+测验 | 含上下文/直觉/做了什么/底部测验 | N6 |
| N8 | 沉淀 | skill + 记忆 | 可复用，下次优先读记忆不重复跑 | N7 |

### 循环终止条件
- 全部节点 done 且 N5 连续一轮 PASS → 标记完成
- 同一阻塞经 2 轮有意义尝试仍无进展 → 报阻塞，不伪造
- 外部受限项（付费 API、真实 GPU、GitHub URL 不可达）→ 写边界说明，不算未闭环

### 并行/串行裁定
- N3a/b/c/d **可并行**（不同文件，无共享状态）
- N3 → N4 **必须串行**（验证依赖修复完成）
- N4 → N5 **必须串行**（审查依赖验证证据）
- N5 → N6 **必须串行**（文档依赖审查 PASS）

---

## D. 进度日志

| 时间 | 节点 | 状态 | 备注 |
|------|------|------|------|
| 09-04 00:12 | N1 | ✅ done | 推演完成 |
| 09-04 00:12 | N2 | ✅ done | 本文件已落盘 |
| 09-04 00:13 | N3 | ⏳ 进行中 | 启动并行修复批 |
