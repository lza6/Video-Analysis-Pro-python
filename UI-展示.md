# Video Analysis Pro · UI 展示

> 桌面端软件 + 官网落地页 完整截图集
> 全部为真实渲染捕获（桌面端：PyQt6 offscreen + 2x retina；官网：Playwright + 2x retina），未做任何后期合成。

---

## 一、桌面端软件（PyQt6 · 深色主题 · 1440 × 900）

### 主界面概览

![桌面端主界面](docs/screenshots/desktop/desktop-overview.png)

三栏布局：左侧数据提取区（拖拽视频 / 提取密度 / 开始按钮）、中央多 Tab 工作区（7 个功能页签）、右侧 Agent 智能助手面板（类 DeepSeek R1 思维链）。深色主题由 qdarktheme 驱动，控件间距遵循 8px 栅格。

### 各功能 Tab 详图

#### 📝 AI 摘要报告

![AI 摘要报告](docs/screenshots/desktop/desktop-tab-report.png)

AI 阅读视频帧与字幕后生成的结构化报告——内容总结、技术分析、情感识别。支持 Markdown 渲染、导出 PDF。

#### 🖼️ 关键帧画廊

![关键帧画廊](docs/screenshots/desktop/desktop-tab-gallery.png)

算法自动识别画面变化最显著的关键时刻，画廊式轮播展示，告别机械等间隔截图。

#### 🎬 摘要媒体 (GIF/Clips)

![摘要媒体](docs/screenshots/desktop/desktop-tab-media.png)

AI 挑选精彩片段，自动拼接短视频或 GIF 动图，便于分享到社交媒体或插入演示文稿。

#### 📊 元数据与画质

![元数据与画质](docs/screenshots/desktop/desktop-tab-metrics.png)

亮度、清晰度、饱和度趋势的专业图表，一眼看懂视频质量曲线。

#### 📜 系统日志

![系统日志](docs/screenshots/desktop/desktop-tab-logs.png)

详细日志面板，FFmpeg 自愈、模型加载、分析进度全程可见，方便排查问题。

#### 📦 模型管理

![模型管理](docs/screenshots/desktop/desktop-tab-models.png)

本地 Ollama（Llama3、Qwen2.5）与云端 OpenAI 格式 API（DeepSeek、GPT-4o、Claude）统一管理，支持模型下载与切换。

#### 💡 获取 API

![获取 API](docs/screenshots/desktop/desktop-tab-api-help.png)

API 申请引导页，降低新手接入云端模型的门槛。

---

## 二、官网落地页（玻璃拟态 2.0 · Next.js + Motion + React Three Fiber）

### 桌面端（1440 × 900）

#### Hero 首屏

![Hero 首屏](docs/screenshots/opt/full/desktop-1440-hero-top.png)

中央 3D 晶体（icosahedron + 物理透射材质 + 虹彩）随指针视差旋转，环绕双轨道环与 220 粒子场。左侧大标题用青→紫→粉极光渐变，CTA 按钮为渐变实心 + 玻璃幽灵按钮双层级。背景三团极光光斑缓慢漂移，整页叠加细颗粒噪点增加"物理感"。

#### 功能区 Features

![功能区](docs/screenshots/opt/section-features.png)

Bento 不对称网格——首张"多模态 AI 分析"占两列宽，其余五张等宽。每张玻璃卡有顶部高光描边（`glass-edge` 伪元素 mask 实现），hover 时上浮 4px + 紫色辉光投影。

#### 工作原理 How It Works

![工作原理](docs/screenshots/opt/section-how.png)

三阶段流水线，中间"AI 分析"卡上移 8px 并升级为强玻璃层，形成视觉焦点。中央有青→紫→粉的垂直渐变连接线（滚动时 scaleY 展开）。每张卡含序号大字（渐变文字）、阶段标签、技术栈 chip。

#### 下载区 Download

![下载区](docs/screenshots/opt/section-download.png)

强玻璃容器 + 顶部紫色光晕。四平台卡片（Windows 标记为主推，边框加亮），hover 上浮。底部标注 Python/FFmpeg/GPL/听风公司。

#### 常见问题 FAQ

![常见问题](docs/screenshots/opt/section-faq.png)

手风琴折叠，`+` 号 hover 旋转 45° 变 `×`。展开用 AnimatePresence + height 动画，reduced-motion 下直接显示。

### 平板端（768 × 1024）

![平板整页](docs/screenshots/opt/full/tablet-768.png)

导航汉堡菜单，Bento 网格降为 2 列，三阶段流程纵向堆叠，所有玻璃质感与动效保留。

### 移动端（375 × 812）

![移动端整页](docs/screenshots/opt/full/mobile-375.png)

单列布局，Hero 标题降至 5xl，3D 场景高度收缩至 340px。导航折叠为顶部汉堡 + 抽屉菜单。所有触控目标 ≥ 44px，对比度满足 WCAG AA。

---

## 三、设计系统速览

| 维度 | 桌面端（PyQt6） | 官网（Next.js） |
|------|----------------|----------------|
| 色彩 | qdarktheme 深色主题 | oklch 色彩空间 · 墨蓝底 + 青/紫/粉三色极光点缀 |
| 玻璃 | N/A（原生控件） | `backdrop-blur(20-28px) saturate(150-160%)` + 双层渐变 + inset 高光 + mask 棱边 |
| 字体 | 系统字体栈 | Noto Sans SC + JetBrains Mono，`font-display: swap` |
| 动效 | QThread 异步刷新 | Motion 声明式 · stagger 编排 · 滚动联动 · reduced-motion 全降级 |
| 3D | N/A | R3F + drei · 动态 import · SSR fallback · WebGL 失败降级 |
| 无障碍 | 原生 Qt a11y | 语义 HTML · `focus-visible` ring · 对比度 ≥ 4.5:1 · 3D `aria-hidden` |
| SEO | N/A | Metadata API + JSON-LD + sitemap.xml + robots.txt + 动态 OG 图 |

---

## 四、本地预览与重新截图

### 官网

```bash
cd website
npm install              # 首次
npm run dev              # 预览 → http://localhost:3000
node scripts/capture.mjs # 重新捕获官网截图（含自动压缩优化）
```

### 桌面端软件

```bash
# 在项目根目录，用 venv 运行
./venv/Scripts/python.exe capture_desktop_ui.py
```

截图脚本：
- 官网：`website/scripts/capture.mjs`（Playwright · 3 断点 × 2x retina → sharp 压缩至 1600px 宽）
- 桌面端：`capture_desktop_ui.py`（PyQt6 offscreen · 2x retina · 7 tab + 主界面概览）

原始 2x 截图：`docs/screenshots/desktop/raw/` 与 `docs/screenshots/`；优化版：`docs/screenshots/desktop/` 与 `docs/screenshots/opt/`。
