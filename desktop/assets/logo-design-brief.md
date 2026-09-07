# TingFeng Hermes — Logo 设计简报

> TingFeng Hermes v9.0.0 品牌视觉规范
> 出品：听风公司 (Tingfeng) · 维护者 `lza6`
> 版本：1.0 · 2026-09-06

---

## 一、品牌名与寓意

| 项 | 内容 |
|----|------|
| 品牌名 | **TingFeng Hermes**（听风·赫尔墨斯） |
| 出品方 | 听风公司 (Tingfeng) |
| 定位 | 通用 Agent，通过 IM 网关连接万物、传递消息 |

**寓意拆解**：

- **听风 (TingFeng)** = 感知信号。风是看不见的流动，"听风"即捕捉环境中最微弱的信号流——视频帧、音频、事件、RTSP 流、IM 消息。
- **Hermes（赫尔墨斯）** = 希腊神话信使之神，标志是**双翼帽**（飞行的速度）与**双蛇杖 Caduceus**（信息与传递的权柄）。
- **合义**：感知万物信号并传递的**智能信使**。Agent 不只是分析视频，而是把分析结论通过 IM 网关送达人、送达系统、送达设备——这是 Hermes 的本职。

## 二、配色方案（与 website/ 玻璃拟态 2.0 同源）

所有色值以 OKLCH 为规范（感知线性，跨设备更稳），附 sRGB 近似值供不支持 OKLCH 的工具回退。

| 角色 | OKLCH | sRGB 近似 | 用途 |
|------|-------|-----------|------|
| 深底 | `oklch(18% 0.02 250)` | `#15182a` → `#2a1f4a` 渐变 | 图标底、顶栏底、启动画面底 |
| 主色 · 风蓝紫 | `oklch(68% 0.21 250)` | `#6b7cf6`（渐变 `#8b9bf9` → `#5b6ce0`） | 蛇杖、蛇身、外框强调、主标题 |
| 强调 · 智能绿 | `oklch(72% 0.18 140)` | `#4ee0a0`（核心 `#9af5c8`） | 杖头智能核心球、信号粒子、副标题 |
| 文字 · 高对比 | `oklch(96% 0.01 250)` | `#f4f6ff` → `#c9d0f0` | 横版 logo 主标文字 |

**色彩语义**：

- 风蓝紫 = "风"的可见化 + AI 的冷智能感
- 智能绿 = 信号粒子 + 核心 AI 的"激活态"
- 深底 = 玻璃拟态的"夜空"基底，让信号与翼面浮起

## 三、构图说明

### 元素层

```
┌─────────────────────┐
│      ◆ 核心球        │  ← 杖头智能核心（智能绿）= Hermes 帽顶 + AI 核心
│   ▲▲▲ 翼 ▲▲▲        │  ← 三片几何羽毛（左/右镜像）= 信使之翼 + 风的流动
│      │ 杆            │  ← 中央竖轴 = 蛇杖之杆
│   ( ( 蛇身 ) )       │  ← 双蛇 S 形缠绕（镜像相位）= Caduceus 双蛇
│      │               │
│      ● 杖底球        │  ← 蛇杖收束
│   · ·  ·            │  ← 三颗信号粒子（智能绿）= "感知万物信号"
└─────────────────────┘
   外框：圆角方形 rx=56
```

### 设计原则

1. **简洁现代，线条几何**——不画复杂插画，所有形态用 path/circle 基元拼合，矢量在任何尺寸都不糊。
2. **圆角方形外框**——`rx=56`（256 尺寸下）匹配现代应用图标规范（iOS/macOS/Windows 都偏圆角方）。
3. **垂直中轴对称**——双翼、双蛇左右镜像，强化"信使"的庄重感与平衡感。
4. **留白**——核心球与翼之间、翼与蛇头之间留 6-10px 呼吸空间，避免 16px 小尺寸下糊成一团。
5. **三色克制**——只用 深底/风蓝紫/智能绿 三组色，不引入第四色，保证品牌识别一致性。

### SVG 实现要点

- `viewBox="0 0 256 256"`（icon）与 `viewBox="0 0 640 200"`（logo-mark 横版）
- 渐变用 `linearGradient` + `radialGradient`，`gradientUnits="userSpaceOnUse"` 保证缩放一致
- 横版 logo-mark 用 `<symbol id="tfh-mark">` + `<use>` 复用 icon，单一图源
- 文字用 `system-ui` 系统字体栈（不依赖外部字体加载），中文回退 `PingFang SC` / `Microsoft YaHei`
- 全部 path 手写，无外部资源依赖

## 四、文件清单与用途

| 文件 | 尺寸 | 用途 |
|------|------|------|
| `desktop/assets/icon.svg` | 256×256 viewBox | Electron 窗口图标、electron-builder 打包源、任务栏 |
| `desktop/assets/logo-mark.svg` | 640×200 viewBox | 品牌横版标识（icon + "TingFeng Hermes" + 副标），用于关于页/启动画面/README |
| `webapp/public/logo.svg` | 256×256 viewBox | Web 端顶栏 logo（与 icon 同源） |
| `desktop/assets/logo-design-brief.md` | — | 本文件，设计规范与生成指南 |

## 五、应用场景

| 场景 | 使用哪个 SVG | 备注 |
|------|--------------|------|
| Electron 窗口图标 | `desktop/assets/icon.svg` | electron-builder 会从 SVG 渲染多尺寸 PNG/.ico |
| 任务栏 / 系统托盘 | `desktop/assets/icon.svg` | 需 16/32 尺寸，从 SVG 渲染 |
| 安装包图标（NSIS） | `desktop/assets/icon.svg` | electron-builder 配 `icon` 字段 |
| Web 顶栏 | `webapp/public/logo.svg` | Next.js `/public/logo.svg` 直接引用 |
| 启动画面 / 关于页 | `desktop/assets/logo-mark.svg` | 横版带文字 |
| README / 文档 | `desktop/assets/logo-mark.svg` | Markdown `![](...)` 直接嵌入 |

## 六、后续真位图生成建议

SVG 是源，生产环境需要位图（PNG/ICO/ICNS）。**全部从 SVG 渲染，不要手画位图**，保证品牌一致性。

### 推荐工具链

```bash
# 方案 A：sharp（Node.js，跨平台，推荐）
npx sharp-cli -i desktop/assets/icon.svg -o desktop/assets/icon-256.png resize 256 256
npx sharp-cli -i desktop/assets/icon.svg -o desktop/assets/icon-512.png resize 512 512
npx sharp-cli -i desktop/assets/icon.svg -o desktop/assets/icon-1024.png resize 1024 1024

# 方案 B：ffmpeg（项目已自带 imageio-ffmpeg，零额外依赖）
ffmpeg -i desktop/assets/icon.svg -s 256x256 desktop/assets/icon-256.png
ffmpeg -i desktop/assets/icon.svg -s 512x512 desktop/assets/icon-512.png
ffmpeg -i desktop/assets/icon.svg -s 1024x1024 desktop/assets/icon-1024.png

# 方案 C：librsvg（Linux/Docker 环境）
rsvg-convert -w 1024 -h 1024 desktop/assets/icon.svg -o desktop/assets/icon-1024.png
```

### Windows .ico（安装包用）

```bash
# 用 png-to-ico 或 ImageMagick 从 256 PNG 生成多尺寸 ico
npx png-to-ico desktop/assets/icon-256.png > desktop/assets/icon.ico
# 或
magick desktop/assets/icon-256.png -define icon:auto-resize=256,128,64,48,32,16 desktop/assets/icon.ico
```

### macOS .icns（若未来出 Mac 版）

```bash
# 需要 iconutil（macOS 原生）
mkdir icon.iconset
sips -z 16 16     icon-1024.png --out icon.iconset/icon_16x16.png
sips -z 32 32     icon-1024.png --out icon.iconset/icon_16x16@2x.png
# ... 其余尺寸 ...
iconutil -c icns icon.iconset
```

### electron-builder 配置（Builder-A 的 electron-builder.yml 已有 `icon` 字段）

```yaml
# electron-builder.yml 片段（Builder-A 维护，此处仅参考）
win:
  icon: desktop/assets/icon.svg   # 或 icon.ico（生成后）
```

electron-builder 25+ 支持 SVG 直接作为源，会自动渲染多尺寸；若打 NSIS 报错，回退到 `.ico`。

## 七、设计自检

- [x] 三色克制（深底/风蓝紫/智能绿），无第四色
- [x] 双翼 + 双蛇杖 + 核心球，Hermes 语义完整
- [x] 信号粒子（智能绿）呼应"感知万物信号"
- [x] 圆角方形外框，现代应用图标规范
- [x] 系统字体栈，无外部字体依赖
- [x] `<symbol>` 复用，icon 与 logo-mark 图源单一
- [x] viewBox + xmlns 合法，`xml.etree.ElementTree.parse` 通过
- [x] 与 website/ 玻璃拟态 2.0 同源配色（深底 + 蓝紫 + 绿强调）
- [x] 16px 小尺寸下核心球+杆+双翼仍可辨识（手测：主要轮廓 stroke-width ≥ 4）

## 八、版权

- Logo 与设计简报 © 听风公司 (Tingfeng)，随项目 GPL-3.0 开源。
- Hermes / Caduceus 为公共领域神话符号，无商标冲突。
- 配色与构图原创，不引用任何第三方 logo 设计。
