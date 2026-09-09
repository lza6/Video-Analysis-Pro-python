# 反 AI-slop 设计自查清单 (DESIGN.md)

> v10.2 P3-2「反 AI-slop 设计升级」— 数据可视化即设计系统。
> 作用:后续所有 webapp 页面 / 组件对照此清单自查,避免生成"千篇一律 AI 界面"。

## 反模式禁令(永不引入)

- [ ] **禁 Inter 默认字体栈** — 一律用 `--font-sans`(Noto Sans SC 栈)+ 数据场景 `--font-data`(JetBrains Mono)
- [ ] **禁紫蓝渐变当主信号** — 主信号固定电光青 `--color-accent`;紫蓝(`--color-accent-2/3`)只做点缀,不抢主
- [ ] **禁卡中卡(嵌套卡片层叠)** — 一个视觉面最多一层卡片
- [ ] **禁均匀圆角 + 大阴影** — 容器 `rounded-card`(弧形)与操作件/图表 `radius-btn`(锐角)软硬对比,阴影用玻璃拟态弱阴影
- [ ] **禁千篇一律 card-grid** — 网格必须有层次/节奏差异(双端定标、基线刻度、局部高光)
- [ ] 禁 emoji 当图表图标(用 SVG)、禁无悬停态的静态图表、禁把第三方图表库塞进核心依赖

## 10 品质自检(每项 >=4 达标)

| # | 品质 | 落地点 |
|---|------|--------|
| 1 | 层次(scale contrast) | `text-2xl font-black` 数值 + `text-[11px] uppercase` 标签 + `text-xs` 说明,三级字号梯度 |
| 2 | 节奏(spacing) | 间距用 token 体系,块间 `gap-4/6`,内部 `p-4/5`,不一刀切 `p-5` |
| 3 | 纵深(depth/layering) | `glass-strong` 卡片 + `glass-edge` 描边 + 顶部电光青 1px 高光(局部层次,非卡中卡) |
| 4 | 字体(typography) | 数字永远是 `--font-data`(JetBrains Mono);标签 uppercase + 0.14em 字距 |
| 5 | 色彩语义(color semantics) | 语义色阶:`accent`(主)/`accent-dim`(弱)/`ok`/`danger`/`warn`/`mute`,不用纯装饰色 |
| 6 | 悬停态(hover/focus/active) | 图表 hover 抬升/提亮/放大,过渡用 `--duration-fast(150ms)` 或 `--duration-normal(300ms)` + `--ease-out-expo` |
| 7 | 编辑构成(editorial/bento) | 图表用双端定标(标签左对齐/值右对齐)、基线刻度线、折线末端动态点,打破卡片方阵 |
| 8 | 质感(texture/atmosphere) | 玻璃拟态 blur+saturate + 电光青顶部高光 + body 背景青色微晕,禁纯色单调底 |
| 9 | 动效(motion) | 过渡只动 transform/opacity/border-color(合成器友好),禁 width/height/top 动画;`prefers-reduced-motion` 已全局降级 |
| 10 | 数据可视化即设计系统 | 图表是设计系统一部分:全部走 token(色彩/字体/动效/描边),纯 SVG 零第三方依赖,新图表组件进 `src/components/charts/` 复用 |

## 数据可视化 token 对照

| token | 值 | 用途 |
|-------|-----|------|
| `--color-accent-strong` | `oklch(0.93 0.13 190)` | 图表主线 hover 提亮/终点高亮 |
| `--color-accent-dim` | `oklch(0.72 0.11 190)` | 图表弱化/渐变起点/平铺色 |
| `--color-surface-raised` | `oklch(0.24 0.04 268)` | 卡片抬升分层 |
| `--color-border-glass` | `oklch(1 0 0 / 0.14)` | 玻璃描边/图表基线 |
| `--duration-fast` | `150ms` | hover 反馈 |
| `--duration-normal` | `300ms` | 位移/抬升 |
| `--ease-out-expo` | `cubic-bezier(0.16,1,0.3,1)` | 统一缓动 |

## 已接入组件

- `src/components/charts/Sparkline.tsx` — 迷你趋势线(电光青渐变主线 + 数据字体 + 末端语义色动态点 + hover 放大)
- `src/components/charts/MetricTile.tsx` — KPI 磁贴(数值 JetBrains Mono + SVG 增量箭头 + 底部迷你条形趋势 + 顶部电光青高光 + hover 抬升)
- `src/components/charts/BarStrip.tsx` — 横向条形分布(track 双渐变 depth + fill 语义色 + hover 整行提亮)
- `webapp/src/app/(app)/metrics/page.tsx` — P3-2 演示接入:指标卡换 `MetricTile`,时间序列区块加 `BarStrip`

## 自查时机

每次新增/修改页面、组件、图表时,逐项对照。红线反模式一票否决;10 品质至少满足 4 项(图表/核心页面建议 >=6 项)。改 globals.css token 时同步本表。
