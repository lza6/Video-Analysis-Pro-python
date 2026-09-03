# Vibe Coding — 现代前端应用开发计划

> 版本 v1.1 · 2026-09-03 · 状态：**Phase 0 已确认，Phase 1 待启动**

## Phase 0 决策记录（2026-09-03 stakeholder 确认）

| 决策项 | 结论 |
|--------|------|
| 产品定位 | **Video Analysis Pro 的官网落地页**（功能展示 + 下载入口） |
| 技术路线 | **Next.js (App Router) + Motion 库 + React Three Fiber** |
| 视觉风格 | **B. 玻璃拟态 2.0**（毛玻璃层叠、真实景深、渐变光斑） |
| 3D 强度 | **中度**：Hero 一个 3D 场景（指针视差/粒子场），其余区域 2D 动效 |
| 多语言 | 待定（默认单语简体中文） |
| 部署 | 待定（默认 Vercel） |

> 理解修正：任务中 "Framer" 按 **Motion 动画库**（原 Framer Motion）实现，非 Framer 建站平台。

---

## 0. 三个必须先澄清的前提

| # | 问题 | 说明 | 默认假设 |
|---|------|------|---------|
| 1 | **"Framer" 指什么** | ① Framer 平台：可视化建站工具，适合营销落地页，代码自由度低；② Framer Motion：开源动画库（现已更名为 **Motion**，import 为 `motion/react`），适合真正可开发的应用 | 按 **Motion 动画库** 理解 |
| 2 | **产品是什么** | 当前仓库是 Python/PyQt 桌面应用（Video Analysis Pro）。新 Web 应用可以是：(a) 本项目的 Web 端；(b) 本项目的官网/落地页；(c) 独立新产品 | 待 Phase 0 确认 |
| 3 | **"mobile" 的深度** | 响应式 Web（含 PWA 可选）即可覆盖多数场景；真原生 App 需换 React Native/Expo 技术栈 | 默认响应式 Web |

---

## 1. 目标与验收标准（先锁定再动手）

**目标**：交付一个视觉上现代、动效与 3D 沉浸感强、SEO 友好、无障碍且无已知安全漏洞的 Web 应用。

| 维度 | 验收标准 | 验证方式 |
|------|---------|---------|
| 性能 (CWV) | LCP < 2.5s · INP < 200ms · CLS < 0.1 | Lighthouse / PageSpeed |
| SEO | Lighthouse SEO ≥ 95；结构化数据校验通过 | Lighthouse + Rich Results Test |
| 无障碍 | WCAG 2.2 AA；axe 扫描 0 critical | axe-core / Playwright |
| 安全 | 无 CRITICAL/HIGH 依赖漏洞；CSP 与安全头齐备 | npm audit + 安全 checklist |
| 响应式 | 320 / 768 / 1024 / 1440 无溢出、布局正常 | 视觉回归截图 |
| 动效降级 | `prefers-reduced-motion` 下全部降级可用 | Playwright emulate |
| 功能 | 关键用户路径真实浏览器走通（成功 + 失败 + 返回路径） | Playwright E2E |

---

## 2. 技术选型

| 层 | 选型 | 说明 | 选择理由 |
|----|------|------|---------|
| 框架 | **Next.js（App Router）** | 最新稳定版 | RSC / SSG / ISR 原生支持 — SEO 与性能的地基 |
| 语言 | **TypeScript** | strict 模式 | 类型安全 |
| 样式 | **Tailwind CSS v4** + CSS 自定义属性 token | oklch 色彩空间 | 设计 token 化，主题可切换 |
| 动效 | **Motion（原 Framer Motion）** | `motion/react` | 声明式动画、layout 动画、滚动联动、手势 |
| 3D | **React Three Fiber + drei** | three 最新版 | React 生态内做 WebGL，组件化 |
| 表单 | react-hook-form + zod | — | 系统边界校验 + 类型推导 |
| 测试 | Vitest + Testing Library + Playwright | — | 单测 + 真实浏览器 E2E |
| SEO | Next.js Metadata API + JSON-LD + next-sitemap | — | 官方方案 |
| 部署 | Vercel（或自托管） | — | Phase 0 确认 |

**体积预算**（gzip）：落地页 JS < 150kb / CSS < 30kb；应用页 JS < 300kb / CSS < 50kb。3D 通过动态 import 按需加载，不计入首屏。

---

## 3. 设计系统

### 3.1 风格方向（Phase 1 供 stakeholder 选择，附参考站点）

| 方向 | 关键词 | 适合 |
|------|--------|------|
| A. 深色奢华 Dark Luxury | 深底、高对比金/翠点缀、衬线+无衬线大字号配对 | 高端产品、AI 工具 |
| B. 玻璃拟态 Glassmorphism 2.0 | 毛玻璃层叠、真实景深、渐变光斑 | 创意工具、SaaS |
| C. Bento 编辑风 | 便当格布局、克制留白、数据卡片编辑排版 | 产品官网、数据型应用 |
| D. 新粗野主义 Neo-brutalism | 硬边框、高饱和撞色、粗黑投影 | 潮流品牌、开发者工具 |

**反模板承诺**：不做"默认模板脸"——统一间距铺满、灰色安全配色、无 hover 状态、居中标题+渐变光斑的通用 hero。每个页面至少体现：清晰的比例层次、有节奏的间距、真实的层深、有性格的排版配对、语义化用色、设计过的 hover/focus/active 状态。

### 3.2 设计 Token（oklch）

```css
:root {
  --color-surface: oklch(98% 0 0);
  --color-text: oklch(18% 0 0);
  --color-accent: oklch(68% 0.21 250);   /* 具体色相 Phase 1 定 */
  --text-hero: clamp(3rem, 1rem + 7vw, 8rem);
  --space-section: clamp(4rem, 3rem + 5vw, 10rem);
  --duration-fast: 150ms;
  --duration-normal: 300ms;
  --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
}
```

- 只用语义 token，不散落硬编码色值
- 字体最多两族，`font-display: swap`，关键字重 preload
- 若做明暗双主题，两套都必须"设计过"，不是简单反色

---

## 4. 动效系统（Motion / Framer Motion）

### 4.1 动效清单

| 类型 | 实现 | 位置 |
|------|------|------|
| 入场编排 | `initial/animate` + stagger | 首屏、各 section |
| 滚动联动 | `useScroll` + `useTransform` | 视差、进度条、数字滚动 |
| 共享元素 | `layoutId` | 列表 → 详情过渡 |
| 存在性动画 | `AnimatePresence` | 模态、抽屉、列表增删 |
| 手势 | `whileHover` / `whileTap` / `drag` | 卡片、按钮、看板 |
| 页面过渡 | `template.tsx` + AnimatePresence | 路由切换 |

### 4.2 性能与可访问性红线

- 只动 **transform / opacity / clip-path / filter**，不动 layout 属性（width/height/top/margin…）
- `will-change` 窄范围使用，用完即除
- 滚动处理用 rAF / IntersectionObserver，不写裸 scroll handler
- `useReducedMotion` 全局 hook：`prefers-reduced-motion: reduce` 时动画全部降级为淡入或直接跳过

---

## 5. 3D 体验（React Three Fiber）

- **Hero 3D 场景**：低多边形几何体 / 玻璃材质 / 粒子场 + 指针视差；canvas 骨架屏占位，`dynamic import` 按需加载
- **加载策略**：SSR 输出静态 fallback，WebGL 可用后水合；不可用（低端设备）自动降级为静态图
- **资产纪律**：Draco 压缩几何体 + KTX2 纹理；单场景 GPU 预算 < 50ms/帧（60fps）
- **深度克制**：3D 只在 Hero 与关键交互出现，不做成"到处都是的 demo"

---

## 6. SEO 策略

### 6.1 技术 SEO（一次做对）

| 项 | 做法 |
|----|------|
| 渲染 | 首屏 SSG/SSR；3D/动效组件全部 client 组件且不阻塞内容 |
| Meta | Metadata API：title template、description、canonical、OG image（1200×630）、Twitter Card |
| 结构化数据 | JSON-LD：`Organization` + `WebSite` + `FAQPage` + `BreadcrumbList` + `SoftwareApplication`（视产品） |
| 站点地图 | `sitemap.xml` 自动生成 + `robots.txt` |
| 多语言 | 如需：hreflang + next-intl（Phase 0 确认） |
| AEO（AI 搜索趋势） | 清晰的问答式内容结构 + `llms.txt` |

### 6.2 关键词运营（落实"定期更新"机制）

1. **Phase 0-1**：种子词研究（按搜索意图分：信息型 / 商业型 / 交易型），产出词库表（词、意图、月搜索量估计、目标页面）
2. **上线时**：每页 1 主词 + 2-3 辅词，落进 title / H1 / 首段 / URL；建立内链集群（pillar → cluster）
3. **月度例程**：Google Search Console 数据回流 → 淘汰零展示词 → 补充新趋势词 → 更新 `keywords.md` 词库 → 对应页面微调；可接入 rank tracker
4. 词库与变更记录进仓库，全程可追溯

---

## 7. 无障碍与安全基线

### 7.1 无障碍（WCAG 2.2 AA）

- 语义 HTML 优先（header/nav/main/footer），不留通用 div 堆叠
- 键盘全路径可达，`:focus-visible` 自定义 focus ring（设计过的，不是浏览器默认）
- 文本对比度 ≥ 4.5:1（token 定义阶段就用工具校验）
- 动效尊重 reduced-motion；3D 场景提供文字等价描述
- 表单错误信息关联 `aria-describedby`

### 7.2 安全（每阶段 Gate 复查）

- **CSP**：nonce 版 `script-src`，禁 `unsafe-inline`；全套安全头（HSTS、`X-Frame-Options: DENY`、`X-Content-Type-Options: nosniff`、Referrer-Policy、Permissions-Policy）
- **输入**：zod 边界校验，永不信任外部数据；表单 honeypot + 速率限制 + CSRF 防护
- **输出**：不用 `dangerouslySetInnerHTML`（必要时经净化库）；富文本白名单净化
- **依赖**：上线前 `npm audit` + lockfile 审计；CDN 脚本带 SRI
- **错误信息**：不泄露堆栈、路径、密钥

---

## 8. 分阶段路线图（每阶段一个确认 Gate）

| 阶段 | 内容 | 交付物 | Gate（stakeholder 确认） |
|------|------|--------|------------------------|
| **P0 需求澄清** | 产品定位 / 受众 / 平台 / 多语言 / 部署 | 决策表 | 需求签字 ← **本次对话即在此** |
| **P1 设计方向** | 风格方向选定 + 竞品参考 + 色板草案 | 风格定案文档 | 方向选定 |
| **P2 原型** | 设计 token + 首页线框（ASCII/HTML 原型）+ 组件清单 | 可预览原型 | 布局确认 |
| **P3 实现** | 页面骨架 + 组件 + 响应式（320→1440） | 可运行站点 | 桌面/移动截图确认 |
| **P4 动效** | Motion 集成 + reduced-motion 降级 | 动效完成版 | 动效 review（快慢/幅度） |
| **P5 3D** | R3F 场景 + 降级策略 + 性能达标 | 3D 完成版 | CWV 达标证据 |
| **P6 SEO** | meta / JSON-LD / sitemap / OG image / 词库落地 | SEO 检查表全绿 | Rich Results 校验通过 |
| **P7 验证** | Playwright E2E + axe + 视觉回归 + 安全 checklist | 测试报告 | 全部验收标准通过 |
| **P8 交付+运营** | 部署 + GSC 接入 + 关键词月度例程 | 上线站点 | 上线确认 |

**依赖关系**：P0→P1→P2 严格串行（方向未定不写代码）；P3 之后 P4/P5 可并行；P6 依赖 P3 稳定；P7 收口。

---

## 9. 验证方式（不伪造完成）

- **真实浏览器**（Playwright）：关键路径点击/输入/等待，覆盖成功、失败、返回路径，关键状态截图
- **性能**：Lighthouse CI 跑 CWV，达标数值贴进报告
- **无障碍**：axe-core 扫描 + 键盘走查
- **视觉回归**：320 / 768 / 1024 / 1440 截图对比
- **结论标注**：已验证 / 静态确认 / 合理推断 / 待验证 四级标签，不虚报

---

## 10. 风险清单

| 风险 | 影响 | 对策 |
|------|------|------|
| 3D 与性能预算冲突 | LCP 超标 | 动态 import + SSR fallback；3D 不参与首屏 LCP |
| Motion 包名迁移混乱（framer-motion → motion） | 构建问题 | 统一用 `motion/react` 新包名 |
| SEO vs 重客户端渲染 | 收录差 | 内容层 SSG/SSR，只有交互层是 client |
| Framer 平台 vs 库的理解偏差 | 返工 | P0 澄清（见 §0） |
| Windows 本地构建内存 | 构建慢/失败 | NODE_OPTIONS=512MB 已配置，必要时调高 |

---

## 11. 当前待确认决策（Phase 0 Gate）

1. 产品定位：Web 版应用 / 官网落地页 / 独立新产品 / 演示原型？
2. 技术路线：Next.js + Motion 库（推荐）/ Framer 平台建站 / 原生 App（Expo）？
3. 视觉风格方向：A / B / C / D？
4. 3D 强度：重度（Hero 全 3D）/ 中度（关键区域）/ 轻量（仅微交互）？
5. 是否多语言（影响 hreflang 与 i18n 工作量）？
6. 部署目标：Vercel / 自托管？
