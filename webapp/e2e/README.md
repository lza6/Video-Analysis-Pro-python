# E2E 冒烟测试(Playwright)

验证 webapp 前端(Next.js 静态导出)+ FastAPI 后端全栈可用性。当前仅 1 个样板 spec
(`smoke.spec.ts`),覆盖:首页加载、`/api/health` 200、内页导航不崩。

## 前置

### 1. 安装依赖

```bash
cd webapp
npm install                     # 装 package.json 里所有依赖(含 @playwright/test)
npx playwright install chromium # 下载 Chromium(首次约 150MB)
```

> `@playwright/test` 已在 `webapp/package.json` 的 `devDependencies` 声明,但**尚未 install**。
> 首次使用前必须先 `npm install`,否则 `npx playwright` 不可用。

### 2. 启动后端(E2E 专用 :8001)

E2E 用 8001 端口,避开开发实例(默认 :8000)。`playwright.config.ts` **不启动服务**,
假设外部已就绪。

```bash
# (a) 构建前端静态产物(若 webapp/out/ 不存在)
cd webapp && npm run build          # 产物输出到 webapp/out/

# (b) 启 FastAPI 同源挂载前端,绑 8001
cd ..
VAP_PORT=8001 python -m src.web.serve --no-browser
```

服务就绪后访问 http://127.0.0.1:8001 应见首页。

### 3. 鉴权(可选)

若后端启用了 Bearer Token(`VAP_HEADLESS_TOKEN` 非空),`/api/health` 会返 401。
设同名 token 给 E2E:

```bash
VAP_E2E_TOKEN=<同 token> npx playwright test
```

未设此变量时,测试假设鉴权关闭(token 为空),直接裸调 `/api/health`。

## 运行

```bash
cd webapp
npx playwright test              # 跑全部 e2e/*.spec.ts
npx playwright test --headed     # 可视化看浏览器(本地调试用)
npx playwright show-report       # 打开 HTML 报告(playwright-report/)
```

## 配置要点

| 项 | 值 | 说明 |
|----|----|----|
| baseURL | `http://127.0.0.1:8001` | 后端同源挂载前端,统一入口 |
| browserName | chromium | 单浏览器,冒烟层够用 |
| trace | on-first-retry | 首次重试时录 trace,便于排障 |
| screenshot | only-on-failure | 失败才截图,省空间 |
| retries | CI=2 / 本地=1 | 偶发抖动容错 |
| webServer | **未配** | 不自动启服务,假设外部已就绪 |

## 产物(已 gitignore)

- `webapp/playwright-report/` — HTML 报告
- `webapp/test-results/` — 失败截图 / trace / video
- `webapp/.playwright/` — 浏览器缓存

均已在仓库根 `.gitignore` 排除。

## 扩展

新增 spec 放 `webapp/e2e/*.spec.ts` 即被自动收录。建议下一步覆盖:
- `/api/analyze` 提交流水线(需小视频 fixture)
- Agent 对话往返(`/api/agent/chat`)
- 批量页 `/batch/` 列表加载
