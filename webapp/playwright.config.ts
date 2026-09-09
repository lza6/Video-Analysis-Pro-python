import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright E2E 配置(冒烟层)。
 *
 * 前置:后端(FastAPI + 前端静态导出)已跑在 http://127.0.0.1:8002。
 * 本配置**不启动服务**(不配 webServer),假设外部已就绪 —— 避免与开发实例
 * (默认 :8000)端口冲突,也避免在 CI 里重建前端产物的耗时。
 *
 * 启动后端的步骤见 ./e2e/README.md。
 */
export default defineConfig({
  testDir: "./e2e",
  outputDir: "./test-results",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 1,
  workers: process.env.CI ? 1 : undefined,

  reporter: process.env.CI
    ? [["github"], ["html", { open: "never" }]]
    : "list",

  use: {
    baseURL: "http://127.0.0.1:8002",
    browserName: "chromium",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    viewport: { width: 1440, height: 900 },
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
