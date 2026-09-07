import { test, expect } from "@playwright/test";

/**
 * /analyze AI 视频分析页:验证上传/配置/按钮区域可见,不触发真实分析提交。
 * 页面骨架不依赖 /api/* 数据,后端未就绪时整组跳过。
 */
let ready = false;
test.beforeAll(async ({ request }) => {
  try {
    const r = await request.get("/api/health", { timeout: 3000 });
    ready = r.status() < 500;
  } catch {
    ready = false;
  }
});

test.describe("/analyze AI 视频分析", () => {
  test("页面标题可见", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/analyze/");
    await expect(page.locator("h1").first()).toContainText("视频分析");
  });

  test("VideoPicker 上传/本机路径切换可见", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/analyze/");
    await expect(page.getByRole("button", { name: "上传视频" })).toBeVisible();
    await expect(page.getByRole("button", { name: "本机路径" })).toBeVisible();
  });

  test("ConfigPanel 抽帧密度标签可见", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/analyze/");
    await expect(page.getByText("抽帧密度").first()).toBeVisible();
  });

  test("开始分析按钮可见(未触发提交)", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/analyze/");
    await expect(page.getByRole("button", { name: "开始分析" })).toBeVisible();
  });

  test("页面返回 <500(无服务端错误)", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    const res = await page.goto("/analyze/");
    expect(res?.status() ?? 500).toBeLessThan(500);
  });
});
