import { test, expect } from "@playwright/test";

/**
 * 侧边栏导航:从 /dashboard 点击 NAV 各项,验证目标路由可达不 404。
 * 静态导出走 <a> 全页导航,点击后整页重载。
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

const NAV = 'nav[aria-label="模块导航"]';

test.describe("侧边栏导航", () => {
  test("dashboard → Agent 对话", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/dashboard/");
    await page.locator(`${NAV} a[href="/agent/"]`).click();
    await expect(page).toHaveURL(/\/agent\/?$/);
    await expect(page.locator("h1").first()).toContainText("Agent 对话");
  });

  test("dashboard → AI 分析(根路由)", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/dashboard/");
    await page.locator(`${NAV} a[href="/"]`).click();
    await expect(page).toHaveURL(/\/?$/);
    await expect(page.locator("h1").first()).toContainText("视频分析");
  });

  test("dashboard → 批量处理", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/dashboard/");
    await page.locator(`${NAV} a[href="/batch/"]`).click();
    await expect(page).toHaveURL(/\/batch\/?$/);
    await expect(page.locator("h1").first()).toContainText("批量处理");
  });

  test("侧边栏在 /dashboard 高亮总览项", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/dashboard/");
    const overview = page.locator(`${NAV} a[href="/dashboard/"]`);
    await expect(overview).toHaveAttribute("aria-current", "page");
  });
});
