import { test, expect } from "@playwright/test";

/**
 * 错误边界 / 404:访问不存在路由,应见 not-found 或 error 边界(不白屏)。
 * 静态导出时 Next.js 生成 404.html,非存在路径回退到该页。
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

test.describe("error boundary / 404", () => {
  test("不存在的一级路由显示 404(不白屏)", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    const res = await page.goto("/this-route-does-not-exist-xyz");
    // 静态导出:FastAPI 挂载的静态回退返 404 状态码 + 404 页内容
    expect(res?.status() ?? 200).toBe(404);
    await expect(page.locator("body")).not.toBeEmpty();
    await expect(page.getByText("404").first()).toBeVisible();
  });

  test("不存在的子路由显示 404 或错误边界(不白屏)", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/dashboard/nonexistent-sub-path-xyz");
    await expect(page.locator("body")).not.toBeEmpty();
    // 404 页或 error 边界均含 "404" 文本
    await expect(page.getByText("404").first()).toBeVisible();
  });

  test("根路径首页可加载(基线对照)", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    const res = await page.goto("/");
    expect(res?.status() ?? 500).toBeLessThan(400);
    await expect(page.locator("h1").first()).toBeVisible();
  });
});
