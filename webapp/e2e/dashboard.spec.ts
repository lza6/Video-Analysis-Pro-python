import { test, expect } from "@playwright/test";

/**
 * /dashboard 控制总览页:验证页面骨架渲染,不依赖 /api/* 返回真值。
 * 后端未就绪时整组跳过(不因 goto 连接失败误报)。
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

test.describe("/dashboard 控制总览", () => {
  test("标题与侧边栏总览高亮", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/dashboard/");
    await expect(page.locator("h1").first()).toContainText("总览");
    const overview = page.locator('nav[aria-label="模块导航"] a[href="/dashboard/"]');
    await expect(overview).toHaveAttribute("aria-current", "page");
  });

  test("StatCard 状态卡区域可见", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/dashboard/");
    // 四张状态卡之一:"后端健康" 标签始终渲染(值可为"离线")
    await expect(page.getByText("后端健康").first()).toBeVisible();
  });

  test("最近作业区块骨架可见", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/dashboard/");
    await expect(page.getByText("最近分析作业").first()).toBeVisible();
  });

  test("页面返回 <500(无服务端错误)", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    const res = await page.goto("/dashboard/");
    expect(res?.status() ?? 500).toBeLessThan(500);
  });
});
