import { test, expect } from "@playwright/test";

/**
 * /agent Agent 对话页:验证对话输入区可见,不发送真实消息(避免触发 LLM 调用)。
 * 页面骨架纯客户端,后端未就绪时整组跳过。
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

test.describe("/agent Agent 对话", () => {
  test("页面标题可见", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/agent/");
    await expect(page.locator("h1").first()).toContainText("Agent 对话");
  });

  test("对话输入框可见", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/agent/");
    await expect(page.getByPlaceholder("问我任何关于视频的问题…")).toBeVisible();
  });

  test("发送按钮可见(未点击)", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/agent/");
    await expect(page.getByRole("button", { name: "发送" })).toBeVisible();
  });

  test("页面返回 <500(无服务端错误)", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    const res = await page.goto("/agent/");
    expect(res?.status() ?? 500).toBeLessThan(500);
  });
});
