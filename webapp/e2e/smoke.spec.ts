import { test, expect } from "@playwright/test";

/**
 * 冒烟测试:验证全栈(前端静态导出 + FastAPI 后端)在 :8001 可用。
 *
 * 前置:后端已跑在 http://127.0.0.1:8001(见 ./README.md)。
 * 若后端启用了 Bearer Token 鉴权(VAP_HEADLESS_TOKEN 非空),
 * 设 VAP_E2E_TOKEN=<同 token>,测试会带 Authorization 头;留空则假设鉴权关闭。
 */
const TOKEN = process.env.VAP_E2E_TOKEN ?? "";

function authHeaders(): Record<string, string> | undefined {
  return TOKEN ? { Authorization: `Bearer ${TOKEN}` } : undefined;
}

test.describe("smoke: 全栈可用性", () => {
  test("首页能加载", async ({ page }) => {
    await page.goto("/");
    const h1 = page.locator("h1").first();
    await expect(h1).toBeVisible();
    await expect(h1).toContainText("视频分析");
  });

  test("/api/health 返回 200", async ({ request }) => {
    const res = await request.get("/api/health", { headers: authHeaders() });
    expect(res.status()).toBe(200);
    const body = (await res.json()) as { status: string };
    expect(body.status).toBe("ok");
  });

  test("导航到 Agent 对话页不崩", async ({ page }) => {
    await page.goto("/");
    // 静态导出走 <a> 全页导航;点击后整页重载到 /agent/
    await page.click('a[href="/agent/"]');
    await expect(page).toHaveURL(/\/agent\/?$/);
    await expect(page.locator("h1").first()).toHaveText("Agent 对话");
  });
});
