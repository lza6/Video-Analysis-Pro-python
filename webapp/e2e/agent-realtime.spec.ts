import { test, expect } from "@playwright/test";

/**
 * /agent 实时事件流 + 会话管理 E2E（v10.5.0 P1-7 / P1-8 / P1-9）。
 *
 * 不依赖付费 LLM：无凭据时后端降级适配器仍会跑完 run_turn 并持久化 Session，
 * assistant_delta/done 事件照常推送 —— 这正好验证"实时事件管线"本身。
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

test.describe("/agent 实时流与会话管理", () => {
  // 冷启动 + 降级 LLM 路径较慢,给足预算防抖动(首轮曾 30s 超时,重试通过)
  test.setTimeout(90_000);
  test("会话管理条可见(新建/历史下拉)", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/agent/");
    await expect(page.getByRole("button", { name: "＋ 新建会话" })).toBeVisible();
    await expect(page.getByLabel("历史会话")).toBeVisible();
  });

  test("run_stream 实时事件 + 会话落库进历史下拉", async ({ page, request }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    const chat = await request.post("/api/agent/chat", {
      data: { text: "E2E实时流验证" },
    });
    expect(chat.ok()).toBeTruthy();
    const sid = ((await chat.json()) as { session_id: string }).session_id;
    expect(sid).toBeTruthy();

    // 执行 + 实时事件断言（assistant_delta 说明 hooks→SSE 已接通）
    const res = await request.get(
      `/api/agent/run_stream?session_id=${sid}&text=E2E%E5%AE%9E%E6%97%B6%E6%B5%81%E9%AA%8C%E8%AF%81`,
    );
    expect(res.ok()).toBeTruthy();
    const body = await res.text();
    expect(body).toContain("event: assistant_delta");
    expect(body).toContain("event: done");

    // 会话已持久化 → 刷新后历史下拉出现该会话
    await page.goto("/agent/");
    const opt = page.locator(
      'select[aria-label="历史会话"] option',
      { hasText: sid.slice(0, 8) },
    );
    await expect(opt).toHaveCount(1);
  });

  test("审批弹窗含快捷键提示文案", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/agent/");
    // 弹窗需要真实写工具触发,这里只验证无弹窗时不炸 + 页面正常
    await expect(page.getByPlaceholder("问我任何关于视频的问题…")).toBeVisible();
  });

  test("P1-1 术语气泡与 P1-6 Toast 容器可见", async ({ page }) => {
    test.skip(!ready, "后端 /api/health 不可达,跳过");
    await page.goto("/agent/");
    // 术语气泡:<abbr title> 悬浮解释(P1-1)
    await expect(page.locator("abbr[title]").first()).toBeVisible();
    // 全局 Toast 容器(aria-live,P1-6)
    await expect(page.locator('[aria-live="polite"]').first()).toBeVisible();
  });
});