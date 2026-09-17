// desktop/test-cua-service.js — CUA 服务单测（v10.4.0 P0-4）
// 运行: node desktop/test-cua-service.js
// 覆盖: 四个 IPC handler 的注册/卸载 + mock 标注契约（绝不伪造真实桌面操作）
//      + capturePage 失败降级 + main.js 生产接线。
//
// 无需 Electron：用一个假的 ipcMain 收集 handler，直接调它们断言返回值。
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const {
  mountCuaService,
  capturePageScreenshot,
  mockScreenshot,
  isMockShot,
} = require("./cua-service");

let passed = 0;
function check(name, fn) {
  return Promise.resolve(fn()).then(() => {
    passed += 1;
    console.log(`  ok — ${name}`);
  });
}

function fakeIpcMain() {
  const handlers = new Map();
  return {
    handlers,
    handle(channel, fn) {
      handlers.set(channel, fn);
    },
    removeHandler(channel) {
      handlers.delete(channel);
    },
  };
}

async function main() {
  console.log("== CUA service ==");

  await check("mockScreenshot 带 mock:true 标注", () => {
    const s = mockScreenshot("测试原因");
    assert.strictEqual(s.mock, true);
    assert.strictEqual(s.reason, "测试原因");
    assert.strictEqual(s.dataUrl, null);
    assert.strictEqual(isMockShot(s), true);
  });

  await check("capturePageScreenshot(webContents=null) → mock 降级", async () => {
    const s = await capturePageScreenshot(null);
    assert.strictEqual(s.mock, true);
    assert.ok(s.reason.includes("不可用"));
  });

  await check("capturePageScreenshot(已销毁) → mock 降级", async () => {
    const s = await capturePageScreenshot({ isDestroyed: () => true });
    assert.strictEqual(s.mock, true);
  });

  await check("capturePageScreenshot 真实路径 → mock:false + dataUrl", async () => {
    const wc = {
      isDestroyed: () => false,
      capturePage: async () => ({
        toDataURL: () => "data:image/png;base64,AAAA",
        getSize: () => ({ width: 100, height: 50 }),
      }),
    };
    const s = await capturePageScreenshot(wc);
    assert.strictEqual(s.mock, false);
    assert.strictEqual(s.dataUrl, "data:image/png;base64,AAAA");
    assert.strictEqual(s.width, 100);
    assert.strictEqual(s.height, 50);
  });

  await check("capturePage 抛异常 → mock 降级且带原因", async () => {
    const wc = {
      isDestroyed: () => false,
      capturePage: async () => {
        throw new Error("boom");
      },
    };
    const s = await capturePageScreenshot(wc);
    assert.strictEqual(s.mock, true);
    assert.ok(s.reason.includes("boom"));
  });

  await check("mountCuaService 注册 4 个 handler", () => {
    const ipc = fakeIpcMain();
    mountCuaService(ipc, () => null);
    assert.deepStrictEqual(
      [...ipc.handlers.keys()].sort(),
      ["cua:click", "cua:getForeground", "cua:screenshot", "cua:type"]
    );
  });

  await check("click/type/getForeground 恒为 mock 且不伪造", async () => {
    const ipc = fakeIpcMain();
    mountCuaService(ipc, () => null);
    const click = await ipc.handlers.get("cua:click")(null, { x: 1, y: 2 });
    assert.strictEqual(click.mock, true);
    assert.ok(click.note.includes("mock"));
    assert.strictEqual(click.x, 1);
    assert.strictEqual(click.y, 2);

    const type = await ipc.handlers.get("cua:type")(null, { text: "hi" });
    assert.strictEqual(type.mock, true);
    const fg = await ipc.handlers.get("cua:getForeground")();
    assert.strictEqual(fg.mock, true);
  });

  await check("cua:type 截断超长文本（防 IPC 载荷爆炸）", async () => {
    const ipc = fakeIpcMain();
    mountCuaService(ipc, () => null);
    const out = await ipc.handlers.get("cua:type")(null, { text: "x".repeat(500) });
    assert.strictEqual(out.text.length, 40);
  });

  await check("screenshot handler 走 getWebContents()", async () => {
    const ipc = fakeIpcMain();
    let called = 0;
    mountCuaService(ipc, () => {
      called += 1;
      return null;
    });
    const s = await ipc.handlers.get("cua:screenshot")();
    assert.strictEqual(called, 1);
    assert.strictEqual(s.mock, true);
  });

  await check("unmount 移除全部 handler", () => {
    const ipc = fakeIpcMain();
    const unmount = mountCuaService(ipc, () => null);
    assert.strictEqual(ipc.handlers.size, 4);
    unmount();
    assert.strictEqual(ipc.handlers.size, 0);
  });

  await check("unmount 可重复调用（不抛异常）", () => {
    const ipc = fakeIpcMain();
    const unmount = mountCuaService(ipc, () => null);
    unmount();
    unmount();
  });

  console.log("== main.js 生产接线（P0-4）==");

  await check("main.js 真实 require cua-service", () => {
    const src = fs.readFileSync(path.join(__dirname, "main.js"), "utf8");
    assert.ok(src.includes('require("./cua-service")'), "main.js 未挂载 CUA 服务");
    assert.ok(src.includes("VAP_CUA_ENABLED"), "VAP_CUA_ENABLED 无消费方");
    assert.ok(src.includes("mountCuaService("), "未调用 mountCuaService");
    assert.ok(src.includes("unmountCua"), "退出时未卸载 CUA handler");
  });

  await check("cua-service.js 导出四个符号", () => {
    assert.strictEqual(typeof mountCuaService, "function");
    assert.strictEqual(typeof capturePageScreenshot, "function");
    assert.strictEqual(typeof mockScreenshot, "function");
    assert.strictEqual(typeof isMockShot, "function");
  });

  console.log(`\n${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAILED:", e && e.message ? e.message : e);
  process.exit(1);
});
