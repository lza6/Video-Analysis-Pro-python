// cua-service.js
// CUA（Computer Use Agent）服务 — Electron 主进程 IPC handler。
//
// P2-3 最小实现：暴露 `screenshot() / click(x,y) / type(text) / getForeground()`
// 四个 IPC handler（`ipcMain.handle('cua:*', ...)`）。
//
// 红线（遵守项目付费/真实 GUI 约束）：
//   - **不真点桌面**：无真实 GUI 自动化环境（无 pyautogui / 无 OS 级 CUA）。
//   - `screenshot()` 走 Electron 主进程 `webContents.capturePage`（真实截当前
//     应用窗口），不可用则返回 mock 截图对象并带 `mock: true` 标注。
//   - `click / type / getForeground` 始终返回 mock 结果 + 标注
//     "mock：无真实桌面 CUA 环境"，**不伪造真实操作**。
//
// 可选挂载：main.js 的 registerIpc() 末尾调用 `mountCuaService(ipcMain, mainWindow)`。
// 主控决定是否挂载；本文件独立提供，未挂载时 import 亦无副作用。

"use strict";

/**
 * 判断截图对象是否为 mock 占位。
 * @param {object} shot 截图对象
 * @returns {boolean}
 */
function isMockShot(shot) {
  return !!(shot && shot.mock === true);
}

/**
 * 组装 mock 截图对象（带标注，不伪造真实截图）。
 * @param {string} reason 标注原因
 * @returns {{mock: boolean, reason: string, dataUrl: null, width: null, height: null}}
 */
function mockScreenshot(reason) {
  return {
    mock: true,
    reason,
    dataUrl: null,
    width: null,
    height: null,
  };
}

/**
 * 经 webContents.capturePage 截当前应用窗口（真实能力）。
 * @param {object} webContents Electron webContents（可 null）
 * @returns {Promise<object>} 截图对象（含 dataUrl / width / height 或 mock 标注）
 */
async function capturePageScreenshot(webContents) {
  if (!webContents || webContents.isDestroyed()) {
    return mockScreenshot("webContents 不可用（窗口未就绪或被销毁）");
  }
  try {
    const image = await webContents.capturePage();
    const dataUrl = image.toDataURL(); // data:image/png;base64,...
    return {
      mock: false,
      reason: "capturePage 真实截图",
      dataUrl,
      width: image.getSize().width,
      height: image.getSize().height,
    };
  } catch (err) {
    return mockScreenshot(`capturePage 失败: ${err.message}`);
  }
}

/**
 * 挂载 CUA IPC handlers。
 * @param {import("electron").IpcMain} ipcMain
 * @param {() => object | null} getWebContents 返回当前 webContents 的 getter
 */
function mountCuaService(ipcMain, getWebContents) {
  // 截图：真实 capturePage 或 mock（带标注）
  ipcMain.handle("cua:screenshot", async () => {
    const wc = getWebContents();
    return capturePageScreenshot(wc);
  });

  // 点击：无真实桌面 CUA 环境 → mock（绝不伪造）
  ipcMain.handle("cua:click", (_event, { x, y } = {}) => ({
    ok: true,
    mock: true,
    note: "mock：无真实桌面 CUA 环境，未执行真实点击",
    x,
    y,
  }));

  // 输入：同上 mock
  ipcMain.handle("cua:type", (_event, { text } = {}) => ({
    ok: true,
    mock: true,
    note: "mock：无真实桌面 CUA 环境，未执行真实输入",
    text: String(text || "").slice(0, 40),
  }));

  // 前台应用：返回占位（无真实 OS 级能力）
  ipcMain.handle("cua:getForeground", () => ({
    ok: true,
    mock: true,
    note: "mock：无真实桌面 CUA 环境，未查询真实前台窗口",
    app: "unknown",
  }));

  return () => {
    try {
      ipcMain.removeHandler("cua:screenshot");
      ipcMain.removeHandler("cua:click");
      ipcMain.removeHandler("cua:type");
      ipcMain.removeHandler("cua:getForeground");
    } catch (_) {
      /* ignore */
    }
  };
}

module.exports = {
  mountCuaService,
  capturePageScreenshot,
  mockScreenshot,
  isMockShot,
};