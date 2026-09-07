// main.js
// Electron 主进程入口:
//  1. 单实例锁(第二实例聚焦已有窗口)
//  2. spawn python src/web/serve.py 子进程(runtime-controller)
//  3. 探测就绪 + 实际端口
//  4. BrowserWindow loadURL(端口)
//  5. 系统托盘(Tray + 退出菜单)
//  6. before-quit 杀子进程(Windows taskkill /T 防孤进程)

"use strict";

const { app, BrowserWindow, Tray, Menu, shell } = require("electron");
const path = require("path");
const fs = require("fs");
const os = require("os");
const cp = require("child_process");
const { RuntimeController } = require("./runtime-controller");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const ICON_PATH = path.join(__dirname, "assets", "icon.png");

// 文件日志(electron 主进程 console.log 可能不输出 stdout,写文件做黑匣子)
const MAIN_LOG = path.join(PROJECT_ROOT, "logs", "main.log");
function mlog(msg) {
  try {
    const line = "[" + new Date().toISOString() + "] " + msg + "\n";
    fs.appendFileSync(MAIN_LOG, line, { encoding: "utf8" });
  } catch (e) { /* ignore */ }
}
mlog("=== main.js loaded ===");
/** @type {BrowserWindow | null} */
let mainWindow = null;
/** @type {Tray | null} */
let tray = null;
/** @type {RuntimeController | null} */
let runtime = null;
/** @type {number | null} */
let currentPort = null;
/** @type {boolean} */
let isQuitting = false;

// ---------- 启动前清理上次残留 ----------
// 杀掉上次崩溃/强制关闭遗留的 electron.exe + 跑 serve.py 的 python.exe,
// 避免端口 8000-8019 被占导致新实例起不来。不动系统服务(如 Manager.exe)。
//
// 关键:不能杀当前进程自己!taskkill /IM electron.exe 会杀掉自己。
// 只杀:占 8000-8019 端口的 python(不动 electron 自己)。
// electron 残留靠单实例锁 + before-quit cleanup 兜底,不在 ready 里杀 electron。
function killStaleInstances() {
  const isWin = process.platform === "win32";
  try {
    if (isWin) {
      // 不杀 electron.exe(会自杀)——只清占端口的 python。
      // 残留 electron 靠单实例锁(second-instance 聚焦已有窗口)兜底。
      // 查 8000-8019 LISTENING PID,杀占这些端口的 python(不动系统服务)
      try {
        const out = cp.execSync("netstat -ano", {
          encoding: "utf8",
          maxBuffer: 1024 * 1024 * 8,
          stdio: ["pipe", "pipe", "ignore"],
        });
        const stalePids = new Set();
        for (const line of out.split(/\r?\n/)) {
          const m = line.match(/:(80(?:0[0-9]|1[0-9]))\s+.*LISTENING\s+(\d+)/);
          if (m) {
            const pid = m[2].trim();
            if (pid && pid !== "0" && pid !== String(process.pid)) stalePids.add(pid);
          }
        }
        for (const pid of stalePids) {
          try {
            const nameOut = cp.execSync(
              `tasklist /FI "PID eq ${pid}" /FO CSV /NH`,
              { encoding: "utf8", stdio: ["pipe", "pipe", "ignore"] }
            ).trim();
            if (nameOut.toLowerCase().includes("python")) {
              cp.execSync(`taskkill /F /PID ${pid} /T`, { stdio: "ignore" });
              mlog(`[cleanup] killed python PID=${pid}`);
            }
          } catch (_) { /* PID 已退出 */ }
        }
      } catch (_) { /* netstat 失败,跳过(serve.py 端口被占会自动跳端口) */ }
    } else {
      // macOS/Linux:按命令行匹配 pkill,排除自己
      try {
        cp.execSync(`pkill -f 'src.web.serve' || true`, { stdio: "ignore" });
      } catch (_) {}
    }
  } catch (e) {
    mlog(`[cleanup] 异常(忽略): ${e.message}`);
  }
}

// ---------- 单实例锁 ----------
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  // 第二实例:直接退出,让已有窗口被聚焦(second-instance 事件里做)
  mlog("second instance, quitting");
  app.quit();
} else {
  app.on("second-instance", () => {
    // 用户又双击了一次:聚焦已有窗口
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      if (!mainWindow.isVisible()) mainWindow.show();
      mainWindow.focus();
    }
  });
  app.on("ready", () => {
    mlog("app ready event fired");
    killStaleInstances();
    boot();
  });
  app.on("before-quit", () => {
    mlog("before-quit event fired");
    cleanupAndQuit();
  });
  app.on("window-all-closed", () => {
    // 托盘应用:窗口全关后不退出(留在托盘)。除非已主动退出。
    if (isQuitting) return;
    // macOS 惯例留在托盘,其他平台也保留(后端仍在跑)
  });
  app.on("activate", () => {
    if (mainWindow === null && runtime) {
      createWindow(currentPort);
    }
  });
}

async function boot() {
  mlog("boot() called");
  runtime = new RuntimeController();
  mlog("runtime created");
  try {
    mlog("calling runtime.start");
    const port = await runtime.start({
      onLog: (l) => process.stdout.write(`[serve] ${l}`),
      onReady: (p) => {
        console.log(`[runtime] ready port=${p}`);
        mlog("runtime onReady port=" + p);
      },
      onFailed: (e) => {
        console.error(`[runtime] failed: ${e.message}`);
        mlog("runtime onFailed: " + e.message);
      },
    });
    currentPort = port;
    mlog("runtime.start returned port=" + port);
    createTray();
    mlog("tray created");
    createWindow(port);
    mlog("window created");
  } catch (e) {
    mlog("boot FAILED: " + (e.message || String(e)) + " | stack: " + (e.stack || ""));
    // 后端起不来也要开个窗口给用户看错误(避免黑屏)
    console.error(`[runtime] 启动失败: ${e.message}`);
    createTray();
    createErrorWindow(e.message || String(e));
  }
}

function createWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 640,
    title: "TingFeng Hermes",
    icon: fs.existsSync(ICON_PATH) ? ICON_PATH : undefined,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      sandbox: false, // preload 需要 require 读 package.json
      nodeIntegration: false,
    },
  });

  const url = `http://127.0.0.1:${port}/`;
  mainWindow.loadURL(url).catch((e) => {
    console.error(`[window] loadURL 失败: ${e.message}`);
  });

  // 外部链接用系统浏览器打开,不在应用内导航走丢
  mainWindow.webContents.setWindowOpenHandler(({ url: target }) => {
    if (target.startsWith("http://127.0.0.1") || target.startsWith("http://localhost")) {
      return { action: "allow" };
    }
    shell.openExternal(target);
    return { action: "deny" };
  });

  mainWindow.on("close", (e) => {
    if (!isQuitting) {
      // 关窗时最小化到托盘,而不是退出
      e.preventDefault();
      mainWindow.hide();
    }
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function createErrorWindow(msg) {
  mainWindow = new BrowserWindow({
    width: 720,
    height: 480,
    title: "TingFeng Hermes — 启动失败",
  });
  mainWindow.loadURL(
    "data:text/html;charset=utf-8," +
      encodeURIComponent(
        `<html><body style="font-family:sans-serif;padding:2em;color:#222">
<h2>后端服务启动失败</h2>
<pre style="white-space:pre-wrap;background:#f4f4f4;padding:1em;border-radius:6px">${escapeHtml(
          msg
        )}</pre>
<p>请检查:项目根 venv 是否存在;端口 8000-8019 是否被占;src/web/serve.py 是否能独立跑通。</p>
</body></html>`
      )
  );
}

function createTray() {
  if (!fs.existsSync(ICON_PATH)) {
    console.warn(`[tray] 图标缺失: ${ICON_PATH} — 托盘将使用默认图标`);
  }
  try {
    tray = new Tray(fs.existsSync(ICON_PATH) ? ICON_PATH : undefined);
  } catch (e) {
    console.warn(`[tray] 创建托盘失败: ${e.message}`);
    return;
  }
  const menu = Menu.buildFromTemplate([
    {
      label: "显示主窗口",
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          mainWindow.focus();
        } else if (currentPort) {
          createWindow(currentPort);
        }
      },
    },
    {
      label: `后端端口: ${currentPort ?? "—"}`,
      enabled: false,
    },
    { type: "separator" },
    {
      label: "退出",
      click: () => {
        isQuitting = true;
        app.quit();
      },
    },
  ]);
  tray.setToolTip("TingFeng Hermes");
  tray.setContextMenu(menu);
  tray.on("click", () => {
    if (mainWindow) {
      mainWindow.isVisible() ? mainWindow.focus() : mainWindow.show();
    } else if (currentPort) {
      createWindow(currentPort);
    }
  });
}

function cleanupAndQuit() {
  if (runtime) {
    runtime.kill();
    runtime = null;
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
