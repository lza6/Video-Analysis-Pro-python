// main.js
// Electron 主进程入口:
//  1. 单实例锁(第二实例聚焦已有窗口)
//  2. spawn python src/web/serve.py 子进程(runtime-controller)
//  3. 探测就绪 + 实际端口
//  4. BrowserWindow loadURL(端口)
//  5. 系统托盘(Tray + 退出菜单)
//  6. before-quit 杀子进程(Windows taskkill /T 防孤进程)
//  7. 黑匣子日志 LogStore(installConsoleCapture)
//  8. autoUpdater 检查更新(electron-updater,仅生产 + feed 配置存在才启用)
//  9. IPC:diagnostics:collect / logs:query / logs:subscribe

"use strict";

const { app, BrowserWindow, Tray, Menu, shell, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const os = require("os");
const cp = require("child_process");
const { RuntimeController } = require("./runtime-controller");
const { LogStore, installConsoleCapture } = require("./log-store");
const { collectDiagnostics } = require("./crash-diagnostics");

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

// 黑匣子内存日志(主进程控制台 + 子进程 stdout/stderr + 系统事件)
const logStore = new LogStore({ maxEntries: 2000 });
installConsoleCapture(logStore);
logStore.append("main", "info", "main.js loaded, log-store ready");

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

// ---------- autoUpdater(electron-updater) ----------
// 仅生产 && feed 配置存在才启用;否则只打日志(安全降级,不做真实更新)。
function setupAutoUpdater() {
  let autoUpdater = null;
  try {
    const isDev = !!process.env.ELECTRON_IS_DEV || !app.isPackaged;
    const updaterPath = path.join(os.homedir(), ".tingfeng-hermes-updates", "config.json");
    let feedConfig = null;
    if (fs.existsSync(updaterPath)) {
      try {
        feedConfig = JSON.parse(fs.readFileSync(updaterPath, "utf8"));
      } catch (_) { /* ignore */ }
    }
    if (!feedConfig) {
      // 尝试从 package.json build.publish 读 feed
      try {
        const pkg = JSON.parse(
          fs.readFileSync(path.join(__dirname, "package.json"), "utf8")
        );
        if (pkg.build && pkg.build.publish && pkg.build.publish.length > 0) {
          feedConfig = pkg.build.publish;
        }
      } catch (_) { /* ignore */ }
    }
    if (isDev) {
      logStore.append("updater", "info", "autoUpdater disabled (dev mode)");
      mlog("autoUpdater disabled (dev mode)");
      return null;
    }
    if (!feedConfig) {
      logStore.append(
        "updater",
        "warn",
        "autoUpdater enabled but feed not configured — 不执行更新检查"
      );
      mlog("autoUpdater enabled but feed not configured");
      return null;
    }
    // 实际启用:动态 require(electron-updater)(避免开发态加载失败)
    autoUpdater = require("electron-updater").autoUpdater;
    autoUpdater.setFeedURL(feedConfig.url || feedConfig[0]?.url || feedConfig, {});
    autoUpdater.checkForUpdatesAndNotify();
    autoUpdater.on("error", (err) => {
      logStore.append("updater", "error", `autoUpdater error: ${err.message}`);
    });
    logStore.append(
      "updater",
      "info",
      `autoUpdater enabled (feed=${JSON.stringify(feedConfig).slice(0, 120)})`
    );
    mlog("autoUpdater enabled, checkForUpdatesAndNotify()");
  } catch (e) {
    logStore.append("updater", "error", `autoUpdater setup error: ${e.message}`);
    mlog("autoUpdater setup error: " + e.message);
  }
  return autoUpdater;
}

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
    setupAutoUpdater();
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

// ---------- IPC:黑匣子日志 + 诊断导出 ----------
function registerIpc() {
  // 诊断导出:返回 zip Buffer(渲染层 Blob 下载)
  ipcMain.handle("diagnostics:collect", async () => {
    const pids = [];
    if (runtime && runtime.child && runtime.child.pid) pids.push(runtime.child.pid);
    pids.push(process.pid);
    try {
      const diag = collectDiagnostics({ pids });
      logStore.append("main", "info", `diagnostics collected: ${diag.entries.length} entries, ${diag.zip.length} bytes`);
      return { ok: true, buffer: diag.zip, entries: diag.entries };
    } catch (e) {
      logStore.append("main", "error", `diagnostics collect error: ${e.message}`);
      return { ok: false, error: e.message };
    }
  });

  // 日志查询:level 过滤 + 搜索 + 上限
  ipcMain.handle("logs:query", (_e, opts) => {
    try {
      const q = logStore.query({
        level: opts?.level,
        search: opts?.search,
        limit: opts?.limit,
      });
      return { ok: true, logs: q };
    } catch (e) {
      return { ok: false, error: e.message };
    }
  });

  // 日志订阅:new listener on store 的 append 事件(复用已有 EventEmitter)
  ipcMain.handle("logs:subscribe", (_e) => {
    const listener = (entry) => {
      // 推给渲染层:由于 handle 单次返回,这里用 send 广播
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send("logs:append", entry);
      }
    };
    logStore.on("append", listener);
    return { ok: true };
  });
}

async function boot() {
  mlog("boot() called");
  runtime = new RuntimeController();
  mlog("runtime created");
  registerIpc();
  try {
    mlog("calling runtime.start");
    // onLog 同时写 LogStore(主进程黑匣子) + 原 process.stdout 输出
    const port = await runtime.start({
      onLog: (l) => {
        logStore.append("serve", "info", l);
        process.stdout.write(`[serve] ${l}`);
      },
      onReady: (p) => {
        console.log(`[runtime] ready port=${p}`);
        logStore.append("main", "info", `runtime onReady port=${p}`);
        mlog("runtime onReady port=" + p);
      },
      onFailed: (e) => {
        console.error(`[runtime] failed: ${e.message}`);
        logStore.append("main", "error", `runtime onFailed: ${e.message}`);
        mlog("runtime onFailed: " + e.message);
      },
      // 崩溃断路器观察:非主动退出 → 触发 UI 提示 + 自动重启(若未冻结)
      onExit: (code, signal) => {
        const frozen = runtime && runtime._breaking;
        logStore.append(
          "main",
          "error",
          `serve.py 子进程退出 exit=${code} signal=${signal} frozen=${!!frozen}`
        );
        mlog(`serve.py exit code=${code} signal=${signal} frozen=${!!frozen}`);
        if (frozen) {
          // 冻结:等 re-arm。给用户可手动重启入口(tray)。
          if (tray) {
            const menu = Menu.buildFromTemplate([
              { label: "显示主窗口", click: () => mainWindow && mainWindow.show() },
              { label: `后端状态: 已冻结(5min 后自动恢复)`, enabled: false },
              { label: "立即重启后端", click: () => { void runtime.restart(); } },
              { label: "退出", click: () => { isQuitting = true; app.quit(); } },
            ]);
            tray.setContextMenu(menu);
          }
        } else {
          // 非冻结:自动重启(限 1 次/次崩溃,由 onExit 触发 main.js 控制)
          // 这里只记录,实际重启在 main.js 层做(避免多路重启)
          if (runtime && runtime.state === "restarting") return;
          scheduleAutoRestart();
        }
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
    logStore.append("main", "error", `boot failed: ${e.message}`);
    createTray();
    createErrorWindow(e.message || String(e));
  }
}

// 自动重启调度(断路器未冻结时)。限 1 次(每次崩溃后单发)。
let autoRestartTimer = null;
function scheduleAutoRestart() {
  if (autoRestartTimer) return;
  logStore.append("main", "warn", "serve.py 崩溃,2s 后自动重启");
  mlog("scheduleAutoRestart 2s");
  autoRestartTimer = setTimeout(() => {
    autoRestartTimer = null;
    if (isQuitting) return;
    if (!runtime || runtime._breaking) {
      logStore.append("main", "warn", "autoRestart skipped (breaker frozen or stopped)");
      return;
    }
    logStore.append("main", "info", "auto restarting runtime...");
    mlog("auto restarting runtime");
    void runtime.restart({}).then((port) => {
      currentPort = port;
      mlog("auto restart ok port=" + port);
      if (!mainWindow || mainWindow.isDestroyed()) {
        createWindow(port);
      } else {
        mainWindow.loadURL(`http://127.0.0.1:${port}/`).catch((e) => {
          logStore.append("main", "error", `loadURL after restart failed: ${e.message}`);
        });
      }
    }).catch((e) => {
      logStore.append("main", "error", `auto restart failed: ${e.message}`);
    });
  }, 2000);
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
  // loadURL 失败重试 5 次,指数退避
  let loadAttempts = 0;
  const loadWithRetry = () => {
    mainWindow.loadURL(url).catch((e) => {
      loadAttempts += 1;
      logStore.append("main", "error", `loadURL 失败 (attempt ${loadAttempts}): ${e.message}`);
      mlog(`loadURL fail attempt ${loadAttempts}: ${e.message}`);
      if (loadAttempts < 5) {
        setTimeout(loadWithRetry, 1000 * loadAttempts); // 1s/2s/3s/4s
      } else {
        createErrorWindow(`loadURL 重试 5 次仍失败: ${e.message}`);
      }
    });
  };
  loadWithRetry();

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

  // 渲染进程崩溃记录到黑匣子
  mainWindow.webContents.on("render-process-gone", (_e, details) => {
    logStore.append("renderer", "error", `renderer gone: ${JSON.stringify(details)}`);
  });
  app.on("crash", (_e, _killed) => {
    logStore.append("system", "error", "app crash event fired");
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
      label: "导出诊断包",
      click: () => {
        void exportDiagnostics();
      },
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

// 导出诊断包到磁盘(托盘入口)。IPC 版本返回 Buffer 给前端下载。
function exportDiagnostics() {
  const pids = [];
  if (runtime && runtime.child && runtime.child.pid) pids.push(runtime.child.pid);
  pids.push(process.pid);
  try {
    const diag = collectDiagnostics({ pids });
    const outPath = path.join(PROJECT_ROOT, "logs", `diagnostics-${Date.now()}.zip`);
    fs.writeFileSync(outPath, diag.zip);
    logStore.append("main", "info", `diagnostics exported to ${outPath} (${diag.zip.length} bytes)`);
    mlog(`diagnostics exported to ${outPath}`);
    if (tray) {
      tray.displayBalloon("诊断包已导出", outPath);
    }
    return outPath;
  } catch (e) {
    logStore.append("main", "error", `export diagnostics failed: ${e.message}`);
    mlog("export diagnostics failed: " + e.message);
    return null;
  }
}

function cleanupAndQuit() {
  if (runtime) {
    runtime.stop();
    runtime = null;
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}