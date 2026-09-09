// preload.js
// contextBridge 暴露最小 API 给渲染层(主窗口 webapp)。当前 webapp 不依赖这些
// API 也能跑(它直接打 FastAPI),这里只暴露元信息供 About 页/调试展示。

const { contextBridge, ipcRenderer } = require("electron");
const path = require("path");
const fs = require("fs");

// preload 运行在沙箱外、有 Node 访问,读 package.json 拿版本。
// 渲染层(Next.js webapp)无 Node,经 contextBridge 拿到只读值。
let version = "1.0.0";
try {
  const pkg = JSON.parse(
    fs.readFileSync(path.join(__dirname, "package.json"), "utf-8")
  );
  if (pkg && typeof pkg.version === "string") version = pkg.version;
} catch (_) {
  /* fallback to default */
}

contextBridge.exposeInMainWorld("vapDesktop", {
  getAppVersion: () => version,
  platform: process.platform,
  // 黑匣子日志 + 诊断导出(Builder P1-3:logs 页黑匣子实时面板 + 导出诊断包)
  logsQuery: (opts) => ipcRenderer.invoke("logs:query", opts),
  logsSubscribe: () => ipcRenderer.invoke("logs:subscribe"),
  onLogsAppend: (cb) => {
    ipcRenderer.on("logs:append", (_e, entry) => cb(entry));
  },
  collectDiagnostics: () => ipcRenderer.invoke("diagnostics:collect"),
});
