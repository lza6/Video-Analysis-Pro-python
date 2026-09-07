// desktop/cleanup-stale.js
// 启动前清理上次残留:杀 electron.exe + 占 8000-8019 端口的 python.exe(不动系统服务)。
// 被 start-desktop.bat 调用(避开 CMD 内嵌 node -e 的引号转义坑)。
"use strict";

const cp = require("child_process");

function killStale() {
  // 1. 杀残留 electron.exe
  try {
    cp.execSync("taskkill /F /IM electron.exe /T", { stdio: "ignore" });
    console.log("  [INFO] killed stale electron.exe");
  } catch (_) {
    /* 没有残留 */
  }

  // 2. netstat 查 8000-8019 LISTENING PID,只杀 python(不动 Manager.exe 等系统服务)
  try {
    const out = cp.execSync("netstat -ano", {
      encoding: "utf8",
      maxBuffer: 8 * 1024 * 1024,
      stdio: ["pipe", "pipe", "ignore"],
    });
    const stalePids = new Set();
    for (const line of out.split(/\r?\n/)) {
      const m = line.match(/:(80(?:0[0-9]|1[0-9]))\s+.*LISTENING\s+(\d+)/);
      if (m && m[2] !== "0") stalePids.add(m[2].trim());
    }
    for (const pid of stalePids) {
      try {
        const nameOut = cp
          .execSync(`tasklist /FI "PID eq ${pid}" /FO CSV /NH`, {
            encoding: "utf8",
            stdio: ["pipe", "pipe", "ignore"],
          })
          .trim();
        if (nameOut.toLowerCase().includes("python")) {
          cp.execSync(`taskkill /F /PID ${pid} /T`, { stdio: "ignore" });
          console.log(`  [INFO] killed stale python PID=${pid}`);
        }
      } catch (_) {
        /* PID 已退出 */
      }
    }
  } catch (e) {
    console.error("  [WARN] cleanup netstat err:", e.message);
  }
}

killStale();
