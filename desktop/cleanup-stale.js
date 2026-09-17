// desktop/cleanup-stale.js
// 启动前清理上次残留：只清理**本项目**的残留进程。
// 被 start-desktop.bat 调用(避开 CMD 内嵌 node -e 的引号转义坑)。
//
// v10.5.0 (P0-B) 加固：旧版第一步会**全机**杀 Electron(按镜像名 /IM 匹配,
// 把 VS Code 等用户应用一起干掉),与"只杀 python"的注释自相矛盾。
// 现改为：按进程命令行匹配本项目路径/后端标识，只杀本项目；端口清理同样过滤。
// PowerShell 脚本经临时 .ps1 文件执行(规避 cmd -> powershell.exe 的嵌套引号坑)。
// 新增 `--dry-run`：只列出将清理的 PID，不实际执行(便于人工复核)。
"use strict";

const cp = require("child_process");
const os = require("os");
const fs = require("fs");
const path = require("path");

// 本项目标识(命令行匹配用)——只清理本项目残留，绝不误伤其他应用。
const PROJECT_MARKER = "Video-Analysis-Pro-python";
const BACKEND_MARKER = "src.web.serve"; // 后端启动模块
const VENV_MARKER = "venv"; // 本项目虚拟环境路径(win 安装版自带 venv)

// 纯函数：从进程行里选出"本项目"的 PID(可单测)。
// rows: [{pid, name, commandLine}]
function selectProjectPids(rows) {
  const marker = [
    PROJECT_MARKER.toLowerCase(),
    BACKEND_MARKER.toLowerCase(),
    VENV_MARKER.toLowerCase(),
  ];
  return rows
    .filter((r) => {
      const low = String(r.commandLine || "").toLowerCase();
      return marker.some((m) => low.includes(m));
    })
    .map((r) => String(r.pid).trim())
    .filter(Boolean);
}

// 通过临时 .ps1 执行 PowerShell(规避 cmd -> powershell.exe 嵌套引号被拆坏)。
function ps(script) {
  const tmp = path.join(os.tmpdir(), "hermes-cleanup-" + process.pid + ".ps1");
  fs.writeFileSync(tmp, script, "utf8");
  try {
    return cp.execSync(
      'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + tmp + '"',
      { encoding: "utf8", maxBuffer: 8 * 1024 * 1024, stdio: ["pipe", "pipe", "ignore"] }
    );
  } finally {
    try { fs.unlinkSync(tmp); } catch (_) { /* 临时文件清理失败可忽略 */ }
  }
}

function killPids(pids, label) {
  for (const pid of pids) {
    try {
      cp.execSync("taskkill /F /PID " + pid + " /T", { stdio: "ignore" });
      console.log("  [INFO] killed stale " + label + " PID=" + pid);
    } catch (_) {
      /* 已退出 */
    }
  }
}

const PS_ELECTRON = "Get-CimInstance Win32_Process -Filter \"Name='electron.exe'\" | " +
  'ForEach-Object { "$($_.ProcessId)`t$($_.CommandLine)" }';

function killStale(dryRun) {
  // 1. 本项目 electron(按命令行含本项目路径)——不再全机杀 Electron
  let electronRows = [];
  try {
    electronRows = ps(PS_ELECTRON)
      .split(/\r?\n/)
      .filter(Boolean)
      .map((ln) => {
        const i = ln.indexOf("\t");
        return {
          pid: i >= 0 ? ln.slice(0, i) : ln,
          name: "electron.exe",
          commandLine: i >= 0 ? ln.slice(i + 1) : "",
        };
      });
  } catch (e) {
    console.error("  [WARN] electron enum err:", e.message);
  }
  const ePids = selectProjectPids(electronRows);
  if (ePids.length === 0) {
    console.log("  [INFO] no stale project electron.exe found");
  }
  dryRun
    ? console.log("  [DRY-RUN] would kill electron: " + (ePids.join(", ") || "(none)"))
    : killPids(ePids, "project electron");

  // 2. netstat 查 8000-8019 LISTENING PID；仅当进程命令行属于本项目才杀
  try {
    const out = cp.execSync("netstat -ano", {
      encoding: "utf8",
      maxBuffer: 8 * 1024 * 1024,
      stdio: ["pipe", "pipe", "ignore"],
    });
    const stale = new Set();
    for (const line of out.split(/\r?\n/)) {
      const m = line.match(/:(80(?:0[0-9]|1[0-9]))\s+.*LISTENING\s+(\d+)/);
      if (m && m[2] !== "0") stale.add(m[2].trim());
    }
    const rows = [];
    for (const pid of stale) {
      try {
        const cmdline = ps(
          "(Get-CimInstance Win32_Process -Filter \"ProcessId=" + pid + "\").CommandLine"
        ).trim();
        rows.push({ pid: pid, name: "python", commandLine: cmdline });
      } catch (_) {
        /* PID 已退出 */
      }
    }
    const pPids = selectProjectPids(rows);
    if (pPids.length === 0) {
      console.log("  [INFO] no stale project python on 8000-8019");
    }
    dryRun
      ? console.log("  [DRY-RUN] would kill python: " + (pPids.join(", ") || "(none)"))
      : killPids(pPids, "project python");
  } catch (e) {
    console.error("  [WARN] cleanup netstat err:", e.message);
  }
}

const dryRun = process.argv.includes("--dry-run");
killStale(dryRun);
