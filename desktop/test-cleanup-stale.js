// desktop/test-cleanup-stale.js — selectProjectPids 纯函数单测(不杀任何进程)
// 验证 cleanup-stale.js(P0-B) 的"精确清理"语义：只匹配本项目，绝不误杀其他应用。
"use strict";

const assert = require("assert");
const src = require("fs").readFileSync(require("path").join(__dirname, "cleanup-stale.js"), "utf8");

// 通过 Function 构造器取出纯函数(脚本尾部 killStale(dryRun) 会执行,因此只取函数定义段)
const fnDef = src.slice(src.indexOf("const PROJECT_MARKER"), src.indexOf("function ps("));
// eslint-disable-next-line no-new-func
const selectProjectPids = new Function(fnDef + "\nreturn selectProjectPids;")();

const VSCode = { pid: "1111", name: "electron.exe", commandLine: "C:\\Program Files\\Microsoft VS Code\\Code.exe --type=renderer" };
const OtherElectron = { pid: "2222", name: "electron.exe", commandLine: "C:\\Users\\x\\AppData\\Local\\Slack\\app-4.40.0\\slack.exe" };
const ProjectElectron = { pid: "3333", name: "electron.exe", commandLine: "C:\\Users\\x\\Desktop\\agent\\视觉分析\\Video-Analysis-Pro-python\\desktop\\node_modules\\electron\\dist\\electron.exe --project" };
const ProjectPython = { pid: "4444", name: "python.exe", commandLine: "C:\\Users\\x\\Desktop\\agent\\视觉分析\\Video-Analysis-Pro-python\\venv\\Scripts\\python.exe -m src.web.serve --port 8001" };
const OtherPython = { pid: "5555", name: "python.exe", commandLine: "C:\\Python39\\python.exe -m http.server 8088" };
const BackendMarkerOnly = { pid: "6666", name: "python.exe", commandLine: "python -m src.web.serve --no-browser" };

const result = selectProjectPids([VSCode, OtherElectron, ProjectElectron, ProjectPython, OtherPython, BackendMarkerOnly]);

// 必须命中：本项目 electron / 本项目 python / 仅后端标识也命中
for (const pid of ["3333", "4444", "6666"]) {
  assert(result.includes(pid), "应命中本项目 PID " + pid + "，实际=" + JSON.stringify(result));
}
// 必须排除：VS Code / 其他 Electron / 其他非本项目 python
for (const pid of ["1111", "2222", "5555"]) {
  assert(!result.includes(pid), "不得命中非本项目 PID " + pid + "，实际=" + JSON.stringify(result));
}

// 边界：空输入 / 无命令行
assert.deepStrictEqual(selectProjectPids([]), []);
assert.deepStrictEqual(selectProjectPids([{ pid: "9", name: "python.exe", commandLine: "" }]), []);

// 防回归：脚本源码不得再出现全机杀 Electron(旧版 taskkill /F /IM electron.exe)
assert(!/taskkill\s+\/F\s+\/IM\s+electron/i.test(src), "出现全机杀 electron 的 taskkill /IM，违反 P0-B 安全边界");

console.log("cleanup-stale checks passed (" + result.length + " matched, 3 excluded + 2 edge cases)");
