// runtime-controller.js
// 职责:spawn `python src/web/serve.py` 子进程,从 stdout 探测实际端口,
//      轮询 /api/health 兜底,提供 kill 清理(Windows taskkill /T 防孤进程)。
//
// 不依赖任何 npm 包,纯 Node 标准库 + child_process。Electron 主进程 require 它。

"use strict";

const { spawn } = require("child_process");
const path = require("path");
const http = require("http");

// serve.py stdout 中的就绪信号:
//   "TingFeng Hermes Web UI: http://127.0.0.1:<port>"
// 也兼容 uvicorn 自身的 "Uvicorn running on http://127.0.0.1:<port>"
const READY_RE = /http:\/\/127\.0\.0\.1:(\d+)/;
const FAIL_RE = /端口\s*\d+\s*-\s*\d+\s*全被占用|无法启动/;

const PROJECT_ROOT = path.resolve(__dirname, "..");

/**
 * 定位 venv python 可执行文件。找不到则回退系统 python/python3。
 * @returns {string}
 */
function resolvePythonExe() {
  const fs = require("fs");
  const candidates = [
    path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe"), // Windows
    path.join(PROJECT_ROOT, "venv", "bin", "python"), // macOS/Linux
  ];
  for (const c of candidates) {
    try {
      if (fs.existsSync(c)) return c;
    } catch (_) {
      /* ignore */
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}

/**
 * 等待 http://127.0.0.1:port/api/health 返回 200。兜底探测。
 * @param {number} port
 * @param {number} timeoutMs
 * @returns {Promise<boolean>}
 */
function waitForHealth(port, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve) => {
    function probe() {
      if (Date.now() > deadline) return resolve(false);
      const req = http.get(
        `http://127.0.0.1:${port}/api/health`,
        { timeout: 2000 },
        (res) => {
          res.resume();
          resolve(res.statusCode === 200);
        }
      );
      req.on("error", () => setTimeout(probe, 400));
      req.on("timeout", () => {
        req.destroy();
        setTimeout(probe, 400);
      });
    }
    probe();
  });
}

class RuntimeController {
  constructor() {
    /** @type {import("child_process").ChildProcess | null} */
    this.child = null;
    /** @type {number|null} */
    this.port = null;
    /** @type {(line: string) => void} */
    this.onLog = () => {};
    /** @type {(port: number) => void} */
    this.onReady = () => {};
    /** @type {(err: Error) => void} */
    this.onFailed = () => {};
  }

  /**
   * 启动 python serve.py 子进程,探测端口。
   * @param {{onLog?: (l: string) => void, onReady?: (p: number) => void, onFailed?: (e: Error) => void}} [opts]
   * @returns {Promise<number>} 实际端口
   */
  async start(opts = {}) {
    this.onLog = opts.onLog || (() => {});
    this.onReady = opts.onReady || (() => {});
    this.onFailed = opts.onFailed || (() => {});

    const exe = resolvePythonExe();
    // 用 `python -m src.web.serve` 而非 `python src/web/serve.py`:
    // serve.py 内部用相对导入(from .config import get_settings),
    // 当脚本直接跑会 ImportError(attempted relative import with no
    // known parent package)。改走包模块入口,与 launcher.py 一致。
    const env = {
      ...process.env,
      PYTHONIOENCODING: "utf-8",
      PYTHONUNBUFFERED: "1",
      // 桌面壳自己开窗口,不让 serve.py 再开浏览器
      VAP_NO_BROWSER: "1",
    };

    // Windows 默认 GBK,显式指定 encoding:utf-8 + errors:replace 防 UnicodeDecodeError
    this.child = spawn(exe, ["-m", "src.web.serve", "--no-browser"], {
      cwd: PROJECT_ROOT,
      env,
      windowsHide: false,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdoutBuf = "";
    let stderrBuf = "";
    let settled = false;
    /** @type {((port:number)=>void)|null} */
    let resolveReady = null;
    /** @type {((err:Error)=>void)|null} */
    let rejectFail = null;

    const settleReady = (port) => {
      if (settled) return;
      settled = true;
      this.port = port;
      this._log(`settleReady port=${port}`);
      this.onReady(port);
      if (resolveReady) resolveReady(port);
    };

    const settleFail = (err) => {
      if (settled) return;
      settled = true;
      this._log(`settleFail: ${err.message}`);
      this.onFailed(err);
      if (rejectFail) rejectFail(err);
    };

    const scan = (chunk) => {
      const text = chunk.toString("utf-8");
      this.onLog(text);
      stdoutBuf += text;
      let m;
      if ((m = READY_RE.exec(stdoutBuf))) {
        const port = parseInt(m[1], 10);
        if (port > 0) settleReady(port);
      }
      if (FAIL_RE.test(stdoutBuf)) {
        settleFail(new Error("serve.py 报告端口全被占用,无法启动"));
      }
    };

    this.child.stdout.on("data", scan);
    this.child.stderr.on("data", (chunk) => {
      const text = chunk.toString("utf-8");
      stderrBuf += text;
      this.onLog(`[stderr] ${text}`);
      // uvicorn 默认 stderr,虽然 serve.py 重定向到 stdout,兜底也扫一下 stderr
      const m = READY_RE.exec(text);
      if (m) settleReady(parseInt(m[1], 10));
    });

    this.child.on("error", (e) => settleFail(e));
    this.child.on("exit", (code, signal) => {
      if (!settled) {
        settleFail(
          new Error(`serve.py 子进程提前退出(exit=${code} signal=${signal})`)
        );
      }
    });

    // 兜底:5s 内 stdout 还没给端口,主动轮询默认 8000-8019
    setTimeout(async () => {
      if (settled) return;
      this.onLog("[runtime] stdout 未直接给端口,启动健康兜底探测\n");
      for (let p = 8000; p <= 8019; p++) {
        const ok = await waitForHealth(p, 800);
        if (ok) {
          settleReady(p);
          return;
        }
      }
    }, 5000);

    // 总超时 90s
    // settleReady/settleFail 调 onReady/onFailed 后,resolve/reject Promise
    // 让 start() 能真正返回 port(之前只 settle 没 resolve,Promise 永挂)。
    return new Promise((resolve, reject) => {
      resolveReady = resolve;
      rejectFail = reject;
      const t = setTimeout(() => {
        if (!settled) {
          settleFail(new Error("serve.py 90s 内未就绪"));
        }
      }, 90000);
      this.child.once("exit", () => clearTimeout(t));
    }).then((port) => {
      this._log(`start() then: port=${port}`);
      return port;
    }).catch((e) => {
      this._log(`start() catch: ${e.message}`);
      throw e;
    });
  }

  /**
   * 写文件日志(主进程 stdout 可能不输出,写 logs/runtime.log 黑匣子)。
   * @param {string} msg
   */
  _log(msg) {
    try {
      const fs = require("fs");
      const p = require("path");
      const logFile = p.join(PROJECT_ROOT, "logs", "runtime.log");
      const line = "[" + new Date().toISOString() + "] " + msg + "\n";
      fs.appendFileSync(logFile, line, { encoding: "utf8" });
    } catch (_) { /* ignore */ }
  }

  /**
   * 杀掉子进程。Windows 用 taskkill /T /F 杀整个进程树(防孤进程)。
   */
  kill() {
    if (!this.child || this.child.exitCode !== null) return;
    const pid = this.child.pid;
    try {
      if (process.platform === "win32" && pid) {
        // /T:杀子树 /F:强制
        spawn("taskkill", ["/PID", String(pid), "/T", "/F"], {
          windowsHide: true,
          stdio: "ignore",
        });
      } else if (pid) {
        try {
          process.kill(pid, "SIGTERM");
        } catch (_) {
          /* already dead */
        }
      }
    } catch (_) {
      /* ignore */
    }
    this.child = null;
  }
}

module.exports = { RuntimeController, READY_RE, waitForHealth };
