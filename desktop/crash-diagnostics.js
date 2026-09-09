// crash-diagnostics.js — 崩溃 / 诊断上下文收集
//
// 职责:collectDiagnostics() 一键打包以下内容为 zip buffer(或降级对象):
//  1. logs/runtime.log + logs/main.log + logs/desktop.log(存在则读,自动脱敏)
//  2. .env 是否存在(next 检查是否有 VAP_NV_API_KEYS 且已脱敏)
//  3. gc_guard 进程内存快照(psutil 或 pids;node 主进程自身 RSS + 已知 PID)
//  4. 系统信息(platform / node / electron / package 版本 / uptime / mem)
//
// 实现:最少 zip 打包 —— 用 node 内置 zlib + 手写 store/ 存储条目(NM 头+
//                   中央目录+EOCD),支持 UTF-8 文件名与二进制内容。
//                   不依赖 archiver/jszip 等第三方包。
//
// 纯 Node 标准库。Electron 主进程(getDiagnostics)与 IPC 均可调用。

"use strict";

const fs = require("fs");
const path = require("path");
const os = require("os");
const zlib = require("zlib");
const { version: nodeVersion } = process;

// ============================== 最小 zip 打包 ==============================
// ZIP 结构(存储法,仅 LocalFileHeader + Data + CentralDirectory + EOCD)。
// 不支持分卷 / 加密 / 数据描述符;目录项为目录条目。足够诊断包。
const CRC32_TABLE = (() => {
  const table = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  return table;
})();

function crc32(buf) {
  let crc = -1;
  for (let i = 0; i < buf.length; i++) {
    crc = (crc >>> 8) ^ CRC32_TABLE[(crc ^ buf[i]) & 0xff];
  }
  return (crc ^ -1) >>> 0;
}

/** 编码 UTF-8 文件名的二进制(追加返回)。 */
function encodeFilename(name) {
  return Buffer.from(name, "utf8");
}

/** 构造一个存储法 ZIP 的字节块(ze=nsxt shell 不返回流式,返 Buffer)。 */
function buildZip(entries) {
  // entries: [{ name: string, data: Buffer }]
  const localParts = [];
  const centralParts = [];
  let offset = 0;

  for (const e of entries) {
    const nameBuf = encodeFilename(e.name);
    const data = Buffer.isBuffer(e.data) ? e.data : Buffer.from(String(e.data), "utf8");
    const crc = crc32(data);
    const size = data.length;

    // -- Local File Header(通用标志只设 bit 11 = UTF-8 文件名) --
    const lh = Buffer.alloc(30);
    lh.writeUInt32LE(0x04034b50, 0); // signature
    lh.writeUInt16LE(20, 4); // version needed (2.0)
    lh.writeUInt16LE(0x0800, 6); // general purpose flag: UTF-8
    lh.writeUInt16LE(0, 8); // compression: 0 = store
    lh.writeUInt16LE(0, 10); // mod time (0)
    lh.writeUInt16LE(0, 12); // mod date (0)
    lh.writeUInt32LE(crc, 14);
    lh.writeUInt32LE(size, 18);
    lh.writeUInt32LE(size, 22);
    lh.writeUInt16LE(nameBuf.length, 26);
    lh.writeUInt16LE(0, 28); // extra len
    localParts.push(lh, nameBuf, data);

    // -- Central Directory --
    const ch = Buffer.alloc(46);
    ch.writeUInt32LE(0x02014b50, 0); // signature
    ch.writeUInt16LE(20, 4); // version made by
    ch.writeUInt16LE(20, 6); // version needed
    ch.writeUInt16LE(0x0800, 8); // flag UTF-8
    ch.writeUInt16LE(0, 10); // method store
    ch.writeUInt16LE(0, 12); // mod time
    ch.writeUInt16LE(0, 14); // mod date
    ch.writeUInt32LE(crc, 16);
    ch.writeUInt32LE(size, 20);
    ch.writeUInt32LE(size, 24);
    ch.writeUInt16LE(nameBuf.length, 28);
    ch.writeUInt16LE(0, 30); // extra
    ch.writeUInt16LE(0, 32); // comment
    ch.writeUInt16LE(0, 34); // disk # → 0。outrev 远古bug,填0
    ch.writeUInt16LE(0, 36); // internal attrs
    ch.writeUInt32LE(0x20, 38); // external attrs: 8 = regular file
    ch.writeUInt32LE(offset, 42); // local header offset
    centralParts.push(ch, nameBuf);
    offset += 30 + nameBuf.length + data.length;
  }

  const centralEnd = Buffer.alloc(22);
  centralEnd.writeUInt32LE(0x06054b50, 0); // EOCD signature
  centralEnd.writeUInt16LE(0, 4); // disk
  centralEnd.writeUInt16LE(0, 6); // central disk
  centralEnd.writeUInt16LE(entries.length, 8);
  centralEnd.writeUInt16LE(entries.length, 10);
  // cdSize = Σ(46 + nameLen),cdOffset = Σ local sizes(offset 已累计)
  let centralSize = 0;
  for (const e of entries) centralSize += 46 + encodeFilename(e.name).length;
  centralEnd.writeUInt32LE(centralSize, 12); // cdSize
  centralEnd.writeUInt32LE(offset, 16); // cdOffset
  centralEnd.writeUInt16LE(0, 20); // comment

  return Buffer.concat([...localParts, ...centralParts, centralEnd]);
}

// ============================== 诊断收集 ==============================
/**
 * 读取文件内容,失败回退空 Buffer(不抛)。
 * @param {string} filePath
 * @returns {Buffer}
 */
function readFileGraceful(filePath, maxBytes = 5 * 1024 * 1024) {
  try {
    const st = fs.statSync(filePath);
    if (!st.isFile() || st.size === 0) return Buffer.alloc(0);
    let fd;
    try {
      fd = fs.openSync(filePath, "r");
      const buf = Buffer.alloc(Math.min(st.size, maxBytes));
      fs.readSync(fd, buf, 0, buf.length, 0);
      return buf;
    } finally {
      if (fd !== undefined) fs.closeSync(fd);
    }
  } catch (_) {
    return Buffer.alloc(0);
  }
}

/**
 * 尝试读取配置/快照对象。用于打包"config 快照(不含密钥)"。
 * @returns {object}
 */
function collectConfigSnapshot() {
  const out = {};
  const projectRoot = path.resolve(__dirname, "..");
  // app_config.ini 运行时文件,已 gitignore,读其存在与否 + 脱敏行(去掉 api_key 值)
  const iniPath = path.join(projectRoot, "config", "app_config.ini");
  try {
    if (fs.existsSync(iniPath)) {
      const raw = fs.readFileSync(iniPath, "utf8");
      out.app_config_ini = redactAll(raw).split(/\r?\n/).slice(0, 80);
    }
  } catch (_) { /* ignore */ }
  // .env 只记录存在性与键名(绝不读值)
  const envPath = path.join(projectRoot, ".env");
  try {
    if (fs.existsSync(envPath)) {
      const raw = fs.readFileSync(envPath, "utf8");
      const keys = (raw.match(/^\s*([A-Z][A-Z0-9_]*)\s*=/gm) || []).map(
        (line) => line.trim().split("=")[0]
      );
      out.dotenv = { exists: true, keys };
    } else {
      out.dotenv = { exists: false };
    }
  } catch (_) {
    out.dotenv = { exists: false, note: "unreadable" };
  }
  out.environment = {
    VAP_LLM_PROVIDER: process.env.VAP_LLM_PROVIDER || null,
    VAP_LLM_BASE_URL: process.env.VAP_LLM_BASE_URL || null,
    VAP_TUNNEL_PROVIDER: process.env.VAP_TUNNEL_PROVIDER || null,
    // 只列键名,不列值
    env_keys_with_vap_prefix: Object.keys(process.env).filter((k) =>
      k.toUpperCase().startsWith("VAP_")
    ),
  };
  return out;
}

/**
 * gc_guard / 运行中相关进程的内存快照。
 * 数据可能不可用(psutil 缺失/权限),所有读都 try/catch 降级。
 * @returns {Array<{pid:number,cmd:string,rss_mb:number|null}>}
 */
function collectProcessMemorySnapshot() {
  const out = [];
  // 1. 当前 node/electron 主进程自身
  try {
    out.push({
      pid: process.pid,
      cmd: "electron-main",
      rss_mb: Math.round((process.memoryUsage().rss / 1024 / 1024) * 10) / 10,
    });
  } catch (_) { /* ignore */ }
  // 2. 尝试载入 psutil(python),用 python -c 读指定 PID 的 RSS —— 目标进程是
  //    serve.py(python subprocess)。没有 psutil 就跳过(降级对象足够)。
  //    真实 Electron 主进程里,子进程 PID 由 RuntimeController 持有。
  const pids = (global.__TF_DIAG_PIDS__ || []).filter((p) => Number.isFinite(p));
  if (pids.length > 0) {
    try {
      const { spawnSync } = require("child_process");
      const script =
        "import sys,json" +
        ";import psutil" +
        ";d={};" +
        "[d.update({str(p):{'name':psutil.Process(p).name(),'rss_mb':round(psutil.Process(p).memory_info().rss/1048576,1)}})" +
        "for p in list(map(int,sys.argv[1:])) if psutil.pid_exists(p)]" +
        ";print(json.dumps(d))";
      const r = spawnSync("python", ["-c", script, ...pids.map(String)], {
        encoding: "utf8",
        timeout: 8000,
        windowsHide: true,
      });
      if (r.status === 0 && r.stdout) {
        const parsed = JSON.parse(r.stdout.trim());
        for (const pid of pids) {
          const info = parsed[String(pid)];
          if (info) {
            out.push({
              pid,
              cmd: info.name || "unknown",
              rss_mb: info.rss_mb ?? null,
            });
          }
        }
      }
    } catch (_) { /* psutil 不可用,跳过 */ }
  }
  return out;
}

/**
 * 收集系统信息。
 * @returns {object}
 */
function collectSystemInfo() {
  return {
    platform: process.platform,
    arch: process.arch,
    node: nodeVersion,
    electron: (() => {
      try {
        return require("electron/package.json").version;
      } catch (_) {
        return null;
      }
    })(),
    desktop_pkg: (() => {
      try {
        const p = require(path.join(__dirname, "package.json"));
        return { name: p.name, version: p.version };
      } catch (_) {
        return null;
      }
    })(),
    cpus: os.cpus().length,
    total_mem_mb: Math.round((os.totalmem() / 1024 / 1024) * 10) / 10,
    free_mem_mb: Math.round((os.freemem() / 1024 / 1024) * 10) / 10,
    uptime_sec: Math.round(os.uptime()),
    loadavg: (() => {
      try {
        return os.loadavg();
      } catch (_) {
        return null;
      }
    })(),
    home: os.homedir(),
  };
}

/**
 * 主入口:收集诊断并打包为 zip Buffer。
 * @param {{ pids?: number[], outDir?: string }} [opts]
 * @returns {{ zip: Buffer, entries: string[], system: object }}
 */
function collectDiagnostics(opts = {}) {
  const projectRoot = path.resolve(__dirname, "..");
  const pids = opts.pids || [];

  const files = [
    ["runtime.log", path.join(projectRoot, "logs", "runtime.log")],
    ["main.log", path.join(projectRoot, "logs", "main.log")],
    ["desktop.log", path.join(projectRoot, "logs", "desktop.log")],
  ];
  const entries = [];
  for (const [name, full] of files) {
    let data = readFileGraceful(full);
    // 日志含密钥风险:统一脱敏
    data = Buffer.from(redactAll(data.toString("utf8")), "utf8");
    entries.push({ name: `logs/${name}`, data });
  }

  const system = collectSystemInfo();
  const config = collectConfigSnapshot();
  const mem = collectProcessMemorySnapshot();

  entries.push({
    name: "system.json",
    data: Buffer.from(JSON.stringify(system, null, 2), "utf8"),
  });
  entries.push({
    name: "config-snapshot.json",
    data: Buffer.from(JSON.stringify(config, null, 2), "utf8"),
  });
  entries.push({
    name: "process-memory.json",
    data: Buffer.from(JSON.stringify(mem, null, 2), "utf8"),
  });

  const zip = buildZip(entries);
  return { zip, entries: entries.map((e) => e.name), system };
}

// 脱敏复用 —— 把 LogStore 的 redact 引用过来,避免两处维护正则。
const { redact: redactAll } = require("./log-store");

module.exports = {
  collectDiagnostics,
  collectSystemInfo,
  collectProcessMemorySnapshot,
  collectConfigSnapshot,
  buildZip,
  crc32,
  readFileGraceful,
};