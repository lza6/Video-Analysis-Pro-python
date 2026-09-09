// log-store.js — 主进程黑匣子日志(内存环形缓冲 + 控制台捕获)
//
// 职责:
//  1. LogStore:append/query/search/clear/snapshot 环形缓冲(默认 2000 条滚动)
//  2. 密钥脱敏:VAP_*/api_key/token/authorization/bearer/sk-* 正则掩码
//  3. installConsoleCapture():patch console.* + process.stdout/stderr.write
//     + uncaughtException/unhandledRejection,喂给 LogStore
//
// 纯 Node 标准库,不依赖 npm。Electron 主进程 require 它。

"use strict";

const { EventEmitter } = require("events");

// ---------------------------------------------------------------------------
// 密钥脱敏(正则掩码)
// ---------------------------------------------------------------------------
const SECRET_PATTERNS = [
  // 先跑 Bearer(带 \b 大小写不敏感),避免被下面的 authorization=X 规则抢先吃掉
  /(\bBearer\s+)[A-Za-z0-9._+/=-]+/gi,
  // .env 风格:VAP_LLM_API_KEY=sk-xxx(整体掩掉整个 value)
  /(VAP_[A-Z0-9_]+)\s*[=:]\s*(['"]?)[^\s'";,]*\2/g,
  // key=value / key: value / JSON "key":"value"(不区分大小写的敏感键名)
  /((?:api[_-]?key|access[_-]?key|secret|token|authorization|password|auth|passwd)\s*[=:]\s*)(['"]?)([^\s'";,&}]+)\2/gi,
  // 云端 Key 形态:sk-xxx…(保留前 3 位便于识别前缀)
  /(sk-[A-Za-z0-9]{3})[A-Za-z0-9]+/g,
];

const REDACT_REPLACER = (m, prefix) => `${prefix}****`;

/**
 * 对单条文本做密钥脱敏。不可变:返回新字符串。
 * @param {string} text
 * @returns {string}
 */
function redact(text) {
  let out = String(text);
  for (const re of SECRET_PATTERNS) {
    out = out.replace(re, (m, ...groups) => {
      const prefix = groups[0];
      return prefix ? REDACT_REPLACER(m, prefix) : "****";
    });
  }
  return out;
}

/** @returns {string} 将一个参数格式化成单行字面值(Error 取 stack,对象 JSON)。 */
function formatArg(value) {
  if (value instanceof Error) return value.stack || value.message;
  if (typeof value === "object" && value !== null) {
    try {
      return JSON.stringify(value);
    } catch (_) {
      return String(value);
    }
  }
  return String(value);
}

/**
 * 主进程黑匣子日志存储。
 *
 * 条目: { ts: number, source: string, level: 'debug'|'info'|'warn'|'error',
 *          msg: string(已脱敏) }
 */
class LogStore extends EventEmitter {
  /**
   * @param {{ maxEntries?: number }} [opts]
   */
  constructor(opts = {}) {
    super();
    /** @type {Array<{ts:number,source:string,level:string,msg:string}>} */
    this._entries = [];
    this._max = Number.isFinite(opts.maxEntries) ? opts.maxEntries : 2000;
  }

  /** @returns {number} 环形缓冲上限 */
  get maxEntries() {
    return this._max;
  }

  /**
   * 追加一条日志(自动脱敏 + 滚动裁剪)。
   * @param {string} source 来源标记:'main'|'serve'|'console'|'renderer'|'system'
   * @param {string} level  log 级别
   * @param {string} msg    原始消息(将脱敏)
   * @returns {{ts:number,source:string,level:string,msg:string}}
   */
  append(source, level, msg) {
    const entry = {
      ts: Date.now(),
      source: String(source || "system"),
      level: String(level || "info"),
      msg: redact(msg),
    };
    this._entries.push(entry);
    if (this._entries.length > this._max) {
      this._entries.splice(0, this._entries.length - this._max);
    }
    this.emit("append", entry);
    return entry;
  }

  /**
   * 查询(支持级别过滤 + 上限)。
   * @param {{ limit?: number, level?: string, source?: string, search?: string, after?: number }} [opts]
   * @returns {Array<{ts:number,source:string,level:string,msg:string}>}
   */
  query(opts = {}) {
    const { limit, level, source, search, after } = opts;
    let out = this._entries;
    if (level) {
      const want = level.toLowerCase();
      out = out.filter((e) => e.level === want);
    }
    if (source) out = out.filter((e) => e.source === source);
    if (search) {
      const q = String(search).toLowerCase();
      out = out.filter(
        (e) => e.msg.toLowerCase().includes(q) || e.source.toLowerCase().includes(q)
      );
    }
    if (after !== undefined && Number.isFinite(after)) {
      out = out.filter((e) => e.ts >= after);
    }
    if (Number.isFinite(limit) && limit > 0) out = out.slice(-limit);
    return out.slice();
  }

  /**
   * 文本搜索(跨 source/msg 子串,大小写不敏感)。
   * @param {string} text
   * @param {{ limit?: number }} [opts]
   * @returns {Array<{ts:number,source:string,level:string,msg:string}>}
   */
  search(text, opts = {}) {
    return this.query({ search: text, limit: opts.limit });
  }

  /** 清空缓冲。 */
  clear() {
    this._entries.length = 0;
    this.emit("cleared");
  }

  /** 返回只读快照(浅拷贝)。 */
  snapshot() {
    return this._entries.slice();
  }
}

// ---------------------------------------------------------------------------
// 控制台捕获
// ---------------------------------------------------------------------------
let _captureInstalled = false;

/**
 * 挂 process stdout/stderr 与 console.*、进程级异常事件到 LogStore。
 * 幂等:重复调用只装一次。
 * @param {LogStore} store
 */
function installConsoleCapture(store) {
  if (_captureInstalled) return;
  _captureInstalled = true;

  const patch = (fn, source, level) => {
    const original = fn.bind(console);
    console[fn.name] = (...args) => {
      store.append(source, level, args.map(formatArg).join(" "));
      return original(...args);
    };
  };
  patch(console.log, "console", "info");
  patch(console.info, "console", "info");
  patch(console.warn, "console", "warn");
  patch(console.error, "console", "error");
  patch(console.debug, "console", "debug");

  // 其它模块直接写 process.stdout/stderr 的原始字节(如 runtime-controller
  // 的 process.stdout.write)。捕获后仍转发原输出,不改变行为。
  const origWrite = {
    stdout: process.stdout.write.bind(process.stdout),
    stderr: process.stderr.write.bind(process.stderr),
  };
  process.stdout.write = (chunk, ...rest) => {
    store.append("stdout", "info", String(chunk));
    return origWrite.stdout(chunk, ...rest);
  };
  process.stderr.write = (chunk, ...rest) => {
    store.append("stderr", "error", String(chunk));
    return origWrite.stderr(chunk, ...rest);
  };

  process.on("uncaughtException", (err) => {
    store.append("system", "error", `[uncaughtException] ${formatArg(err)}`);
  });
  process.on("unhandledRejection", (reason) => {
    store.append("system", "error", `[unhandledRejection] ${formatArg(reason)}`);
  });
}

module.exports = { LogStore, installConsoleCapture, redact };