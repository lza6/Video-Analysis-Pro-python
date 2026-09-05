"""多 provider 多 key 路由层 (T1)

在 `llm_gateway.py` 之上加一层「按 provider 分组的多 key 轮换 + 每 key 限速 +
无限重试切 key」的路由器。llm_gateway 仍是单 provider 单 key 的协议后端抽象，
本模块负责运维层：key 池调度、40 req/min 滑动窗口、429/5xx 无限重试切下一个 key、
401/403/404/422 标记 key 失效并 raise。

设计要点（用户硬性要求）：
  - 40 req/min 每 key 限速要管理好 → 滑动窗口令牌桶，acquire 阻塞等到有空位
  - 429/5xx **无限重试**（不退避，0.5s 间隔），切下一个 key
  - 仅 401/403/404/422 放弃（标记 key 失效并 raise，不切 key）
  - 多 key 轮换提并发 → select_key 按 (priority desc, 最久未用) 选可用 key

NVIDIA 实测约束：
  - 端点 POST https://integrate.api.nvidia.com/v1/chat/completions
  - 认证 Authorization: Bearer nvapi-xxx
  - 思考用 chat_template_kwargs.enable_thinking + reasoning_budget
    拍平到顶层（raw REST 不接受 extra_body / thinking_token_budget）

不重写 llm_gateway.py，本文件只新增。
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock, Semaphore
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("VideoAnalyzerCore")

# ---- 常量 ----
NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
KILO_DEFAULT_BASE_URL = "https://api.kilocode.io/v1"
# 每 key 每分钟请求数上限（NVIDIA 公开配额）
RATE_LIMIT_PER_MIN = 40
# 429/5xx 无限重试间隔（秒），用户明确要求不退避、0.5s 间隔
RETRY_INTERVAL = 0.5
# 503/5xx 短退避：worker 空槽再重试同 key（不立即切 key，减少空转）
# 实测 NVIDIA 503 "Worker 16/16" 频繁，立即切 key 只会让 11 个 key 同时撞墙
# 退避 1.5s × 重试 2 次同 key，给 worker 池留出空槽时间
SERVER_ERROR_BACKOFF_SEC = 1.5
SERVER_ERROR_SAME_KEY_RETRIES = 2
# 放弃状态码：401/403/404/422 → 标记 key 失效并 raise（不切 key）
FATAL_STATUS_CODES = {401, 403, 404, 422}
# 重试状态码：429/5xx → 切下一个 key 无限重试
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
# 单 key 并发上限（默认 2）：11 key 全打满反互挤，每 key 并发 2 更稳
DEFAULT_MAX_CONCURRENT_PER_KEY = 2


@dataclass
class ProviderKey:
    """单个 provider 的单个 API key 运行态。

    静态字段来自配置（.env / 9router json），运行态字段由 ProviderRouter 维护。
    """
    id: str
    name: str
    provider: str  # "nvidia" | "kilo" | "openai-compatible" | "ollama" | ...
    api_key: str
    base_url: str = ""
    priority: int = 0
    isActive: bool = True  # 配置态开关（9router 字段名沿用）
    # 运行态
    consecutive_429: int = 0
    last_used: float = 0.0  # monotonic 时间戳
    backoff_until: float = 0.0  # monotonic 时间戳，到点前该 key 不被选中
    last_error: str = ""

    @property
    def is_available(self) -> bool:
        """是否可被 select_key 选中：配置态 active + 限速窗口未到。"""
        if not self.isActive:
            return False
        # backoff_until 用 monotonic；0 = 未设
        if self.backoff_until and time.monotonic() < self.backoff_until:
            return False
        return True


class RateLimiter:
    """每 key 滑动窗口令牌桶：40 req/min，acquire(key_id) 阻塞等到有空位。

    线程安全。用 monotonic 时间戳记请求时刻的 deque，超过窗口的旧记录弹出。
    测试用 monkeypatch time.monotonic / time.sleep 快进。

    另含 per-key 并发信号量（max_concurrent_per_key，默认 2）：限制单 key 同时
    在飞的请求数，避免 11 key 全打满反互挤（NVIDIA worker 池有限，每 key 并发
    2 更稳）。acquire_concurrent 拿信号量，release_concurrent 释放。
    """

    def __init__(self, limit_per_min: int = RATE_LIMIT_PER_MIN,
                 window_sec: float = 60.0,
                 max_concurrent_per_key: int = DEFAULT_MAX_CONCURRENT_PER_KEY):
        self.limit = max(1, limit_per_min)
        self.window_sec = window_sec
        self.max_concurrent_per_key = max(1, max_concurrent_per_key)
        self._buckets: Dict[str, deque] = {}
        self._semaphores: Dict[str, Semaphore] = {}
        self._lock = Lock()

    def _now(self) -> float:
        """可被 monkeypatch 的时钟入口。"""
        return time.monotonic()

    def _sleep(self, seconds: float) -> None:
        """可被 monkeypatch 的睡眠入口。"""
        time.sleep(seconds)

    def _bucket(self, key_id: str) -> deque:
        b = self._buckets.get(key_id)
        if b is None:
            b = deque()
            self._buckets[key_id] = b
        return b

    def _semaphore(self, key_id: str) -> Semaphore:
        """惰性创建 per-key 信号量（线程安全）。"""
        s = self._semaphores.get(key_id)
        if s is None:
            s = Semaphore(self.max_concurrent_per_key)
            self._semaphores[key_id] = s
        return s

    def acquire(self, key_id: str, timeout: float = 120.0) -> bool:
        """阻塞直到该 key 有空位可用，返回 True；超时返回 False。

        实现：弹出窗口外旧记录，若当前窗口内 < limit 立即放行；否则睡到最早记录
        超出窗口为止，醒来再检查。
        """
        deadline = self._now() + timeout
        while True:
            now = self._now()
            with self._lock:
                b = self._bucket(key_id)
                # 弹出窗口外旧记录
                while b and now - b[0] >= self.window_sec:
                    b.popleft()
                if len(b) < self.limit:
                    b.append(now)
                    return True
                # 计算最早记录何时超出窗口
                wait = self.window_sec - (now - b[0])
            if now + wait > deadline:
                return False
            self._sleep(max(0.0, wait))

    def available(self, key_id: str) -> int:
        """查询当前窗口剩余额度（非阻塞，仅供诊断/测试）。"""
        now = self._now()
        with self._lock:
            b = self._bucket(key_id)
            while b and now - b[0] >= self.window_sec:
                b.popleft()
            return max(0, self.limit - len(b))

    # ---- per-key 并发信号量 ----
    def acquire_concurrent(self, key_id: str,
                           timeout: Optional[float] = None) -> bool:
        """拿 per-key 并发槽位。Semaphore(n) 惰性创建，acquire 阻塞到有空槽。

        与 acquire（限速窗口）正交：acquire 管「每分钟 40 个」，acquire_concurrent
        管「同时在飞 ≤ 2 个」。timeout=None 阻塞到有空槽；超时返回 False。
        """
        sem = self._semaphore(key_id)
        if timeout is None:
            # Semaphore.acquire(None) 阻塞到有空槽
            sem.acquire()
            return True
        return sem.acquire(timeout=timeout)

    def release_concurrent(self, key_id: str) -> None:
        """释放 per-key 并发槽位。成对调用（acquire → release）。"""
        sem = self._semaphores.get(key_id)
        if sem is not None:
            sem.release()

    def concurrent_slots(self, key_id: str) -> int:
        """该 key 的并发上限（诊断/测试用）。"""
        return self.max_concurrent_per_key

    def concurrent_available(self, key_id: str) -> int:
        """该 key 当前剩余并发槽位（诊断/测试用，非阻塞）。

        Semaphore 不暴露剩余值，用 _value 内部字段读取（CPython 实现）。
        测试用 monkeypatch 验证 acquire/release 行为，不依赖此值。
        """
        sem = self._semaphores.get(key_id)
        if sem is None:
            return self.max_concurrent_per_key
        return getattr(sem, "_value", self.max_concurrent_per_key)


class ProviderRouter:
    """多 provider 多 key 路由器。

    - __init__ 接受一组 ProviderKey，按 (priority desc) 排序保留。
    - select_key(provider) 选一个可用 key（active + 未到 backoff + 优先级最高 +
      最久未用优先，提并发）。
    - post_nvidia(payload) 用轮换 key 发 NVIDIA，429/5xx 无限重试切下一个 key，
      401/403/404/422 标记 key 失效并 raise。
    - record_result(key_id, status_code, error) 更新运行态。
    """

    def __init__(self, keys: List[ProviderKey], *,
                 rate_limit_per_min: int = RATE_LIMIT_PER_MIN,
                 max_concurrent_per_key: int = DEFAULT_MAX_CONCURRENT_PER_KEY,
                 backoff_sec: float = SERVER_ERROR_BACKOFF_SEC,
                 same_key_retries: int = SERVER_ERROR_SAME_KEY_RETRIES):
        # 按 priority 降序排（priority 大的优先）；同 priority 保持入参顺序
        self._keys: List[ProviderKey] = sorted(
            keys, key=lambda k: -k.priority)
        self._by_id: Dict[str, ProviderKey] = {k.id: k for k in self._keys}
        self._limiter = RateLimiter(
            limit_per_min=rate_limit_per_min,
            max_concurrent_per_key=max_concurrent_per_key,
        )
        self._session = requests.Session()
        self._lock = Lock()  # 保护运行态更新
        # 503/5xx 短退避参数（I5.8-router-1：可由 .env 配置）。
        # 默认与模块级常量一致，保持向后兼容；调用方（batch_tab._build_router）
        # 可从 load_router_config_from_env 读 .env 后显式传入。
        self._backoff_sec = max(0.0, float(backoff_sec))
        self._same_key_retries = max(0, int(same_key_retries))

    # ---- 查询 ----
    def list_keys(self, provider: Optional[str] = None) -> List[ProviderKey]:
        if provider is None:
            return list(self._keys)
        return [k for k in self._keys if k.provider == provider]

    def get_key(self, key_id: str) -> Optional[ProviderKey]:
        return self._by_id.get(key_id)

    # ---- 选择 ----
    def select_key(self, provider: str = "nvidia",
                   exclude: Optional[set] = None) -> Optional[ProviderKey]:
        """选一个可用 key。

        规则：provider 匹配 + isActive + 未到 backoff_until + 限速窗口有额度（非
        阻塞探测，没额度就跳过选下一个）；优先级高的先；同优先级最久未用优先。
        exclude: 调用方本轮要跳过的 key id 集合（避免同一轮重试复用）。
        """
        exclude = exclude or set()
        candidates = [k for k in self._keys
                      if k.provider == provider and k.id not in exclude
                      and k.is_available]
        if not candidates:
            return None
        # 优先级已降序；同优先级下选 last_used 最小的（最久未用）
        candidates.sort(key=lambda k: (-k.priority, k.last_used))
        # 限速窗口：探测是否有额度，无额度的跳过（避免 acquire 阻塞在某 key）
        for k in candidates:
            if self._limiter.available(k.id) > 0:
                return k
        # 所有 key 都打满限速：返回优先级最高那个，acquire 时会阻塞等
        return candidates[0]

    # ---- 记录结果 ----
    def record_result(self, key_id: str, status_code: Optional[int],
                      error: str = "") -> None:
        """更新 key 运行态。调用方在每次请求后调用。"""
        with self._lock:
            k = self._by_id.get(key_id)
            if k is None:
                return
            k.last_used = time.monotonic()
            k.last_error = error
            if status_code in FATAL_STATUS_CODES:
                # 401/403/404/422：标记 key 失效，不再轮换
                k.isActive = False
                logger.error(
                    f"[router] key {k.name} 标记失效：HTTP {status_code} {error}")
            elif status_code == 429:
                k.consecutive_429 += 1
                # 429 给一个短 backoff（避免刚被限又立刻选中），但仍是无限重试
                # 用户要求不退避，这里只做 60s 窗口冷却（下一轮 select 跳过它）
                k.backoff_until = time.monotonic() + 60.0
            elif status_code is not None and status_code >= 500:
                # 5xx：短暂 backoff 1.0s（post_nvidia 同 key 重试时由
                # SERVER_ERROR_BACKOFF_SEC 控制，这里只设 select_key 跳过窗口）
                k.backoff_until = time.monotonic() + 1.0
            elif status_code is not None and 200 <= status_code < 300:
                # 成功：清 429 计数与 backoff
                k.consecutive_429 = 0
                k.backoff_until = 0.0
            # 其它状态码（如 2xx 之外的网络异常 status_code=None）：不动 active，
            # 只记 last_error，由重试循环切下一个 key

    # ---- NVIDIA 原生 POST ----
    def post_nvidia(self, payload: Dict[str, Any],
                    *, timeout: int = 120,
                    max_attempts: Optional[int] = None) -> Dict[str, Any]:
        """用轮换 key 发 NVIDIA chat/completions。

        503/5xx 退避策略（M2 优化，减少 11 key 全撞墙空转）：
          - 不立即切 key：先短退避 1.5s（SERVER_ERROR_BACKOFF_SEC）重试**同 key**
            最多 2 次（SERVER_ERROR_SAME_KEY_RETRIES），给 worker 池留空槽时间。
          - 同 key 重试 2 次仍 5xx：才切下一个 key（无限重试语义保留）。
          - 429 仍 60s 每 key 冷却（保留）；401/403/404/422 标记失效并 raise。
          - 网络异常（ConnectionError/Timeout）：切下一个 key，0.5s 间隔。

        per-key 并发：每次发请求前 acquire_concurrent（≤ max_concurrent_per_key
        个同时在飞），结束 release。避免 11 key 全打满反互挤。

        - max_attempts=None 表示无限重试直到成功或所有 key 失效；显式数字用于
          测试断言上限。
        """
        nvidia_keys = [k for k in self._keys if k.provider == "nvidia"]
        if not nvidia_keys:
            raise RuntimeError("[router] 无 nvidia key 可用")

        attempted: set = set()
        attempts = 0
        last_error: str = ""
        while True:
            attempts += 1
            if max_attempts is not None and attempts > max_attempts:
                raise RuntimeError(
                    f"[router] 达 max_attempts={max_attempts} 仍失败：{last_error}")
            key = self.select_key(provider="nvidia", exclude=attempted)
            if key is None:
                # 所有 key 本轮都试过了：清空 attempted 重新来过（无限重试语义）
                if not attempted:
                    # 一个可用 key 都没有
                    active = [k for k in nvidia_keys if k.isActive]
                    if not active:
                        raise RuntimeError(
                            f"[router] 所有 nvidia key 已失效：{last_error}")
                    # 全部在 backoff：睡 0.5s 等窗口刷新
                    time.sleep(RETRY_INTERVAL)
                    attempted.clear()
                    continue
                attempted.clear()
                continue

            # 限速阻塞等到该 key 有空位（40/min 滑动窗口）
            self._limiter.acquire(key.id)
            # per-key 并发槽位（≤ max_concurrent_per_key 同时在飞）
            self._limiter.acquire_concurrent(key.id)
            url = f"{key.base_url.rstrip('/')}/chat/completions"
            headers = {"Authorization": f"Bearer {key.api_key}",
                       "Content-Type": "application/json",
                       "Accept": "application/json"}

            # 同 key 5xx 短退避重试（不切 key，给 worker 池留空槽）
            same_key_tries = 0
            try:
                while True:
                    try:
                        resp = self._session.post(
                            url, headers=headers, json=payload, timeout=timeout)
                    except (requests.ConnectionError, requests.Timeout) as e:
                        last_error = f"{type(e).__name__}: {e}"
                        logger.warning(
                            f"[router] nvidia key {key.name} 网络错误 "
                            f"{last_error}，切下一个 key")
                        self.record_result(key.id, None, last_error)
                        break  # 跳出同 key 循环，外层切下一个 key

                    code = resp.status_code
                    if 200 <= code < 300:
                        self.record_result(key.id, code, "")
                        try:
                            return resp.json()
                        finally:
                            resp.close()
                    elif code in FATAL_STATUS_CODES:
                        # 401/403/404/422：放弃，标记 key 失效
                        body = self._safe_body(resp)
                        err = f"HTTP {code} {body}"
                        resp.close()
                        self.record_result(key.id, code, err)
                        raise RuntimeError(
                            f"[router] nvidia key {key.name} 放弃"
                            f"（HTTP {code}）：{body}")
                    elif code == 429:
                        body = self._safe_body(resp)
                        resp.close()
                        last_error = f"HTTP {code} {body}"
                        logger.warning(
                            f"[router] nvidia key {key.name} {last_error}，"
                            f"60s 冷却后切下一个 key")
                        self.record_result(key.id, code, last_error)
                        break  # 429 走 60s 冷却，外层切下一个 key
                    elif code in RETRY_STATUS_CODES or code >= 500:
                        # 503/5xx：不立即切 key，先短退避重试同 key 2 次
                        body = self._safe_body(resp)
                        resp.close()
                        last_error = f"HTTP {code} {body}"
                        self.record_result(key.id, code, last_error)
                        if same_key_tries < self._same_key_retries:
                            same_key_tries += 1
                            logger.info(
                                f"[router] nvidia key {key.name} {last_error}，"
                                f"短退避 {self._backoff_sec}s 重试同 key "
                                f"({same_key_tries}/{self._same_key_retries})")
                            time.sleep(self._backoff_sec)
                            continue  # 同 key 重试
                        logger.warning(
                            f"[router] nvidia key {key.name} {last_error}，"
                            f"同 key 重试 {self._same_key_retries} 次仍 5xx，"
                            f"切下一个 key（无限重试）")
                        break  # 同 key 重试用尽，外层切下一个 key
                    else:
                        # 其它 4xx（如 400）：不可重试，raise
                        body = self._safe_body(resp)
                        resp.close()
                        self.record_result(key.id, code, f"HTTP {code} {body}")
                        raise RuntimeError(
                            f"[router] nvidia key {key.name} HTTP {code}：{body}")
                # 同 key 循环正常 break 出来（需切 key）：加 attempted，0.5s 后重选
                attempted.add(key.id)
                time.sleep(RETRY_INTERVAL)
                continue
            finally:
                self._limiter.release_concurrent(key.id)

    @staticmethod
    def _safe_body(resp, limit: int = 500) -> str:
        try:
            t = resp.text
        except Exception:
            return "<unreadable body>"
        return t[:limit]

    # ---- 思考参数拍平（NVIDIA 实测） ----
    @staticmethod
    def build_nvidia_payload(model: str, messages: List[Dict[str, Any]],
                             *, enable_thinking: bool = False,
                             reasoning_budget: int = 4096,
                             max_tokens: int = 1024,
                             temperature: float = 0.2,
                             stream: bool = False,
                             extra: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, Any]:
        """构造 NVIDIA 原生 payload，思考参数拍平到顶层。

        raw REST 不接受 extra_body / thinking_token_budget，必须把
        chat_template_kwargs.enable_thinking 和 reasoning_budget 放顶层。
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if enable_thinking:
            # 拍平到顶层（实测 NVIDIA 端点接受这种写法）
            payload["chat_template_kwargs"] = {"enable_thinking": True}
            payload["reasoning_budget"] = reasoning_budget
        if extra:
            payload.update(extra)
        return payload


# ---- 配置加载 ----
def _split_keys(raw: str) -> List[str]:
    """逗号/空白分隔的多 key 字符串拆分，去空去重保序。"""
    out: List[str] = []
    seen: set = set()
    for part in raw.replace(",", " ").split():
        part = part.strip()
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return out


def load_from_env(env_path: Optional[str] = None) -> List[ProviderKey]:
    """从 .env 读 VAP_NV_API_KEYS / VAP_KILO_API_KEYS（逗号分隔多 key）。

    也兼容单数 VAP_NV_API_KEY / VAP_KILO_API_KEY（旧字段）。
    env_path 不给则只读进程已加载的环境变量。
    给了 env_path：把文件里的变量读进一个本地 dict，进程环境变量优先（覆盖
    文件同名变量），不污染 os.environ。
    """
    local_env: Dict[str, str] = {}
    if env_path:
        from pathlib import Path
        p = Path(env_path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k:
                    local_env[k] = v

    def _get(name: str) -> str:
        # 进程环境变量优先于文件
        return os.environ.get(name, local_env.get(name, ""))

    keys: List[ProviderKey] = []

    def _add(provider: str, raw: str, base_url: str):
        for i, k in enumerate(_split_keys(raw)):
            keys.append(ProviderKey(
                id=f"{provider}-{i+1}",
                name=f"{provider}-{i+1}",
                provider=provider,
                api_key=k,
                base_url=base_url,
                priority=100 - i,  # 先列出的优先级高
                isActive=True,
            ))

    nv_raw = _get("VAP_NV_API_KEYS") or _get("VAP_NV_API_KEY")
    if nv_raw:
        _add("nvidia", nv_raw, NVIDIA_DEFAULT_BASE_URL)

    kilo_raw = _get("VAP_KILO_API_KEYS") or _get("VAP_KILO_API_KEY")
    if kilo_raw:
        _add("kilo", kilo_raw, KILO_DEFAULT_BASE_URL)

    return keys


def load_router_config_from_env(
        env_path: Optional[str] = None) -> Dict[str, Any]:
    """从 .env 读 ProviderRouter 调优参数（I5.8-router-1/2）。

    返回 dict（键固定，调用方按需取用，不传给 ProviderRouter 就用默认值）：
      - backoff_sec: 503/5xx 短退避秒数（默认 1.5 = SERVER_ERROR_BACKOFF_SEC）
      - same_key_retries: 同 key 重试次数（默认 2 = SERVER_ERROR_SAME_KEY_RETRIES）
      - max_concurrent_per_key: 每 key 并发上限（默认 2 =
        DEFAULT_MAX_CONCURRENT_PER_KEY）

    env_path 不给则只读进程已加载的环境变量；给了则文件 + 进程环境变量
    合并（进程环境变量优先），不污染 os.environ。值为空或非法时回退默认。
    本函数不改变 load_from_env 的返回签名（仍返回 List[ProviderKey]），
    仅作为独立入口供调用方（batch_tab._build_router）读取路由器调优参数。
    """
    local_env: Dict[str, str] = {}
    if env_path:
        from pathlib import Path
        p = Path(env_path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k:
                    local_env[k] = v

    def _get(name: str) -> str:
        # 进程环境变量优先于文件
        return os.environ.get(name, local_env.get(name, ""))

    def _as_float(name: str, default: float) -> float:
        raw = _get(name).strip()
        if not raw:
            return default
        try:
            val = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                f"[router] {name}={raw!r} 非法，回退默认 {default}")
            return default
        if val < 0:
            logger.warning(
                f"[router] {name}={val} 为负，回退默认 {default}")
            return default
        return val

    def _as_int(name: str, default: int) -> int:
        raw = _get(name).strip()
        if not raw:
            return default
        try:
            val = int(raw)
        except (TypeError, ValueError):
            logger.warning(
                f"[router] {name}={raw!r} 非法，回退默认 {default}")
            return default
        if val < 0:
            logger.warning(
                f"[router] {name}={val} 为负，回退默认 {default}")
            return default
        return val

    return {
        "backoff_sec": _as_float(
            "VAP_NV_BACKOFF_SEC", SERVER_ERROR_BACKOFF_SEC),
        "same_key_retries": _as_int(
            "VAP_NV_SAME_KEY_RETRIES", SERVER_ERROR_SAME_KEY_RETRIES),
        "max_concurrent_per_key": _as_int(
            "VAP_NV_MAX_CONCURRENT_PER_KEY", DEFAULT_MAX_CONCURRENT_PER_KEY),
    }


def load_from_9router(path: str) -> List[ProviderKey]:
    """从 9router providerConnections json 导入。

    9router 实测结构（list 或 {"providerConnections":[...]}）：
      每条含 provider/apiKey/priority/isActive/testStatus/backoffLevel/
      lastErrorAt/consecutiveUseCount。
    本地 _nv_keys.json 是简化变体（name/key/priority/last_status），
    provider 默认 "nvidia"，base_url 默认 NVIDIA 端点。
    """
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error(f"[router] 9router json 解析失败 {path}: {e}")
        return []

    if isinstance(data, dict):
        conns = data.get("providerConnections", data.get("connections", []))
    elif isinstance(data, list):
        conns = data
    else:
        return []

    keys: List[ProviderKey] = []
    for i, c in enumerate(conns):
        if not isinstance(c, dict):
            continue
        # 9router 标准：apiKey + provider；本地变体：key + name + last_status
        api_key = c.get("apiKey") or c.get("key") or ""
        if not api_key:
            continue
        provider = (c.get("provider") or "nvidia").lower()
        name = c.get("name") or f"{provider}-{i+1}"
        priority = c.get("priority", 0)
        # isActive 优先取配置态，否则看 testStatus
        is_active = c.get("isActive", True)
        test_status = (c.get("testStatus") or c.get("last_status") or "").lower()
        if test_status in ("unavailable", "failed", "error"):
            is_active = False
        base_url = c.get("baseUrl") or c.get("base_url") or ""
        if not base_url:
            base_url = (NVIDIA_DEFAULT_BASE_URL if provider == "nvidia"
                        else KILO_DEFAULT_BASE_URL)
        keys.append(ProviderKey(
            id=c.get("id") or f"{provider}-{i+1}",
            name=name,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            priority=priority,
            isActive=is_active,
        ))
    return keys
