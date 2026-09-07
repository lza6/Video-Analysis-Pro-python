"""工具沙箱：为 Agent 工具（视频处理 / 文件操作）提供进程级资源隔离。

- `Sandbox` Protocol：`enter()` / `exit()` / `run(coro)` 三件套
- `WindowsJobSandbox`：pywin32 Job Object，限内存/CPU + 崩溃自动回收
- `LinuxLandlockSandbox`：stdlib `landlock`（3.13+ Linux 内置）限文件访问
- `NullSandbox`：降级实现，直接 await（macOS / 依赖缺失时）

依赖策略（匹配 `im_gateway/cipher.py` 的 cryptography 处理模式）：
pywin32 / landlock 是**条件依赖**，不写进 requirements.txt，缺失时降级
NullSandbox + 告警。调用方用 `build_sandbox()` 自动选型。

已知坑：WindowsJobSandbox assign 当前进程到 Job 在已运行进程上可能失败
（已属别的 Job / 不支持 nested），失败 raise `SandboxError`，建议改用
ForkBackend；landlock 一旦 apply 无法解除（进程级），exit 为 no-op。
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class SandboxError(RuntimeError):
    """沙箱装配 / 应用失败（如 assign 进程到 Job 被拒）。"""


@runtime_checkable
class Sandbox(Protocol):
    """工具沙箱协议：enter / exit / run 三件套。"""

    async def enter(self) -> None: ...
    async def exit(self) -> None: ...
    async def run(self, coro: Any) -> Any: ...


class NullSandbox:
    """无操作沙箱：直接 await，无 OS 级隔离。macOS / 依赖缺失 / 测试用。"""

    async def enter(self) -> None: return None
    async def exit(self) -> None: return None
    async def run(self, coro: Any) -> Any: return await coro


class WindowsJobSandbox:
    """Windows Job Object 沙箱（pywin32 条件依赖）。

    设 `JOB_OBJECT_LIMIT_PROCESS_MEMORY` + `JOB_OBJECT_LIMIT_CPU_RATE`
    + `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`，assign 当前进程到 Job。
    assign 失败（已属别的 Job / 不支持 nested）raise `SandboxError`，
    建议改用 ForkBackend。pywin32 缺失时构造 raise `ImportError`。
    """

    def __init__(self, *, memory_limit_mb: int = 4096, cpu_rate: float = 0.5) -> None:
        try:
            import win32job, win32api, win32con  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "WindowsJobSandbox 需要 pywin32（win32job），未安装。"
                "请 pip install pywin32 或降级 NullSandbox"
            ) from exc
        if memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb 必须 > 0")
        if not 0.0 < cpu_rate <= 1.0:
            raise ValueError("cpu_rate 必须在 (0, 1] 区间")
        self._win32job, self._win32api, self._win32con = win32job, win32api, win32con
        self._memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self._cpu_rate_permille = int(cpu_rate * 10000)
        self._job_handle: Any = None

    async def enter(self) -> None:
        """创建 Job + 设限制 + assign 当前进程。assign 失败 raise SandboxError。"""
        win32job, win32con = self._win32job, self._win32con
        self._job_handle = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(
            self._job_handle, win32job.JobObjectExtendedLimitInformation
        )
        info["BasicLimitInformation"]["LimitFlags"] = (
            win32con.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | win32con.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info["ProcessMemoryLimit"] = self._memory_limit_bytes
        win32job.SetInformationJobObject(
            self._job_handle, win32job.JobObjectExtendedLimitInformation, info
        )
        # CPU rate 走独立 API（Windows 8+），失败不阻塞沙箱装配
        try:
            win32job.SetInformationJobObject(
                self._job_handle, win32job.JobObjectCpuRateControlInformation,
                {"ControlFlags": (win32con.JOB_OBJECT_CPU_RATE_CONTROL_ENABLE
                                  | win32con.JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP),
                 "CpuRate": self._cpu_rate_permille},
            )
        except Exception:  # noqa: BLE001
            logger.warning("WindowsJobSandbox: 设 CPU rate 失败，仅保留内存+KILL限制")
        # assign 当前进程（可能失败：已属别的 Job / 不支持 nested）
        try:
            self._win32job.AssignProcessToJobObject(
                self._job_handle, self._win32api.GetCurrentProcess()
            )
        except Exception as exc:  # noqa: BLE001 — AssignProcess 错误是已知坑
            try: self._job_handle.Close()
            except Exception:  # noqa: BLE001
                pass
            self._job_handle = None
            raise SandboxError(
                "assign 当前进程到 Job 失败（已属别的 Job / 不支持 nested job）。"
                "建议改用 ForkBackend 让子进程跑在沙箱内"
            ) from exc
        logger.info("WindowsJobSandbox 已生效: memory_limit=%dMB cpu_rate=%.2f",
                    self._memory_limit_bytes // (1024 * 1024),
                    self._cpu_rate_permille / 10000.0)

    async def exit(self) -> None:
        """关闭 Job handle（KILL_ON_JOB_CLOSE 自动回收归属进程）。"""
        if self._job_handle is not None:
            try: self._job_handle.Close()
            except Exception:  # noqa: BLE001
                logger.warning("WindowsJobSandbox: 关闭 Job handle 失败")
            self._job_handle = None

    async def run(self, coro: Any) -> Any: return await coro


class LinuxLandlockSandbox:
    """Linux landlock 沙箱（stdlib `landlock`，3.13+ Linux 编译内置）。

    landlock 进程级，apply 后无法解除（exit no-op）。stdlib 无 landlock
    （Python < 3.13 / 非 Linux）构造 raise `ImportError`，降级 NullSandbox。
    `allowed_paths` 为文件访问白名单（绝对路径），空列表 = 完全禁写。
    """

    def __init__(self, *, allowed_paths: list[str] | None = None) -> None:
        try:
            import landlock  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "LinuxLandlockSandbox 需要 stdlib landlock（Python 3.13+ Linux 内置）。"
                "当前 Python 缺 landlock，请降级 NullSandbox"
            ) from exc
        self._landlock = landlock
        self._allowed_paths = list(allowed_paths) if allowed_paths else []
        self._applied = False

    async def enter(self) -> None:
        """创建 ruleset + 加白名单 + apply。apply 失败（无 CAP）raise SandboxError。"""
        landlock = self._landlock
        ruleset = landlock.ruleset()
        for path in self._allowed_paths:
            try: ruleset.add_rule(path, landlock.AccessFile.read_write)
            except Exception:  # noqa: BLE001
                logger.warning("landlock 添加白名单路径失败: %s", path)
        try:
            ruleset.apply()
            self._applied = True
            logger.info("LinuxLandlockSandbox 已生效: %d 条白名单路径",
                        len(self._allowed_paths))
        except Exception as exc:  # noqa: BLE001
            raise SandboxError(
                "landlock apply 失败（需 CAP_SYS_ADMIN 或 user namespace）"
            ) from exc

    async def exit(self) -> None:
        """no-op：landlock 进程级，无法解除。"""
        return None

    async def run(self, coro: Any) -> Any: return await coro


def build_sandbox(platform: str | None = None) -> Sandbox:
    """按平台自动选沙箱。

    - `win32` → WindowsJobSandbox（pywin32 缺降级 NullSandbox + 告警）
    - `linux` → LinuxLandlockSandbox（stdlib landlock 缺降级 + 告警）
    - 其余（darwin / 未知）→ NullSandbox

    Args:
        platform: 平台字符串，默认读 sys.platform（便于测试注入）。
    """
    plat = platform if platform is not None else sys.platform
    if plat == "win32":
        try:
            return WindowsJobSandbox()
        except ImportError:
            logger.warning("build_sandbox: pywin32 缺失，降级 NullSandbox"
                           "（工具将在主进程直接跑，无资源隔离）")
            return NullSandbox()
    if plat == "linux":
        try:
            return LinuxLandlockSandbox()
        except ImportError:
            logger.warning("build_sandbox: stdlib landlock 缺失"
                           "（Python<3.13 或非 Linux），降级 NullSandbox")
            return NullSandbox()
    logger.info("build_sandbox: 平台 %s 无原生沙箱，用 NullSandbox", plat)
    return NullSandbox()
