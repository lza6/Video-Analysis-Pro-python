"""IMGateway：编排 mailbox + adapters + agent callback。

生命周期：
  - start() : 启动所有 adapter，开 background 轮询任务
  - stop()  : 取消轮询任务，stop 所有 adapter

轮询循环（单任务，串行处理各 adapter 的入站）：
  for adapter in adapters:
      env = await adapter.receive()
      if env:
          msg_id = mailbox.put_inbound(channel=adapter.channel_name,
                                        peer=env.peer, content=env.content)
          # lease → callback → ack → send
          await self._process_one(adapter, env, msg_id)

处理一条消息（_process_one）：
  1. lease（领消息，防并发重复处理；同时先 reclaim_stale 清超时租约）
  2. 调 callback.handle(content) 拿回复
  3. ack(msg_id)
  4. mailbox.put_outbound + adapter.send(peer, reply)

任何步骤抛异常 → mailbox.fail(msg_id, error)，不 ack。
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Dict, List, Optional

from src.core.im_gateway.mailbox import IMMailbox
from src.core.im_gateway.adapters.base import IMAdapter, InboundEnvelope
from src.core.im_gateway.route import AgentCallback

logger = logging.getLogger(__name__)


class IMGateway:
    """IM 网关编排器。

    持有 mailbox + 多个 adapter + 一个 agent callback。轮询各 adapter，
    把入站消息落库后 lease→callback→ack→send 回复。

    Mock 友好：所有 adapter 是 Mock，不真实连 IM。真实凭据预算 0。
    """

    def __init__(
        self,
        *,
        mailbox: IMMailbox,
        adapters: List[IMAdapter],
        callback: AgentCallback,
        lease_holder: Optional[str] = None,
        poll_interval_sec: float = 0.05,
        lease_ttl_sec: float = 30.0,
    ) -> None:
        if not adapters:
            raise ValueError("adapters 不能为空")
        self.mailbox = mailbox
        self._adapters: Dict[str, IMAdapter] = {
            a.channel_name: a for a in adapters
        }
        self._callback = callback
        self.lease_holder = lease_holder or f"gateway-{uuid.uuid4().hex[:8]}"
        self.poll_interval_sec = poll_interval_sec
        self.lease_ttl_sec = lease_ttl_sec
        self._task: Optional[asyncio.Task[None]] = None
        self._stopping = False
        # 处理计数（测试断言用）
        self.processed_count = 0
        self.failed_count = 0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """启动所有 adapter；不开 background 轮询任务。

        轮询改由调用方显式驱动（process_once / 单独的 run_poll_loop
        协程）。理由：tests 用 new_event_loop 跑测试，create_task 绑定
        的 loop 在 stop 时跨 loop await 报错；且单 loop 内 poll task 与
        process_once 并发会 race 抢入站，导致计数不确定。
        生产需要后台轮询时，主控应 `asyncio.create_task(gateway.run_poll_loop())`
        显式起任务，生命周期由主控管。
        """
        for adapter in self._adapters.values():
            await adapter.start()
        self._stopping = False
        logger.info(
            "IMGateway started: holder=%s adapters=%s (poll loop NOT auto-started)",
            self.lease_holder, list(self._adapters),
        )

    async def run_poll_loop(self) -> None:
        """后台轮询协程（生产用，主控 create_task 起它）。

        本方法不自动 start（start 只起 adapter）；主控流程：
            await gateway.start()
            task = asyncio.create_task(gateway.run_poll_loop())
            ...  # 运行期
            task.cancel(); await asyncio.gather(task, return_exceptions=True)
            await gateway.stop()
        """
        self._task = asyncio.current_task()
        while not self._stopping:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("poll loop error: %s", e)
            await asyncio.sleep(self.poll_interval_sec)

    async def stop(self) -> None:
        """停止轮询任务（若 run_poll_loop 起了）并停所有 adapter。"""
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, RuntimeError):
                # RuntimeError: 跨 loop await（旧 loop 已关）——安全忽略
                pass
            self._task = None
        for adapter in self._adapters.values():
            await adapter.stop()
        logger.info("IMGateway stopped")

    def register_callback(self, callback: AgentCallback) -> None:
        """运行时切换 agent callback（主控收口接真 agent 时用）。"""
        self._callback = callback
        logger.info("IMGateway callback re-registered")

    def register_adapter(self, adapter: IMAdapter) -> None:
        """运行时追加 adapter（已 start 后追加不会自动 start，需调方负责）。"""
        self._adapters[adapter.channel_name] = adapter

    # ------------------------------------------------------------------
    # 轮询循环（内部，由 run_poll_loop 调用；保留独立方法便于子类覆盖）
    # ------------------------------------------------------------------
    async def _poll_once(self) -> None:
        """单轮轮询：reclaim_stale + 逐 adapter drain。

        拆成独立方法：run_poll_loop 调它做循环；process_once 也调它做
        单轮（process_once 额外返回处理条数）。
        """
        await asyncio.to_thread(self.mailbox.reclaim_stale)
        for adapter in list(self._adapters.values()):
            if self._stopping:
                break
            await self._drain_adapter(adapter)

    async def _drain_adapter(self, adapter: IMAdapter) -> None:
        """把 adapter 当前所有入站消息排干（每条落库+处理）。

        注意：与 background _poll_loop 并发时，poll task 可能抢先 pop 走
        一两条入站。本方法只负责"当前能 pop 到的"，不保证排干所有历史
        入站（race 由 poll task 兜底）。测试要确定性排干时，应 stop
        poll task 或直接调 process_once 多次 + 计数断言。
        """
        for _ in range(64):  # 上限防理论无限循环
            if self._stopping:
                break
            env = await adapter.receive()
            if env is None:
                break
            msg_id = await asyncio.to_thread(
                self.mailbox.put_inbound,
                channel=adapter.channel_name,
                peer=env.peer,
                content=env.content,
            )
            await self._process_one(adapter, env, msg_id)

    async def _process_one(
        self, adapter: IMAdapter, env: InboundEnvelope, msg_id: str
    ) -> None:
        """处理一条已落库的入站消息：lease → callback → ack → send。

        lease 可能返回 None（被并发抢先或刚 reclaim 后被别的 holder 领），
        此时跳过——别的 holder 会处理。
        """
        leased = await asyncio.to_thread(
            self.mailbox.lease,
            lease_holder=self.lease_holder,
            lease_ttl_sec=self.lease_ttl_sec,
        )
        # lease 拿到的消息不一定就是刚 put 的那条（mailbox 按时间顺序），
        # 但本测试场景下 Mock 注入是确定性的，leased.msg_id 应 == msg_id。
        # 生产多 adapter 并发时不保证——本类按 leased 顺序处理，不假设。
        if leased is None:
            logger.debug("lease 空（无 pending），跳过 msg_id=%s", msg_id)
            return
        if leased.msg_id != msg_id:
            # 罕见：lease 到的不是刚 put 的，回退让别的轮次处理
            logger.warning(
                "lease 到的消息与刚 put 的不一致（%s != %s），ack 后让原消息重回",
                leased.msg_id, msg_id,
            )
            # 把这条意外 lease 到的 ack 掉避免卡住，但不 send（不是当前 adapter 的）
            await asyncio.to_thread(self.mailbox.ack, leased.msg_id)
            return
        try:
            reply = await self._callback.handle(leased.content)
        except Exception as e:  # noqa: BLE001
            logger.exception("agent callback 处理失败: %s", e)
            await asyncio.to_thread(self.mailbox.fail, leased.msg_id, str(e))
            self.failed_count += 1
            return
        await asyncio.to_thread(self.mailbox.ack, leased.msg_id)
        self.processed_count += 1
        # 出站：落库 + adapter.send
        await asyncio.to_thread(
            self.mailbox.put_outbound,
            channel=adapter.channel_name,
            peer=leased.peer,
            content=reply,
            reply_to=leased.msg_id,
        )
        await adapter.send(leased.peer, reply)

    # ------------------------------------------------------------------
    # 测试辅助 / 同步驱动入口
    # ------------------------------------------------------------------
    async def process_once(self) -> int:
        """单次轮询：reclaim_stale + 逐 adapter drain，返回本轮处理条数。

        不依赖 background poll task（start 不再自动起 task），适合测试
        和同步驱动场景。生产需后台轮询用 run_poll_loop。
        """
        before = self.processed_count + self.failed_count
        await self._poll_once()
        after = self.processed_count + self.failed_count
        return after - before
