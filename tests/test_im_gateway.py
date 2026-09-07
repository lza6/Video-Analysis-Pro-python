"""IM 网关（Builder-C）测试。

覆盖：
  - mailbox: put_inbound / lease / ack / fail / reclaim_stale / 超时重 lease
  - cipher : 加解密往返 + HMAC 篡改检测 + 不同 master_key 不可解
  - Mock adapter : send/receive 闭环 + 多 adapter 隔离
  - gateway : 完整流程 receive → lease → callback(Mock) → ack → send 回复

全部 Mock，不真实连 IM，不真实付费。
不依赖 pytest-asyncio：用 asyncio.run() 在同步测试中驱动 async 代码
（与 tests/test_remote_tunnel.py 同一模式）。

注意 gateway 测试：start/process_once/stop 必须在**同一个** event loop
里跑（gateway.start 把 background poll task 绑在当前 loop 上，跨 loop
await 会抛 "attached to a different loop"）。本模块提供 gateway_test()，
在一个新 loop 里跑完整个 async 闭包再关闭。
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable, TypeVar

import pytest

# 让 tests 能 import src 包（conftest 已插根路径，保险再插一次）
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.im_gateway.mailbox import IMMailbox  # noqa: E402
from src.core.im_gateway.cipher import (  # noqa: E402
    GatewayCipher,
    generate_master_key,
    is_cryptography_available,
)
from src.core.im_gateway.adapters.wechat import WechatAdapter  # noqa: E402
from src.core.im_gateway.adapters.telegram import TelegramAdapter  # noqa: E402
from src.core.im_gateway.adapters.discord import DiscordAdapter  # noqa: E402
from src.core.im_gateway.route import (  # noqa: E402
    AgentCallback,
    EchoAgentCallback,
    route_message,
)
from src.core.im_gateway.gateway import IMGateway  # noqa: E402

T = TypeVar("T")


def run(coro: Awaitable[T]) -> T:
    """同步驱动单次 async 调用（不依赖 pytest-asyncio）。

    用于无状态 adapter 测试（每次 run 一个新 loop）。gateway 测试因为
    background task 绑定 loop，不能用本函数，改用 run_gw。
    """
    return asyncio.new_event_loop().run_until_complete(coro)  # type: ignore[arg-type]


def run_gw(fn: Callable[[], Awaitable[T]]) -> T:
    """同步驱动一个 gateway 测试闭包：单个 loop 跑完 start→process→stop。

    fn 返回一个 async 闭包，内部所有 await 都共享本函数创建的同一个 loop
    （gateway.start 的 create_task 绑在此 loop 上，stop 时 await task 才不
    跨 loop）。与 test_remote_tunnel.py 的 run() 区别：run() 是每调一次
    新 loop，run_gw 是整个闭包共用一个 loop。
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(fn())
    finally:
        # 关 loop 前确保无残留 task（gateway.stop 应已 cancel poll task）
        loop.close()


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def tmp_mailbox(tmp_path: Path) -> IMMailbox:
    """每个测试用独立 sqlite 临时文件，避免并发污染。"""
    return IMMailbox(config_dir=str(tmp_path), db_filename="test_im.db")


# ----------------------------------------------------------------------
# mailbox: 基本读写
# ----------------------------------------------------------------------
def test_put_inbound_initial_status_pending(tmp_mailbox: IMMailbox) -> None:
    msg_id = tmp_mailbox.put_inbound(
        channel="wechat", peer="wxid_a", content="hello"
    )
    msg = tmp_mailbox.get(msg_id)
    assert msg is not None
    assert msg.direction == "inbound"
    assert msg.channel == "wechat"
    assert msg.peer == "wxid_a"
    assert msg.content == "hello"
    assert msg.status == "pending"
    assert msg.leased_by is None
    assert msg.lease_expires_at is None


def test_put_outbound_is_done_terminal(tmp_mailbox: IMMailbox) -> None:
    msg_id = tmp_mailbox.put_outbound(
        channel="telegram", peer="chat_1", content="reply", reply_to="abc"
    )
    msg = tmp_mailbox.get(msg_id)
    assert msg is not None
    assert msg.direction == "outbound"
    assert msg.status == "done"
    assert msg.processed_at is not None


def test_list_pending_orders_by_created_at(tmp_mailbox: IMMailbox) -> None:
    a = tmp_mailbox.put_inbound(channel="wechat", peer="p1", content="1")
    b = tmp_mailbox.put_inbound(channel="wechat", peer="p2", content="2")
    c = tmp_mailbox.put_inbound(channel="wechat", peer="p3", content="3")
    pending = tmp_mailbox.list_pending()
    assert [m.msg_id for m in pending] == [a, b, c]


# ----------------------------------------------------------------------
# mailbox: lease / ack / fail
# ----------------------------------------------------------------------
def test_lease_returns_none_when_empty(tmp_mailbox: IMMailbox) -> None:
    assert tmp_mailbox.lease(lease_holder="agent-1") is None


def test_lease_marks_processing_and_sets_lease_holder(tmp_mailbox: IMMailbox) -> None:
    msg_id = tmp_mailbox.put_inbound(channel="wechat", peer="p1", content="hi")
    leased = tmp_mailbox.lease(lease_holder="agent-1", lease_ttl_sec=10)
    assert leased is not None
    assert leased.msg_id == msg_id
    assert leased.status == "processing"
    assert leased.leased_by == "agent-1"
    assert leased.lease_expires_at is not None


def test_lease_skips_already_leased_pending_empty(tmp_mailbox: IMMailbox) -> None:
    """一条消息被 lease 后，再 lease 应返回 None（pending 队列空）。"""
    tmp_mailbox.put_inbound(channel="wechat", peer="p1", content="hi")
    first = tmp_mailbox.lease(lease_holder="agent-1")
    assert first is not None
    second = tmp_mailbox.lease(lease_holder="agent-2")
    assert second is None


def test_ack_marks_done_and_clears_lease(tmp_mailbox: IMMailbox) -> None:
    tmp_mailbox.put_inbound(channel="wechat", peer="p1", content="hi")
    leased = tmp_mailbox.lease(lease_holder="agent-1")
    assert leased is not None
    ok = tmp_mailbox.ack(leased.msg_id)
    assert ok is True
    msg = tmp_mailbox.get(leased.msg_id)
    assert msg is not None
    assert msg.status == "done"
    assert msg.leased_by is None
    assert msg.lease_expires_at is None
    assert msg.processed_at is not None


def test_ack_returns_false_for_unknown_or_non_processing(
    tmp_mailbox: IMMailbox,
) -> None:
    assert tmp_mailbox.ack("nonexistent") is False
    msg_id = tmp_mailbox.put_inbound(channel="wechat", peer="p1", content="hi")
    assert tmp_mailbox.ack(msg_id) is False


def test_fail_marks_failed_with_error(tmp_mailbox: IMMailbox) -> None:
    tmp_mailbox.put_inbound(channel="wechat", peer="p1", content="hi")
    leased = tmp_mailbox.lease(lease_holder="agent-1")
    assert leased is not None
    ok = tmp_mailbox.fail(leased.msg_id, "agent crashed")
    assert ok is True
    msg = tmp_mailbox.get(leased.msg_id)
    assert msg is not None
    assert msg.status == "failed"
    assert msg.error == "agent crashed"


# ----------------------------------------------------------------------
# mailbox: 超时重 lease（投递语义核心）
# ----------------------------------------------------------------------
def test_reclaim_stale_returns_processing_to_pending(
    tmp_mailbox: IMMailbox,
) -> None:
    """processing 且超时的消息应被 reclaim 回 pending，可重新 lease。"""
    tmp_mailbox.put_inbound(channel="wechat", peer="p1", content="hi")
    leased = tmp_mailbox.lease(lease_holder="agent-1", lease_ttl_sec=0.001)
    assert leased is not None
    time.sleep(0.05)
    n = tmp_mailbox.reclaim_stale()
    assert n == 1
    re = tmp_mailbox.lease(lease_holder="agent-2")
    assert re is not None
    assert re.msg_id == leased.msg_id
    assert re.leased_by == "agent-2"


def test_reclaim_stale_skips_non_expired(tmp_mailbox: IMMailbox) -> None:
    """未超时的 processing 不应被 reclaim。"""
    tmp_mailbox.put_inbound(channel="wechat", peer="p1", content="hi")
    tmp_mailbox.lease(lease_holder="agent-1", lease_ttl_sec=60.0)
    n = tmp_mailbox.reclaim_stale()
    assert n == 0


def test_requeue_failed_returns_to_pending(tmp_mailbox: IMMailbox) -> None:
    tmp_mailbox.put_inbound(channel="wechat", peer="p1", content="hi")
    leased = tmp_mailbox.lease(lease_holder="agent-1")
    assert leased is not None
    tmp_mailbox.fail(leased.msg_id, "boom")
    ok = tmp_mailbox.requeue_failed(leased.msg_id)
    assert ok is True
    msg = tmp_mailbox.get(leased.msg_id)
    assert msg is not None
    assert msg.status == "pending"


# ----------------------------------------------------------------------
# cipher: 加解密往返
# ----------------------------------------------------------------------
def test_cipher_roundtrip() -> None:
    cipher = GatewayCipher(master_key="test-master-key-xyz")
    for plaintext in ["", "a", "wx_token_abc123", "discord_token_x" * 100]:
        ct = cipher.encrypt(plaintext)
        assert ct != plaintext or plaintext == ""
        pt = cipher.decrypt(ct)
        assert pt == plaintext


def test_cipher_ciphertext_is_not_plaintext() -> None:
    cipher = GatewayCipher(master_key="k1")
    pt = "super_secret_token_12345"
    ct = cipher.encrypt(pt)
    assert pt not in ct
    assert ct.startswith(("v1:", "v0:"))


def test_cipher_tamper_detection() -> None:
    """降级模式下 HMAC 校验应抓住篡改（cryptography 模式 Fernet 自带校验）。"""
    cipher = GatewayCipher(master_key="k1")
    ct = cipher.encrypt("token")
    tampered = ct[:-2] + ("AA" if ct[-2:] != "AA" else "BB")
    with pytest.raises(Exception):
        cipher.decrypt(tampered)


def test_cipher_different_master_key_cannot_decrypt() -> None:
    """不同 master_key 解不开彼此的密文。"""
    c1 = GatewayCipher(master_key="key-A")
    c2 = GatewayCipher(master_key="key-B")
    ct = c1.encrypt("token")
    with pytest.raises(Exception):
        c2.decrypt(ct)


def test_cipher_empty_master_key_raises() -> None:
    with pytest.raises(ValueError):
        GatewayCipher(master_key="")


def test_generate_master_key_is_urlsafe_b64() -> None:
    k = generate_master_key()
    assert isinstance(k, str) and len(k) > 0
    GatewayCipher(master_key=k)


def test_is_cryptography_available_returns_bool() -> None:
    assert isinstance(is_cryptography_available(), bool)


# ----------------------------------------------------------------------
# Mock adapters: send/receive 闭环
# ----------------------------------------------------------------------
def test_wechat_adapter_send_receive_roundtrip() -> None:
    adapter = WechatAdapter()
    run(adapter.start())
    run(adapter.inject_inbound("wxid_1", "分析视频"))
    env = run(adapter.receive())
    assert env is not None
    assert env.peer == "wxid_1"
    assert env.content == "分析视频"
    assert run(adapter.receive()) is None
    ok = run(adapter.send("wxid_1", "收到"))
    assert ok is True
    assert len(adapter.sent_log) == 1
    assert adapter.sent_log[0].peer == "wxid_1"
    assert adapter.sent_log[0].content == "收到"
    run(adapter.stop())


def test_telegram_adapter_send_receive_roundtrip() -> None:
    adapter = TelegramAdapter()
    run(adapter.start())
    run(adapter.inject_inbound("chat_99", "/analyze"))
    env = run(adapter.receive())
    assert env is not None and env.peer == "chat_99"
    run(adapter.send("chat_99", "done"))
    assert len(adapter.sent_log) == 1
    run(adapter.stop())


def test_discord_adapter_send_receive_roundtrip() -> None:
    adapter = DiscordAdapter()
    run(adapter.start())
    run(adapter.inject_inbound("user_42", "hello"))
    env = run(adapter.receive())
    assert env is not None and env.peer == "user_42"
    run(adapter.send("user_42", "hi back"))
    assert len(adapter.sent_log) == 1
    run(adapter.stop())


def test_adapters_are_isolated() -> None:
    """三个 adapter 各自独立，channel_name 不串。"""
    a, b, c = WechatAdapter(), TelegramAdapter(), DiscordAdapter()
    assert a.channel_name == "wechat"
    assert b.channel_name == "telegram"
    assert c.channel_name == "discord"
    run(a.inject_inbound("p", "msg"))
    assert run(b.receive()) is None
    assert run(c.receive()) is None
    env = run(a.receive())
    assert env is not None


# ----------------------------------------------------------------------
# route: EchoAgentCallback
# ----------------------------------------------------------------------
def test_echo_callback() -> None:
    cb = EchoAgentCallback()
    assert isinstance(cb, AgentCallback)  # runtime_checkable
    assert run(route_message(cb, "ping")) == "echo: ping"


# ----------------------------------------------------------------------
# gateway: 完整流程（run_gw 单 loop 跑完 start→process→stop）
# ----------------------------------------------------------------------
def test_gateway_full_pipeline_wechat(tmp_mailbox: IMMailbox) -> None:
    """receive → lease → callback → ack → send 完整闭环（wechat）。"""
    adapter = WechatAdapter()

    async def scenario() -> None:
        await adapter.start()
        gateway = IMGateway(
            mailbox=tmp_mailbox,
            adapters=[adapter],
            callback=EchoAgentCallback(),
            lease_holder="gw-test",
        )
        await gateway.start()
        try:
            await adapter.inject_inbound("wxid_x", "分析这段视频")
            await gateway.process_once()
            assert gateway.processed_count == 1
            assert gateway.failed_count == 0
            assert len(adapter.sent_log) == 1
            assert adapter.sent_log[0].peer == "wxid_x"
            assert adapter.sent_log[0].content == "echo: 分析这段视频"
            outbound = tmp_mailbox.list_outbound_for_peer("wechat", "wxid_x")
            assert len(outbound) == 1
            assert outbound[0].content == "echo: 分析这段视频"
            assert tmp_mailbox.list_pending() == []
        finally:
            await gateway.stop()
            await adapter.stop()

    run_gw(scenario)


def test_gateway_multiple_adapters_parallel_drain(
    tmp_mailbox: IMMailbox,
) -> None:
    """多 adapter 同时有入站时，gateway 应分别处理。"""
    wechat = WechatAdapter()
    tg = TelegramAdapter()
    dc = DiscordAdapter()

    async def scenario() -> None:
        await asyncio.gather(wechat.start(), tg.start(), dc.start())
        gateway = IMGateway(
            mailbox=tmp_mailbox,
            adapters=[wechat, tg, dc],
            callback=EchoAgentCallback(),
        )
        await gateway.start()
        try:
            await wechat.inject_inbound("wxid_1", "w-msg")
            await tg.inject_inbound("chat_1", "t-msg")
            await dc.inject_inbound("u_1", "d-msg")
            # process_once 每轮每 adapter 只 pop 一条；3 个 adapter 各 1 条
            # 但 _drain_adapter 内部 while 会排干当前队列，理论上一次够。
            # 实际若 race（background poll task 也在 pop），补一次确保排干。
            await gateway.process_once()
            if gateway.processed_count < 3:
                await gateway.process_once()
            assert gateway.processed_count == 3
            assert len(wechat.sent_log) == 1
            assert len(tg.sent_log) == 1
            assert len(dc.sent_log) == 1
            assert wechat.sent_log[0].content == "echo: w-msg"
            assert tg.sent_log[0].content == "echo: t-msg"
            assert dc.sent_log[0].content == "echo: d-msg"
        finally:
            await gateway.stop()
            await asyncio.gather(wechat.stop(), tg.stop(), dc.stop())

    run_gw(scenario)


def test_gateway_callback_exception_marks_failed(
    tmp_mailbox: IMMailbox,
) -> None:
    """agent callback 抛异常时，消息应标 failed，不 send。"""

    class BoomCallback:
        async def handle(self, text: str) -> str:
            raise RuntimeError("agent down")

    adapter = WechatAdapter()

    async def scenario() -> None:
        await adapter.start()
        gateway = IMGateway(
            mailbox=tmp_mailbox,
            adapters=[adapter],
            callback=BoomCallback(),
        )
        await gateway.start()
        try:
            await adapter.inject_inbound("wxid_x", "hello")
            await gateway.process_once()
            assert gateway.processed_count == 0
            assert gateway.failed_count == 1
            assert len(adapter.sent_log) == 0
            assert tmp_mailbox.list_pending() == []
            with sqlite3.connect(tmp_mailbox.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM im_messages WHERE direction='inbound'"
                ).fetchone()
                assert row["status"] == "failed"
                assert "agent down" in (row["error"] or "")
        finally:
            await gateway.stop()
            await adapter.stop()

    run_gw(scenario)


def test_gateway_no_inbound_returns_zero(tmp_mailbox: IMMailbox) -> None:
    """无入站消息时 process_once 返回 0。"""
    adapter = WechatAdapter()

    async def scenario() -> None:
        await adapter.start()
        gateway = IMGateway(
            mailbox=tmp_mailbox,
            adapters=[adapter],
            callback=EchoAgentCallback(),
        )
        await gateway.start()
        try:
            n = await gateway.process_once()
            assert n == 0
            assert gateway.processed_count == 0
        finally:
            await gateway.stop()
            await adapter.stop()

    run_gw(scenario)


def test_gateway_register_callback_swaps_agent(
    tmp_mailbox: IMMailbox,
) -> None:
    """运行时切换 callback（模拟主控收口接真 agent）。"""

    class UpperCallback:
        async def handle(self, text: str) -> str:
            return text.upper()

    adapter = WechatAdapter()

    async def scenario() -> None:
        await adapter.start()
        gateway = IMGateway(
            mailbox=tmp_mailbox,
            adapters=[adapter],
            callback=EchoAgentCallback(),
        )
        await gateway.start()
        try:
            await adapter.inject_inbound("p", "hi")
            await gateway.process_once()
            assert adapter.sent_log[0].content == "echo: hi"
            gateway.register_callback(UpperCallback())
            await adapter.inject_inbound("p", "hi")
            await gateway.process_once()
            assert adapter.sent_log[1].content == "HI"
        finally:
            await gateway.stop()
            await adapter.stop()

    run_gw(scenario)


def test_gateway_rejects_empty_adapters(tmp_mailbox: IMMailbox) -> None:
    with pytest.raises(ValueError):
        IMGateway(
            mailbox=tmp_mailbox,
            adapters=[],
            callback=EchoAgentCallback(),
        )
