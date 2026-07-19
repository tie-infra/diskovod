from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from diskovod.agent_actions import DeliveryRecord
from diskovod.agent_types import AgentRuntimeContext, CapabilityProfile
from diskovod.durable_actions import DurableActionGateway, SideEffectLedger
from diskovod.events import DiscordEventQueue
from diskovod.persistence import open_checkpointer


class Transport:
    def __init__(self, *, raises: bool = False):
        self.raises = raises
        self.calls = 0

    async def send_messages(self, context, messages):
        self.calls += 1
        if self.raises:
            raise TimeoutError("unknown delivery state")
        return [
            DeliveryRecord("accepted", index, discord_message_id=f"message-{index}")
            for index, _ in enumerate(messages)
        ]


class ClaimSignalingLedger(SideEffectLedger):
    def __init__(self, path: Path, loop: asyncio.AbstractEventLoop):
        super().__init__(path)
        self.loop = loop
        self.claim_started = asyncio.Event()

    def claim(self, *args, **kwargs):
        self.loop.call_soon_threadsafe(self.claim_started.set)
        return super().claim(*args, **kwargs)


def context(trace_id: str = "run-1") -> AgentRuntimeContext:
    return AgentRuntimeContext(
        account_id="account",
        channel_id="chat",
        participant_ids=("peer",),
        owner_id="owner",
        ui_locale="en",
        prompt_locale="en",
        assistant_name="Diskovod",
        automation_mode="inline",
        force_reply=False,
        provider_id="test",
        model_id="test",
        transport_profile="test",
        capabilities=CapabilityProfile(),
        trace_id=trace_id,
    )


@pytest.mark.asyncio
async def test_completed_side_effect_is_replayed_without_duplicate_delivery(tmp_path: Path):
    ledger = SideEffectLedger(tmp_path / "diskovod.sqlite3")
    transport = Transport()
    gateway = DurableActionGateway(ledger, transport)

    first = await gateway.send_messages(context(), ("hello",), tool_call_id="call-1")
    second = await gateway.send_messages(context(), ("hello",), tool_call_id="call-1")

    assert first == second
    assert transport.calls == 1
    ledger.close()


@pytest.mark.asyncio
async def test_ambiguous_transport_failure_is_recorded_and_never_retried(tmp_path: Path):
    ledger = SideEffectLedger(tmp_path / "diskovod.sqlite3")
    transport = Transport(raises=True)
    gateway = DurableActionGateway(ledger, transport)

    first = await gateway.send_messages(context(), ("hello",), tool_call_id="call-1")
    second = await gateway.send_messages(context(), ("hello",), tool_call_id="call-1")

    assert first[0].status == "ambiguous"
    assert second == first
    assert transport.calls == 1
    ledger.close()


@pytest.mark.asyncio
async def test_tool_sends_do_not_block_checkpoint_commits(tmp_path: Path):
    path = tmp_path / "diskovod.sqlite3"
    ledger = ClaimSignalingLedger(path, asyncio.get_running_loop())
    transport = Transport()
    gateway = DurableActionGateway(ledger, transport)

    try:
        async with open_checkpointer(path, "x" * 32) as checkpointer:
            original_commit = checkpointer.conn.commit
            for index in range(12):
                ledger.claim_started.clear()
                commit_started = asyncio.Event()

                async def commit_after_ledger_yields() -> None:
                    commit_started.set()
                    await ledger.claim_started.wait()
                    await original_commit()

                checkpointer.conn.commit = commit_after_ledger_yields
                checkpoint_task = asyncio.create_task(
                    checkpointer.aput(
                        {"configurable": {"thread_id": "thread", "checkpoint_ns": ""}},
                        empty_checkpoint(),
                        {"source": "loop", "step": index, "parents": {}},
                        {},
                    )
                )
                await commit_started.wait()
                try:
                    records = await gateway.send_messages(
                        context(f"run-{index}"),
                        (f"message {index}",),
                        tool_call_id=f"call-{index}",
                    )
                    assert records[0].accepted
                finally:
                    await checkpoint_task
                    checkpointer.conn.commit = original_commit
    finally:
        ledger.close()

    assert transport.calls == 12


def test_event_queue_is_ordered_deduplicated_and_isolated_by_chat(tmp_path: Path):
    queue = DiscordEventQueue(tmp_path / "diskovod.sqlite3")
    assert queue.thread_id("account", "chat-a") == "discord:account:chat-a:g1"
    assert queue.thread_id("account", "chat-b") == "discord:account:chat-b:g1"
    assert queue.ingest("event-2", "chat-a", "message", {"content": "second"}) is True
    assert queue.ingest("event-1", "chat-a", "message", {"content": "first observed later"}) is True
    assert queue.ingest("event-2", "chat-a", "message", {"content": "duplicate"}) is False
    queue.ingest("other", "chat-b", "message", {"content": "other chat"})

    claimed = queue.claim_ready("chat-a", "request-1", injection_batch=1)

    assert [event.id for event in claimed] == ["event-2", "event-1"]
    assert queue.claim_ready("chat-a", "request-1") == []
    assert queue.claimed("chat-a", "request-1") == claimed
    assert queue.complete("chat-a", "request-1") == 2
    assert [event.id for event in queue.claim_ready("chat-b", "request-2")] == ["other"]
    queue.close()


def test_event_queue_persists_live_steering_and_generation_rollover(tmp_path: Path):
    queue = DiscordEventQueue(tmp_path / "diskovod.sqlite3")
    queue.set_live_steering("account", "chat", False)
    assert queue.live_steering("chat") is False
    assert queue.roll_generation("account", "chat") == "discord:account:chat:g2"
    assert queue.thread_id("account", "chat") == "discord:account:chat:g2"
    queue.close()
