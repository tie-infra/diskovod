from __future__ import annotations

from pathlib import Path

import pytest

from diskovod.agent_actions import DeliveryRecord
from diskovod.agent_types import AgentRuntimeContext, CapabilityProfile
from diskovod.durable_actions import DurableActionGateway, SideEffectLedger
from diskovod.events import DiscordEventQueue


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
