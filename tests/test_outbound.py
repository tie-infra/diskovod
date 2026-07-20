from pathlib import Path

import pytest

from diskovod.agent_types import AgentRuntimeContext, CapabilityProfile
from diskovod.outbound import DeliveryRecord, OutboundPublisher
from diskovod.store import Store


class Transport:
    def __init__(self, *, raises: bool = False):
        self.raises = raises
        self.messages: list[str] = []
        self.reactions: list[tuple[str, str]] = []

    async def send_messages(self, context, messages):
        del context
        if self.raises:
            raise TimeoutError("unknown delivery state")
        self.messages.extend(messages)
        return [DeliveryRecord("accepted", 0, f"discord:{len(self.messages)}")]

    async def react_to_message(self, context, message_id, emoji):
        del context
        self.reactions.append((message_id, emoji))
        return DeliveryRecord("accepted", 0, f"reaction:{message_id}:{emoji}")


def context() -> AgentRuntimeContext:
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
        trace_id="run",
        thread_id="thread",
    )


@pytest.mark.asyncio
async def test_outbound_message_replay_is_idempotent(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport()
    publisher = OutboundPublisher(store.database, transport)

    first = await publisher.publish_messages(
        context(), ("hello",), source_kind="assistant_text", source_id="ai-1"
    )
    second = await publisher.publish_messages(
        context(), ("hello",), source_kind="assistant_text", source_id="ai-1"
    )

    assert first == second
    assert transport.messages == ["hello"]
    await store.aclose()


@pytest.mark.asyncio
async def test_outbound_ambiguous_result_is_not_retried(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport(raises=True)
    publisher = OutboundPublisher(store.database, transport)

    first = await publisher.publish_messages(
        context(), ("hello",), source_kind="assistant_text", source_id="ai-1"
    )
    transport.raises = False
    second = await publisher.publish_messages(
        context(), ("hello",), source_kind="assistant_text", source_id="ai-1"
    )

    assert first[0].status == "ambiguous"
    assert second == first
    assert transport.messages == []
    await store.aclose()


@pytest.mark.asyncio
async def test_reaction_target_is_part_of_idempotent_action(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport()
    publisher = OutboundPublisher(store.database, transport)

    first = await publisher.react(context(), "🎉", "message-1", source_id="tool-1")
    second = await publisher.react(context(), "🎉", "message-1", source_id="tool-1")

    assert first == second
    assert transport.reactions == [("message-1", "🎉")]
    await store.aclose()
