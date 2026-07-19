from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage

from diskovod.agent_actions import DeliveryRecord
from diskovod.providers import ModelConfiguration, ProviderCapabilities
from diskovod.runtime import AgentService
from diskovod.store import Store

from test_agent import ScriptedChatModel, UnusedPublicHTTP, tool_call


class FakeModels:
    def __init__(self, model):
        self.model = model
        self.configuration = ModelConfiguration(
            provider_id="test",
            model_id="test-model",
            transport_profile="test",
            credential_profile="test",
            capabilities=ProviderCapabilities(),
        )
        self.ready = True

    def build_model(self):
        return self.model


class RecordingTransport:
    def __init__(self):
        self.messages: list[tuple[str, tuple[str, ...]]] = []

    async def send_messages(self, context, messages):
        self.messages.append((context.channel_id, messages))
        return [
            DeliveryRecord("accepted", index, discord_message_id=f"message-{index}")
            for index, _ in enumerate(messages)
        ]

    async def react_to_message(self, context, message_id, emoji):
        return DeliveryRecord("accepted", 0, discord_message_id=f"reaction:{message_id}:{emoji}")


async def wait_for_idle(service: AgentService) -> None:
    while tasks := tuple(service.tasks.values()):
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_agent_service_persists_a_chat_thread_and_delivers_a_tool_send(tmp_path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_app_settings(replace(store.app_settings(), enabled=True, debounce_seconds=0))
    await store.aupsert_conversation("channel", "peer", "Peer")
    transport = RecordingTransport()
    model = ScriptedChatModel(
        responses=[
            tool_call(
                "send_messages",
                {"messages": ["Hello from the graph"], "continue_after_sending": False},
                "send-1",
            )
        ]
    )
    service = AgentService(store, FakeModels(model), transport, "x" * 32, UnusedPublicHTTP())
    await service.start()

    await service.submit_message(
        message_id="discord-1",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="Hello?",
        attachments=[],
        observed_at=time.time(),
    )
    await wait_for_idle(service)

    assert transport.messages == [("channel", ("Hello from the graph",))]
    async with store.database.transaction() as connection:
        run = await (await connection.execute("SELECT status, thread_id FROM agent_runs")).fetchone()
        queue = await (await connection.execute("SELECT disposition FROM chat_event_queue")).fetchone()
        checkpoint_count = (await (await connection.execute("SELECT COUNT(*) FROM checkpoints")).fetchone())[
            0
        ]
    assert dict(run) == {"status": "completed", "thread_id": "discord:owner:channel:g1"}
    assert queue["disposition"] == "completed"
    assert checkpoint_count > 0

    await service.close()
    await store.aclose()


@pytest.mark.asyncio
async def test_historical_replay_uses_emulated_discord_actions(tmp_path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_app_settings(replace(store.app_settings(), enabled=True, debounce_seconds=0))
    await store.aupsert_conversation("channel", "peer", "Peer")
    transport = RecordingTransport()
    model = ScriptedChatModel(responses=[AIMessage(content="Initial turn")])
    service = AgentService(store, FakeModels(model), transport, "x" * 32, UnusedPublicHTTP())
    await service.start()
    await service.submit_message(
        message_id="discord-replay",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="Replay me",
        attachments=[],
        observed_at=time.time(),
    )
    await wait_for_idle(service)
    checkpoint = (await service.checkpoint_views())[0]["checkpoints"][0]
    model.responses.append(
        tool_call(
            "send_messages",
            {"messages": ["This must be emulated"], "continue_after_sending": False},
            "replay-send",
        )
    )

    replay_id = await service.replay_checkpoint("discord:owner:channel:g1", checkpoint["checkpoint_id"])

    assert transport.messages == []
    async with store.database.transaction() as connection:
        trace = await (
            await connection.execute(
                "SELECT payload FROM agent_trace_events WHERE run_id=? AND kind='emulated_actions'",
                (replay_id,),
            )
        ).fetchone()
    assert "This must be emulated" in trace["payload"]
    await service.close()
    await store.aclose()


@pytest.mark.asyncio
async def test_model_change_rolls_checkpoint_to_portable_summary(tmp_path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_app_settings(replace(store.app_settings(), enabled=True, debounce_seconds=0))
    await store.aupsert_conversation("channel", "peer", "Peer")
    model = ScriptedChatModel(responses=[AIMessage(content="Initial answer")])
    models = FakeModels(model)
    service = AgentService(store, models, RecordingTransport(), "x" * 32, UnusedPublicHTTP())
    await service.start()
    await service.submit_message(
        message_id="discord-rollover",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="Keep this context",
        attachments=[],
        observed_at=time.time(),
    )
    await wait_for_idle(service)
    previous = models.configuration
    models.configuration = replace(previous, provider_id="other", model_id="other-model")
    model.responses.append(AIMessage(content="Portable summary of the prior conversation"))

    assert await service.apply_configuration_transition(previous) == 1

    thread = await store.achat_thread_for_channel("channel")
    assert {"generation": thread["generation"], "thread_id": thread["thread_id"]} == {
        "generation": 2,
        "thread_id": "discord:owner:channel:g2",
    }
    snapshot = await service.checkpointer.aget_tuple({"configurable": {"thread_id": thread["thread_id"]}})
    summary = snapshot.checkpoint["channel_values"]["messages"][0]
    assert summary.content == "Portable summary of the prior conversation"
    assert summary.additional_kwargs["diskovod_generation_summary"]["source_thread_id"].endswith(":g1")
    await service.close()
    await store.aclose()


@pytest.mark.asyncio
async def test_agent_service_allows_a_zero_message_turn(tmp_path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_app_settings(replace(store.app_settings(), enabled=True, debounce_seconds=0))
    await store.aupsert_conversation("channel", "peer", "Peer")
    transport = RecordingTransport()
    service = AgentService(
        store,
        FakeModels(ScriptedChatModel(responses=[AIMessage(content="No visible action needed")])),
        transport,
        "x" * 32,
        UnusedPublicHTTP(),
    )
    await service.start()
    await service.submit_message(
        message_id="discord-2",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="FYI",
        attachments=[],
        observed_at=time.time(),
    )
    await wait_for_idle(service)

    assert transport.messages == []
    async with store.database.transaction() as connection:
        run = await (await connection.execute("SELECT status FROM agent_runs")).fetchone()
    assert run["status"] == "completed"
    await service.close()
    await store.aclose()


@pytest.mark.asyncio
async def test_escalation_interrupt_resumes_without_resending_acknowledgement(tmp_path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_app_settings(replace(store.app_settings(), enabled=True, debounce_seconds=0))
    await store.aupsert_conversation("channel", "peer", "Peer")
    transport = RecordingTransport()
    model = ScriptedChatModel(
        responses=[
            tool_call(
                "escalate_to_owner",
                {
                    "reason": "peer_requested_owner",
                    "acknowledgement": "I marked this for the owner.",
                },
                "escalate-1",
            ),
            AIMessage(content="The owner resolved the handoff."),
        ]
    )
    service = AgentService(store, FakeModels(model), transport, "x" * 32, UnusedPublicHTTP())
    await service.start()
    await service.submit_message(
        message_id="discord-3",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="Can I speak to the owner?",
        attachments=[],
        observed_at=time.time(),
    )
    await wait_for_idle(service)

    escalation = (await store.aactive_interrupts())[0]
    assert transport.messages == [("channel", ("I marked this for the owner.",))]
    assert await service.claim_escalation(escalation["id"]) is True
    assert (
        await service.resume_escalation(
            escalation["id"],
            action="owner_reply",
            owner_message="I am here now.",
            owner_message_id="owner-message",
            owner_author_id="owner",
            owner_author_name="Owner",
        )
        is True
    )

    assert transport.messages == [("channel", ("I marked this for the owner.",))]
    assert await store.aactive_interrupts() == []
    async with store.database.transaction() as connection:
        run = await (await connection.execute("SELECT status FROM agent_runs")).fetchone()
    assert run["status"] == "completed"
    thread_id = await service.events.thread_id("owner", "channel")
    checkpoint = await service.checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    owner = next(
        message
        for message in checkpoint.checkpoint["channel_values"]["messages"]
        if message.id == "owner-message"
    )
    assert owner.additional_kwargs["diskovod_participant"]["role"] == "owner"
    await service.close()
    await store.aclose()
