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

from test_agent import ScriptedChatModel, tool_call


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
    for _ in range(100):
        if not service.tasks:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("agent service did not become idle")


@pytest.mark.asyncio
async def test_agent_service_persists_a_chat_thread_and_delivers_a_tool_send(tmp_path):
    store = Store(tmp_path / "diskovod.sqlite3", "x" * 32)
    store.set_app_settings(replace(store.app_settings(), enabled=True, debounce_seconds=0))
    store.upsert_conversation("channel", "peer", "Peer")
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
    service = AgentService(store, FakeModels(model), transport, "x" * 32)
    await service.start()

    service.submit_message(
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
    run = store._db.execute("SELECT status, thread_id FROM agent_runs").fetchone()
    assert dict(run) == {"status": "completed", "thread_id": "discord:owner:channel:g1"}
    queue = store._db.execute("SELECT disposition FROM chat_event_queue").fetchone()
    assert queue["disposition"] == "completed"
    assert store._db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0

    await service.close()
    store.close()


@pytest.mark.asyncio
async def test_agent_service_allows_a_zero_message_turn(tmp_path):
    store = Store(tmp_path / "diskovod.sqlite3", "x" * 32)
    store.set_app_settings(replace(store.app_settings(), enabled=True, debounce_seconds=0))
    store.upsert_conversation("channel", "peer", "Peer")
    transport = RecordingTransport()
    service = AgentService(
        store,
        FakeModels(ScriptedChatModel(responses=[AIMessage(content="No visible action needed")])),
        transport,
        "x" * 32,
    )
    await service.start()
    service.submit_message(
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
    assert store._db.execute("SELECT status FROM agent_runs").fetchone()["status"] == "completed"
    await service.close()
    store.close()
