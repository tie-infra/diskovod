from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage

from diskovod.outbound import DeliveryRecord
from diskovod.providers import ModelConfiguration, ProviderCapabilities
from diskovod.runtime import AgentService
from diskovod.store import Store

from test_agent import ScriptedChatModel, UnusedPublicHTTP


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
async def test_agent_service_persists_a_chat_thread_and_delivers_assistant_text(tmp_path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
    await store.aupsert_conversation("channel", "peer", "Peer")
    transport = RecordingTransport()
    model = ScriptedChatModel(responses=[AIMessage(content="Hello from the graph")])
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
        run = await (await connection.execute("SELECT id, status, thread_id FROM agent_runs")).fetchone()
        mailbox = await (await connection.execute("SELECT state FROM conversation_mailbox")).fetchone()
        checkpoint_count = (await (await connection.execute("SELECT COUNT(*) FROM checkpoints")).fetchone())[
            0
        ]
        checkpoint_index = await (
            await connection.execute(
                "SELECT run_id, message_count FROM checkpoint_index ORDER BY created_at DESC LIMIT 1"
            )
        ).fetchone()
    assert run["status"] == "completed"
    assert run["thread_id"] == "discord:owner:channel:g1"
    assert mailbox["state"] == "completed"
    assert checkpoint_count > 0
    assert checkpoint_index["run_id"] == run["id"]
    assert checkpoint_index["message_count"] > 0

    await service.close()
    await store.aclose()


@pytest.mark.asyncio
async def test_public_output_cutover_reconciles_claims_from_legacy_graph_threads(tmp_path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    service = AgentService(
        store,
        FakeModels(ScriptedChatModel(responses=[])),
        RecordingTransport(),
        "x" * 32,
        UnusedPublicHTTP(),
    )
    run_ids = ("release", "complete", "review", "escalation")
    for run_id in run_ids:
        await store.astart_agent_run(
            run_id=run_id,
            thread_id="legacy-thread",
            channel_id="channel",
            trace_id=f"trace-{run_id}",
        )
        await store.afinish_agent_run(run_id, "interrupted")
    async with store.database.transaction() as connection:
        for sequence, run_id in enumerate(run_ids[:3], 1):
            await connection.execute(
                """
                INSERT INTO conversation_mailbox(
                  id, channel_id, sequence, kind, available_at, observed_at,
                  payload, state, run_id, claimed_at
                ) VALUES(?, 'channel', ?, 'message', 0, 0, '{}', 'claimed', ?, 0)
                """,
                (f"event-{run_id}", sequence, run_id),
            )
        for run_id, state, result in (
            (
                "complete",
                "succeeded",
                '{"status":"accepted","message_index":0,"discord_message_id":"42",'
                '"error_code":null,"error_detail":null}',
            ),
            (
                "review",
                "ambiguous",
                '{"status":"ambiguous","message_index":0,"discord_message_id":null,'
                '"error_code":"transport_exception","error_detail":null}',
            ),
        ):
            await connection.execute(
                """
                INSERT INTO outbound_actions(
                  id, batch_id, ordinal, thread_id, channel_id, run_id,
                  source_kind, source_id, kind, payload, state, result, created_at
                ) VALUES(?, ?, 0, 'legacy-thread', 'channel', ?,
                  'legacy_tool', ?, 'discord_message', '{}', ?, ?, 0)
                """,
                (f"action-{run_id}", f"batch-{run_id}", run_id, run_id, state, result),
            )
        await connection.execute(
            """
            INSERT INTO escalation_interrupts(
              id, thread_id, channel_id, state, payload, created_at, updated_at
            ) VALUES(
              'legacy-escalation', 'legacy-thread', 'channel', 'pending',
              '{"trace_id":"trace-escalation","trigger_message_id":"trigger"}', 0, 0
            )
            """
        )

    await service._reconcile_public_output_cutover({"legacy-thread"})

    async with store.database.transaction() as connection:
        events = {
            row["id"]: dict(row)
            for row in await (
                await connection.execute("SELECT * FROM conversation_mailbox")
            ).fetchall()
        }
        runs = {
            row["id"]: dict(row)
            for row in await (await connection.execute("SELECT * FROM agent_runs")).fetchall()
        }
    assert events["event-release"]["state"] == "pending"
    assert events["event-release"]["run_id"] is None
    assert runs["release"]["status"] == "cancelled"
    assert events["event-complete"]["state"] == "completed"
    assert runs["complete"]["status"] == "completed"
    assert events["event-review"]["state"] == "failed"
    assert runs["review"]["status"] == "failed"
    escalation = await store.aescalation_interrupt("legacy-escalation")
    assert escalation["payload"]["resume_strategy"] == "mailbox"
    assert runs["escalation"]["status"] == "cancelled"

    assert await service.resume_escalation(
        "legacy-escalation",
        action="owner_reply",
        owner_message="Please continue from here",
        owner_message_id="owner-reply",
        owner_author_id="owner",
        owner_author_name="Owner",
    )
    resolution = await store.aescalation_interrupt("legacy-escalation")
    assert resolution["state"] == "resolved"
    async with store.database.transaction() as connection:
        reply = await (
            await connection.execute(
                "SELECT state, payload FROM conversation_mailbox WHERE id='discord:message:owner-reply'"
            )
        ).fetchone()
    assert reply["state"] == "pending"
    assert "Please continue from here" in reply["payload"]
    await store.aclose()


@pytest.mark.asyncio
async def test_abandoned_ordinary_run_releases_only_claims_without_visible_effects(tmp_path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    service = AgentService(
        store,
        FakeModels(ScriptedChatModel(responses=[])),
        RecordingTransport(),
        "x" * 32,
        UnusedPublicHTTP(),
    )
    await store.astart_agent_run(
        run_id="abandoned",
        thread_id="current-thread",
        channel_id="channel",
        trace_id="trace-abandoned",
    )
    async with store.database.transaction() as connection:
        await connection.execute(
            """
            INSERT INTO conversation_mailbox(
              id, channel_id, sequence, kind, available_at, observed_at,
              payload, state, run_id, claimed_at
            ) VALUES('event', 'channel', 1, 'message', 0, 0, '{}', 'claimed', 'abandoned', 0)
            """
        )

    await service._reconcile_abandoned_agent_runs()

    async with store.database.transaction() as connection:
        event = await (
            await connection.execute("SELECT state, run_id FROM conversation_mailbox WHERE id='event'")
        ).fetchone()
        run = await (
            await connection.execute("SELECT status FROM agent_runs WHERE id='abandoned'")
        ).fetchone()
    assert dict(event) == {"state": "pending", "run_id": None}
    assert run["status"] == "cancelled"
    partial = [
        {
            "state": "failed",
            "result": '[{"status":"accepted"},{"status":"failed"}]',
        }
    ]
    assert service._claimed_run_outcome(partial) == (
        "failed",
        "failed",
        "failed_for_delivery_review",
    )
    await store.aclose()


@pytest.mark.asyncio
async def test_historical_replay_uses_emulated_discord_actions(tmp_path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
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
    transport.messages.clear()
    model.responses.append(AIMessage(content="This must be emulated"))

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
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
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
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
    await store.aupsert_conversation("channel", "peer", "Peer")
    transport = RecordingTransport()
    service = AgentService(
        store,
        FakeModels(ScriptedChatModel(responses=[AIMessage(content="")])),
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
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
    await store.aupsert_conversation("channel", "peer", "Peer")
    transport = RecordingTransport()
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="I marked this for the owner.",
                tool_calls=[
                    {
                        "name": "escalate_to_owner",
                        "args": {},
                        "id": "escalate-1",
                        "type": "tool_call",
                    }
                ],
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

    assert transport.messages == [
        ("channel", ("I marked this for the owner.",)),
        ("channel", ("The owner resolved the handoff.",)),
    ]
    assert await store.aactive_interrupts() == []
    async with store.database.transaction() as connection:
        run = await (await connection.execute("SELECT status FROM agent_runs")).fetchone()
    assert run["status"] == "completed"
    thread_id = await service.mailbox.thread_id("owner", "channel")
    checkpoint = await service.checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    owner = next(
        message
        for message in checkpoint.checkpoint["channel_values"]["messages"]
        if message.id == "owner-message"
    )
    assert owner.additional_kwargs["diskovod_participant"]["role"] == "owner"
    await service.close()
    await store.aclose()
