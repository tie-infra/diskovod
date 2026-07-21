from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from diskovod.interaction import AvailabilitySchedule, ActiveTurnInput, TriggerRule, preset_policy
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


class CapturingModel(ScriptedChatModel):
    seen_inputs: list[list[str]] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_inputs.append(
            [str(message.content) for message in messages if isinstance(message, HumanMessage)]
        )
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


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
        event = await (await connection.execute("SELECT context_state FROM conversation_events")).fetchone()
        work = await (await connection.execute("SELECT state FROM agent_work")).fetchone()
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
    assert event["context_state"] == "applied"
    assert work["state"] == "completed"
    assert checkpoint_count > 0
    assert checkpoint_index["run_id"] == run["id"]
    assert checkpoint_index["message_count"] > 0

    await service.close()
    await store.aclose()


@pytest.mark.asyncio
async def test_invocation_turn_captures_passive_context_without_passive_provider_calls(tmp_path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
    await store.aupsert_conversation("channel", "peer", "Peer")
    await store.aset_interaction_policy("channel", preset_policy("on_invocation", prompt_locale="en"))
    model = CapturingModel(responses=[AIMessage(content="A compromise is 6:30.")])
    transport = RecordingTransport()
    service = AgentService(store, FakeModels(model), transport, "x" * 32, UnusedPublicHTTP())
    await service.start()

    for values in (
        ("one", "peer", "I can arrive around six."),
        ("two", "owner", "Seven would work better for me."),
    ):
        await service.submit_message(
            message_id=values[0],
            channel_id="channel",
            account_id="owner",
            author_id=values[1],
            author_name=values[1].title(),
            participant_role=values[1],
            content=values[2],
            attachments=[],
            observed_at=time.time(),
        )
    assert model.index == 0
    assert service.tasks == {}

    await service.submit_message(
        message_id="three",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="Hey, Diskovod, find a compromise.",
        attachments=[],
        observed_at=time.time(),
    )
    await wait_for_idle(service)

    assert model.index == 1
    assert model.seen_inputs[-1][-3:] == [
        "I can arrive around six.",
        "Seven would work better for me.",
        "Hey, Diskovod, find a compromise.",
    ]
    assert transport.messages == [("channel", ("A compromise is 6:30.",))]
    async with store.database.transaction() as connection:
        assert (await (await connection.execute("SELECT COUNT(*) FROM agent_work")).fetchone())[0] == 1
        assert {
            row[0]
            for row in await (
                await connection.execute("SELECT context_state FROM conversation_events ORDER BY sequence")
            ).fetchall()
        } == {"applied"}
    await service.close()
    await store.aclose()


@pytest.mark.asyncio
async def test_configured_reaction_starts_a_turn_and_is_added_to_context(tmp_path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
    await store.aupsert_conversation("channel", "peer", "Peer")
    await store.aset_interaction_policy(
        "channel",
        replace(
            preset_policy("on_invocation"),
            trigger_rules=(TriggerRule("reaction_invocation", reactions=("👀",)),),
        ),
    )
    model = CapturingModel(responses=[AIMessage(content="I’ll take a look.")])
    transport = RecordingTransport()
    service = AgentService(store, FakeModels(model), transport, "x" * 32, UnusedPublicHTTP())
    await service.start()

    await service.submit_reaction(
        message_id="discord-1",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        emoji="👀",
        observed_at=time.time(),
    )
    await wait_for_idle(service)

    assert model.index == 1
    assert model.seen_inputs[-1][-1] == "Peer reacted to message discord-1 with 👀."
    assert transport.messages == [("channel", ("I’ll take a look.",))]
    async with store.database.transaction() as connection:
        event = await (
            await connection.execute(
                "SELECT kind, json_extract(admission_decision, '$.reason'), context_state "
                "FROM conversation_events"
            )
        ).fetchone()
    assert tuple(event) == ("reaction", "reaction_invocation", "applied")

    await service.close()
    await store.aclose()


@pytest.mark.asyncio
async def test_outside_availability_schedule_admits_context_without_starting_a_turn(tmp_path, monkeypatch):
    monday_evening = 1_774_288_800.0  # 2026-03-23 18:00:00 UTC
    monkeypatch.setattr("diskovod.runtime.time.time", lambda: monday_evening)
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
    await store.aupsert_conversation("channel", "peer", "Peer")
    await store.aset_interaction_policy(
        "channel",
        replace(
            preset_policy("autonomous"),
            availability_schedule=AvailabilitySchedule(
                enabled=True,
                weekdays=frozenset({0}),
                start_minute=9 * 60,
                end_minute=17 * 60,
                timezone="UTC",
            ),
        ),
    )
    model = ScriptedChatModel(responses=[])
    service = AgentService(store, FakeModels(model), RecordingTransport(), "x" * 32, UnusedPublicHTTP())
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
        observed_at=monday_evening,
    )

    assert service.tasks == {}
    async with store.database.transaction() as connection:
        event = await (
            await connection.execute(
                "SELECT json_extract(admission_decision, '$.reason') FROM conversation_events"
            )
        ).fetchone()
    assert event[0] == "outside_schedule"
    await service.close()
    await store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timing", "expected_reason"),
    (
        ("inject_at_safe_points", "active_input"),
        ("queue_for_next_turn", "active_input_queued"),
    ),
)
async def test_active_turn_input_never_opens_a_second_turn(tmp_path, timing, expected_reason):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
    await store.aupsert_conversation("channel", "peer", "Peer")
    policy = replace(
        preset_policy("on_invocation"),
        active_turn_input=ActiveTurnInput(timing=timing),
    )
    await store.aset_interaction_policy("channel", policy)
    service = AgentService(
        store,
        FakeModels(ScriptedChatModel(responses=[])),
        RecordingTransport(),
        "x" * 32,
        UnusedPublicHTTP(),
    )
    release = asyncio.Event()
    active = asyncio.create_task(release.wait())
    service.tasks["channel"] = active

    await service.submit_message(
        message_id="during-turn",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="ordinary follow-up without an invocation",
        attachments=[],
        observed_at=time.time(),
    )

    async with store.database.transaction() as connection:
        event = await (
            await connection.execute("SELECT admission_decision, context_state FROM conversation_events")
        ).fetchone()
        work_count = (await (await connection.execute("SELECT COUNT(*) FROM agent_work")).fetchone())[0]
    assert expected_reason in event["admission_decision"]
    assert event["context_state"] == "unapplied"
    assert work_count == 0
    release.set()
    await active
    await store.aclose()


@pytest.mark.asyncio
async def test_owner_handoff_cancels_a_pending_debounced_autonomous_turn(tmp_path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=60)
    )
    await store.aupsert_conversation("channel", "peer", "Peer")
    model = ScriptedChatModel(responses=[])
    service = AgentService(
        store,
        FakeModels(model),
        RecordingTransport(),
        "x" * 32,
        UnusedPublicHTTP(),
    )
    await service.start()
    await service.submit_message(
        message_id="peer-trigger",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="Please answer",
        attachments=[],
        observed_at=time.time(),
    )
    await asyncio.sleep(0)
    # The already accepted turn keeps the autonomous handoff policy even if
    # the chat is reconfigured before the debounce window closes.
    await store.aset_interaction_policy("channel", preset_policy("shared"))
    await service.submit_message(
        message_id="owner-takeover",
        channel_id="channel",
        account_id="owner",
        author_id="owner",
        author_name="Owner",
        participant_role="owner",
        content="I will handle this.",
        attachments=[],
        observed_at=time.time(),
    )
    await asyncio.gather(*service.tasks.values(), return_exceptions=True)

    conversation = await store.aconversation("channel")
    async with store.database.transaction() as connection:
        work = await (await connection.execute("SELECT state FROM agent_work")).fetchone()
    assert conversation["snoozed_until"] > time.time()
    assert work["state"] == "cancelled"
    assert model.index == 0
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
                INSERT INTO conversation_events(
                  id, channel_id, sequence, kind, payload, observed_at,
                  context_state, run_id, claimed_at
                ) VALUES(?, 'channel', ?, 'message', '{}', 0, 'claimed', ?, 0)
                """,
                (f"event-{run_id}", sequence, run_id),
            )
            await connection.execute(
                """
                INSERT INTO agent_work(
                  id, channel_id, kind, source_event_id, trigger_kind, policy_version,
                  policy_snapshot, available_at, state, run_id, decision, created_at, claimed_at
                ) VALUES(?, 'channel', 'turn', ?, 'message', 1, '{}', 0,
                  'claimed', ?, '{}', 0, 0)
                """,
                (f"work-{run_id}", f"event-{run_id}", run_id),
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
        work = {
            row["id"]: dict(row)
            for row in await (await connection.execute("SELECT * FROM agent_work")).fetchall()
        }
        runs = {
            row["id"]: dict(row)
            for row in await (await connection.execute("SELECT * FROM agent_runs")).fetchall()
        }
    assert work["work-release"]["state"] == "pending"
    assert work["work-release"]["run_id"] is None
    assert runs["release"]["status"] == "cancelled"
    assert work["work-complete"]["state"] == "completed"
    assert runs["complete"]["status"] == "completed"
    assert work["work-review"]["state"] == "failed"
    assert runs["review"]["status"] == "failed"
    escalation = await store.aescalation_interrupt("legacy-escalation")
    assert escalation["payload"]["resume_strategy"] == "journal"
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
                "SELECT context_state, payload FROM conversation_events "
                "WHERE id='discord:message:owner-reply'"
            )
        ).fetchone()
    assert reply["context_state"] == "unapplied"
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
            INSERT INTO conversation_events(
              id, channel_id, sequence, kind, payload, observed_at,
              context_state, run_id, claimed_at
            ) VALUES('event', 'channel', 1, 'message', '{}', 0, 'claimed', 'abandoned', 0)
            """
        )
        await connection.execute(
            """
            INSERT INTO agent_work(
              id, channel_id, kind, source_event_id, trigger_kind, policy_version,
              policy_snapshot, available_at, state, run_id, decision, created_at, claimed_at
            ) VALUES('work', 'channel', 'turn', 'event', 'message', 1, '{}', 0,
              'claimed', 'abandoned', '{}', 0, 0)
            """
        )

    await service._reconcile_abandoned_agent_runs()

    async with store.database.transaction() as connection:
        event = await (
            await connection.execute("SELECT context_state, run_id FROM conversation_events WHERE id='event'")
        ).fetchone()
        run = await (
            await connection.execute("SELECT status FROM agent_runs WHERE id='abandoned'")
        ).fetchone()
    assert dict(event) == {"context_state": "unapplied", "run_id": None}
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
        owner_event = await (
            await connection.execute(
                "SELECT context_state FROM conversation_events WHERE id='discord:message:owner-message'"
            )
        ).fetchone()
    assert run["status"] == "completed"
    assert owner_event["context_state"] == "applied"
    thread_id = await service.journal.thread_id("owner", "channel")
    checkpoint = await service.checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    owner = next(
        message
        for message in checkpoint.checkpoint["channel_values"]["messages"]
        if message.id == "owner-message"
    )
    assert owner.additional_kwargs["diskovod_participant"]["role"] == "owner"
    await service.close()
    await store.aclose()
