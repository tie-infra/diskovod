from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from diskovod.agent import build_agent
from diskovod.models import AssistantProfile
from diskovod.store import Store
from diskovod.waits import ConversationWaits

from test_agent import RecordingGateway, ScriptedChatModel, UnusedPublicHTTP, prompt, runtime_context
from test_runtime import AgentService, FakeModels, RecordingTransport, wait_for_idle


def wait_call(
    message: str = "One thought…",
    call_id: str = "followup-1",
    pause: str = "brief",
) -> AIMessage:
    return AIMessage(
        content=message,
        tool_calls=[
            {
                "name": "wait_before_followup",
                "args": {"pause": pause},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


class FakeFollowupScheduler:
    def __init__(self):
        self.resolved: list[str] = []

    async def arm_followup(self, context, state, *, tool_call_id, pause, max_duration):
        del context, state, tool_call_id, pause, max_duration
        return "wait-1", 123.0, 2.0

    async def resolve_followup(self, wait_id):
        self.resolved.append(wait_id)


class BudgetFollowupScheduler(FakeFollowupScheduler):
    def __init__(self):
        super().__init__()
        self.maximums: list[float] = []
        self.armed: dict[str, tuple[str, float, float]] = {}

    async def arm_followup(self, context, state, *, tool_call_id, pause, max_duration):
        del context, state, pause
        if tool_call_id in self.armed:
            return self.armed[tool_call_id]
        self.maximums.append(max_duration)
        result = (f"wait:{tool_call_id}", 123.0, min(8.0, max_duration))
        self.armed[tool_call_id] = result
        return result


@pytest.mark.asyncio
async def test_wait_tool_suspends_then_returns_to_the_ordinary_agent_loop():
    scheduler = FakeFollowupScheduler()
    gateway = RecordingGateway()
    model = ScriptedChatModel(responses=[wait_call(), AIMessage(content="Actually, one caveat.")])
    agent = build_agent(
        model,
        gateway,
        prompt(),
        UnusedPublicHTTP(),
        checkpointer=InMemorySaver(),
        followup_scheduler=scheduler,
    )
    context = replace(runtime_context(), allow_conversational_followups=True)
    config = {"configurable": {"thread_id": "thread"}}

    suspended = await agent.ainvoke(
        {"messages": [HumanMessage("hello")], "logical_request_id": "request"},
        config=config,
        context=context,
    )
    assert suspended["__interrupt__"]
    assert [call[1] for call in gateway.calls] == [("One thought…",)]

    result = await agent.ainvoke(
        Command(resume={"reason": "deadline", "wait_id": "wait-1"}),
        config=config,
        context=context,
    )

    assert [call[1] for call in gateway.calls] == [
        ("One thought…",),
        ("Actually, one caveat.",),
    ]
    assert scheduler.resolved == ["wait-1"]
    assert result["followup_wait_count"] == 1
    assert result["followup_wait_seconds"] == 2.0


@pytest.mark.asyncio
async def test_wait_with_a_sibling_tool_fails_before_publication():
    gateway = RecordingGateway()
    response = wait_call()
    response.tool_calls.append(
        {
            "name": "calculate",
            "args": {"expression": "2 + 2"},
            "id": "calculation",
            "type": "tool_call",
        }
    )
    agent = build_agent(
        ScriptedChatModel(responses=[response]),
        gateway,
        prompt(),
        UnusedPublicHTTP(),
        followup_scheduler=FakeFollowupScheduler(),
    )

    with pytest.raises(RuntimeError, match="only tool call"):
        await agent.ainvoke(
            {"messages": [HumanMessage("hello")]},
            context=replace(runtime_context(), allow_conversational_followups=True),
        )
    assert gateway.calls == []


@pytest.mark.asyncio
async def test_followup_wait_count_and_cumulative_delay_survive_resumes():
    scheduler = BudgetFollowupScheduler()
    gateway = RecordingGateway()
    model = ScriptedChatModel(
        responses=[
            wait_call("First thought.", "wait-1", "short"),
            wait_call("Second thought.", "wait-2", "short"),
            wait_call("No third pause.", "wait-3", "brief"),
            AIMessage("Finished."),
        ]
    )
    agent = build_agent(
        model,
        gateway,
        prompt(),
        UnusedPublicHTTP(),
        checkpointer=InMemorySaver(),
        followup_scheduler=scheduler,
    )
    context = replace(runtime_context(), allow_conversational_followups=True)
    config = {"configurable": {"thread_id": "thread"}}

    first = await agent.ainvoke(
        {"messages": [HumanMessage("hello")], "logical_request_id": "request"},
        config=config,
        context=context,
    )
    assert first["__interrupt__"]
    second = await agent.ainvoke(
        Command(resume={"reason": "deadline", "wait_id": "wait:wait-1"}),
        config=config,
        context=context,
    )
    assert second["__interrupt__"]
    result = await agent.ainvoke(
        Command(resume={"reason": "deadline", "wait_id": "wait:wait-2"}),
        config=config,
        context=context,
    )

    assert "__interrupt__" not in result
    assert scheduler.maximums == [10.0, 2.0]
    assert result["followup_wait_count"] == 2
    assert result["followup_wait_seconds"] == 10.0
    assert [call[1] for call in gateway.calls] == [
        ("First thought.",),
        ("Second thought.",),
        ("No third pause.",),
        ("Finished.",),
    ]


@pytest.mark.asyncio
async def test_wait_repository_wakes_early_and_survives_reopening(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    waits = ConversationWaits(store.database)
    context = replace(runtime_context(), thread_id="thread")
    wait = await waits.arm(
        context,
        run_id="run",
        tool_call_id="tool",
        duration=60,
        payload={"duration": 60},
    )
    assert await waits.schedule(wait.id)
    assert await waits.wake_for_input("channel")

    claimed = await waits.claim_ready("channel")

    assert claimed is not None
    assert claimed.id == wait.id
    assert claimed.resume_reason == "new_input"
    await store.aclose()


class SignallingTransport(RecordingTransport):
    def __init__(self):
        super().__init__()
        self.followup_sent = asyncio.Event()

    async def send_messages(self, context, messages):
        result = await super().send_messages(context, messages)
        if len(self.messages) >= 2:
            self.followup_sent.set()
        return result


@pytest.mark.asyncio
async def test_service_resumes_a_durable_wait_when_new_input_arrives(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("diskovod.runtime.random.uniform", lambda minimum, maximum: maximum)
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
    await store.aset_assistant_profile(replace(AssistantProfile(), allow_conversational_followups=True))
    await store.aupsert_conversation("channel", "peer", "Peer")
    transport = SignallingTransport()
    model = ScriptedChatModel(responses=[wait_call("I may add something."), AIMessage("Never mind—got it.")])
    service = AgentService(store, FakeModels(model), transport, "x" * 32, UnusedPublicHTTP())
    await service.start()
    await service.submit_message(
        message_id="first",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="Think about this",
        attachments=[],
        observed_at=time.time(),
    )
    await wait_for_idle(service)
    assert (await service.waits.active("channel")).state == "scheduled"

    await service.submit_message(
        message_id="second",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="Additional context",
        attachments=[],
        observed_at=time.time(),
    )
    await asyncio.wait_for(transport.followup_sent.wait(), timeout=3)
    await wait_for_idle(service)

    assert [messages for _, messages in transport.messages] == [
        ("I may add something.",),
        ("Never mind—got it.",),
    ]
    assert await service.waits.active("channel") is None
    async with store.database.transaction() as connection:
        trace_kinds = {
            str(row["kind"])
            for row in await (
                await connection.execute("SELECT kind FROM agent_trace_events")
            ).fetchall()
        }
    assert {
        "followup_wait_armed",
        "followup_wait_scheduled",
        "followup_wait_woken",
        "followup_wait_resume",
        "mailbox_injection",
        "followup_wait_result",
    } <= trace_kinds
    await service.close()
    await store.aclose()


@pytest.mark.asyncio
async def test_owner_can_cancel_a_durable_followup_without_another_model_call(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("diskovod.runtime.random.uniform", lambda minimum, maximum: maximum)
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
    await store.aset_assistant_profile(replace(AssistantProfile(), allow_conversational_followups=True))
    await store.aupsert_conversation("channel", "peer", "Peer")
    model = ScriptedChatModel(responses=[wait_call("I may follow up.")])
    service = AgentService(
        store,
        FakeModels(model),
        RecordingTransport(),
        "x" * 32,
        UnusedPublicHTTP(),
    )
    await service.start()
    await service.submit_message(
        message_id="first",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="Think about this",
        attachments=[],
        observed_at=time.time(),
    )
    await wait_for_idle(service)
    wait = await service.waits.active("channel")

    assert wait is not None
    assert await service.cancel_followup("channel")
    assert await service.waits.active("channel") is None
    assert model.index == 1
    async with store.database.transaction() as connection:
        saved_wait = await (
            await connection.execute("SELECT state FROM conversation_waits WHERE id=?", (wait.id,))
        ).fetchone()
        run = await (
            await connection.execute("SELECT status FROM agent_runs WHERE id=?", (wait.run_id,))
        ).fetchone()
    assert saved_wait["state"] == "cancelled"
    assert run["status"] == "cancelled"
    async with store.database.transaction() as connection:
        cancellation = await (
            await connection.execute(
                "SELECT payload FROM agent_trace_events "
                "WHERE run_id=? AND kind='followup_wait_cancelled'",
                (wait.run_id,),
            )
        ).fetchone()
    assert "cancelled_by_owner" in cancellation["payload"]
    await service.close()
    await store.aclose()


@pytest.mark.parametrize("recovered_state", ["resuming", "completed"])
async def test_service_recovers_a_followup_interrupted_during_resume(
    tmp_path: Path,
    monkeypatch,
    recovered_state: str,
):
    monkeypatch.setattr("diskovod.runtime.random.uniform", lambda minimum, maximum: maximum)
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
    await store.aset_assistant_profile(replace(AssistantProfile(), allow_conversational_followups=True))
    await store.aupsert_conversation("channel", "peer", "Peer")
    transport = SignallingTransport()
    model = ScriptedChatModel(
        responses=[wait_call("I may follow up."), AIMessage("Recovered follow-up.")]
    )
    service = AgentService(store, FakeModels(model), transport, "x" * 32, UnusedPublicHTTP())
    await service.start()
    await service.submit_message(
        message_id="first",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="Think about this",
        attachments=[],
        observed_at=time.time(),
    )
    await wait_for_idle(service)
    wait = await service.waits.active("channel")
    assert wait is not None
    await service.close()

    assert await service.waits.wake_for_input("channel")
    claimed = await service.waits.claim_ready("channel")
    assert claimed is not None and claimed.state == "resuming"
    if recovered_state == "completed":
        assert await service.waits.resolve(wait.id)

    recovered = AgentService(store, FakeModels(model), transport, "x" * 32, UnusedPublicHTTP())
    await recovered.start()
    await asyncio.wait_for(transport.followup_sent.wait(), timeout=3)
    await wait_for_idle(recovered)

    assert [messages for _, messages in transport.messages] == [
        ("I may follow up.",),
        ("Recovered follow-up.",),
    ]
    assert await recovered.waits.active("channel") is None
    assert await recovered.waits.incomplete_resume("channel") is None
    await recovered.close()
    await store.aclose()


async def test_pausing_a_chat_cancels_its_durable_followup(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("diskovod.runtime.random.uniform", lambda minimum, maximum: maximum)
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
    await store.aset_assistant_profile(replace(AssistantProfile(), allow_conversational_followups=True))
    await store.aupsert_conversation("channel", "peer", "Peer")
    model = ScriptedChatModel(responses=[wait_call("I may follow up.")])
    service = AgentService(
        store,
        FakeModels(model),
        RecordingTransport(),
        "x" * 32,
        UnusedPublicHTTP(),
    )
    await service.start()
    await service.submit_message(
        message_id="first",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="Think about this",
        attachments=[],
        observed_at=time.time(),
    )
    await wait_for_idle(service)
    wait = await service.waits.active("channel")
    assert wait is not None

    await service.permanently_pause("channel")
    await service._resume_pending("channel")

    assert await service.waits.active("channel") is None
    assert service.tasks == {}
    async with store.database.transaction() as connection:
        saved = await (
            await connection.execute(
                "SELECT state, failure FROM conversation_waits WHERE id=?",
                (wait.id,),
            )
        ).fetchone()
    assert dict(saved) == {"state": "cancelled", "failure": "conversation_paused"}
    assert model.index == 1
    await service.close()
    await store.aclose()


async def test_scheduled_followup_resumes_after_restart_at_its_deadline(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("diskovod.runtime.random.uniform", lambda minimum, maximum: maximum)
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_automation_settings(
        replace(store.automation_settings(), enabled=True, debounce_seconds=0)
    )
    await store.aset_assistant_profile(replace(AssistantProfile(), allow_conversational_followups=True))
    await store.aupsert_conversation("channel", "peer", "Peer")
    transport = SignallingTransport()
    model = ScriptedChatModel(responses=[wait_call("I may follow up."), AIMessage("Deadline follow-up.")])
    service = AgentService(store, FakeModels(model), transport, "x" * 32, UnusedPublicHTTP())
    await service.start()
    await service.submit_message(
        message_id="first",
        channel_id="channel",
        account_id="owner",
        author_id="peer",
        author_name="Peer",
        participant_role="peer",
        content="Think about this",
        attachments=[],
        observed_at=time.time(),
    )
    await wait_for_idle(service)
    wait = await service.waits.active("channel")
    assert wait is not None and wait.state == "scheduled"
    await service.close()
    async with store.database.transaction() as connection:
        await connection.execute(
            "UPDATE conversation_waits SET resume_at=0 WHERE id=?",
            (wait.id,),
        )
        await connection.execute(
            "UPDATE conversation_mailbox SET available_at=0 WHERE id=?",
            (wait.wake_event_id,),
        )

    recovered = AgentService(store, FakeModels(model), transport, "x" * 32, UnusedPublicHTTP())
    await recovered.start()
    await asyncio.wait_for(transport.followup_sent.wait(), timeout=3)
    await wait_for_idle(recovered)

    assert [messages for _, messages in transport.messages] == [
        ("I may follow up.",),
        ("Deadline follow-up.",),
    ]
    async with store.database.transaction() as connection:
        resume = await (
            await connection.execute(
                "SELECT payload FROM agent_trace_events "
                "WHERE run_id=? AND kind='followup_wait_resume'",
                (wait.run_id,),
            )
        ).fetchone()
    assert '"reason":"deadline"' in resume["payload"]
    await recovered.close()
    await store.aclose()


async def test_startup_cancels_arming_wait_without_persisted_interrupt(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.astart_agent_run(
        run_id="run",
        thread_id="thread",
        channel_id="channel",
        trace_id="trace",
    )
    await store.afinish_agent_run("run", "interrupted")
    waits = ConversationWaits(store.database)
    wait = await waits.arm(
        replace(runtime_context(), thread_id="thread", trace_id="trace"),
        run_id="run",
        tool_call_id="tool",
        duration=60,
        payload={"duration": 60},
    )
    service = AgentService(
        store,
        FakeModels(ScriptedChatModel(responses=[])),
        RecordingTransport(),
        "x" * 32,
        UnusedPublicHTTP(),
    )

    await service.start()

    assert await service.waits.active("channel") is None
    async with store.database.transaction() as connection:
        saved = await (
            await connection.execute(
                "SELECT state, failure FROM conversation_waits WHERE id=?",
                (wait.id,),
            )
        ).fetchone()
        trace = await (
            await connection.execute(
                "SELECT payload FROM agent_trace_events "
                "WHERE run_id='run' AND kind='followup_wait_reconciled'"
            )
        ).fetchone()
    assert dict(saved) == {
        "state": "cancelled",
        "failure": "arming_without_persisted_interrupt",
    }
    assert "cancelled_without_interrupt" in trace["payload"]
    await service.close()
    await store.aclose()


async def test_disabling_automation_cancels_all_active_followups(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    service = AgentService(
        store,
        FakeModels(ScriptedChatModel(responses=[])),
        RecordingTransport(),
        "x" * 32,
        UnusedPublicHTTP(),
    )
    for index in range(2):
        channel_id = f"channel-{index}"
        run_id = f"run-{index}"
        await store.astart_agent_run(
            run_id=run_id,
            thread_id=f"thread-{index}",
            channel_id=channel_id,
            trace_id=f"trace-{index}",
        )
        await store.afinish_agent_run(run_id, "interrupted")
        wait = await service.waits.arm(
            replace(
                runtime_context(),
                channel_id=channel_id,
                thread_id=f"thread-{index}",
                trace_id=f"trace-{index}",
            ),
            run_id=run_id,
            tool_call_id=f"tool-{index}",
            duration=60,
            payload={"duration": 60},
        )
        assert await service.waits.schedule(wait.id)

    assert await service.cancel_all_followups("automation_disabled") == 2
    for index in range(2):
        await service._resume_pending(f"channel-{index}")
    assert await service.waits.resumable() == []
    assert service.tasks == {}
    async with store.database.transaction() as connection:
        rows = await (
            await connection.execute("SELECT state, failure FROM conversation_waits ORDER BY id")
        ).fetchall()
    assert [dict(row) for row in rows] == [
        {"state": "cancelled", "failure": "automation_disabled"},
        {"state": "cancelled", "failure": "automation_disabled"},
    ]
    await store.aclose()
