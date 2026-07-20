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


def wait_call(message: str = "One thought…") -> AIMessage:
    return AIMessage(
        content=message,
        tool_calls=[
            {
                "name": "wait_before_followup",
                "args": {"pause": "brief"},
                "id": "followup-1",
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
    await service.close()
    await store.aclose()
