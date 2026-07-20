from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from diskovod.agent import build_agent
from diskovod.localization import escalation_fallback
from diskovod.outbound import DeliveryRecord, OutboundPublisher
from diskovod.store import Store

from test_agent import (
    RecordingGateway,
    ScriptedChatModel,
    UnusedPublicHTTP,
    prompt,
    runtime_context,
    tool_call,
)


@pytest.mark.asyncio
async def test_invalid_escalation_uses_fixed_fallback_without_retry():
    gateway = RecordingGateway()
    model = ScriptedChatModel(
        responses=[tool_call("escalate_to_owner", {"reason": "made_up"}, "escalate-invalid")]
    )
    agent = build_agent(model, gateway, prompt(), UnusedPublicHTTP())

    result = await agent.ainvoke(
        {"messages": [HumanMessage("get the owner")]},
        context=runtime_context(),
    )

    assert model.index == 1
    assert gateway.calls[0][1] == (escalation_fallback("en"),)
    assert result["__interrupt__"]


@pytest.mark.asyncio
async def test_valid_escalation_without_text_uses_fallback_and_captures_context():
    gateway = RecordingGateway()
    model = ScriptedChatModel(responses=[tool_call("escalate_to_owner", {}, "escalate-without-text")])
    agent = build_agent(model, gateway, prompt(), UnusedPublicHTTP())

    result = await agent.ainvoke(
        {
            "messages": [HumanMessage("Please get the owner", id="trigger")],
            "logical_request_id": "run",
        },
        context=runtime_context(),
    )

    assert result["__interrupt__"]
    assert gateway.calls[0][1] == (escalation_fallback("en"),)
    assert gateway.escalations[0]["acknowledgement"] == escalation_fallback("en")
    assert gateway.escalations[0]["run_id"] == "run"
    assert gateway.escalations[0]["recent_conversation"][0]["text"] == "Please get the owner"


class EscalationTransport:
    def __init__(self):
        self.messages: list[str] = []

    async def send_messages(self, context, messages):
        self.messages.extend(messages)
        return [
            DeliveryRecord("accepted", index, discord_message_id=f"message-{len(self.messages)}-{index}")
            for index, _ in enumerate(messages)
        ]

    async def react_to_message(self, context, message_id, emoji):
        raise AssertionError("not used")


@pytest.mark.asyncio
async def test_valid_escalation_interrupts_and_resumes_without_resending(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = EscalationTransport()
    gateway = OutboundPublisher(store.database, transport)
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="I’ve marked this for the owner.",
                tool_calls=[
                    {
                        "name": "escalate_to_owner",
                        "args": {},
                        "id": "escalate-valid",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done after owner resolution"),
        ]
    )
    agent = build_agent(model, gateway, prompt(), UnusedPublicHTTP(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "discord:account:channel:g1"}}
    context = runtime_context()

    suspended = await agent.ainvoke(
        {"messages": [HumanMessage("get the owner")]},
        config=config,
        context=context,
    )
    assert suspended["__interrupt__"]
    assert transport.messages == ["I’ve marked this for the owner."]

    resumed = await agent.ainvoke(
        Command(resume={"action": "resolved"}),
        config=config,
        context=context,
    )

    assert resumed["messages"][-1].content == "done after owner resolution"
    assert transport.messages == [
        "I’ve marked this for the owner.",
        "done after owner resolution",
    ]
    async with store.database.transaction() as connection:
        row = await (await connection.execute("SELECT state FROM escalation_interrupts")).fetchone()
    assert row["state"] == "pending"
    await store.aclose()
