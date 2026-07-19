from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from diskovod.agent import build_agent
from diskovod.agent_actions import DeliveryRecord
from diskovod.durable_actions import DurableActionGateway, SideEffectLedger
from diskovod.localization import escalation_fallback
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
    assert result["successful_written_sends"] == 1


class EscalationTransport:
    def __init__(self):
        self.messages = 0

    async def send_messages(self, context, messages):
        self.messages += 1
        return [DeliveryRecord("accepted", 0, discord_message_id="acknowledgement")]

    async def react_to_message(self, context, message_id, emoji):
        raise AssertionError("not used")


@pytest.mark.asyncio
async def test_valid_escalation_interrupts_and_resumes_without_resending(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    ledger = SideEffectLedger(store.database)
    transport = EscalationTransport()
    gateway = DurableActionGateway(ledger, transport)
    model = ScriptedChatModel(
        responses=[
            tool_call(
                "escalate_to_owner",
                {
                    "reason": "peer_requested_owner",
                    "acknowledgement": "I’ve marked this for the owner.",
                },
                "escalate-valid",
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
    assert transport.messages == 1

    resumed = await agent.ainvoke(
        Command(resume={"action": "resolved"}),
        config=config,
        context=context,
    )

    assert resumed["messages"][-1].content == "done after owner resolution"
    assert transport.messages == 1
    async with store.database.transaction() as connection:
        row = await (await connection.execute("SELECT state FROM escalation_interrupts")).fetchone()
    assert row["state"] == "pending"
    await store.aclose()
