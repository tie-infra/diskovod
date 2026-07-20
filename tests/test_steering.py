from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from diskovod.agent import build_agent
from diskovod.mailbox import ConversationMailbox
from diskovod.store import Store

from test_agent import RecordingGateway, ScriptedChatModel, UnusedPublicHTTP, prompt, runtime_context


class InjectingModel(ScriptedChatModel):
    mailbox: object
    injected: bool = False

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if not self.injected:
            self.injected = True
            await self.mailbox.ingest(
                "steer-1",
                "channel",
                "message",
                {
                    "message_id": "discord-message-2",
                    "author_id": "peer",
                    "author_name": "Peer",
                    "participant_role": "peer",
                    "content": "Actually, use 8 instead.",
                },
            )
        return result


def mailbox_injector(mailbox: ConversationMailbox):
    async def inject(state, context):
        run_id = str(state.get("logical_request_id") or "")
        known = set(state.get("claimed_event_ids", []))
        recovered = [
            event for event in await mailbox.claimed(context.channel_id, run_id) if event.id not in known
        ]
        events = recovered + await mailbox.claim_ready(context.channel_id, run_id)
        if not events:
            return None
        return {
            "messages": [
                HumanMessage(str(event.payload.get("content") or ""), id=event.payload.get("message_id"))
                for event in events
            ],
            "claimed_event_ids": [event.id for event in events],
        }

    return inject


@pytest.mark.asyncio
async def test_new_input_is_injected_at_the_safe_point_after_tool_execution(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    mailbox = ConversationMailbox(store.database)
    await mailbox.thread_id("account", "channel")
    model = InjectingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "6 * 7"},
                        "id": "calculation",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="reconsidered after steering"),
        ],
        mailbox=mailbox,
    )
    agent = build_agent(
        model,
        RecordingGateway(),
        prompt(),
        UnusedPublicHTTP(),
        input_injector=mailbox_injector(mailbox),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Use 6 * 7")], "logical_request_id": "request-1"},
        context=runtime_context(),
    )

    calculation = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "calculation"
    )
    assert "42" in calculation.content
    assert any(
        isinstance(message, HumanMessage) and message.content == "Actually, use 8 instead."
        for message in result["messages"]
    )
    assert model.index == 2
    assert result["claimed_event_ids"] == ["steer-1"]
    await store.aclose()


@pytest.mark.asyncio
async def test_recovery_reapplies_claimed_but_uncheckpointed_events(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    mailbox = ConversationMailbox(store.database)
    await mailbox.thread_id("account", "channel")
    await mailbox.ingest(
        "event",
        "channel",
        "message",
        {"message_id": "message", "content": "recovered"},
    )
    assert (await mailbox.claim_ready("channel", "request-1"))[0].id == "event"

    update = await mailbox_injector(mailbox)(
        {"messages": [], "logical_request_id": "request-1"}, runtime_context()
    )

    assert update["claimed_event_ids"] == ["event"]
    assert update["messages"][0].content == "recovered"
    await store.aclose()
