from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from diskovod.agent import build_agent
from diskovod.conversation import ConversationJournal
from diskovod.interaction import TriggerDecision, preset_policy
from diskovod.store import Store

from test_agent import RecordingGateway, ScriptedChatModel, UnusedPublicHTTP, prompt, runtime_context


class InjectingModel(ScriptedChatModel):
    journal: object
    injected: bool = False

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if not self.injected:
            self.injected = True
            await self.journal.admit(
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
                observed_at=1,
                schedule=False,
                trigger_kind="active_input",
                trigger_participant="peer",
                policy=preset_policy("shared"),
                policy_version=1,
                decision=TriggerDecision(False, "active_input"),
            )
        return result


def journal_injector(journal: ConversationJournal):
    async def inject(state, context):
        run_id = str(state.get("logical_request_id") or "")
        known = set(state.get("claimed_event_ids", []))
        recovered = [
            event for event in await journal.claimed(context.channel_id, run_id) if event.id not in known
        ]
        events = recovered + await journal.claim_injection(
            context.channel_id,
            run_id,
            injection_batch=1,
            participants=frozenset({"owner", "peer"}),
        )
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
    journal = ConversationJournal(store.database)
    await journal.thread_id("account", "channel")
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
        journal=journal,
    )
    agent = build_agent(
        model,
        RecordingGateway(),
        prompt(),
        UnusedPublicHTTP(),
        input_injector=journal_injector(journal),
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
    journal = ConversationJournal(store.database)
    await journal.thread_id("account", "channel")
    await journal.admit(
        "event",
        "channel",
        "message",
        {"message_id": "message", "content": "recovered", "participant_role": "peer"},
        observed_at=1,
        schedule=False,
        trigger_kind="active_input",
        trigger_participant="peer",
        policy=preset_policy("shared"),
        policy_version=1,
        decision=TriggerDecision(False, "active_input"),
    )
    assert (
        await journal.claim_injection(
            "channel", "request-1", injection_batch=1, participants=frozenset({"peer"})
        )
    )[0].id == "event"

    update = await journal_injector(journal)(
        {"messages": [], "logical_request_id": "request-1"}, runtime_context()
    )

    assert update["claimed_event_ids"] == ["event"]
    assert update["messages"][0].content == "recovered"
    await store.aclose()
