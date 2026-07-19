from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from diskovod.agent import build_agent
from diskovod.events import DiscordEventQueue
from diskovod.steering import LiveConversationMiddleware

from test_agent import RecordingGateway, ScriptedChatModel, prompt, runtime_context, tool_call


class InjectingModel(ScriptedChatModel):
    queue: object
    injected: bool = False

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if not self.injected:
            self.injected = True
            self.queue.ingest(
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


class InjectingGateway(RecordingGateway):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    async def send_messages(self, context, messages, *, tool_call_id):
        result = await super().send_messages(context, messages, tool_call_id=tool_call_id)
        self.queue.ingest(
            "steer-after-send",
            "channel",
            "message",
            {
                "message_id": "discord-after-send",
                "author_id": "peer",
                "author_name": "Peer",
                "participant_role": "peer",
                "content": "One more thing",
            },
        )
        return result


@pytest.mark.asyncio
async def test_new_input_cancels_unstarted_tool_and_returns_to_model(tmp_path: Path):
    queue = DiscordEventQueue(tmp_path / "diskovod.sqlite3")
    queue.thread_id("account", "channel")
    model = InjectingModel(
        responses=[
            tool_call("calculate", {"expression": "6 * 7"}, "stale-calculation"),
            AIMessage(content="reconsidered after steering"),
        ],
        queue=queue,
    )
    agent = build_agent(
        model,
        RecordingGateway(),
        prompt(),
        extra_middleware=[LiveConversationMiddleware(queue, "en")],
    )

    result = await agent.ainvoke(
        {
            "messages": [HumanMessage("Use 6 * 7")],
            "logical_request_id": "request-1",
        },
        context=runtime_context(),
    )

    cancellation = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "stale-calculation"
    )
    assert cancellation.status == "error"
    assert "new conversation input" in cancellation.content
    assert any(
        isinstance(message, HumanMessage) and message.content == "Actually, use 8 instead."
        for message in result["messages"]
    )
    assert model.index == 2
    assert result["claimed_event_ids"] == ["steer-1"]
    assert queue.claimed("channel", "request-1")[0].id == "steer-1"
    queue.close()


def test_recovery_reapplies_claimed_but_uncheckpointed_events(tmp_path: Path):
    queue = DiscordEventQueue(tmp_path / "diskovod.sqlite3")
    queue.thread_id("account", "channel")
    queue.ingest("event", "channel", "message", {"content": "recovered"})
    assert queue.claim_ready("channel", "request-1")[0].id == "event"
    middleware = LiveConversationMiddleware(queue, "en")

    class Runtime:
        context = runtime_context()

    update = middleware.before_model(
        {"messages": [], "logical_request_id": "request-1"},
        Runtime(),
    )

    assert update["claimed_event_ids"] == ["event"]
    assert update["messages"][0].content == "recovered"
    queue.close()


@pytest.mark.asyncio
async def test_live_input_wins_race_with_explicit_send_termination(tmp_path: Path):
    queue = DiscordEventQueue(tmp_path / "diskovod.sqlite3")
    queue.thread_id("account", "channel")
    gateway = InjectingGateway(queue)
    model = ScriptedChatModel(
        responses=[
            tool_call(
                "send_messages",
                {"messages": ["Initial final answer"], "continue_after_sending": False},
                "final-send",
            ),
            AIMessage(content="observed the racing input"),
        ]
    )
    agent = build_agent(
        model,
        gateway,
        prompt(),
        extra_middleware=[LiveConversationMiddleware(queue, "en")],
    )

    result = await agent.ainvoke(
        {
            "messages": [HumanMessage("Start")],
            "logical_request_id": "request-race",
        },
        context=runtime_context(),
    )

    assert model.index == 2
    assert any(
        isinstance(message, HumanMessage) and message.id == "discord-after-send"
        for message in result["messages"]
    )
    assert result["terminate_after_send"] is False
    queue.close()
