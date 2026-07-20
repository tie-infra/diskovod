from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from diskovod.agent import AgentPrompt, OutboundDeliveryError, build_agent
from diskovod.agent_tools import localized_agent_tools
from diskovod.agent_types import AgentRuntimeContext, CapabilityProfile
from diskovod.localization import SUPPORTED_LOCALES, tool_text
from diskovod.outbound import DeliveryRecord


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "diskovod-scripted-test-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        del tools, tool_choice, kwargs
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del messages, stop, run_manager, kwargs
        response = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class RecordingGateway:
    def __init__(self, *, fail: bool = False, reaction_fail: bool = False):
        self.fail = fail
        self.reaction_fail = reaction_fail
        self.calls: list[tuple[str, tuple[str, ...], str]] = []
        self.escalations: list[dict[str, object]] = []

    async def publish_messages(self, context, messages, *, source_kind, source_id):
        self.calls.append((context.channel_id, messages, source_id))
        return [
            DeliveryRecord(
                status="failed" if self.fail else "accepted",
                message_index=index,
                discord_message_id=None if self.fail else f"discord-{len(self.calls)}-{index}",
                error_code="delivery_failed" if self.fail else None,
            )
            for index, _ in enumerate(messages)
        ]

    async def react(self, context, emoji, message_id, *, source_id):
        self.calls.append((context.channel_id, (f"reaction:{message_id}:{emoji}",), source_id))
        return DeliveryRecord(
            "failed" if self.reaction_fail else "accepted",
            0,
            discord_message_id=None if self.reaction_fail else f"reaction:{emoji}",
            error_code="discord_reaction_failed" if self.reaction_fail else None,
        )

    async def record_escalation(self, context, *, source_id, payload):
        self.escalations.append(payload)


class UnusedPublicHTTP:
    async def get(self, url, *, max_bytes, timeout=None):
        del url, max_bytes, timeout
        raise AssertionError("This test must not perform external HTTP requests")


def runtime_context(locale: str = "en") -> AgentRuntimeContext:
    return AgentRuntimeContext(
        account_id="account",
        channel_id="channel",
        participant_ids=("peer",),
        owner_id="owner",
        ui_locale=locale,
        prompt_locale=locale,
        assistant_name="Diskovod",
        automation_mode="inline",
        force_reply=False,
        provider_id="test",
        model_id="test",
        transport_profile="test",
        capabilities=CapabilityProfile(),
        trace_id="trace",
        thread_id="thread",
        trigger_message_id="trigger",
    )


def prompt(locale: str = "en") -> AgentPrompt:
    return AgentPrompt(locale, "Diskovod", "Be helpful.")


def tool_call(name: str, arguments: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": call_id, "type": "tool_call"}],
    )


@pytest.mark.asyncio
async def test_ordinary_final_text_is_sent_to_discord():
    gateway = RecordingGateway()
    model = ScriptedChatModel(responses=[AIMessage(content="internal final text")])
    agent = build_agent(model, gateway, prompt(), UnusedPublicHTTP())

    result = await agent.ainvoke(
        {"messages": [HumanMessage("hello")]},
        context=runtime_context(),
    )

    assert result["messages"][-1].content == "internal final text"
    assert [call[1] for call in gateway.calls] == [("internal final text",)]


@pytest.mark.asyncio
async def test_public_text_is_delivered_before_tool_work_and_the_final_reply():
    gateway = RecordingGateway()
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="I’ll check.",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "6 * 7"},
                        "id": "calculation",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="It’s 42."),
        ]
    )
    agent = build_agent(model, gateway, prompt(), UnusedPublicHTTP())

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Check 6 * 7")]},
        context=runtime_context(),
    )

    assert [call[1] for call in gateway.calls] == [("I’ll check.",), ("It’s 42.",)]
    assert model.index == 2
    assert result["outbound_delivery_count"] == 2
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert any(message.name == "calculate" and "42" in message.content for message in tool_messages)


@pytest.mark.asyncio
async def test_failed_publication_fails_before_tools_execute():
    gateway = RecordingGateway(fail=True)
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="Result",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "6 * 7"},
                        "id": "must-not-run",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    agent = build_agent(model, gateway, prompt(), UnusedPublicHTTP())

    with pytest.raises(OutboundDeliveryError):
        await agent.ainvoke(
            {"messages": [HumanMessage("hello")]},
            context=runtime_context(),
        )
    assert model.index == 1


@pytest.mark.asyncio
async def test_verified_hosted_search_is_bound_as_a_server_tool():
    model = ScriptedChatModel(responses=[AIMessage(content="server search completed")])
    traces: list[tuple[str, str, dict[str, Any]]] = []
    agent = build_agent(
        model,
        RecordingGateway(),
        prompt(),
        UnusedPublicHTTP(),
        hosted_web_search=True,
        diagnostics=lambda trace_id, kind, payload: traces.append((trace_id, kind, payload)),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Search for this")]},
        context=runtime_context(),
    )

    assert result["messages"][-1].content == "server search completed"
    request_trace = next(payload for _, kind, payload in traces if kind == "model_request")
    assert "send_messages" not in request_trace["tools"]
    assert "web_search" in request_trace["tools"]


def test_tool_schemas_and_validation_messages_are_localized():
    gateway = RecordingGateway()
    for locale in SUPPORTED_LOCALES:
        tools = {tool.name: tool for tool in localized_agent_tools(locale, gateway, UnusedPublicHTTP())}
        text = tool_text(locale)
        assert "send_messages" not in tools
        assert tools["react_to_message"].description == text["react"]
        reaction_schema = tools["react_to_message"].tool_call_schema.model_json_schema()
        assert set(reaction_schema["properties"]) == {"emoji"}
        escalation_schema = tools["escalate_to_owner"].tool_call_schema.model_json_schema()
        assert escalation_schema.get("properties", {}) == {}
        assert tools["react_to_message"].handle_validation_error(None) == text["invalid_arguments"]


@pytest.mark.asyncio
async def test_only_standard_public_text_blocks_are_delivered():
    gateway = RecordingGateway()
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content=[
                    {"type": "reasoning", "reasoning": "private chain"},
                    {"type": "text", "text": "Visible answer"},
                    {"type": "vendor_private", "value": "not public"},
                ]
            )
        ]
    )
    agent = build_agent(model, gateway, prompt(), UnusedPublicHTTP())

    await agent.ainvoke({"messages": [HumanMessage("hello")]}, context=runtime_context())

    assert [call[1] for call in gateway.calls] == [("Visible answer",)]


@pytest.mark.asyncio
async def test_successful_reaction_only_step_ends_without_an_extra_model_call():
    gateway = RecordingGateway()
    model = ScriptedChatModel(responses=[tool_call("react_to_message", {"emoji": "🎉"}, "reaction")])
    agent = build_agent(model, gateway, prompt(), UnusedPublicHTTP())

    await agent.ainvoke(
        {"messages": [HumanMessage("we did it", id="trigger")]},
        context=runtime_context(),
    )

    assert model.index == 1
    assert [call[1] for call in gateway.calls] == [("reaction:trigger:🎉",)]


@pytest.mark.asyncio
async def test_text_is_published_before_its_reaction_and_both_are_terminal():
    gateway = RecordingGateway()
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="Nice, that was a tricky one.",
                tool_calls=[
                    {
                        "name": "react_to_message",
                        "args": {"emoji": "🔥"},
                        "id": "reaction",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    agent = build_agent(model, gateway, prompt(), UnusedPublicHTTP())

    await agent.ainvoke({"messages": [HumanMessage("fixed", id="trigger")]}, context=runtime_context())

    assert [call[1] for call in gateway.calls] == [
        ("Nice, that was a tricky one.",),
        ("reaction:trigger:🔥",),
    ]
    assert model.index == 1


@pytest.mark.asyncio
async def test_failed_reaction_returns_control_to_the_model():
    gateway = RecordingGateway(reaction_fail=True)
    model = ScriptedChatModel(
        responses=[
            tool_call("react_to_message", {"emoji": "👍"}, "reaction"),
            AIMessage("I couldn't add the reaction, but yes."),
        ]
    )
    agent = build_agent(model, gateway, prompt(), UnusedPublicHTTP())

    await agent.ainvoke({"messages": [HumanMessage("okay?", id="trigger")]}, context=runtime_context())

    assert model.index == 2
    assert gateway.calls[-1][1] == ("I couldn't add the reaction, but yes.",)


@pytest.mark.asyncio
async def test_forced_reply_rejects_reaction_only_before_side_effects():
    gateway = RecordingGateway()
    agent = build_agent(
        ScriptedChatModel(responses=[tool_call("react_to_message", {"emoji": "👍"}, "reaction")]),
        gateway,
        prompt(),
        UnusedPublicHTTP(),
    )

    with pytest.raises(RuntimeError, match="forced reply"):
        await agent.ainvoke(
            {"messages": [HumanMessage("reply", id="trigger")]},
            context=replace(runtime_context(), force_reply=True),
        )
    assert gateway.calls == []
