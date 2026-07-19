from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from diskovod.agent import AgentPrompt, build_agent
from diskovod.agent_actions import DeliveryRecord
from diskovod.agent_tools import localized_agent_tools
from diskovod.agent_types import AgentRuntimeContext, CapabilityProfile
from diskovod.localization import SUPPORTED_LOCALES, tool_text


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
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[str, tuple[str, ...], str]] = []

    async def send_messages(self, context, messages, *, tool_call_id):
        self.calls.append((context.channel_id, messages, tool_call_id))
        return [
            DeliveryRecord(
                status="failed" if self.fail else "accepted",
                message_index=index,
                discord_message_id=None if self.fail else f"discord-{len(self.calls)}-{index}",
                error_code="delivery_failed" if self.fail else None,
            )
            for index, _ in enumerate(messages)
        ]

    async def react_to_message(self, context, emoji, *, tool_call_id):
        return DeliveryRecord("accepted", 0, discord_message_id=f"reaction:{emoji}")

    async def record_escalation(self, context, payload, *, tool_call_id):
        return None


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
    )


def prompt(locale: str = "en") -> AgentPrompt:
    return AgentPrompt(locale, "Diskovod", "Be helpful.")


def tool_call(name: str, arguments: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": call_id, "type": "tool_call"}],
    )


@pytest.mark.asyncio
async def test_ordinary_final_text_stops_without_sending_to_discord():
    gateway = RecordingGateway()
    model = ScriptedChatModel(responses=[AIMessage(content="internal final text")])
    agent = build_agent(model, gateway, prompt())

    result = await agent.ainvoke(
        {"messages": [HumanMessage("hello")]},
        context=runtime_context(),
    )

    assert result["messages"][-1].content == "internal final text"
    assert gateway.calls == []


@pytest.mark.asyncio
async def test_send_messages_returns_control_and_supports_explicit_final_send():
    gateway = RecordingGateway()
    model = ScriptedChatModel(
        responses=[
            tool_call(
                "send_messages",
                {"messages": ["I’ll check."], "continue_after_sending": True},
                "progress",
            ),
            tool_call("calculate", {"expression": "6 * 7"}, "calculation"),
            tool_call(
                "send_messages",
                {"messages": ["It’s 42."], "continue_after_sending": False},
                "final",
            ),
        ]
    )
    agent = build_agent(model, gateway, prompt())

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Check 6 * 7")]},
        context=runtime_context(),
    )

    assert [call[1] for call in gateway.calls] == [("I’ll check.",), ("It’s 42.",)]
    assert model.index == 3
    assert result["successful_written_sends"] == 2
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert any(message.name == "calculate" and "42" in message.content for message in tool_messages)
    assert '"termination_honored":true' in tool_messages[-1].content


@pytest.mark.asyncio
async def test_failed_final_send_does_not_terminate_the_agent():
    gateway = RecordingGateway(fail=True)
    model = ScriptedChatModel(
        responses=[
            tool_call(
                "send_messages",
                {"messages": ["Result"], "continue_after_sending": False},
                "failed-send",
            ),
            AIMessage(content="stop after observing the failure"),
        ]
    )
    agent = build_agent(model, gateway, prompt())

    result = await agent.ainvoke(
        {"messages": [HumanMessage("hello")]},
        context=runtime_context(),
    )

    assert model.index == 2
    assert result["successful_written_sends"] == 0
    tool_result = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert '"termination_honored":false' in tool_result.content


def test_tool_schemas_and_validation_messages_are_localized():
    gateway = RecordingGateway()
    for locale in SUPPORTED_LOCALES:
        tools = {tool.name: tool for tool in localized_agent_tools(locale, gateway)}
        text = tool_text(locale)
        assert tools["send_messages"].description == text["send_messages"]
        schema = tools["send_messages"].tool_call_schema.model_json_schema()
        assert schema["properties"]["messages"]["description"] == text["messages"]
        assert schema["properties"]["continue_after_sending"]["description"] == text["continue_after_sending"]
        assert tools["send_messages"].handle_validation_error(None) == text["invalid_arguments"]
