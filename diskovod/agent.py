from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolCallRequest,
    hook_config,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from langgraph.store.base import BaseStore
from langgraph.runtime import Runtime

from .agent_actions import AgentActionGateway
from .agent_tools import localized_agent_tools
from .agent_types import AgentRuntimeContext, DiskovodAgentState
from .localization import assistant_identity, escalation_fallback, prompts_for, tool_policy


@dataclass(frozen=True, slots=True)
class AgentPrompt:
    locale: str
    assistant_name: str
    base_instructions: str
    personality: str = ""
    owner_details: str = ""

    def stable_prefix(self) -> str:
        prompts = prompts_for(self.locale)
        parts = [
            assistant_identity(self.locale, self.assistant_name),
            self.base_instructions.strip() or prompts.base,
            prompts.dm_style,
            prompts.terminal_roleplay,
            tool_policy(self.locale),
        ]
        if self.personality.strip():
            parts.append(prompts.cached_personality.format(profile=self.personality.strip()))
        if self.owner_details.strip():
            parts.append(prompts.owner_details.format(details=self.owner_details.strip()))
        return "\n\n".join(parts)


class RuntimePromptMiddleware(AgentMiddleware[DiskovodAgentState, AgentRuntimeContext]):
    """Append only trusted invocation context after the stable cacheable prefix."""

    async def awrap_model_call(self, request, handler):
        context = request.runtime.context
        prompts = prompts_for(context.prompt_locale)
        suffix = [
            f"Automation mode: {context.automation_mode}.",
            f"Participant roles are supplied in trusted message metadata for channel {context.channel_id}.",
        ]
        if context.force_reply:
            suffix.append(prompts.forced_reply)
        system = request.system_message
        stable = system.text if system is not None else ""
        return await handler(
            request.override(system_message=SystemMessage(content=stable + "\n\n" + "\n".join(suffix)))
        )


class ExplicitSendTerminationMiddleware(AgentMiddleware[DiskovodAgentState, AgentRuntimeContext]):
    """End before another model call after a constrained successful final send."""

    state_schema = DiskovodAgentState

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, Any] | None:
        del runtime
        if state.get("terminate_after_send"):
            return {"jump_to": "end", "terminate_after_send": False}
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)


class EscalationValidationMiddleware(AgentMiddleware[DiskovodAgentState, AgentRuntimeContext]):
    """Apply the fixed fallback directly; malformed escalation is never repaired."""

    def __init__(self, gateway: AgentActionGateway, locale: str):
        self.gateway = gateway
        self.locale = locale

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        call = request.tool_call
        if call["name"] != "escalate_to_owner" or _valid_escalation_arguments(call["args"]):
            return await handler(request)
        fallback = escalation_fallback(self.locale)
        deliveries = await self.gateway.send_messages(
            request.runtime.context,
            (fallback,),
            tool_call_id=call["id"],
        )
        result = {
            "ok": False,
            "error": "invalid_arguments",
            "fallback_deliveries": [delivery.to_dict() for delivery in deliveries],
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=str(result),
                        tool_call_id=call["id"],
                        name=call["name"],
                        status="error",
                    )
                ],
                "successful_written_sends": int(
                    bool(deliveries) and all(item.accepted for item in deliveries)
                ),
                "terminate_after_send": True,
            }
        )


def build_agent(
    model: BaseChatModel,
    gateway: AgentActionGateway,
    prompt: AgentPrompt,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    store: BaseStore | None = None,
    model_call_limit: int = 12,
    tool_call_limit: int = 24,
    extra_middleware: Sequence[AgentMiddleware] = (),
):
    """Build Diskovod's provider-neutral LangChain agent loop."""
    return create_agent(
        model=model,
        tools=localized_agent_tools(prompt.locale, gateway),
        system_prompt=prompt.stable_prefix(),
        middleware=(
            *extra_middleware,
            RuntimePromptMiddleware(),
            EscalationValidationMiddleware(gateway, prompt.locale),
            ExplicitSendTerminationMiddleware(),
            ModelCallLimitMiddleware(run_limit=model_call_limit, exit_behavior="error"),
            ToolCallLimitMiddleware(run_limit=tool_call_limit, exit_behavior="error"),
        ),
        state_schema=DiskovodAgentState,
        context_schema=AgentRuntimeContext,
        checkpointer=checkpointer,
        store=store,
        name="diskovod",
    )


def _valid_escalation_arguments(arguments: object) -> bool:
    if not isinstance(arguments, dict) or set(arguments) != {"reason", "acknowledgement"}:
        return False
    acknowledgement = arguments["acknowledgement"]
    return (
        arguments["reason"] in {"peer_requested_owner", "owner_only_information", "other_explicit_request"}
        and isinstance(acknowledgement, str)
        and 1 <= len(acknowledgement.strip()) <= 2000
        and not any(ord(character) < 32 and character not in "\n\t" for character in acknowledgement)
    )
