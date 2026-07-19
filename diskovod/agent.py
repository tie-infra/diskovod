from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    hook_config,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.runtime import Runtime

from .agent_actions import AgentActionGateway
from .agent_tools import localized_agent_tools
from .agent_types import AgentRuntimeContext, DiskovodAgentState
from .localization import assistant_identity, prompts_for, tool_policy


@dataclass(frozen=True, slots=True)
class AgentPrompt:
    locale: str
    assistant_name: str
    base_instructions: str
    personality: str = ""

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
        return "\n\n".join(parts)


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


def build_agent(
    model: BaseChatModel,
    gateway: AgentActionGateway,
    prompt: AgentPrompt,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    store: BaseStore | None = None,
    model_call_limit: int = 12,
    tool_call_limit: int = 24,
):
    """Build Diskovod's provider-neutral LangChain agent loop."""
    return create_agent(
        model=model,
        tools=localized_agent_tools(prompt.locale, gateway),
        system_prompt=prompt.stable_prefix(),
        middleware=(
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
