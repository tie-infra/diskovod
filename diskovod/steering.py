from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.runtime import Runtime

from .agent_types import AgentRuntimeContext, DiskovodAgentState
from .events import DiscordEventQueue, QueuedDiscordEvent
from .localization import runtime_context_text, tool_text


class LiveConversationMiddleware(AgentMiddleware[DiskovodAgentState, AgentRuntimeContext]):
    state_schema = DiskovodAgentState

    def __init__(self, queue: DiscordEventQueue, locale: str, *, max_batches: int = 4):
        self.queue = queue
        self.text = tool_text(locale)
        self.runtime_text = runtime_context_text(locale)
        self.max_batches = max_batches

    @hook_config(can_jump_to=["model"])
    def before_model(
        self,
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, Any] | None:
        return self._inject(state, runtime, cancel_tools=False)

    @hook_config(can_jump_to=["model"])
    async def abefore_model(
        self,
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, Any] | None:
        return self._inject(state, runtime, cancel_tools=True)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self,
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    def _inject(
        self,
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
        *,
        cancel_tools: bool,
    ) -> dict[str, Any] | None:
        request_id = state.get("logical_request_id")
        context = runtime.context
        if (
            not request_id
            or not self.queue.live_steering(context.channel_id)
            or state.get("live_injection_batches", 0) >= self.max_batches
        ):
            return None
        already_applied = set(state.get("claimed_event_ids", []))
        recovered = [
            event
            for event in self.queue.claimed(context.channel_id, request_id)
            if event.id not in already_applied
        ]
        claimed = self.queue.claim_ready(
            context.channel_id,
            request_id,
            injection_batch=state.get("live_injection_batches", 0) + 1,
        )
        events = recovered + claimed
        if not events:
            return None
        messages: list[Any] = []
        latest = state.get("messages", [])[-1] if state.get("messages") else None
        if cancel_tools and isinstance(latest, AIMessage):
            messages.extend(
                ToolMessage(
                    content=self.text["tool_cancelled_by_new_input"],
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
                for call in latest.tool_calls
            )
        messages.extend(self._message(event) for event in events)
        return {
            "messages": messages,
            "claimed_event_ids": [event.id for event in events],
            "live_injection_batches": 1,
            # Newly claimed conversation input wins a race with an explicit final send.
            "terminate_after_send": False,
            "jump_to": "model"
            if cancel_tools and isinstance(latest, AIMessage) and latest.tool_calls
            else None,
        }

    def _message(self, event: QueuedDiscordEvent):
        if event.kind == "delete":
            return RemoveMessage(id=str(event.payload["message_id"]))
        message_id = str(event.payload.get("message_id") or event.id)
        content = str(event.payload.get("content") or "")
        attachments = event.payload.get("attachments") or []
        if attachments:
            names = [
                str(item.get("filename") or "attachment") for item in attachments if isinstance(item, dict)
            ]
            content += "\n\n" + self.runtime_text["attachments"] + " " + ", ".join(names)
        return HumanMessage(
            content=content,
            id=message_id,
            additional_kwargs={
                "diskovod_participant": {
                    "id": str(event.payload.get("author_id") or "unknown"),
                    "name": str(event.payload.get("author_name") or self.runtime_text["unknown_participant"]),
                    "role": str(event.payload.get("participant_role") or "peer"),
                    "discord_event_id": event.id,
                    "observed_at": event.observed_at,
                    "edited": event.kind == "edit",
                },
                "diskovod_attachments": attachments,
            },
        )
