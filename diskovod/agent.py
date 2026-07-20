from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any
import json
import uuid

from langchain.agents.middleware import (
    AgentMiddleware,
    SummarizationMiddleware,
    ToolCallRequest,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langgraph.store.base import BaseStore
from langgraph.runtime import Runtime

from .agent_tools import FollowupScheduler, localized_agent_tools
from .agent_types import AgentRuntimeContext, DiskovodAgentState
from .attachments import AttachmentRepository
from .http_client import PublicHTTP
from .localization import (
    assistant_identity,
    escalation_fallback,
    prompts_for,
    runtime_context_text,
    summarization_prompt,
    tool_policy,
    tool_text,
)
from .outbound import OutboundActions


@dataclass(frozen=True, slots=True)
class AgentPrompt:
    locale: str
    assistant_name: str
    base_instructions: str
    personality: str = ""
    owner_details: str = ""

    def stable_prefix(self, *, allow_conversational_followups: bool = False) -> str:
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
        if allow_conversational_followups:
            parts.append(tool_text(self.locale)["followup_policy"])
        return "\n\n".join(parts)


class LocalTracingMiddleware(AgentMiddleware[DiskovodAgentState, AgentRuntimeContext]):
    """Capture exact normalized model/tool exchanges in the local correlated trace."""

    def __init__(self, diagnostics: Callable[[str, str, dict[str, Any]], None]):
        self.diagnostics = diagnostics

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        context = request.runtime.context
        self.diagnostics(
            context.trace_id,
            "tool_request",
            _trace_value(request.tool_call),
        )
        try:
            response = await handler(request)
        except Exception as error:
            self.diagnostics(
                context.trace_id,
                "tool_error",
                {
                    "tool_call": _trace_value(request.tool_call),
                    "type": type(error).__name__,
                    "detail": str(error)[:8000],
                },
            )
            raise
        self.diagnostics(
            context.trace_id,
            "tool_response",
            {"tool_call": _trace_value(request.tool_call), "response": _trace_value(response)},
        )
        return response


def build_agent(
    model: BaseChatModel,
    gateway: OutboundActions,
    prompt: AgentPrompt,
    http: PublicHTTP,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    store: BaseStore | None = None,
    model_call_limit: int = 12,
    tool_call_limit: int = 24,
    input_injector: Callable[[DiskovodAgentState, AgentRuntimeContext], Awaitable[dict[str, Any] | None]]
    | None = None,
    attachments: AttachmentRepository | None = None,
    diagnostics: Callable[[str, str, dict[str, Any]], None] | None = None,
    hosted_web_search: bool = False,
    followup_scheduler: FollowupScheduler | None = None,
    native_tools: bool = True,
):
    """Build Diskovod's explicit provider-neutral conversation graph."""
    client_tools = (
        localized_agent_tools(
            prompt.locale,
            gateway,
            http,
            attachments,
            followup_scheduler=followup_scheduler,
            include_web_search=not hosted_web_search,
        )
        if native_tools
        else []
    )
    model_tools: list[Any] = list(client_tools)
    if hosted_web_search:
        model_tools.append({"type": "web_search"})
    bound_model = model.bind_tools(model_tools) if model_tools else model
    summarizer = SummarizationMiddleware(
        model,
        trigger=("messages", 80),
        keep=("messages", 30),
        summary_prompt=summarization_prompt(prompt.locale),
    )
    tracer = LocalTracingMiddleware(diagnostics) if diagnostics is not None else None

    async def input_node(
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, Any] | None:
        if input_injector is None:
            return None
        return await input_injector(state, runtime.context)

    async def summarize_node(
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, Any] | None:
        return await summarizer.abefore_model(state, runtime)

    async def model_node(
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, Any]:
        request_id = str(state.get("logical_request_id") or "")
        same_run = state.get("counter_run_id") == request_id
        count = int(state.get("model_call_count", 0)) if same_run else 0
        if count >= model_call_limit:
            raise RuntimeError(f"Model call limit exceeded ({model_call_limit})")
        messages = _attachment_messages(state.get("messages", []), runtime.context)
        system = _runtime_system_message(prompt, runtime.context)
        if diagnostics is not None:
            diagnostics(
                runtime.context.trace_id,
                "model_request",
                {
                    "model_class": type(model).__name__,
                    "system_message": _trace_value(system),
                    "messages": [_trace_value(message) for message in messages],
                    "tools": [_trace_tool_name(tool) for tool in model_tools],
                    "model_settings": {},
                    "observability": "normalized_langchain_exchange; raw_provider_transport_unavailable",
                },
            )
        try:
            response = await bound_model.ainvoke([system, *messages])
        except Exception as error:
            if diagnostics is not None:
                diagnostics(
                    runtime.context.trace_id,
                    "model_error",
                    {"type": type(error).__name__, "detail": str(error)[:8000]},
                )
            raise
        if not isinstance(response, AIMessage):
            raise TypeError(f"Chat model returned {type(response).__name__}, expected AIMessage")
        if not response.id:
            response.id = f"ai:{uuid.uuid4()}"
        if diagnostics is not None:
            diagnostics(runtime.context.trace_id, "model_response", _trace_value(response))
        update = {
            "messages": [response],
            "model_call_count": count + 1,
            "counter_run_id": request_id,
            "model_step_route": "validate",
        }
        if not same_run:
            update["tool_call_count"] = 0
        return update

    async def validate_node(
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, Any]:
        latest = _latest_ai_message(state)
        calls = latest.tool_calls
        request_id = str(state.get("logical_request_id") or "")
        count = int(state.get("tool_call_count", 0)) if state.get("counter_run_id") == request_id else 0
        if count + len(calls) > tool_call_limit:
            raise RuntimeError(f"Tool call limit exceeded ({tool_call_limit})")
        escalations = [call for call in calls if call["name"] == "escalate_to_owner"]
        if escalations:
            if len(calls) != 1:
                raise RuntimeError("escalate_to_owner must be the only tool call in a model step")
            if escalations[0].get("args") != {}:
                return {
                    "tool_call_count": count + 1,
                    "counter_run_id": request_id,
                    "model_step_route": "malformed_escalation",
                }
        waits = [call for call in calls if call["name"] == "wait_before_followup"]
        if waits and (len(waits) != 1 or len(calls) != 1):
            raise RuntimeError("wait_before_followup must be the only tool call in a model step")
        if (
            runtime.context.force_reply
            and not _public_text(latest)
            and calls
            and all(call["name"] == "react_to_message" for call in calls)
        ):
            raise RuntimeError("A forced reply requires public assistant text")
        if runtime.context.force_reply and not _public_text(latest) and not calls:
            raise RuntimeError("A forced reply requires public assistant text")
        return {
            "tool_call_count": count + len(calls),
            "counter_run_id": request_id,
            "model_step_route": "publish",
        }

    async def publish_node(
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, Any]:
        latest = _latest_ai_message(state)
        content = _public_text(latest)
        if not content:
            return {"model_step_route": "tools" if latest.tool_calls else "end"}
        records = await gateway.publish_messages(
            runtime.context,
            (content,),
            source_kind="assistant_text",
            source_id=str(latest.id),
        )
        if not records or not all(record.accepted for record in records):
            raise OutboundDeliveryError(records)
        return {
            "outbound_delivery_count": len(records),
            "model_step_route": "tools" if latest.tool_calls else "end",
        }

    async def malformed_escalation_node(
        state: DiskovodAgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, Any]:
        latest = _latest_ai_message(state)
        call = latest.tool_calls[0]
        fallback = escalation_fallback(prompt.locale)
        records = await gateway.publish_messages(
            runtime.context,
            (fallback,),
            source_kind="localized_fallback",
            source_id=f"{call['id']}:malformed",
        )
        if not records or not all(record.accepted for record in records):
            raise OutboundDeliveryError(records)
        payload: dict[str, object] = {
            "channel_id": runtime.context.channel_id,
            "thread_id": runtime.context.thread_id,
            "run_id": str(state.get("logical_request_id") or ""),
            "trigger_message_id": runtime.context.trigger_message_id,
            "trace_id": runtime.context.trace_id,
            "tool_call_id": call["id"],
            "acknowledgement": fallback,
            "malformed_tool_call": True,
            "arguments": _trace_value(call.get("args")),
            "participant_ids": list(runtime.context.participant_ids),
            "recent_conversation": [
                {
                    "type": message.type,
                    "id": str(message.id or ""),
                    "text": message.text[:2000],
                }
                for message in state.get("messages", [])[-12:]
            ],
        }
        await gateway.record_escalation(runtime.context, source_id=call["id"], payload=payload)
        resolution = interrupt(payload)
        return {
            "messages": [
                ToolMessage(
                    json.dumps({"ok": True, "resolution": resolution}, ensure_ascii=False),
                    tool_call_id=call["id"],
                    name="escalate_to_owner",
                )
            ],
            "outbound_delivery_count": len(records),
        }

    async def trace_tool_call(request: ToolCallRequest, handler):
        if tracer is None:
            return await handler(request)
        return await tracer.awrap_tool_call(request, handler)

    tool_node = ToolNode(client_tools, awrap_tool_call=trace_tool_call)

    def after_validation(state: DiskovodAgentState) -> str:
        return str(state.get("model_step_route") or "publish")

    def after_publication(state: DiskovodAgentState) -> str:
        return str(state.get("model_step_route") or "end")

    def after_tools(state: DiskovodAgentState) -> str:
        latest_ai = _latest_ai_message(state)
        if len(latest_ai.tool_calls) != 1 or latest_ai.tool_calls[0]["name"] != "react_to_message":
            return "input"
        call_id = latest_ai.tool_calls[0]["id"]
        result = next(
            (
                message
                for message in reversed(state.get("messages", []))
                if isinstance(message, ToolMessage) and message.tool_call_id == call_id
            ),
            None,
        )
        if result is not None and _tool_result_ok(result):
            return "end"
        return "input"

    builder = StateGraph(DiskovodAgentState, context_schema=AgentRuntimeContext)
    builder.add_node("input", input_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("model", model_node)
    builder.add_node("validate", validate_node)
    builder.add_node("publish", publish_node)
    builder.add_node("malformed_escalation", malformed_escalation_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "input")
    builder.add_edge("input", "summarize")
    builder.add_edge("summarize", "model")
    builder.add_edge("model", "validate")
    builder.add_conditional_edges(
        "validate",
        after_validation,
        {"publish": "publish", "malformed_escalation": "malformed_escalation"},
    )
    builder.add_conditional_edges("publish", after_publication, {"tools": "tools", "end": END})
    builder.add_conditional_edges("tools", after_tools, {"input": "input", "end": END})
    builder.add_edge("malformed_escalation", END)
    return builder.compile(checkpointer=checkpointer, store=store, name="diskovod")


class OutboundDeliveryError(RuntimeError):
    """A visible assistant action did not reach a definitive successful state."""

    def __init__(self, records: Sequence[Any]):
        self.records = tuple(records)
        statuses = ", ".join(str(getattr(record, "status", "missing")) for record in records)
        super().__init__(f"Outbound delivery did not succeed: {statuses or 'missing result'}")


def _latest_ai_message(state: DiskovodAgentState) -> AIMessage:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return message
    raise RuntimeError("Conversation graph has no AIMessage for this model step")


def _public_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    parts: list[str] = []
    for block in message.content:
        if not isinstance(block, dict) or block.get("type") not in {"text", "output_text"}:
            continue
        value = block.get("text")
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts).strip()


def _tool_result_ok(message: ToolMessage) -> bool:
    content = message.content
    if isinstance(content, str):
        try:
            value = json.loads(content)
        except (TypeError, ValueError):
            return False
        return isinstance(value, dict) and value.get("ok") is True
    return False


def _runtime_system_message(prompt: AgentPrompt, context: AgentRuntimeContext) -> SystemMessage:
    prompts = prompts_for(context.prompt_locale)
    runtime_text = runtime_context_text(context.prompt_locale)
    mode = runtime_text.get(f"mode_{context.automation_mode}", context.automation_mode)
    suffix = [
        runtime_text["mode"].format(mode=mode),
        runtime_text["participants"].format(channel=context.channel_id),
    ]
    if context.force_reply:
        suffix.append(prompts.forced_reply)
    return SystemMessage(
        content=prompt.stable_prefix(allow_conversational_followups=context.allow_conversational_followups)
        + "\n\n"
        + "\n".join(suffix)
    )


def _attachment_messages(
    messages: Sequence[BaseMessage],
    context: AgentRuntimeContext,
) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for message in messages:
        if not isinstance(message, HumanMessage) or message.id != context.trigger_message_id:
            result.append(message)
            continue
        attachments = message.additional_kwargs.get("diskovod_attachments") or []
        blocks: list[dict[str, Any]] = [{"type": "text", "text": message.text}]
        for attachment in attachments:
            if not isinstance(attachment, dict) or not attachment.get("url"):
                continue
            media_type = str(attachment.get("content_type") or "")
            if context.capabilities.image_input and media_type.startswith("image/"):
                blocks.append({"type": "image", "url": str(attachment["url"]), "mime_type": media_type})
            elif context.capabilities.file_input:
                blocks.append({"type": "file", "url": str(attachment["url"]), "mime_type": media_type})
        result.append(message.model_copy(update={"content": blocks}) if len(blocks) > 1 else message)
    return result


def _trace_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _trace_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_value(item) for item in value]
    return str(value)


def _trace_tool_name(tool: Any) -> str:
    """Return a stable diagnostic name for client and provider-hosted tools."""
    if isinstance(tool, dict):
        name = tool.get("name") or tool.get("type")
    else:
        name = getattr(tool, "name", None)
    return str(name) if name else type(tool).__name__
