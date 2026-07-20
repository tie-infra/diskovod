from datetime import datetime
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from pydantic import Field, StringConstraints

from .agent_types import AgentRuntimeContext, DiskovodAgentState
from .attachments import AttachmentRepository
from .calculation import evaluate_expression
from .http_client import PublicHTTP
from .localization import escalation_fallback, tool_text
from .outbound import DeliveryRecord, OutboundActions
from .web_access import WebAccessError, fetch_url as fetch_public_url, search_web as search_public_web


MessageText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
ExpressionText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
ReactionEmoji = Literal[
    "👍", "❤️", "😂", "🔥", "🎉", "😮", "😢", "🙏", "👀", "✅", "💯", "🤝", "👌", "😊", "😅", "🤔", "🙌"
]


def localized_agent_tools(
    locale: str,
    outbound: OutboundActions,
    http: PublicHTTP,
    attachments: AttachmentRepository | None = None,
    *,
    include_web_search: bool = True,
) -> list[BaseTool]:
    text = tool_text(locale)
    invalid = lambda _: text["invalid_arguments"]  # noqa: E731

    def delivery_result(delivery: DeliveryRecord) -> dict[str, object]:
        result = delivery.to_dict()
        if delivery.error_code:
            result["error"] = text.get(
                f"delivery_error_{delivery.error_code}",
                text["delivery_error_unknown"],
            )
        return result

    async def get_current_datetime(
        timezone: Annotated[str | None, Field(description=text["timezone"])],
        runtime: ToolRuntime[AgentRuntimeContext, DiskovodAgentState],
    ) -> dict[str, Any]:
        zone_name = timezone or runtime.context.owner_timezone
        try:
            zone = ZoneInfo(zone_name)
        except (ZoneInfoNotFoundError, ValueError):
            return {"ok": False, "error": text["unknown_timezone"]}
        current = datetime.now(zone)
        offset = current.strftime("%z")
        return {
            "ok": True,
            "iso": current.isoformat(timespec="seconds"),
            "date": current.date().isoformat(),
            "time": current.time().isoformat(timespec="seconds"),
            "weekday": text["weekdays"][current.weekday()],
            "utc_offset": offset[:3] + ":" + offset[3:],
            "timezone": zone_name,
        }

    async def calculate(
        expression: Annotated[ExpressionText, Field(description=text["expression"])],
        runtime: ToolRuntime[AgentRuntimeContext, DiskovodAgentState],
    ) -> dict[str, Any]:
        del runtime
        try:
            value = evaluate_expression(expression)
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
            return {"ok": False, "error": text["invalid_expression"]}
        return {"ok": True, "result": value}

    async def web_search(
        query: Annotated[MessageText, Field(description=text["web_query"])],
        runtime: ToolRuntime[AgentRuntimeContext, DiskovodAgentState],
    ) -> dict[str, Any]:
        del runtime
        try:
            results = await search_public_web(http, query)
        except WebAccessError as error:
            return {"ok": False, "error": text["web_error"], "code": str(error)}
        return {"ok": True, "results": results}

    async def fetch_url(
        url: Annotated[MessageText, Field(description=text["url"])],
        runtime: ToolRuntime[AgentRuntimeContext, DiskovodAgentState],
    ) -> dict[str, Any]:
        del runtime
        try:
            result = await fetch_public_url(http, url)
        except WebAccessError as error:
            return {"ok": False, "error": text["web_error"], "code": str(error)}
        return {"ok": True, **result}

    async def search_chat_attachments(
        query: Annotated[MessageText, Field(description=text["attachment_query"])],
        runtime: ToolRuntime[AgentRuntimeContext, DiskovodAgentState],
    ) -> dict[str, Any]:
        if attachments is None:
            return {"ok": False, "error": text["attachment_unavailable"]}
        results = await attachments.search(runtime.context.channel_id, query)
        return {"ok": True, "results": results}

    async def search_chat_memory(
        query: Annotated[MessageText, Field(description=text["memory_query"])],
        runtime: ToolRuntime[AgentRuntimeContext, DiskovodAgentState],
    ) -> dict[str, Any]:
        if runtime.store is None:
            return {"ok": False, "error": text["memory_unavailable"]}
        namespace = ("chat", runtime.context.account_id, runtime.context.channel_id, "memory")
        items = await runtime.store.asearch(namespace, query=query, limit=8)
        return {
            "ok": True,
            "memories": [{"key": item.key, "value": item.value} for item in items],
        }

    async def remember_chat_memory(
        key: Annotated[MessageText, Field(description=text["memory_key"])],
        value: Annotated[MessageText, Field(description=text["memory_value"])],
        runtime: ToolRuntime[AgentRuntimeContext, DiskovodAgentState],
    ) -> dict[str, Any]:
        if runtime.store is None:
            return {"ok": False, "error": text["memory_unavailable"]}
        namespace = ("chat", runtime.context.account_id, runtime.context.channel_id, "memory")
        normalized_key = "-".join(key.casefold().split())[:120]
        await runtime.store.aput(
            namespace,
            normalized_key,
            {
                "fact": value,
                "source_message_id": runtime.context.trigger_message_id,
                "trace_id": runtime.context.trace_id,
            },
        )
        return {"ok": True, "key": normalized_key}

    async def forget_chat_memory(
        key: Annotated[MessageText, Field(description=text["memory_key"])],
        runtime: ToolRuntime[AgentRuntimeContext, DiskovodAgentState],
    ) -> dict[str, Any]:
        if runtime.store is None:
            return {"ok": False, "error": text["memory_unavailable"]}
        namespace = ("chat", runtime.context.account_id, runtime.context.channel_id, "memory")
        normalized_key = "-".join(key.casefold().split())[:120]
        await runtime.store.adelete(namespace, normalized_key)
        return {"ok": True, "key": normalized_key}

    async def react_to_message(
        emoji: Annotated[ReactionEmoji, Field(description=text["emoji"])],
        runtime: ToolRuntime[AgentRuntimeContext, DiskovodAgentState],
    ) -> dict[str, Any]:
        call_id = runtime.tool_call_id or "missing-tool-call-id"
        target = str(runtime.state.get("reaction_target_message_id") or runtime.context.trigger_message_id)
        delivery = await outbound.react(
            runtime.context,
            emoji,
            target,
            source_id=call_id,
        )
        return {"ok": delivery.accepted, "delivery": delivery_result(delivery)}

    async def escalate_to_owner(
        runtime: ToolRuntime[AgentRuntimeContext, DiskovodAgentState],
    ) -> dict[str, Any]:
        call_id = runtime.tool_call_id or "missing-tool-call-id"
        latest = next(
            (
                message
                for message in reversed(runtime.state.get("messages", []))
                if isinstance(message, AIMessage)
                and any(call.get("id") == call_id for call in message.tool_calls)
            ),
            None,
        )
        acknowledgement = latest.text.strip() if latest is not None else ""
        if not acknowledgement:
            acknowledgement = escalation_fallback(locale)
            deliveries = await outbound.publish_messages(
                runtime.context,
                (acknowledgement,),
                source_kind="localized_fallback",
                source_id=f"{call_id}:acknowledgement",
            )
            if not deliveries or not all(delivery.accepted for delivery in deliveries):
                return {"ok": False, "deliveries": [delivery_result(item) for item in deliveries]}
        payload: dict[str, object] = {
            "channel_id": runtime.context.channel_id,
            "trigger_message_id": runtime.context.trigger_message_id,
            "trace_id": runtime.context.trace_id,
            "acknowledgement": acknowledgement,
            "tool_call_id": call_id,
        }
        await outbound.record_escalation(runtime.context, source_id=call_id, payload=payload)
        resolution = interrupt(payload)
        return {
            "ok": True,
            "resolution": resolution,
        }

    tools = [
        StructuredTool.from_function(
            coroutine=get_current_datetime,
            name="get_current_datetime",
            description=text["current_datetime"],
            handle_validation_error=invalid,
        ),
        StructuredTool.from_function(
            coroutine=calculate,
            name="calculate",
            description=text["calculate"],
            handle_validation_error=invalid,
        ),
        StructuredTool.from_function(
            coroutine=web_search,
            name="web_search",
            description=text["web_search"],
            handle_validation_error=invalid,
        ),
        StructuredTool.from_function(
            coroutine=fetch_url,
            name="fetch_url",
            description=text["fetch_url"],
            handle_validation_error=invalid,
        ),
        StructuredTool.from_function(
            coroutine=search_chat_attachments,
            name="search_chat_attachments",
            description=text["attachment_search"],
            handle_validation_error=invalid,
        ),
        StructuredTool.from_function(
            coroutine=search_chat_memory,
            name="search_chat_memory",
            description=text["memory_search"],
            handle_validation_error=invalid,
        ),
        StructuredTool.from_function(
            coroutine=remember_chat_memory,
            name="remember_chat_memory",
            description=text["remember_memory"],
            handle_validation_error=invalid,
        ),
        StructuredTool.from_function(
            coroutine=forget_chat_memory,
            name="forget_chat_memory",
            description=text["forget_memory"],
            handle_validation_error=invalid,
        ),
        StructuredTool.from_function(
            coroutine=react_to_message,
            name="react_to_message",
            description=text["react"],
            handle_validation_error=invalid,
        ),
        StructuredTool.from_function(
            coroutine=escalate_to_owner,
            name="escalate_to_owner",
            description=text["escalate"],
            handle_validation_error=invalid,
        ),
    ]
    if not include_web_search:
        tools = [tool for tool in tools if tool.name != "web_search"]
    return tools
