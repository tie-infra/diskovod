from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .localization import inline_tool_text, tool_text
from .calculation import evaluate_expression
from .models import FunctionCall

ALLOWED_REACTIONS = frozenset(
    {"👍", "❤️", "😂", "🔥", "🎉", "😮", "😢", "🙏", "👀", "✅", "💯", "🤝", "👌", "😊", "😅", "🤔", "🙌"}
)
MAX_DISCORD_MESSAGE_LENGTH = 2000
MAX_ACTION_MESSAGES = 5

TOOL_SCHEMA_VERSION = "inline-collaboration-tools-v6"
MAX_HOSTED_WEB_SEARCH_CALLS = 2
ESCALATION_REASONS = frozenset({"peer_requested_owner", "owner_only_information", "other_explicit_request"})

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search",
    "search_context_size": "low",
}


def function_tools(locale: str, *, allow_silence: bool = False) -> list[dict[str, Any]]:
    text = tool_text(locale)
    tools = [
        {
            "type": "function",
            "name": "get_current_datetime",
            "description": text["current_datetime"],
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": ["string", "null"],
                        "description": text["timezone"],
                    }
                },
                "required": ["timezone"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "calculate",
            "description": text["calculate"],
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "maxLength": 200,
                        "description": text["expression"],
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "send_messages",
            "description": text["send_messages"],
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 2000},
                        "minItems": 1,
                        "maxItems": 5,
                        "description": text["messages"],
                    }
                },
                "required": ["messages"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "react_to_message",
            "description": text["react"],
            "parameters": {
                "type": "object",
                "properties": {
                    "emoji": {
                        "type": "string",
                        "enum": sorted(ALLOWED_REACTIONS),
                        "description": text["emoji"],
                    }
                },
                "required": ["emoji"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "escalate_to_owner",
            "description": text["escalate"],
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": sorted(ESCALATION_REASONS),
                        "description": text["escalation_reason"],
                    },
                    "acknowledgement": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                        "description": text["acknowledgement"],
                    },
                },
                "required": ["reason", "acknowledgement"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]
    if allow_silence:
        tools.append(
            {
                "type": "function",
                "name": "stay_silent",
                "description": inline_tool_text(locale)["stay_silent"],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        )
    return tools


def action_tools(
    locale: str,
    *,
    web_search: bool,
    allow_silence: bool = False,
) -> list[dict[str, Any]]:
    tools = function_tools(locale, allow_silence=allow_silence)
    return [*tools, WEB_SEARCH_TOOL] if web_search else tools


@dataclass(frozen=True, slots=True)
class DiscordAction:
    kind: str
    messages: tuple[str, ...] = ()
    emoji: str | None = None
    reason: str | None = None
    invalid_arguments: bool = False


def function_call_item(call: FunctionCall) -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": call.call_id,
        "name": call.name,
        "arguments": call.arguments,
    }


def function_output_item(call: FunctionCall, output: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": json.dumps(output, ensure_ascii=False, separators=(",", ":")),
    }


def execute_read_only_tool(
    call: FunctionCall,
    *,
    owner_timezone: str,
    locale: str = "en",
    now: datetime | None = None,
) -> dict[str, Any] | None:
    if call.name not in {"get_current_datetime", "calculate"}:
        return None
    text = tool_text(locale)
    arguments = call.parsed_arguments
    if arguments is None:
        return {"ok": False, "error": text["invalid_arguments"]}
    if call.name == "get_current_datetime":
        timezone = arguments.get("timezone")
        if timezone is not None and not isinstance(timezone, str):
            return {"ok": False, "error": text["timezone_type"]}
        zone_name = timezone or owner_timezone
        try:
            zone = ZoneInfo(zone_name)
        except (ZoneInfoNotFoundError, ValueError):
            return {"ok": False, "error": text["unknown_timezone"]}
        current = (now or datetime.now(zone)).astimezone(zone)
        return {
            "ok": True,
            "iso": current.isoformat(timespec="seconds"),
            "date": current.date().isoformat(),
            "time": current.time().isoformat(timespec="seconds"),
            "weekday": text["weekdays"][current.weekday()],
            "utc_offset": current.strftime("%z")[:3] + ":" + current.strftime("%z")[3:],
            "timezone": zone_name,
        }
    if call.name == "calculate":
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not 1 <= len(expression) <= 200:
            return {"ok": False, "error": text["expression_length"]}
        try:
            value = evaluate_expression(expression)
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
            return {"ok": False, "error": text["invalid_expression"]}
        return {"ok": True, "result": value}
    return None


def validate_discord_action(
    call: FunctionCall,
    *,
    max_messages: int,
    allow_reaction: bool,
    allow_silence: bool = False,
) -> DiscordAction | None:
    arguments = call.parsed_arguments
    if arguments is None:
        return None
    if call.name == "send_messages" and set(arguments) == {"messages"}:
        messages = arguments["messages"]
        if not isinstance(messages, list) or not 1 <= len(messages) <= min(max_messages, 5):
            return None
        normalized: list[str] = []
        for message in messages:
            if not isinstance(message, str):
                return None
            value = message.strip()
            if (
                not value
                or len(value) > MAX_DISCORD_MESSAGE_LENGTH
                or any(ord(character) < 32 and character not in "\n\t" for character in value)
            ):
                return None
            normalized.append(value)
        return DiscordAction("messages", tuple(normalized))
    if call.name == "react_to_message" and set(arguments) == {"emoji"}:
        emoji = arguments["emoji"]
        if allow_reaction and emoji in ALLOWED_REACTIONS:
            return DiscordAction("reaction", emoji=emoji)
    if call.name == "stay_silent" and allow_silence and not arguments:
        return DiscordAction("silent")
    return None


def validate_escalation_action(call: FunctionCall, fallback: str) -> DiscordAction | None:
    if call.name != "escalate_to_owner":
        return None
    arguments = call.parsed_arguments
    if arguments is not None and set(arguments) == {"reason", "acknowledgement"}:
        reason = arguments["reason"]
        acknowledgement = arguments["acknowledgement"]
        if (
            reason in ESCALATION_REASONS
            and isinstance(acknowledgement, str)
            and 1 <= len(acknowledgement.strip()) <= MAX_DISCORD_MESSAGE_LENGTH
            and not any(ord(character) < 32 and character not in "\n\t" for character in acknowledgement)
        ):
            return DiscordAction(
                "escalation",
                (acknowledgement.strip(),),
                reason=str(reason),
            )
    return DiscordAction(
        "escalation",
        (fallback,),
        reason="invalid_tool_arguments",
        invalid_arguments=True,
    )


def validate_hosted_web_search_calls(
    calls: list[Any],
    *,
    enabled: bool,
) -> bool:
    if not calls:
        return True
    if not enabled or len(calls) > MAX_HOSTED_WEB_SEARCH_CALLS:
        return False
    return all(call.kind == "web_search_call" and call.status == "completed" for call in calls)
