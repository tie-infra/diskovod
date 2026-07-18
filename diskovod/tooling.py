from __future__ import annotations

import ast
import json
import math
import operator
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import FunctionCall

ALLOWED_REACTIONS = frozenset(
    {"👍", "❤️", "😂", "🔥", "🎉", "😮", "😢", "🙏", "👀", "✅", "💯", "🤝", "👌", "😊", "😅", "🤔", "🙌"}
)
MAX_DISCORD_MESSAGE_LENGTH = 2000
MAX_ACTION_MESSAGES = 5

TOOL_SCHEMA_VERSION = "native-actions-hosted-search-v3"
MAX_HOSTED_WEB_SEARCH_CALLS = 2
ESCALATION_REASONS = frozenset({"peer_requested_owner", "owner_only_information", "other_explicit_request"})

FUNCTION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_current_datetime",
        "description": (
            "Return the current date, weekday, time, UTC offset, and timezone. Call this whenever "
            "the answer depends on today, tomorrow, a weekday, relative dates, or the exact time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": ["string", "null"],
                    "description": "IANA timezone name, or null to use the owner's configured timezone.",
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
        "description": "Evaluate a bounded arithmetic expression exactly enough for an ordinary DM reply.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "maxLength": 200}},
            "required": ["expression"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "send_messages",
        "description": (
            "Send one to five natural Discord DM messages. Usually send one short message. Use "
            "multiple messages only when the thoughts have natural chat boundaries. After web "
            "search, include useful source links conversationally rather than as a formal report."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "minItems": 1,
                    "maxItems": 5,
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
        "description": (
            "React instead of writing only when the latest message needs no written answer and a "
            "human would naturally acknowledge it with one emoji. Never react to a question, "
            "request, sensitive disclosure, conflict, or unclear context."
        ),
        "parameters": {
            "type": "object",
            "properties": {"emoji": {"type": "string", "enum": sorted(ALLOWED_REACTIONS)}},
            "required": ["emoji"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "escalate_to_owner",
        "description": (
            "Use only when the peer explicitly asks to involve, contact, or hand the conversation "
            "to the account owner. The acknowledgement must be a friendly, concise DM in the "
            "conversation language. It may say the conversation was marked for the owner, but must "
            "not claim the owner has read it, was externally notified, or will respond by any time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "enum": sorted(ESCALATION_REASONS)},
                "acknowledgement": {"type": "string", "minLength": 1, "maxLength": 2000},
            },
            "required": ["reason", "acknowledgement"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search",
    "search_context_size": "low",
}
FUNCTION_AND_WEB_TOOLS: list[dict[str, Any]] = [*FUNCTION_TOOLS, WEB_SEARCH_TOOL]


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
    now: datetime | None = None,
) -> dict[str, Any] | None:
    if call.name not in {"get_current_datetime", "calculate"}:
        return None
    arguments = call.parsed_arguments
    if arguments is None:
        return {"ok": False, "error": "invalid arguments"}
    if call.name == "get_current_datetime":
        timezone = arguments.get("timezone")
        if timezone is not None and not isinstance(timezone, str):
            return {"ok": False, "error": "timezone must be an IANA name or null"}
        zone_name = timezone or owner_timezone
        try:
            zone = ZoneInfo(zone_name)
        except (ZoneInfoNotFoundError, ValueError):
            return {"ok": False, "error": "unknown IANA timezone"}
        current = (now or datetime.now(zone)).astimezone(zone)
        return {
            "ok": True,
            "iso": current.isoformat(timespec="seconds"),
            "date": current.date().isoformat(),
            "time": current.time().isoformat(timespec="seconds"),
            "weekday": current.strftime("%A"),
            "utc_offset": current.strftime("%z")[:3] + ":" + current.strftime("%z")[3:],
            "timezone": zone_name,
        }
    if call.name == "calculate":
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not 1 <= len(expression) <= 200:
            return {"ok": False, "error": "expression must contain 1 to 200 characters"}
        try:
            value = _evaluate_expression(expression)
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
            return {"ok": False, "error": "invalid or unsupported arithmetic expression"}
        return {"ok": True, "result": value}
    return None


def validate_discord_action(
    call: FunctionCall,
    *,
    max_messages: int,
    allow_reaction: bool,
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


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_expression(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    count = 0

    def evaluate(node: ast.AST, depth: int = 0) -> int | float:
        nonlocal count
        count += 1
        if count > 50 or depth > 12:
            raise ValueError("expression is too complex")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left, depth + 1)
            right = evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("exponent is too large")
            value = _BINARY_OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            value = _UNARY_OPERATORS[type(node.op)](evaluate(node.operand, depth + 1))
        else:
            raise ValueError("unsupported expression")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 10**100:
            raise OverflowError("result is too large")
        return value

    return evaluate(tree)
