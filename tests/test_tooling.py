import json
from datetime import UTC, datetime

from diskovod.models import FunctionCall
from diskovod.tooling import (
    execute_read_only_tool,
    validate_discord_action,
    validate_escalation_action,
)


def call(name: str, arguments: object) -> FunctionCall:
    encoded = json.dumps(arguments)
    return FunctionCall("call", name, encoded, arguments if isinstance(arguments, dict) else None)


def test_current_datetime_uses_owner_timezone_without_prompt_injection():
    result = execute_read_only_tool(
        call("get_current_datetime", {"timezone": None}),
        owner_timezone="Europe/Moscow",
        now=datetime(2026, 7, 19, 12, 30, tzinfo=UTC),
    )

    assert result == {
        "ok": True,
        "iso": "2026-07-19T15:30:00+03:00",
        "date": "2026-07-19",
        "time": "15:30:00",
        "weekday": "Sunday",
        "utc_offset": "+03:00",
        "timezone": "Europe/Moscow",
    }


def test_calculator_accepts_arithmetic_and_rejects_code():
    assert execute_read_only_tool(
        call("calculate", {"expression": "(12 + 3) * 4 / 2"}), owner_timezone="UTC"
    ) == {"ok": True, "result": 30.0}
    rejected = execute_read_only_tool(
        call("calculate", {"expression": "__import__('os').system('id')"}),
        owner_timezone="UTC",
    )
    assert rejected == {"ok": False, "error": "invalid or unsupported arithmetic expression"}


def test_message_action_validation_enforces_runtime_limit_and_unknown_fields():
    valid = validate_discord_action(
        call("send_messages", {"messages": ["one", "two"]}),
        max_messages=2,
        allow_reaction=True,
    )
    too_many = validate_discord_action(
        call("send_messages", {"messages": ["one", "two"]}),
        max_messages=1,
        allow_reaction=True,
    )
    unknown = validate_discord_action(
        call("send_messages", {"messages": ["one"], "silent": True}),
        max_messages=2,
        allow_reaction=True,
    )

    assert valid and valid.messages == ("one", "two")
    assert too_many is None
    assert unknown is None


def test_reaction_action_respects_runtime_availability():
    assert validate_discord_action(
        call("react_to_message", {"emoji": "👍"}),
        max_messages=1,
        allow_reaction=True,
    )
    assert (
        validate_discord_action(
            call("react_to_message", {"emoji": "👍"}),
            max_messages=1,
            allow_reaction=False,
        )
        is None
    )


def test_valid_escalation_preserves_the_models_context_aware_acknowledgement():
    action = validate_escalation_action(
        call(
            "escalate_to_owner",
            {
                "reason": "peer_requested_owner",
                "acknowledgement": "Sure — I've marked this for Alex.",
            },
        ),
        "fixed fallback",
    )

    assert action is not None
    assert action.kind == "escalation"
    assert action.messages == ("Sure — I've marked this for Alex.",)
    assert action.reason == "peer_requested_owner"
    assert action.invalid_arguments is False


def test_invalid_escalation_arguments_use_fixed_reply_without_repair():
    action = validate_escalation_action(
        call(
            "escalate_to_owner",
            {"reason": "invented", "acknowledgement": "I paged Alex and they will reply at 3."},
        ),
        "fixed fallback",
    )

    assert action is not None
    assert action.messages == ("fixed fallback",)
    assert action.reason == "invalid_tool_arguments"
    assert action.invalid_arguments is True
