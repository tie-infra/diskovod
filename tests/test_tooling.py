import json
from datetime import UTC, datetime
from types import SimpleNamespace

from diskovod.models import FunctionCall
from diskovod.localization import tool_text
from diskovod.tooling import (
    execute_read_only_tool,
    function_tools,
    validate_discord_action,
    validate_escalation_action,
    validate_hosted_web_search_calls,
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


def test_hosted_web_search_validation_enforces_capability_status_and_budget():
    completed = SimpleNamespace(kind="web_search_call", status="completed")
    failed = SimpleNamespace(kind="web_search_call", status="failed")

    assert validate_hosted_web_search_calls([completed], enabled=True) is True
    assert validate_hosted_web_search_calls([completed], enabled=False) is False
    assert validate_hosted_web_search_calls([failed], enabled=True) is False
    assert validate_hosted_web_search_calls([completed] * 3, enabled=True) is False


def test_function_schema_descriptions_follow_prompt_locale():
    for locale in ("en", "ru", "uk", "ja", "zh", "de", "fr"):
        text = tool_text(locale)
        tools = {tool["name"]: tool for tool in function_tools(locale)}

        assert tools["get_current_datetime"]["description"] == text["current_datetime"]
        assert (
            tools["get_current_datetime"]["parameters"]["properties"]["timezone"]["description"]
            == text["timezone"]
        )
        assert tools["calculate"]["description"] == text["calculate"]
        assert (
            tools["calculate"]["parameters"]["properties"]["expression"]["description"] == text["expression"]
        )
        assert tools["send_messages"]["description"] == text["send_messages"]
        assert (
            tools["send_messages"]["parameters"]["properties"]["messages"]["description"] == text["messages"]
        )
        assert tools["react_to_message"]["description"] == text["react"]
        assert tools["react_to_message"]["parameters"]["properties"]["emoji"]["description"] == text["emoji"]
        assert tools["escalate_to_owner"]["description"] == text["escalate"]
        escalation_properties = tools["escalate_to_owner"]["parameters"]["properties"]
        assert escalation_properties["reason"]["description"] == text["escalation_reason"]
        assert escalation_properties["acknowledgement"]["description"] == text["acknowledgement"]


def test_tool_errors_and_weekdays_follow_prompt_locale():
    invalid = FunctionCall("call", "calculate", "not-json", None)
    assert execute_read_only_tool(invalid, owner_timezone="UTC", locale="fr") == {
        "ok": False,
        "error": "arguments invalides",
    }

    result = execute_read_only_tool(
        call("get_current_datetime", {"timezone": None}),
        owner_timezone="UTC",
        locale="zh",
        now=datetime(2026, 7, 19, 12, 30, tzinfo=UTC),
    )
    assert result["weekday"] == "星期日"
