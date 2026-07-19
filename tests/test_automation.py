import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from diskovod.automation import Automation, build_reply_instructions
from diskovod.chatgpt import ChatGPTClient
from diskovod.localization import inline_tool_text
from diskovod.models import AppSettings, FunctionCall, HostedToolCall, ModelResult, TextOutput
from diskovod.store import Store


def function_result(name: str, arguments: dict, call_id: str = "call-1") -> ModelResult:
    encoded = json.dumps(arguments)
    return ModelResult([], [FunctionCall(call_id, name, encoded, arguments)], [])


class ReplyingChatGPT:
    active_provider = "chatgpt"
    hosted_web_search_available = False

    def __init__(self, results: list[ModelResult]):
        self.results = iter(results)
        self.calls: list[dict] = []

    async def complete_result(self, messages, instructions, model, effort, **kwargs):
        self.calls.append({"instructions": instructions, **kwargs})
        return next(self.results)


class TextChannel:
    me = object()

    def __init__(self, channel_id: str = "dm"):
        self.id = channel_id
        self.sent: list[tuple[str, str, bool]] = []

    async def history(self, limit: int):
        if False:
            yield limit

    @asynccontextmanager
    async def typing(self):
        yield

    async def send(self, content: str, *, nonce: str, silent: bool):
        self.sent.append((content, nonce, silent))
        return SimpleNamespace(
            id=f"sent-{len(self.sent)}",
            author=SimpleNamespace(id="me"),
            created_at=datetime.now(UTC),
        )


class TextTrigger:
    id = "incoming"

    def __init__(self, channel_id: str = "dm"):
        self.channel = TextChannel(channel_id)

    async def add_reaction(self, emoji: str):
        raise AssertionError(f"unexpected reaction: {emoji}")


class ReactionChannel:
    id = "dm"
    me = object()

    async def history(self, limit: int):
        if False:
            yield limit

    @asynccontextmanager
    async def typing(self):
        raise AssertionError("a reaction must not show the typing indicator")
        yield


class ReactionTrigger:
    id = "incoming"

    def __init__(self):
        self.channel = ReactionChannel()
        self.reactions: list[str] = []

    async def add_reaction(self, emoji: str):
        self.reactions.append(emoji)


def reply_store(tmp_path: Path, **overrides) -> Store:
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    settings = AppSettings(
        enabled=True,
        debounce_seconds=0,
        min_delay_seconds=0,
        max_delay_seconds=0,
        min_message_gap_seconds=0,
        max_message_gap_seconds=0,
        **overrides,
    )
    store.set_app_settings(settings)
    store.upsert_conversation("dm", "peer", "Peer")
    store.save_message(
        id="incoming",
        channel_id="dm",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="hello",
        timestamp=time.time(),
    )
    return store


@pytest.mark.asyncio
async def test_human_activity_cancels_inflight_work_and_snoozes(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.upsert_conversation("dm", "peer", "Peer")
    automation = Automation(store, cast(ChatGPTClient, None))
    task = asyncio.create_task(asyncio.sleep(60))
    automation.tasks["dm"] = task

    snoozed_until = automation.human_activity("dm")
    await asyncio.gather(task, return_exceptions=True)

    assert task.cancelled()
    conversation = store.conversation("dm")
    assert conversation["paused"] is False
    assert conversation["snoozed_until"] is not None
    assert store.can_automate("dm") is False
    assert 15 * 60 - 1 <= snoozed_until - time.time() <= 30 * 60 + 1
    assert automation.versions["dm"] == 1
    store.close()


def test_human_activity_does_not_snooze_inline_collaboration(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.upsert_conversation("dm", "peer", "Peer")
    store.set_conversation_mode("dm", "inline")
    automation = Automation(store, cast(ChatGPTClient, None))

    automation.human_activity("dm")

    assert store.conversation("dm")["snoozed_until"] is None
    assert store.can_automate("dm") is True
    store.close()


def test_permanent_pause_is_separate_from_human_quiet_window(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.upsert_conversation("dm", "peer", "Peer")
    automation = Automation(store, cast(ChatGPTClient, None))

    automation.permanently_pause("dm")

    assert store.conversation("dm")["paused"] is True
    assert store.conversation("dm")["snoozed_until"] is None
    store.close()


@pytest.mark.asyncio
async def test_only_the_pending_trigger_is_rescheduled_after_an_edit(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.upsert_conversation("dm", "peer", "Peer")
    store.set_app_settings(AppSettings(enabled=True, debounce_seconds=60))
    automation = Automation(store, cast(ChatGPTClient, None))
    channel = SimpleNamespace(id="dm")
    original = SimpleNamespace(id="latest", channel=channel, content="before")
    older = SimpleNamespace(id="older", channel=channel, content="edited old message")

    automation.schedule(original)
    first_task = automation.tasks["dm"]
    assert automation.reschedule_if_pending(older) is False

    edited = SimpleNamespace(id="latest", channel=channel, content="after")
    assert automation.reschedule_if_pending(edited) is True
    second_task = automation.tasks["dm"]
    assert first_task.cancelling()

    emptied = SimpleNamespace(id="latest", channel=channel, content="")
    assert automation.reschedule_if_pending(emptied) is True
    assert "dm" not in automation.tasks
    await asyncio.gather(first_task, second_task, return_exceptions=True)
    store.close()


def test_reply_instructions_use_manual_style_evidence_and_native_actions():
    history = [
        {"direction": "out", "source": "human", "content": "yeah sounds good"},
        {"direction": "out", "source": "assistant", "content": "Generated style"},
        {"direction": "in", "source": "remote", "content": "when?"},
    ]

    instructions = build_reply_instructions(
        AppSettings(base_instructions="base", owner_details="My name is Alex."),
        {"profile": "profile"},
        history,
    )

    assert "My name is Alex." in instructions
    assert '"yeah sounds good"' in instructions
    assert "Generated style" not in instructions
    assert "send_messages" in instructions
    assert "react_to_message" in instructions
    assert "<message>" not in instructions
    assert "<react>" not in instructions


@pytest.mark.asyncio
async def test_native_message_sequence_uses_delivery_options_and_stores_clean_content(tmp_path: Path):
    store = reply_store(
        tmp_path,
        silent_replies=True,
        robot_prefix=True,
        multi_message_replies=True,
        max_reply_messages=3,
    )
    chatgpt = ReplyingChatGPT([function_result("send_messages", {"messages": ["hey", "what's up?"]})])
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0)

    assert [item[0] for item in trigger.channel.sent] == ["🤖 hey", "🤖 what's up?"]
    assert all(item[2] is True for item in trigger.channel.sent)
    saved = store.history("dm", 10)[-2:]
    assert [item["content"] for item in saved] == ["hey", "what's up?"]
    assert chatgpt.calls[0]["tool_choice"] == "required"
    assert chatgpt.calls[0]["cache_key"].startswith("diskovod:dm-profile:")
    store.close()


@pytest.mark.asyncio
async def test_inline_mode_can_stay_silent_after_an_owner_message(tmp_path: Path):
    store = reply_store(tmp_path)
    store.set_conversation_mode("dm", "inline")
    chatgpt = ReplyingChatGPT([function_result("stay_silent", {})])
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0, owner_trigger=True)

    assert trigger.channel.sent == []
    call_record = chatgpt.calls[0]
    assert inline_tool_text("en")["policy"] in call_record["instructions"]
    assert inline_tool_text("en")["owner_trigger"] in call_record["instructions"]
    assert "stay_silent" in {tool["name"] for tool in call_record["tools"]}
    store.close()


@pytest.mark.asyncio
async def test_inline_mode_always_marks_assistant_messages(tmp_path: Path):
    store = reply_store(tmp_path, robot_prefix=False)
    store.set_conversation_mode("dm", "inline")
    chatgpt = ReplyingChatGPT([function_result("send_messages", {"messages": ["useful context"]})])
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0)

    assert trigger.channel.sent[0][0] == "🤖 useful context"
    store.close()


@pytest.mark.asyncio
async def test_plain_text_is_inert_and_gets_one_forced_native_repair(tmp_path: Path):
    store = reply_store(tmp_path)
    chatgpt = ReplyingChatGPT(
        [
            ModelResult([TextOutput("plain text", [])], [], []),
            function_result("send_messages", {"messages": ["fixed reply"]}),
        ]
    )
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0)

    assert [item[0] for item in trigger.channel.sent] == ["fixed reply"]
    assert chatgpt.calls[1]["tool_choice"] == {"type": "function", "name": "send_messages"}
    store.close()


@pytest.mark.asyncio
async def test_ambiguous_native_repair_rejection_is_annotated_in_request_log(tmp_path: Path):
    store = reply_store(tmp_path)
    request_ids = [
        store.start_model_request(
            provider="test",
            protocol="responses",
            model="model",
            purpose="dm_reply_tool_continuation" if index else "dm_reply",
            request_summary={},
            channel_id="dm",
            attempt=index + 1,
            repair=bool(index),
        )
        for index in range(2)
    ]
    chatgpt = ReplyingChatGPT(
        [
            ModelResult([TextOutput("first plain response", [])], [], [], request_log_id=request_ids[0]),
            ModelResult([TextOutput("second plain response", [])], [], [], request_log_id=request_ids[1]),
        ]
    )
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0)

    logs = {record["id"]: record for record in store.model_request_logs()}
    assert logs[request_ids[0]]["validation_status"] == "repair_requested"
    assert logs[request_ids[0]]["validation_detail"] == "expected_one_function_call_and_no_text"
    assert logs[request_ids[1]]["validation_status"] == "rejected"
    assert logs[request_ids[1]]["validation_detail"] == ("non_terminal_or_ambiguous_output_after_repair")
    assert logs[request_ids[0]]["validation_summary"]["observed"] == {
        "text_output_count": 1,
        "text_characters": 20,
        "response_text_present": True,
        "function_call_count": 0,
        "function_calls": [],
        "hosted_tool_call_count": 0,
        "hosted_tool_calls": [],
    }
    assert chatgpt.calls[1]["request_context"]["parent_request_id"] == request_ids[0]
    assert trigger.channel.sent == []
    store.close()


@pytest.mark.asyncio
async def test_native_reaction_replaces_reply_and_is_recorded(tmp_path: Path):
    store = reply_store(tmp_path)
    chatgpt = ReplyingChatGPT([function_result("react_to_message", {"emoji": "👍"})])
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = ReactionTrigger()

    await automation._reply(trigger, 0)

    assert trigger.reactions == ["👍"]
    assert not store.reaction_allowed("dm")
    assert store.history("dm", 10)[-1]["id"] == "incoming"
    store.close()


@pytest.mark.asyncio
async def test_forced_reply_repairs_reaction_to_written_native_action(tmp_path: Path):
    store = reply_store(tmp_path)
    store.set_permanent_pause("dm", True)
    chatgpt = ReplyingChatGPT(
        [
            function_result("react_to_message", {"emoji": "👍"}),
            function_result("send_messages", {"messages": ["written forced reply"]}),
        ]
    )
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0, force=True)

    assert [item[0] for item in trigger.channel.sent] == ["written forced reply"]
    assert chatgpt.calls[1]["tool_choice"] == {"type": "function", "name": "send_messages"}
    assert store.conversation("dm")["paused"] is True
    store.close()


@pytest.mark.asyncio
async def test_time_tool_output_is_ephemeral_and_terminal_action_follows(tmp_path: Path):
    store = reply_store(tmp_path, owner_timezone="Europe/Moscow")
    chatgpt = ReplyingChatGPT(
        [
            function_result("get_current_datetime", {"timezone": None}, "time-call"),
            function_result("send_messages", {"messages": ["Сегодня воскресенье."]}, "send-call"),
        ]
    )
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0)

    continuation = chatgpt.calls[1]["continuation_items"]
    assert [item["type"] for item in continuation] == ["function_call", "function_call_output"]
    assert '"timezone":"Europe/Moscow"' in continuation[1]["output"]
    assert all("Europe/Moscow" not in item["content"] for item in store.history("dm", 10))
    store.close()


@pytest.mark.asyncio
async def test_owner_escalation_pauses_chat_and_sends_model_acknowledgement(tmp_path: Path):
    store = reply_store(tmp_path, robot_prefix=True)
    chatgpt = ReplyingChatGPT(
        [
            function_result(
                "escalate_to_owner",
                {
                    "reason": "peer_requested_owner",
                    "acknowledgement": "Sure — I've marked this conversation for Alex.",
                },
            )
        ]
    )
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0)

    assert [item[0] for item in trigger.channel.sent] == ["🤖 Sure — I've marked this conversation for Alex."]
    assert store.conversation("dm")["paused"] is True
    escalation = store.active_escalations()[0]
    assert escalation["reason"] == "peer_requested_owner"
    assert escalation["acknowledged_at"] is not None
    assert len(chatgpt.calls) == 1
    store.close()


@pytest.mark.asyncio
async def test_invalid_escalation_arguments_use_localized_fallback_without_repair(tmp_path: Path):
    store = reply_store(tmp_path, prompt_locale="fr")
    chatgpt = ReplyingChatGPT(
        [
            function_result(
                "escalate_to_owner",
                {
                    "reason": "unsupported_reason",
                    "acknowledgement": "Alex has been paged and will reply in ten minutes.",
                },
            )
        ]
    )
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0)

    assert [item[0] for item in trigger.channel.sent] == [
        "J’ai signalé cette conversation au propriétaire du compte."
    ]
    assert store.active_escalations()[0]["reason"] == "invalid_tool_arguments"
    assert len(chatgpt.calls) == 1
    store.close()


@pytest.mark.asyncio
async def test_hosted_search_reaches_discord_only_through_terminal_message_action(tmp_path: Path):
    store = reply_store(tmp_path)
    result = function_result(
        "send_messages",
        {"messages": ["It launched today — https://example.test/announcement"]},
    )
    result.hosted_tool_calls.append(
        HostedToolCall(
            "web_search_call",
            "completed",
            {"action": {"query": "private raw search query"}},
        )
    )
    chatgpt = ReplyingChatGPT([result])
    chatgpt.hosted_web_search_available = True
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0)

    assert [item[0] for item in trigger.channel.sent] == [
        "It launched today — https://example.test/announcement"
    ]
    assert chatgpt.calls[0]["tools"][-1] == {
        "type": "web_search",
        "search_context_size": "low",
    }
    persisted = json.dumps(store.history("dm", 10), ensure_ascii=False)
    assert "private raw search query" not in persisted
    store.close()


@pytest.mark.asyncio
async def test_failed_hosted_search_output_fails_closed_without_repair(tmp_path: Path):
    store = reply_store(tmp_path)
    result = function_result("send_messages", {"messages": ["untrusted reply"]})
    result.hosted_tool_calls.append(HostedToolCall("web_search_call", "failed", {}))
    chatgpt = ReplyingChatGPT([result])
    chatgpt.hosted_web_search_available = True
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0)

    assert trigger.channel.sent == []
    assert len(chatgpt.calls) == 1
    store.close()


def test_profile_cache_key_is_shared_without_sharing_conversation_state(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    chatgpt = ReplyingChatGPT([])
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    settings = AppSettings(model="model", base_instructions="stable")
    personality = {"profile": "style", "source_hash": "profile-v1"}

    first = automation._profile_cache_key(settings, personality)
    second = automation._profile_cache_key(settings, personality)

    assert first == second
    assert first.startswith("diskovod:dm-profile:")
    assert "profile-v1" not in first
    store.close()


def test_profile_cache_key_changes_when_hosted_tool_schema_availability_changes(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    chatgpt = ReplyingChatGPT([])
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    settings = AppSettings(model="model", base_instructions="stable")

    without_search = automation._profile_cache_key(settings, None)
    chatgpt.hosted_web_search_available = True
    with_search = automation._profile_cache_key(settings, None)

    assert without_search != with_search
    store.close()


def test_profile_cache_key_separates_inline_and_automatic_tools(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    automation = Automation(store, cast(ChatGPTClient, ReplyingChatGPT([])))
    settings = AppSettings(model="model", base_instructions="stable")

    automatic = automation._profile_cache_key(settings, None)
    inline = automation._profile_cache_key(settings, None, inline_mode=True)

    assert automatic != inline
    store.close()


def test_profile_cache_key_changes_with_effective_assistant_name(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    automation = Automation(store, cast(ChatGPTClient, ReplyingChatGPT([])))

    english = automation._profile_cache_key(AppSettings(prompt_locale="en"), None)
    russian = automation._profile_cache_key(AppSettings(prompt_locale="ru"), None)
    custom = automation._profile_cache_key(
        AppSettings(prompt_locale="en", assistant_name="Helper"),
        None,
    )

    assert len({english, russian, custom}) == 3
    store.close()
