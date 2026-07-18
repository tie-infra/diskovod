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
from diskovod.models import AppSettings, FunctionCall, ModelResult, TextOutput
from diskovod.store import Store


def function_result(name: str, arguments: dict, call_id: str = "call-1") -> ModelResult:
    encoded = json.dumps(arguments)
    return ModelResult([], [FunctionCall(call_id, name, encoded, arguments)], [])


class ReplyingChatGPT:
    active_provider = "chatgpt"

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
