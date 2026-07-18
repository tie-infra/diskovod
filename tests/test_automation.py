import asyncio
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from diskovod.automation import (
    Automation,
    build_reply_instructions,
    discloses_automated_identity,
    parse_message_sequence,
    parse_reaction,
)
from diskovod.chatgpt import ChatGPTClient
from diskovod.models import AppSettings
from diskovod.store import Store


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


def test_reply_instructions_use_only_manual_owner_messages_as_style_evidence():
    history = [
        {"direction": "out", "source": "human", "content": "yeah sounds good"},
        {
            "direction": "out",
            "source": "assistant",
            "content": "Here are several options:\n\n- First\n- Second",
        },
        {"direction": "in", "source": "remote", "content": "when?"},
    ]

    instructions = build_reply_instructions(
        AppSettings(
            base_instructions="base",
            owner_details="My name is Alex. My dog is called Pixel.",
        ),
        {"profile": "profile"},
        history,
    )

    assert "My name is Alex. My dog is called Pixel." in instructions
    assert "never volunteer unrelated personal" in instructions
    assert '"yeah sounds good"' in instructions
    assert "Here are several options" not in instructions
    assert "Default to one short line" in instructions
    assert "dense and compact" in instructions
    assert "use emoji a little less often" in instructions
    assert "does not restrict the separate reaction action" in instructions
    assert "Identity boundary" in instructions
    assert "A reaction may replace the message" in instructions
    assert "output exactly one Discord message" in instructions
    assert instructions.index("My name is Alex") < instructions.index("profile")
    assert instructions.index("profile") < instructions.index("Default to one short line")
    assert instructions.index("A reaction may replace the message") < instructions.index('"yeah sounds good"')


def test_reply_instructions_offer_model_composed_sequences_only_when_selected():
    instructions = build_reply_instructions(
        AppSettings(base_instructions="base", max_reply_messages=4),
        None,
        [],
        allow_sequence=True,
    )

    assert "sequence of 2–4 Discord messages" in instructions
    assert "mechanically split a sentence" in instructions
    assert "<message>first message</message>" in instructions


@pytest.mark.parametrize(
    ("answer", "maximum", "expected"),
    (
        ("plain reply", 3, ["plain reply"]),
        ("<message>one</message><message>two</message>", 3, ["one", "two"]),
        (" <MESSAGE> one </MESSAGE>\n<message>two</message> ", 3, ["one", "two"]),
        ("<message>one</message><message>two</message><message>three</message>", 2, None),
        ("<message>one</message>", 3, None),
        ("<message>one</message>trailing", 3, None),
        ("<message></message><message>two</message>", 3, None),
        ("<message>one</message><message>two</message>", 1, None),
    ),
)
def test_message_sequence_parser(answer: str, maximum: int, expected: list[str] | None):
    assert parse_message_sequence(answer, maximum) == expected


@pytest.mark.parametrize(
    "answer",
    (
        "I'm an AI, so I don't have personal preferences.",
        "I'm actually an AI assistant.",
        "As a language model, I cannot do that.",
        "I am not a human.",
        "My responses are automated.",
        "You're chatting with an AI.",
        "This reply was generated by AI.",
        "Я — искусственный интеллект.",
        "Я являюсь ИИ.",
        "Я не человек.",
        "Je suis une IA.",
    ),
)
def test_identity_disclosure_detection(answer: str):
    assert discloses_automated_identity(answer)


@pytest.mark.parametrize(
    "answer",
    (
        "openai released another model?",
        "that bot in the server is annoying",
        "yeah, sounds pretty artificial",
    ),
)
def test_identity_disclosure_detection_avoids_topic_false_positives(answer: str):
    assert not discloses_automated_identity(answer)


class ReplyingChatGPT:
    def __init__(self, answers: list[str]):
        self.answers = iter(answers)
        self.calls: list[dict] = []

    async def complete(
        self,
        messages,
        instructions,
        model,
        effort,
        *,
        purpose,
        max_output_tokens=None,
        cache_key=None,
    ):
        self.calls.append(
            {
                "instructions": instructions,
                "purpose": purpose,
                "max_output_tokens": max_output_tokens,
                "cache_key": cache_key,
            }
        )
        return next(self.answers)


@pytest.mark.asyncio
async def test_identity_disclosure_is_repaired_before_release():
    chatgpt = ReplyingChatGPT(["I'm an AI.", "nah, just tired"])
    automation = Automation(cast(Store, None), cast(ChatGPTClient, chatgpt))

    answer = await automation._generate_reply([], "instructions", AppSettings())

    assert answer == "nah, just tired"
    assert [call["purpose"] for call in chatgpt.calls] == [
        "dm_reply",
        "dm_reply_identity_repair",
    ]
    assert all(call["max_output_tokens"] == 256 for call in chatgpt.calls)
    assert all(call["cache_key"] is None for call in chatgpt.calls)
    assert "previous draft was rejected" in chatgpt.calls[1]["instructions"]


@pytest.mark.asyncio
async def test_second_identity_disclosure_is_not_released():
    chatgpt = ReplyingChatGPT(["I'm an AI.", "As a bot, I cannot answer."])
    automation = Automation(cast(Store, None), cast(ChatGPTClient, chatgpt))

    answer = await automation._generate_reply([], "instructions", AppSettings())

    assert answer is None


@pytest.mark.asyncio
async def test_reaction_fallback_requires_plain_text():
    chatgpt = ReplyingChatGPT(["got it", "<react>👍</react>"])
    automation = Automation(cast(Store, None), cast(ChatGPTClient, chatgpt))

    reply = await automation._reaction_fallback([], "instructions", AppSettings())
    rejected = await automation._reaction_fallback([], "instructions", AppSettings())

    assert reply == "got it"
    assert rejected is None
    assert all(call["purpose"] == "dm_reply_reaction_fallback" for call in chatgpt.calls)


@pytest.mark.parametrize("output", ("<react>👍</react>", "👍"))
def test_reaction_parser_accepts_only_an_allowed_single_emoji(output: str):
    assert parse_reaction(output) == "👍"
    assert parse_reaction("<react>🧨</react>") is None
    assert parse_reaction("<react>👍</react> sounds good") is None


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


class TextChannel:
    id = "dm"
    me = object()

    def __init__(self):
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

    def __init__(self):
        self.channel = TextChannel()

    async def add_reaction(self, emoji: str):
        raise AssertionError(f"unexpected reaction: {emoji}")


@pytest.mark.asyncio
async def test_model_composed_sequence_uses_silent_flag_and_is_stored_as_distinct_messages(
    tmp_path: Path,
):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    settings = AppSettings(
        enabled=True,
        silent_replies=True,
        multi_message_replies=True,
        multi_message_chance=100,
        max_reply_messages=3,
        min_message_gap_seconds=0,
        max_message_gap_seconds=0,
        debounce_seconds=0,
        min_delay_seconds=0,
        max_delay_seconds=0,
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
    chatgpt = ReplyingChatGPT(["<message>hey</message><message>what's up?</message>"])
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0)

    assert [item[0] for item in trigger.channel.sent] == ["hey", "what's up?"]
    assert all(item[2] is True for item in trigger.channel.sent)
    saved = store.history("dm", 10)[-2:]
    assert [item["content"] for item in saved] == ["hey", "what's up?"]
    assert all(item["source"] == "assistant" for item in saved)
    assert "sequence of 2–3 Discord messages" in chatgpt.calls[0]["instructions"]
    assert chatgpt.calls[0]["cache_key"].startswith("diskovod:dm:")
    assert "dm" not in chatgpt.calls[0]["cache_key"].removeprefix("diskovod:dm:")
    store.close()


@pytest.mark.asyncio
async def test_unavailable_sequence_markup_is_replaced_with_one_plain_message(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_app_settings(
        AppSettings(
            enabled=True,
            multi_message_replies=False,
            debounce_seconds=0,
            min_delay_seconds=0,
            max_delay_seconds=0,
        )
    )
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
    chatgpt = ReplyingChatGPT(["<message>invalid</message><message>sequence</message>", "fixed reply"])
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = TextTrigger()

    await automation._reply(trigger, 0)

    assert [item[0] for item in trigger.channel.sent] == ["fixed reply"]
    assert [call["purpose"] for call in chatgpt.calls] == [
        "dm_reply",
        "dm_reply_sequence_fallback",
    ]
    store.close()


@pytest.mark.asyncio
async def test_reaction_replaces_reply_and_is_recorded(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    settings = AppSettings(
        enabled=True,
        debounce_seconds=0,
        min_delay_seconds=0,
        max_delay_seconds=0,
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
        content="nice",
        timestamp=time.time(),
    )
    chatgpt = ReplyingChatGPT(["<react>👍</react>"])
    automation = Automation(store, cast(ChatGPTClient, chatgpt))
    automation.versions["dm"] = 0
    trigger = ReactionTrigger()

    await automation._reply(trigger, 0)

    assert trigger.reactions == ["👍"]
    assert not store.reaction_allowed("dm")
    assert store.history("dm", 10)[-1]["id"] == "incoming"
    store.close()
