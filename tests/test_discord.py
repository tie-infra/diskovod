import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import diskovod.discord as discord_module
from diskovod.automation import Automation
from diskovod.discord import CaptchaBroker, DiscordService, PrivateDiscordClient
from diskovod.models import capture_discord_attachments, model_supports_vision
from diskovod.store import Store


class FakeCaptcha(Exception):
    service = "hcaptcha"
    sitekey = "site-key"
    rqdata = "request-data"
    errors = ["captcha-required"]
    should_serve_invisible = False


class FakeAttachment(SimpleNamespace):
    async def read(self, *, use_cached: bool = False) -> bytes:
        assert use_cached is True
        return self.body


@pytest.mark.asyncio
async def test_captures_metadata_and_small_text_attachment_body():
    attachment = FakeAttachment(
        id="attachment-1",
        filename="notes.txt",
        content_type="text/plain; charset=utf-8",
        size=11,
        url="https://cdn.example/notes.txt",
        description="meeting notes",
        body=b"hello world",
    )

    assert await capture_discord_attachments([attachment]) == [
        {
            "id": "attachment-1",
            "filename": "notes.txt",
            "content_type": "text/plain",
            "size": 11,
            "url": "https://cdn.example/notes.txt",
            "description": "meeting notes",
            "text": "hello world",
        }
    ]


def test_vision_capability_check_is_conservative():
    assert model_supports_vision("gpt-5.4-mini") is True
    assert model_supports_vision("gpt-4o-mini") is True
    assert model_supports_vision("o1-mini") is False
    assert model_supports_vision("local-text-model") is False


@pytest.mark.asyncio
async def test_captcha_handler_waits_for_admin_solution():
    broker = CaptchaBroker(timeout=10)
    task = asyncio.create_task(broker.handle(FakeCaptcha(), cast(object, None)))
    await asyncio.sleep(0)

    requests = broker.requests()
    assert len(requests) == 1
    assert requests[0]["service"] == "hcaptcha"
    assert requests[0]["sitekey"] == "site-key"
    assert requests[0]["rqdata"] == "request-data"
    assert broker.solve(requests[0]["id"], "captcha-solution") is True
    assert await task == "captcha-solution"
    assert broker.requests() == []


@pytest.mark.asyncio
async def test_captcha_handler_rejects_expired_request():
    broker = CaptchaBroker(timeout=0.01)
    with pytest.raises(FakeCaptcha):
        await broker.handle(FakeCaptcha(), cast(object, None))
    assert broker.requests() == []


class FakeChannel:
    def __init__(self, messages: list[SimpleNamespace]):
        self.messages = messages
        self.last_message_id = 1000

    async def history(self, limit: int):
        for message in self.messages[:limit]:
            yield message


class EditAutomation:
    def __init__(self):
        self.human_channels: list[str] = []
        self.rescheduled: list[object] = []

    def human_activity(self, channel_id: str):
        self.human_channels.append(channel_id)

    def reschedule_if_pending(self, message: object):
        self.rescheduled.append(message)
        return True


class ForceAutomation:
    def __init__(self):
        self.messages: list[object] = []

    def force_reply(self, message: object):
        self.messages.append(message)


class FakeEditDMChannel:
    id = 42


class RetryClient:
    attempts = 0

    def __init__(self, _store, _automation, _captcha_handler, ready_callback):
        self.ready_callback = ready_callback
        self.user = "reconnected-user"
        self.ready = False
        self.closed = False

    async def start(self, _token: str, *, reconnect: bool):
        assert reconnect is True
        type(self).attempts += 1
        if self.attempts == 1:
            raise OSError("network unavailable")
        self.ready = True
        self.ready_callback()
        await asyncio.Event().wait()

    def is_ready(self):
        return self.ready

    def is_closed(self):
        return self.closed

    async def close(self):
        self.closed = True
        self.ready = False


@pytest.mark.asyncio
async def test_discord_connection_failure_retries_without_stopping_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    RetryClient.attempts = 0
    monkeypatch.setattr(discord_module, "PrivateDiscordClient", RetryClient)
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_discord_token("discord-token")
    service = DiscordService(store, cast(Automation, None))
    service.retry_initial_seconds = 0.001
    service.retry_max_seconds = 0.001

    await service.start()
    for _ in range(100):
        if service.connected:
            break
        await asyncio.sleep(0.001)

    assert RetryClient.attempts == 2
    assert service.connected is True
    assert service.error is None
    assert service.task is not None

    await service.stop()
    assert service.task is None
    store.close()


@pytest.mark.asyncio
async def test_force_reply_fetches_latest_incoming_discord_message(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.upsert_conversation("42", "peer", "Peer")
    store.save_message(
        id="123",
        channel_id="42",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="latest incoming",
        timestamp=time.time(),
    )
    message = SimpleNamespace(id=123)

    class ForceChannel:
        id = 42

        async def fetch_message(self, message_id: int):
            assert message_id == 123
            return message

    channel = ForceChannel()
    automation = ForceAutomation()
    service = DiscordService(store, cast(Automation, automation))
    service.client = cast(
        PrivateDiscordClient,
        SimpleNamespace(
            user=object(),
            is_ready=lambda: True,
            get_channel=lambda channel_id: channel if channel_id == 42 else None,
            private_channels=[channel],
        ),
    )

    await service.force_reply("42")

    assert automation.messages == [message]
    store.close()


@pytest.mark.asyncio
async def test_personality_history_is_limited_and_excludes_generated_messages(tmp_path: Path):
    user = object()
    other = object()
    now = time.time()
    messages = [
        SimpleNamespace(
            id=str(index),
            author=user if index % 2 == 0 else other,
            content=f"human message {index}",
            created_at=datetime.fromtimestamp(now + index, timezone.utc),
        )
        for index in range(30)
    ]
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.save_message(
        id="4",
        channel_id="dm",
        author_id="me",
        author_name="Me",
        direction="out",
        source="assistant",
        content="generated message",
        timestamp=now + 4,
    )
    service = DiscordService(store, cast(Automation, None))
    service.client = cast(
        PrivateDiscordClient,
        SimpleNamespace(
            user=user,
            private_channels=[FakeChannel(messages)],
            is_ready=lambda: True,
        ),
    )

    history = await service.personality_history(20)

    assert len(history) == 9
    assert not any("human message 4" in item for item in history)
    assert any("human message 18" in item for item in history)
    assert not any("human message 20" in item for item in history)
    assert all("standalone owner message" in item for item in history)
    store.close()


@pytest.mark.asyncio
async def test_personality_history_marks_consecutive_owner_message_bursts(tmp_path: Path):
    user = object()
    peer = object()
    now = time.time()
    messages = [
        SimpleNamespace(
            id="peer-1",
            author=peer,
            content="peer content is omitted",
            created_at=datetime.fromtimestamp(now, timezone.utc),
        ),
        SimpleNamespace(
            id="owner-1",
            author=user,
            content="first thought",
            created_at=datetime.fromtimestamp(now + 1, timezone.utc),
        ),
        SimpleNamespace(
            id="owner-2",
            author=user,
            content="continued thought",
            created_at=datetime.fromtimestamp(now + 3, timezone.utc),
        ),
        SimpleNamespace(
            id="peer-2",
            author=peer,
            content="peer content is omitted",
            created_at=datetime.fromtimestamp(now + 4, timezone.utc),
        ),
        SimpleNamespace(
            id="owner-3",
            author=user,
            content="new standalone thought",
            created_at=datetime.fromtimestamp(now + 5, timezone.utc),
        ),
    ]
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    service = DiscordService(store, cast(Automation, None))
    service.client = cast(
        PrivateDiscordClient,
        SimpleNamespace(
            user=user,
            private_channels=[FakeChannel(messages)],
            is_ready=lambda: True,
        ),
    )

    history = await service.personality_history(20)

    assert len(history) == 3
    assert "standalone owner message" in history[0]
    assert "continuation in an owner message burst; 2.0s" in history[1]
    assert "standalone owner message" in history[2]
    assert all("peer content" not in item for item in history)
    store.close()


@pytest.mark.asyncio
async def test_raw_owner_edit_updates_style_history_and_marks_human_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(discord_module.discord, "DMChannel", FakeEditDMChannel)
    user = object()
    automation = EditAutomation()
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.upsert_conversation("42", "peer", "Peer")
    store.save_message(
        id="outgoing",
        channel_id="42",
        author_id="me",
        author_name="Me",
        direction="out",
        source="assistant",
        content="generated draft",
        timestamp=time.time(),
    )
    client = SimpleNamespace(user=user, store=store, automation=automation)
    message = SimpleNamespace(
        id="outgoing",
        channel=FakeEditDMChannel(),
        author=user,
        content="edited by owner",
    )
    payload = SimpleNamespace(data={"content": "edited by owner"}, message=message)

    await PrivateDiscordClient.on_raw_message_edit(client, payload)

    saved = store.history("42", 1)[0]
    assert saved["content"] == "edited by owner"
    assert saved["source"] == "human"
    assert automation.human_channels == ["42"]
    store.close()


@pytest.mark.asyncio
async def test_raw_remote_edit_updates_history_and_only_requests_pending_reschedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(discord_module.discord, "DMChannel", FakeEditDMChannel)
    user = object()
    remote = SimpleNamespace(bot=False)
    automation = EditAutomation()
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.upsert_conversation("42", "peer", "Peer")
    store.save_message(
        id="incoming",
        channel_id="42",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="original text",
        timestamp=time.time(),
    )
    client = SimpleNamespace(user=user, store=store, automation=automation)
    message = SimpleNamespace(
        id="incoming",
        channel=FakeEditDMChannel(),
        author=remote,
        content="corrected text",
    )

    await PrivateDiscordClient.on_raw_message_edit(
        client,
        SimpleNamespace(data={"content": "corrected text"}, message=message),
    )
    await PrivateDiscordClient.on_raw_message_edit(
        client,
        SimpleNamespace(data={"embeds": []}, message=message),
    )

    assert store.history("42", 1)[0]["content"] == "corrected text"
    assert automation.rescheduled == [message]
    store.close()
