import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import diskovod.discord as discord_module
from diskovod.discord import CaptchaBroker, DiscordService, PrivateDiscordClient
from diskovod.models import discord_attachment_metadata
from diskovod.store import Store


class FakeCaptcha(Exception):
    service = "hcaptcha"
    sitekey = "site-key"
    rqdata = "request-data"
    errors = ["captcha-required"]
    should_serve_invisible = False


def test_captures_attachment_metadata_without_downloading():
    attachment = SimpleNamespace(
        id="attachment-1",
        filename="notes.txt",
        content_type="text/plain; charset=utf-8",
        size=11,
        url="https://cdn.example/notes.txt",
        description="meeting notes",
    )

    assert discord_attachment_metadata([attachment]) == [
        {
            "id": "attachment-1",
            "filename": "notes.txt",
            "content_type": "text/plain",
            "size": 11,
            "url": "https://cdn.example/notes.txt",
            "description": "meeting notes",
        }
    ]


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


class EditRuntime:
    def __init__(self):
        self.human_channels: list[str] = []
        self.submitted: list[dict] = []

    async def human_activity(self, channel_id: str):
        self.human_channels.append(channel_id)

    async def submit_message(self, **values):
        self.submitted.append(values)


class ForceRuntime:
    def __init__(self):
        self.requests: list[dict] = []

    async def force_reply(self, **values):
        self.requests.append(values)


class FakeEditDMChannel:
    id = 42


class RetryClient:
    attempts = 0
    ready_event: asyncio.Event

    def __init__(self, _store, _runtime, _captcha_handler, ready_callback):
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
        type(self).ready_event.set()
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
    RetryClient.ready_event = asyncio.Event()
    monkeypatch.setattr(discord_module, "PrivateDiscordClient", RetryClient)
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    await store.aset_discord_token("discord-token")
    service = DiscordService(store)
    service.attach_runtime(cast(object, SimpleNamespace()))
    service.retry_initial_seconds = 0
    service.retry_max_seconds = 0

    await service.start()
    await RetryClient.ready_event.wait()

    assert RetryClient.attempts == 2
    assert service.connected is True
    assert service.error is None
    assert service.task is not None

    await service.stop()
    assert service.task is None
    await store.aclose()


@pytest.mark.asyncio
async def test_force_reply_fetches_latest_incoming_discord_message(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    await store.aupsert_conversation("42", "peer", "Peer")
    await store.asave_message(
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
    runtime = ForceRuntime()
    service = DiscordService(store)
    service.attach_runtime(cast(object, runtime))
    service.client = cast(
        PrivateDiscordClient,
        SimpleNamespace(
            user=SimpleNamespace(id=999),
            is_ready=lambda: True,
            get_channel=lambda channel_id: channel if channel_id == 42 else None,
            private_channels=[channel],
        ),
    )

    await service.force_reply("42")

    assert runtime.requests == [{"channel_id": "42", "account_id": "999", "trigger_message_id": "123"}]
    await store.aclose()


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
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    await store.asave_message(
        id="4",
        channel_id="dm",
        author_id="me",
        author_name="Me",
        direction="out",
        source="assistant",
        content="generated message",
        timestamp=now + 4,
    )
    service = DiscordService(store)
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
    await store.aclose()


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
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    service = DiscordService(store)
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
    await store.aclose()


@pytest.mark.asyncio
async def test_raw_owner_edit_updates_style_history_and_marks_human_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(discord_module.discord, "DMChannel", FakeEditDMChannel)
    user = SimpleNamespace(id=999)
    runtime = EditRuntime()
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    await store.aupsert_conversation("42", "peer", "Peer")
    await store.asave_message(
        id="outgoing",
        channel_id="42",
        author_id="me",
        author_name="Me",
        direction="out",
        source="assistant",
        content="generated draft",
        timestamp=time.time(),
    )
    client = SimpleNamespace(user=user, store=store, runtime=runtime)
    message = SimpleNamespace(
        id="outgoing",
        channel=FakeEditDMChannel(),
        author=user,
        content="edited by owner",
    )
    payload = SimpleNamespace(data={"content": "edited by owner"}, message=message)

    await PrivateDiscordClient.on_raw_message_edit(client, payload)

    saved = (await store.ahistory("42", 1))[0]
    assert saved["content"] == "edited by owner"
    assert saved["source"] == "human"
    assert runtime.human_channels == ["42"]
    assert runtime.submitted[0]["participant_role"] == "owner"
    await store.aclose()


@pytest.mark.asyncio
async def test_raw_owner_edit_reschedules_inline_collaboration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(discord_module.discord, "DMChannel", FakeEditDMChannel)
    user = SimpleNamespace(id=999)
    runtime = EditRuntime()
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    await store.aupsert_conversation("42", "peer", "Peer")
    await store.aset_conversation_mode("42", "inline")
    await store.asave_message(
        id="outgoing",
        channel_id="42",
        author_id="me",
        author_name="Me",
        direction="out",
        source="human",
        content="original",
        timestamp=time.time(),
    )
    client = SimpleNamespace(user=user, store=store, runtime=runtime)
    message = SimpleNamespace(
        id="outgoing",
        channel=FakeEditDMChannel(),
        author=user,
        content="updated",
    )
    payload = SimpleNamespace(data={"content": "updated"}, message=message)

    await PrivateDiscordClient.on_raw_message_edit(client, payload)

    assert runtime.human_channels == []
    assert runtime.submitted[0]["participant_role"] == "owner"
    assert runtime.submitted[0]["edited"] is True
    await store.aclose()


@pytest.mark.asyncio
async def test_raw_remote_edit_updates_history_and_only_requests_pending_reschedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(discord_module.discord, "DMChannel", FakeEditDMChannel)
    user = SimpleNamespace(id=999)
    remote = SimpleNamespace(id=123, bot=False)
    runtime = EditRuntime()
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    await store.aupsert_conversation("42", "peer", "Peer")
    await store.asave_message(
        id="incoming",
        channel_id="42",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="original text",
        timestamp=time.time(),
    )
    client = SimpleNamespace(user=user, store=store, runtime=runtime)
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

    assert (await store.ahistory("42", 1))[0]["content"] == "corrected text"
    assert len(runtime.submitted) == 1
    assert runtime.submitted[0]["participant_role"] == "peer"
    assert runtime.submitted[0]["edited"] is True
    await store.aclose()
