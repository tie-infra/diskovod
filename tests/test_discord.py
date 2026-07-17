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
from diskovod.store import Store


class FakeCaptcha(Exception):
    service = "hcaptcha"
    sitekey = "site-key"
    rqdata = "request-data"
    errors = ["captcha-required"]
    should_serve_invisible = False


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
    assert "human message 4" not in history
    assert "human message 18" in history
    assert "human message 20" not in history
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
