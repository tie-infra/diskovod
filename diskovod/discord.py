from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import discord

from .automation import Automation
from .models import capture_discord_attachments
from .store import Store

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CaptchaRequest:
    id: str
    service: str
    sitekey: str
    rqdata: str | None
    errors: list[str]
    should_serve_invisible: bool
    created_at: float
    expires_at: float


class CaptchaBroker:
    def __init__(self, timeout: float = 600):
        self.timeout = timeout
        self._pending: dict[str, tuple[CaptchaRequest, asyncio.Future[str]]] = {}

    async def handle(self, exception: discord.CaptchaRequired, _client: discord.Client) -> str:
        now = time.time()
        request = CaptchaRequest(
            id=secrets.token_urlsafe(18),
            service=exception.service,
            sitekey=exception.sitekey,
            rqdata=exception.rqdata,
            errors=list(exception.errors),
            should_serve_invisible=exception.should_serve_invisible,
            created_at=now,
            expires_at=now + self.timeout,
        )
        future = asyncio.get_running_loop().create_future()
        self._pending[request.id] = (request, future)
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except TimeoutError:
            raise exception
        finally:
            self._pending.pop(request.id, None)

    def requests(self) -> list[dict]:
        now = time.time()
        result = []
        for request, future in self._pending.values():
            if future.done():
                continue
            value = asdict(request)
            value["expires_in_seconds"] = max(0, int(request.expires_at - now))
            result.append(value)
        return sorted(result, key=lambda item: item["created_at"])

    def solve(self, request_id: str, solution: str) -> bool:
        pending = self._pending.get(request_id)
        if not pending or pending[1].done():
            return False
        pending[1].set_result(solution)
        return True

    def cancel_all(self) -> None:
        for _, future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()


class PrivateDiscordClient(discord.Client):
    def __init__(
        self,
        store: Store,
        automation: Automation,
        captcha_handler,
        ready_callback: Callable[[], None],
    ):
        super().__init__(sync_presence=False, captcha_handler=captcha_handler)
        self.store = store
        self.automation = automation
        self.ready_callback = ready_callback

    async def on_ready(self) -> None:
        self.ready_callback()
        log.info(
            "Connected to Discord as %s (%s)",
            self.user,
            self.user.id if self.user else "?",
        )

    async def on_message(self, message: discord.Message) -> None:
        if not self.user or not isinstance(message.channel, discord.DMChannel):
            return
        channel_id = str(message.channel.id)
        attachments = await capture_discord_attachments(getattr(message, "attachments", ()))
        if message.author == self.user:
            nonce = str(message.nonce) if message.nonce is not None else ""
            if (nonce and self.store.consume_nonce(nonce)) or self.store.is_bot_message(str(message.id)):
                return
            peer = message.channel.recipient
            if peer:
                self.store.upsert_conversation(channel_id, str(peer.id), str(peer))
            self.store.save_message(
                id=str(message.id),
                channel_id=channel_id,
                author_id=str(self.user.id),
                author_name=str(self.user),
                direction="out",
                source="human",
                content=message.content,
                timestamp=message.created_at.timestamp(),
                attachments=attachments,
            )
            if self.store.resolve_escalation_on_owner_reply(channel_id):
                log.info("Manual owner reply resolved the active escalation for %s", channel_id)
            self.automation.human_activity(channel_id)
            return
        if message.author.bot:
            return
        self.store.upsert_conversation(channel_id, str(message.author.id), str(message.author))
        self.store.save_message(
            id=str(message.id),
            channel_id=channel_id,
            author_id=str(message.author.id),
            author_name=str(message.author),
            direction="in",
            source="remote",
            content=message.content,
            timestamp=message.created_at.timestamp(),
            attachments=attachments,
        )
        self.automation.schedule(message)

    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent) -> None:
        if not self.user or "content" not in payload.data:
            return
        message = payload.message
        if not isinstance(message.channel, discord.DMChannel):
            return
        channel_id = str(message.channel.id)
        content = message.content
        if message.author == self.user:
            updated = self.store.update_message_content(str(message.id), content, source="human")
            if updated and updated["changed"]:
                self.automation.human_activity(channel_id)
            return
        if message.author.bot:
            return
        updated = self.store.update_message_content(str(message.id), content)
        if not updated or not updated["changed"]:
            return
        self.automation.reschedule_if_pending(message)


class DiscordService:
    def __init__(self, store: Store, automation: Automation):
        self.store = store
        self.automation = automation
        self.captcha = CaptchaBroker()
        self.client: PrivateDiscordClient | None = None
        self.task: asyncio.Task | None = None
        self.error: str | None = None
        self.retry_initial_seconds = 2.0
        self.retry_max_seconds = 60.0
        self._retry_delay = self.retry_initial_seconds

    @property
    def connected(self) -> bool:
        return bool(self.client and self.client.is_ready())

    @property
    def identity(self) -> str | None:
        return str(self.client.user) if self.client and self.client.user else None

    def captcha_requests(self) -> list[dict]:
        return self.captcha.requests()

    def solve_captcha(self, request_id: str, solution: str) -> bool:
        return self.captcha.solve(request_id, solution)

    async def force_reply(self, channel_id: str) -> None:
        client = self.client
        if not client or not client.user or not client.is_ready():
            raise RuntimeError("Discord must be connected before forcing a reply")
        try:
            numeric_channel_id = int(channel_id)
        except ValueError as exc:
            raise RuntimeError("Invalid Discord channel ID") from exc
        channel = client.get_channel(numeric_channel_id)
        if channel is None:
            channel = next(
                (item for item in client.private_channels if str(getattr(item, "id", "")) == channel_id),
                None,
            )
        if channel is None or not hasattr(channel, "fetch_message"):
            raise RuntimeError("Discord conversation is not available")
        stored = self.store.latest_incoming_message(channel_id)
        if stored is None:
            raise RuntimeError("This conversation has no incoming message to answer")
        try:
            message_id = int(stored["id"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("The latest incoming Discord message ID is invalid") from exc
        message = await channel.fetch_message(message_id)
        self.automation.force_reply(message)

    async def personality_history(self, limit: int) -> list[str]:
        client = self.client
        if not client or not client.user or not client.is_ready():
            raise RuntimeError("Discord must be connected before loading message history")

        requested = max(20, min(limit, 500))
        channels = [channel for channel in client.private_channels if hasattr(channel, "history")]
        channels.sort(key=lambda channel: int(getattr(channel, "last_message_id", 0) or 0), reverse=True)
        channels = channels[: min(20, requested)]
        if not channels:
            return []

        per_channel = max(1, (requested + len(channels) - 1) // len(channels))
        messages: list[tuple[float, str]] = []
        remaining = requested
        for channel_index, channel in enumerate(channels, start=1):
            fetch_limit = min(per_channel, remaining)
            if fetch_limit <= 0:
                break
            remaining -= fetch_limit
            channel_messages = []
            async for message in channel.history(limit=fetch_limit):
                channel_messages.append(message)

            channel_messages.sort(key=lambda message: message.created_at.timestamp())
            previous_was_manual_owner = False
            previous_owner_timestamp: float | None = None
            for message in channel_messages:
                content = message.content.strip()[:4000]
                is_manual_owner = (
                    message.author == client.user
                    and content
                    and not self.store.is_assistant_message(str(message.id))
                )
                if not is_manual_owner:
                    previous_was_manual_owner = False
                    previous_owner_timestamp = None
                    continue
                timestamp = message.created_at.timestamp()
                if previous_was_manual_owner and previous_owner_timestamp is not None:
                    gap = max(0.0, timestamp - previous_owner_timestamp)
                    shape = f"continuation in an owner message burst; {gap:.1f}s after its previous part"
                else:
                    shape = "standalone owner message"
                annotated = f"[anonymous conversation {channel_index}; {shape}]\n{content}"
                messages.append((timestamp, annotated))
                previous_was_manual_owner = True
                previous_owner_timestamp = timestamp

        messages.sort(key=lambda item: item[0])
        selected: list[str] = []
        selected_characters = 0
        for _, content in reversed(messages[-requested:]):
            if selected_characters + len(content) > 80000:
                break
            selected.append(content)
            selected_characters += len(content)
        return list(reversed(selected))

    async def start(self) -> None:
        if self.task:
            return
        token = self.store.discord_token()
        if not token:
            return
        self.error = None
        self._retry_delay = self.retry_initial_seconds
        self.task = asyncio.create_task(self._run(token), name="discord-client")

    async def _run(self, token: str) -> None:
        try:
            while True:
                client = PrivateDiscordClient(
                    self.store,
                    self.automation,
                    self.captcha.handle,
                    self._connected,
                )
                self.client = client
                try:
                    await client.start(token, reconnect=True)
                    self.error = "Discord connection closed; retrying"
                    log.warning("Discord connection closed; retrying")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.error = str(exc)
                    log.exception(
                        "Discord connection failed; retrying in %.1f seconds",
                        self._retry_delay,
                    )
                finally:
                    if not client.is_closed():
                        try:
                            await client.close()
                        except Exception:
                            log.exception("Failed to close Discord client after a connection attempt")
                    if self.client is client:
                        self.client = None
                await asyncio.sleep(self._retry_delay)
                self._retry_delay = min(self._retry_delay * 2, self.retry_max_seconds)
        finally:
            if self.task is asyncio.current_task():
                self.task = None

    def _connected(self) -> None:
        self.error = None
        self._retry_delay = self.retry_initial_seconds

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    async def stop(self) -> None:
        self.captcha.cancel_all()
        client, task = self.client, self.task
        self.client = None
        self.task = None
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        elif client and not client.is_closed():
            await client.close()
