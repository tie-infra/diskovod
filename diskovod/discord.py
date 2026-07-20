from __future__ import annotations

import asyncio
import logging
import random
import secrets
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import discord

from .agent_types import AgentRuntimeContext
from .localization import runtime_context_text
from .outbound import DeliveryRecord
from .runtime import AgentService
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
        runtime: AgentService,
        captcha_handler,
        ready_callback: Callable[[], None],
    ):
        super().__init__(sync_presence=False, captcha_handler=captcha_handler)
        self.store = store
        self.runtime = runtime
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
        attachments = await self.runtime.attachments.capture(
            getattr(message, "attachments", ()),
            channel_id=channel_id,
            message_id=str(message.id),
        )
        if message.author == self.user:
            nonce = str(message.nonce) if message.nonce is not None else ""
            consumed_nonce = bool(nonce) and await self.store.aconsume_nonce(nonce)
            if consumed_nonce or await self.store.ais_bot_message(str(message.id)):
                return
            peer = message.channel.recipient
            if peer:
                await self.store.aupsert_conversation(
                    channel_id,
                    str(peer.id),
                    str(peer),
                )
            await self.store.asave_message(
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
            resumed = await self.runtime.resume_escalation_for_owner_reply(
                channel_id,
                message.content,
                message_id=str(message.id),
                author_id=str(self.user.id),
                author_name=str(self.user),
            )
            if resumed:
                log.info("Manual owner reply resumed the interrupted agent for %s", channel_id)
            conversation = await self.store.aconversation(channel_id)
            await self.runtime.submit_message(
                message_id=str(message.id),
                channel_id=channel_id,
                account_id=str(self.user.id),
                author_id=str(self.user.id),
                author_name=str(self.user),
                participant_role="owner",
                content=message.content,
                attachments=attachments,
                observed_at=message.created_at.timestamp(),
                agent_input=False if resumed else None,
            )
            if not resumed and not (
                conversation and conversation["mode"] == "inline" and not conversation["paused"]
            ):
                await self.runtime.human_activity(channel_id)
            return
        if message.author.bot:
            return
        await self.store.aupsert_conversation(
            channel_id,
            str(message.author.id),
            str(message.author),
        )
        await self.store.asave_message(
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
        await self.runtime.submit_message(
            message_id=str(message.id),
            channel_id=channel_id,
            account_id=str(self.user.id),
            author_id=str(message.author.id),
            author_name=str(message.author),
            participant_role="peer",
            content=message.content,
            attachments=attachments,
            observed_at=message.created_at.timestamp(),
        )

    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent) -> None:
        if not self.user or "content" not in payload.data:
            return
        message = payload.message
        if not isinstance(message.channel, discord.DMChannel):
            return
        channel_id = str(message.channel.id)
        content = message.content
        if message.author == self.user:
            updated = await self.store.aupdate_message_content(
                str(message.id),
                content,
                source="human",
            )
            if updated and updated["changed"]:
                conversation = await self.store.aconversation(channel_id)
                await self.runtime.submit_message(
                    message_id=str(message.id),
                    channel_id=channel_id,
                    account_id=str(self.user.id),
                    author_id=str(self.user.id),
                    author_name=str(self.user),
                    participant_role="owner",
                    content=content,
                    attachments=[],
                    observed_at=time.time(),
                    edited=True,
                )
                if not (conversation and conversation["mode"] == "inline" and not conversation["paused"]):
                    await self.runtime.human_activity(channel_id)
            return
        if message.author.bot:
            return
        updated = await self.store.aupdate_message_content(str(message.id), content)
        if not updated or not updated["changed"]:
            return
        await self.runtime.submit_message(
            message_id=str(message.id),
            channel_id=channel_id,
            account_id=str(self.user.id),
            author_id=str(message.author.id),
            author_name=str(message.author),
            participant_role="peer",
            content=content,
            attachments=[],
            observed_at=time.time(),
            edited=True,
        )

    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        if not self.user:
            return
        channel_id = str(payload.channel_id)
        if await self.store.aconversation(channel_id) is None:
            return
        await self.store.amark_message_deleted(str(payload.message_id))
        await self.runtime.submit_delete(
            message_id=str(payload.message_id),
            channel_id=channel_id,
            account_id=str(self.user.id),
        )


class DiscordService:
    def __init__(self, store: Store):
        self.store = store
        self.runtime: AgentService | None = None
        self.captcha = CaptchaBroker()
        self.client: PrivateDiscordClient | None = None
        self.task: asyncio.Task | None = None
        self.error: str | None = None
        self.retry_initial_seconds = 2.0
        self.retry_max_seconds = 60.0
        self._retry_delay = self.retry_initial_seconds

    def attach_runtime(self, runtime: AgentService) -> None:
        if self.runtime is not None:
            raise RuntimeError("The Discord agent runtime is already attached")
        self.runtime = runtime

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
        stored = await self.store.alatest_incoming_message(channel_id)
        if stored is None:
            raise RuntimeError("This conversation has no incoming message to answer")
        try:
            message_id = int(stored["id"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("The latest incoming Discord message ID is invalid") from exc
        await channel.fetch_message(message_id)
        if self.runtime is None:
            raise RuntimeError("The agent runtime is not available")
        await self.runtime.force_reply(
            channel_id=channel_id,
            account_id=str(client.user.id),
            trigger_message_id=str(message_id),
        )

    async def send_messages(
        self,
        context: AgentRuntimeContext,
        messages: tuple[str, ...],
    ) -> list[DeliveryRecord]:
        channel = self._channel(context.channel_id)
        settings = self.store.automation_settings()
        conversation = await self.store.aconversation(context.channel_id)
        inline = bool(conversation and conversation["mode"] == "inline")
        records: list[DeliveryRecord] = []
        for index, part in enumerate(messages):
            if index:
                await asyncio.sleep(
                    random.uniform(
                        settings.min_message_gap_seconds,
                        settings.max_message_gap_seconds,
                    )
                )
            cps = random.uniform(settings.min_typing_cps, settings.max_typing_cps)
            try:
                async with channel.typing():
                    await asyncio.sleep(min(12.0, max(0.8, len(part) / cps)))
                nonce = secrets.token_hex(12)
                await self.store.aremember_nonce(nonce)
                outbound = f"🤖 {part}" if settings.robot_prefix or inline else part
                sent = await channel.send(
                    outbound,
                    nonce=nonce,
                    silent=settings.silent_replies,
                )
            except Exception as error:
                records.append(
                    DeliveryRecord(
                        "failed",
                        index,
                        error_code="discord_send_failed",
                        error_detail=f"{type(error).__name__}: {error}"[:1000],
                    )
                )
                continue
            await self._record_sent_message(
                context,
                part,
                str(sent.id),
                str(sent.author.id),
                str(sent.author),
                sent.created_at.timestamp(),
            )
            records.append(DeliveryRecord("accepted", index, discord_message_id=str(sent.id)))
        return records

    async def _record_sent_message(
        self,
        context: AgentRuntimeContext,
        content: str,
        message_id: str,
        author_id: str,
        author_name: str,
        timestamp: float,
    ) -> None:
        await self.store.aremember_bot_message(message_id)
        await self.store.asave_message(
            id=message_id,
            channel_id=context.channel_id,
            author_id=author_id,
            author_name=author_name,
            direction="out",
            source="assistant",
            content=content,
            timestamp=timestamp,
        )

    async def react_to_message(
        self,
        context: AgentRuntimeContext,
        message_id: str,
        emoji: str,
    ) -> DeliveryRecord:
        channel = self._channel(context.channel_id)
        try:
            message = await channel.fetch_message(int(message_id))
            await message.add_reaction(emoji)
        except Exception as error:
            return DeliveryRecord(
                "failed",
                0,
                error_code="discord_reaction_failed",
                error_detail=f"{type(error).__name__}: {error}"[:1000],
            )
        await self.store.arecord_assistant_reaction(
            trigger_message_id=message_id,
            channel_id=context.channel_id,
            emoji=emoji,
        )
        return DeliveryRecord("accepted", 0, discord_message_id=f"reaction:{message_id}:{emoji}")

    def _channel(self, channel_id: str):
        client = self.client
        if not client or not client.user or not client.is_ready():
            raise RuntimeError("Discord is not connected")
        try:
            numeric = int(channel_id)
        except ValueError as error:
            raise RuntimeError("Invalid Discord channel ID") from error
        channel = client.get_channel(numeric)
        if channel is None:
            channel = next(
                (item for item in client.private_channels if str(getattr(item, "id", "")) == channel_id),
                None,
            )
        if channel is None or not hasattr(channel, "send"):
            raise RuntimeError("Discord conversation is not available")
        return channel

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
        text = runtime_context_text(self.store.assistant_profile().prompt_locale)
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
                    and not await self.store.ais_assistant_message(str(message.id))
                )
                if not is_manual_owner:
                    previous_was_manual_owner = False
                    previous_owner_timestamp = None
                    continue
                timestamp = message.created_at.timestamp()
                if previous_was_manual_owner and previous_owner_timestamp is not None:
                    gap = max(0.0, timestamp - previous_owner_timestamp)
                    shape = text["personality_sample_continuation"].format(seconds=f"{gap:.1f}")
                else:
                    shape = text["personality_sample_standalone"]
                annotated = (
                    text["personality_sample_header"].format(
                        index=channel_index,
                        shape=shape,
                    )
                    + f"\n{content}"
                )
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
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("The Discord agent runtime has not been attached")
        self.error = None
        self._retry_delay = self.retry_initial_seconds
        self.task = asyncio.create_task(self._run(token, runtime), name="discord-client")

    async def _run(self, token: str, runtime: AgentService) -> None:
        try:
            while True:
                client = PrivateDiscordClient(
                    self.store,
                    runtime,
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
