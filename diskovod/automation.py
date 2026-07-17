from __future__ import annotations

import asyncio
import logging
import random
import secrets
import time
from typing import Any

from .chatgpt import ChatGPTClient
from .store import Store

log = logging.getLogger(__name__)


class Automation:
    """Schedules replies while yielding temporarily or permanently to a human."""

    def __init__(self, store: Store, chatgpt: ChatGPTClient):
        self.store = store
        self.chatgpt = chatgpt
        self.tasks: dict[str, asyncio.Task] = {}
        self.versions: dict[str, int] = {}

    def cancel(self, channel_id: str) -> None:
        self.versions[channel_id] = self.versions.get(channel_id, 0) + 1
        task = self.tasks.pop(channel_id, None)
        if task:
            task.cancel()

    def human_activity(self, channel_id: str) -> float:
        settings = self.store.app_settings()
        quiet_minutes = random.uniform(settings.min_human_quiet_minutes, settings.max_human_quiet_minutes)
        snoozed_until = self.store.snooze(channel_id, quiet_minutes * 60)
        self.cancel(channel_id)
        log.info(
            "Human activity in DM channel %s; automation is quiet for %.1f minutes",
            channel_id,
            quiet_minutes,
        )
        return snoozed_until

    def permanently_pause(self, channel_id: str) -> None:
        self.store.set_permanent_pause(channel_id, True)
        self.cancel(channel_id)

    def schedule(self, message: Any) -> None:
        channel_id = str(message.channel.id)
        self.cancel(channel_id)
        if not self.store.app_settings().enabled:
            return
        if not self.store.can_automate(channel_id):
            return
        version = self.versions[channel_id]
        task = asyncio.create_task(self._reply(message, version), name=f"reply-{channel_id}")
        self.tasks[channel_id] = task
        task.add_done_callback(lambda done: self._finished(channel_id, done))

    def _finished(self, channel_id: str, task: asyncio.Task) -> None:
        if self.tasks.get(channel_id) is task:
            self.tasks.pop(channel_id, None)
        if not task.cancelled() and (error := task.exception()):
            log.error("Reply failed for %s: %s", channel_id, error)

    def _still_allowed(self, channel_id: str, version: int) -> bool:
        return self.store.can_automate(channel_id) and self.versions.get(channel_id) == version

    async def _reply(self, trigger: Any, version: int) -> None:
        settings = self.store.app_settings()
        channel_id = str(trigger.channel.id)
        started_at = time.time()
        await asyncio.sleep(settings.debounce_seconds)
        if not self._still_allowed(channel_id, version):
            return

        history = self.store.history(channel_id, settings.history_limit)
        messages = [
            {"role": "assistant" if item["direction"] == "out" else "user", "content": item["content"]}
            for item in history
            if item["content"].strip()
        ]
        personality = self.store.personality()
        instructions = settings.base_instructions
        if personality:
            instructions += (
                "\n\nCached personality and conversational behavior to follow:\n" + personality["profile"]
            )
        answer = await self.chatgpt.complete(
            messages,
            instructions,
            settings.model,
            settings.reasoning_effort,
            purpose="dm_reply",
        )
        if not self._still_allowed(channel_id, version):
            return

        await asyncio.sleep(random.uniform(settings.min_delay_seconds, settings.max_delay_seconds))
        if not self._still_allowed(channel_id, version):
            return
        if await self._manual_message_exists(trigger.channel, started_at):
            self.human_activity(channel_id)
            return

        cps = random.uniform(settings.min_typing_cps, settings.max_typing_cps)
        async with trigger.channel.typing():
            await asyncio.sleep(min(12.0, max(0.8, len(answer) / cps)))
        if not self._still_allowed(channel_id, version):
            return
        if await self._manual_message_exists(trigger.channel, started_at):
            self.human_activity(channel_id)
            return

        nonce = secrets.token_hex(12)
        self.store.remember_nonce(nonce)
        sent = await trigger.channel.send(answer, nonce=nonce)
        self.store.remember_bot_message(str(sent.id))
        me = sent.author
        self.store.save_message(
            id=str(sent.id),
            channel_id=channel_id,
            author_id=str(me.id),
            author_name=str(me),
            direction="out",
            source="assistant",
            content=answer,
            timestamp=sent.created_at.timestamp(),
        )

    async def _manual_message_exists(self, channel: Any, since: float) -> bool:
        """Gateway delivery can lag; check recent server history immediately before sending."""
        async for message in channel.history(limit=12):
            if message.created_at.timestamp() < since:
                break
            if message.author == channel.me and not self.store.is_bot_message(str(message.id)):
                nonce = str(message.nonce) if getattr(message, "nonce", None) is not None else ""
                if not nonce or not self.store.consume_nonce(nonce):
                    return True
        return False

    async def close(self) -> None:
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
