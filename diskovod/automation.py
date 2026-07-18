from __future__ import annotations

import asyncio
import json
import logging
import random
import secrets
import time
from typing import Any

from .chatgpt import ChatGPTClient, make_prompt_cache_key
from .localization import prompts_for
from .models import AppSettings
from .store import Store
from .tooling import (
    FUNCTION_TOOLS,
    TOOL_SCHEMA_VERSION,
    DiscordAction,
    execute_read_only_tool,
    function_call_item,
    function_output_item,
    validate_discord_action,
)

log = logging.getLogger(__name__)


def build_reply_instructions(
    settings: AppSettings,
    personality: dict | None,
    history: list[dict],
) -> str:
    """Build instructions with trusted human style evidence separate from dialogue history."""
    prompts = prompts_for(settings.prompt_locale)
    sections = [settings.base_instructions]
    if settings.owner_details.strip():
        sections.append(prompts.owner_details.format(details=settings.owner_details.strip()))
    if personality:
        sections.append(prompts.cached_personality.format(profile=personality["profile"]))

    sections.append(prompts.dm_style)
    sections.append(
        "Use the available native tools for every action. Finish with exactly one terminal action: "
        "send_messages for written replies or, only when genuinely appropriate, react_to_message. "
        "Do not return final reply text outside a terminal action. Use get_current_datetime whenever "
        "the answer depends on the current date or time, and calculate for non-trivial arithmetic."
    )

    owner_examples = [
        item["content"]
        for item in history
        if item["direction"] == "out" and item.get("source") == "human" and item["content"].strip()
    ][-12:]
    if owner_examples:
        sections.append(
            prompts.owner_examples.format(examples=json.dumps(owner_examples, ensure_ascii=False))
        )

    return "\n\n".join(sections)


class Automation:
    """Schedules replies while yielding temporarily or permanently to a human."""

    def __init__(self, store: Store, chatgpt: ChatGPTClient):
        self.store = store
        self.chatgpt = chatgpt
        self.tasks: dict[str, asyncio.Task] = {}
        self.trigger_ids: dict[str, str] = {}
        self.versions: dict[str, int] = {}
        self.reaction_lock = asyncio.Lock()

    def cancel(self, channel_id: str) -> None:
        self.versions[channel_id] = self.versions.get(channel_id, 0) + 1
        task = self.tasks.pop(channel_id, None)
        self.trigger_ids.pop(channel_id, None)
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
        self.trigger_ids[channel_id] = str(message.id)
        task.add_done_callback(lambda done: self._finished(channel_id, done))

    def force_reply(self, message: Any) -> None:
        """Schedule one reply that bypasses automation enrollment and quiet-window checks."""
        channel_id = str(message.channel.id)
        self.cancel(channel_id)
        version = self.versions[channel_id]
        task = asyncio.create_task(
            self._reply(message, version, force=True),
            name=f"forced-reply-{channel_id}",
        )
        self.tasks[channel_id] = task
        self.trigger_ids[channel_id] = str(message.id)
        task.add_done_callback(lambda done: self._finished(channel_id, done))

    def reschedule_if_pending(self, message: Any) -> bool:
        channel_id = str(message.channel.id)
        if channel_id not in self.tasks or self.trigger_ids.get(channel_id) != str(message.id):
            return False
        if not message.content.strip() and not getattr(message, "attachments", None):
            self.cancel(channel_id)
            return True
        self.schedule(message)
        return True

    def _finished(self, channel_id: str, task: asyncio.Task) -> None:
        if self.tasks.get(channel_id) is task:
            self.tasks.pop(channel_id, None)
            self.trigger_ids.pop(channel_id, None)
        if not task.cancelled() and (error := task.exception()):
            log.error("Reply failed for %s: %s", channel_id, error)

    def _still_allowed(self, channel_id: str, version: int, *, force: bool = False) -> bool:
        return self.versions.get(channel_id) == version and (force or self.store.can_automate(channel_id))

    async def _reply(self, trigger: Any, version: int, *, force: bool = False) -> None:
        settings = self.store.app_settings()
        channel_id = str(trigger.channel.id)
        started_at = time.time()
        await asyncio.sleep(0 if force else settings.debounce_seconds)
        if not self._still_allowed(channel_id, version, force=force):
            return

        history = self.store.history(channel_id, settings.history_limit)
        prompts = prompts_for(settings.prompt_locale)
        messages = [
            {
                "role": "assistant" if item["direction"] == "out" else "user",
                "content": item["content"],
                "locale": settings.prompt_locale,
                # Discord CDN URLs are signed and native inputs are expensive to replay.
                # Keep historic metadata/retrieval text, but send only the trigger's URLs.
                "attachments": [
                    attachment if item["id"] == str(trigger.id) else attachment | {"url": ""}
                    for attachment in item.get("attachments", [])
                ],
            }
            for item in history
            if item["content"].strip() or item.get("attachments")
        ]
        personality = self.store.personality()
        instructions = build_reply_instructions(settings, personality, history)
        if force:
            instructions += "\n\n" + prompts.forced_reply
        cache_key = self._profile_cache_key(settings, personality)
        max_messages = settings.max_reply_messages if settings.multi_message_replies else 1
        allow_reaction = not force and self.store.reaction_allowed(channel_id)
        action = await self._generate_action(
            messages,
            instructions,
            settings,
            max_messages=max_messages,
            allow_reaction=allow_reaction,
            cache_key=cache_key,
        )
        if action is None:
            return

        if not self._still_allowed(channel_id, version, force=force):
            return

        await asyncio.sleep(
            0 if force else random.uniform(settings.min_delay_seconds, settings.max_delay_seconds)
        )
        if not self._still_allowed(channel_id, version, force=force):
            return
        if await self._manual_message_exists(trigger.channel, started_at):
            self.human_activity(channel_id)
            return

        if action.kind == "reaction":
            async with self.reaction_lock:
                if self.store.reaction_allowed(channel_id):
                    assert action.emoji
                    await trigger.add_reaction(action.emoji)
                    self.store.record_assistant_reaction(
                        trigger_message_id=str(trigger.id),
                        channel_id=channel_id,
                        emoji=action.emoji,
                    )
                    return
            action = await self._generate_action(
                messages,
                instructions + "\n\nA written reply is required because a reaction is unavailable.",
                settings,
                max_messages=max_messages,
                allow_reaction=False,
                cache_key=cache_key,
            )
            if (
                action is None
                or action.kind != "messages"
                or not self._still_allowed(channel_id, version, force=force)
            ):
                return
            if await self._manual_message_exists(trigger.channel, started_at):
                self.human_activity(channel_id)
                return

        for index, part in enumerate(action.messages):
            if index:
                await asyncio.sleep(
                    random.uniform(
                        settings.min_message_gap_seconds,
                        settings.max_message_gap_seconds,
                    )
                )
                if not self._still_allowed(channel_id, version, force=force):
                    return
                if await self._manual_message_exists(trigger.channel, started_at):
                    self.human_activity(channel_id)
                    return

            cps = random.uniform(settings.min_typing_cps, settings.max_typing_cps)
            async with trigger.channel.typing():
                await asyncio.sleep(min(12.0, max(0.8, len(part) / cps)))
            if not self._still_allowed(channel_id, version, force=force):
                return
            if await self._manual_message_exists(trigger.channel, started_at):
                self.human_activity(channel_id)
                return

            nonce = secrets.token_hex(12)
            self.store.remember_nonce(nonce)
            outbound = f"🤖 {part}" if settings.robot_prefix else part
            sent = await trigger.channel.send(outbound, nonce=nonce, silent=settings.silent_replies)
            self.store.remember_bot_message(str(sent.id))
            me = sent.author
            self.store.save_message(
                id=str(sent.id),
                channel_id=channel_id,
                author_id=str(me.id),
                author_name=str(me),
                direction="out",
                source="assistant",
                content=part,
                timestamp=sent.created_at.timestamp(),
            )

    def _profile_cache_key(self, settings: AppSettings, personality: dict | None) -> str:
        provider = getattr(self.chatgpt, "active_provider", "chatgpt")
        custom = self.store.custom_provider() if provider == "custom" else None
        protocol = custom.protocol if custom else "responses"
        profile_hash = (personality or {}).get("source_hash", "no-personality")
        identity = "\0".join(
            (
                provider,
                protocol,
                settings.model,
                settings.prompt_locale,
                TOOL_SCHEMA_VERSION,
                settings.base_instructions,
                settings.owner_details,
                str(profile_hash),
            )
        )
        return make_prompt_cache_key("dm-profile", identity)

    async def _generate_action(
        self,
        messages: list[dict[str, Any]],
        instructions: str,
        settings: AppSettings,
        *,
        max_messages: int,
        allow_reaction: bool,
        cache_key: str | None = None,
    ) -> DiscordAction | None:
        provider = self.store.custom_provider() if self.chatgpt.active_provider == "custom" else None
        if provider and not provider.supports("native_function_calls"):
            log.error("Custom provider %s has not passed native function-call validation", provider.name)
            return None
        continuation: list[dict[str, Any]] = []
        repair_used = False
        read_only_calls = 0
        tool_choice: str | dict[str, Any] = "required"
        for request_index in range(4):
            result = await self.chatgpt.complete_result(
                messages,
                instructions,
                settings.model,
                settings.reasoning_effort,
                purpose="dm_reply" if request_index == 0 else "dm_reply_tool_continuation",
                max_output_tokens=settings.max_reply_tokens,
                cache_key=cache_key,
                locale=settings.prompt_locale,
                tools=FUNCTION_TOOLS,
                tool_choice=tool_choice,
                continuation_items=continuation,
            )
            calls = result.function_calls
            if len(calls) != 1 or result.text:
                if repair_used:
                    log.error("Rejected non-terminal or ambiguous model output after native repair")
                    return None
                repair_used = True
                tool_choice = {"type": "function", "name": "send_messages"}
                continue

            call = calls[0]
            output = execute_read_only_tool(
                call,
                owner_timezone=settings.owner_timezone,
            )
            if output is not None:
                if read_only_calls >= 2:
                    log.error("Rejected reply after exceeding the read-only tool budget")
                    return None
                read_only_calls += 1
                continuation.extend((function_call_item(call), function_output_item(call, output)))
                tool_choice = "required"
                continue

            action = validate_discord_action(
                call,
                max_messages=max_messages,
                allow_reaction=allow_reaction,
            )
            if action is not None:
                return action
            if repair_used:
                log.error("Rejected malformed native Discord action after one repair")
                return None
            repair_used = True
            repair_name = call.name if call.name in {"send_messages", "react_to_message"} else "send_messages"
            if repair_name == "react_to_message" and not allow_reaction:
                repair_name = "send_messages"
            tool_choice = {"type": "function", "name": repair_name}
        log.error("Rejected reply after exceeding the total model request budget")
        return None

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
        self.trigger_ids.clear()
