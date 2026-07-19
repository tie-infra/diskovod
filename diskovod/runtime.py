from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from contextlib import AbstractAsyncContextManager
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from .agent import AgentPrompt, build_agent
from .agent_types import AgentRuntimeContext, CapabilityProfile
from .attachments import AttachmentRepository
from .durable_actions import DiscordActionTransport, DurableActionGateway, SideEffectLedger
from .events import DiscordEventQueue, QueuedDiscordEvent
from .localization import assistant_name_for, runtime_context_text
from .persistence import SQLiteLangGraphStore, open_checkpointer
from .providers import ModelService
from .steering import LiveConversationMiddleware
from .store import Store

log = logging.getLogger(__name__)


class AgentService:
    """Per-chat LangGraph execution service and Discord ingress boundary."""

    def __init__(
        self,
        store: Store,
        models: ModelService,
        transport: DiscordActionTransport,
        checkpoint_secret: str,
    ):
        self.store = store
        self.models = models
        self.transport = transport
        self.checkpoint_secret = checkpoint_secret
        self.events = DiscordEventQueue(store.path)
        self.attachments = AttachmentRepository(store.path)
        self.memory = SQLiteLangGraphStore(store.path)
        self.ledger = SideEffectLedger(store.path)
        self.gateway = DurableActionGateway(self.ledger, transport)
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self._trace_buffers: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._checkpoint_context: AbstractAsyncContextManager[AsyncSqliteSaver] | None = None
        self.checkpointer: AsyncSqliteSaver | None = None

    async def start(self) -> None:
        if self.checkpointer is not None:
            return
        self._checkpoint_context = open_checkpointer(self.store.path, self.checkpoint_secret)
        self.checkpointer = await self._checkpoint_context.__aenter__()
        for channel_id in self._pending_channels():
            self._ensure_task(channel_id)

    async def close(self) -> None:
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
        if self._checkpoint_context is not None:
            await self._checkpoint_context.__aexit__(None, None, None)
            self._checkpoint_context = None
            self.checkpointer = None
        self.events.close()
        self.attachments.close()
        self.memory.close()
        self.ledger.close()

    @property
    def ready(self) -> bool:
        return self.checkpointer is not None and self.models.ready

    def cancel(self, channel_id: str) -> None:
        task = self.tasks.pop(channel_id, None)
        if task is not None:
            task.cancel()

    def permanently_pause(self, channel_id: str) -> None:
        self.store.set_permanent_pause(channel_id, True)
        self.cancel(channel_id)

    def human_activity(self, channel_id: str) -> float:
        conversation = self.store.conversation(channel_id)
        if conversation and conversation["mode"] == "inline" and not conversation["paused"]:
            return time.time()
        settings = self.store.app_settings()
        quiet_minutes = random.uniform(
            settings.min_human_quiet_minutes,
            settings.max_human_quiet_minutes,
        )
        snoozed_until = self.store.snooze(channel_id, quiet_minutes * 60)
        self.cancel(channel_id)
        return snoozed_until

    def submit_message(
        self,
        *,
        message_id: str,
        channel_id: str,
        account_id: str,
        author_id: str,
        author_name: str,
        participant_role: str,
        content: str,
        attachments: list[dict[str, Any]],
        observed_at: float,
        edited: bool = False,
        force: bool = False,
        agent_input: bool | None = None,
    ) -> bool:
        mode = self._mode(channel_id)
        automate = force or (
            self.store.app_settings().enabled
            and self.store.can_automate(channel_id)
            and (participant_role == "peer" or mode == "inline")
        )
        if agent_input is not None:
            automate = agent_input
        event_id = (
            f"discord:edit:{message_id}:{int(observed_at * 1_000_000)}"
            if edited
            else f"discord:message:{message_id}"
        )
        payload = {
            "message_id": message_id,
            "account_id": account_id,
            "author_id": author_id,
            "author_name": author_name,
            "participant_role": participant_role,
            "content": content,
            "attachments": attachments,
        }
        inserted = self.events.ingest(
            event_id,
            channel_id,
            "edit" if edited else "message",
            payload,
            observed_at=observed_at,
            enqueue=automate,
        )
        if inserted and automate:
            self.events.thread_id(account_id, channel_id)
            self._ensure_task(channel_id, force=force)
        return inserted

    def submit_delete(
        self,
        *,
        message_id: str,
        channel_id: str,
        account_id: str,
        observed_at: float | None = None,
    ) -> bool:
        automate = self.store.app_settings().enabled and self.store.can_automate(channel_id)
        timestamp = observed_at or time.time()
        inserted = self.events.ingest(
            f"discord:delete:{message_id}:{int(timestamp * 1_000_000)}",
            channel_id,
            "delete",
            {"message_id": message_id, "account_id": account_id},
            observed_at=timestamp,
            enqueue=automate,
        )
        if inserted and automate:
            self._ensure_task(channel_id)
        return inserted

    def force_reply(self, *, channel_id: str, account_id: str, trigger_message_id: str) -> None:
        event_id = f"diskovod:force:{uuid.uuid4()}"
        self.events.ingest(
            event_id,
            channel_id,
            "force_reply",
            {
                "message_id": trigger_message_id,
                "account_id": account_id,
                "participant_role": "control",
            },
            enqueue=True,
        )
        self.events.thread_id(account_id, channel_id)
        self._ensure_task(channel_id, force=True)

    def set_live_steering(self, channel_id: str, enabled: bool) -> None:
        account_id = self._account_id(channel_id)
        self.events.set_live_steering(account_id, channel_id, enabled)

    async def claim_escalation(self, escalation_id: str) -> bool:
        return self.store.set_interrupt_state(escalation_id, "claimed")

    async def resume_escalation(
        self,
        escalation_id: str,
        *,
        action: str,
        owner_message: str = "",
    ) -> bool:
        if action not in {"resolved", "dismissed", "owner_reply"}:
            raise ValueError("Unknown escalation resolution")
        if self.checkpointer is None or not self.models.ready:
            raise RuntimeError("The agent runtime is not ready")
        escalation = self.store.escalation_interrupt(escalation_id)
        if escalation is None or escalation["state"] not in {"pending", "claimed"}:
            return False
        payload = escalation["payload"]
        tool_call_id = str(payload.get("tool_call_id") or "")
        suffix = f":{tool_call_id}"
        trace_id = escalation_id[: -len(suffix)] if tool_call_id and escalation_id.endswith(suffix) else ""
        with self.store._lock:
            run = self.store._db.execute("SELECT * FROM agent_runs WHERE trace_id=?", (trace_id,)).fetchone()
        if run is None:
            raise RuntimeError("The interrupted agent run cannot be found")
        channel_id = str(escalation["channel_id"])
        thread_id = str(escalation["thread_id"])
        settings = self.store.app_settings()
        configuration = self.models.configuration
        assert configuration is not None
        conversation = self.store.conversation(channel_id)
        account_id = self._account_id(channel_id)
        context = AgentRuntimeContext(
            account_id=account_id,
            channel_id=channel_id,
            participant_ids=(str(conversation["peer_id"]),) if conversation else (),
            owner_id=account_id,
            ui_locale=settings.admin_locale,
            prompt_locale=settings.prompt_locale,
            assistant_name=assistant_name_for(settings.prompt_locale, settings.assistant_name),
            automation_mode=str(conversation["mode"] if conversation else "automatic"),
            force_reply=False,
            provider_id=configuration.provider_id,
            model_id=configuration.model_id,
            transport_profile=configuration.transport_profile,
            capabilities=self._capabilities(configuration),
            trace_id=trace_id,
            thread_id=thread_id,
            owner_timezone=settings.owner_timezone,
            trigger_message_id=str(payload.get("trigger_message_id") or ""),
            permissions=frozenset({"send_messages", "reactions", "owner_escalation"}),
        )
        personality = self.store.personality() or {}
        agent = build_agent(
            self.models.build_model(),
            self.gateway,
            AgentPrompt(
                settings.prompt_locale,
                context.assistant_name,
                settings.base_instructions,
                str(personality.get("profile") or ""),
                settings.owner_details,
            ),
            checkpointer=self.checkpointer,
            store=self.memory,
            extra_middleware=(LiveConversationMiddleware(self.events, settings.prompt_locale),),
            attachments=self.attachments,
            diagnostics=self._buffer_trace,
        )
        resume_payload = {
            "action": action,
            "owner_message": owner_message[:4000],
            "resolved_at": time.time(),
        }
        self.store.record_agent_trace(str(run["id"]), "interrupt_resume", resume_payload)
        result = await agent.ainvoke(
            Command(resume=resume_payload),
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 40},
            context=context,
        )
        self._flush_trace(trace_id, str(run["id"]))
        state = "dismissed" if action == "dismissed" else "resolved"
        self.store.set_interrupt_state(escalation_id, state)
        self.store.finish_agent_run(
            str(run["id"]),
            "interrupted" if result.get("__interrupt__") else "completed",
        )
        if not result.get("__interrupt__") and self._has_pending(channel_id):
            self._ensure_task(channel_id)
        return True

    async def resume_escalation_for_owner_reply(
        self,
        channel_id: str,
        owner_message: str,
    ) -> bool:
        escalation = self.store.active_interrupt_for_channel(channel_id)
        if escalation is None:
            return False
        return await self.resume_escalation(
            str(escalation["id"]),
            action="owner_reply",
            owner_message=owner_message,
        )

    def _ensure_task(self, channel_id: str, *, force: bool = False) -> None:
        if self.checkpointer is None:
            return
        running = self.tasks.get(channel_id)
        if running is not None and not running.done():
            return
        task = asyncio.create_task(self._run(channel_id, force=force), name=f"agent-{channel_id}")
        self.tasks[channel_id] = task
        task.add_done_callback(lambda done: self._finished(channel_id, done))

    def _finished(self, channel_id: str, task: asyncio.Task[None]) -> None:
        if self.tasks.get(channel_id) is task:
            self.tasks.pop(channel_id, None)
        if not task.cancelled() and (error := task.exception()):
            log.error("Agent run failed for %s: %s", channel_id, error)
        if not self._is_interrupted(channel_id) and self._has_pending(channel_id):
            self._ensure_task(channel_id)

    async def _run(self, channel_id: str, *, force: bool) -> None:
        if self.checkpointer is None or not self.models.ready:
            return
        settings = self.store.app_settings()
        if not force:
            await asyncio.sleep(settings.debounce_seconds)
        account_id = self._account_id(channel_id)
        thread_id = self.events.thread_id(account_id, channel_id)
        run_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        claimed = self.events.claim_ready(channel_id, run_id)
        if not claimed:
            return
        force = force or any(event.kind == "force_reply" for event in claimed)
        trigger_message_id = next(
            (
                str(event.payload.get("message_id") or "")
                for event in reversed(claimed)
                if event.payload.get("participant_role") == "peer"
            ),
            str(claimed[-1].payload.get("message_id") or ""),
        )
        configuration = self.models.configuration
        assert configuration is not None
        conversation = self.store.conversation(channel_id)
        participant_ids = tuple(
            dict.fromkeys(
                str(event.payload.get("author_id")) for event in claimed if event.payload.get("author_id")
            )
        )
        context = AgentRuntimeContext(
            account_id=account_id,
            channel_id=channel_id,
            participant_ids=participant_ids,
            owner_id=account_id,
            ui_locale=settings.admin_locale,
            prompt_locale=settings.prompt_locale,
            assistant_name=assistant_name_for(settings.prompt_locale, settings.assistant_name),
            automation_mode=str(conversation["mode"] if conversation else "automatic"),
            force_reply=force,
            provider_id=configuration.provider_id,
            model_id=configuration.model_id,
            transport_profile=configuration.transport_profile,
            capabilities=self._capabilities(configuration),
            trace_id=trace_id,
            thread_id=thread_id,
            owner_timezone=settings.owner_timezone,
            trigger_message_id=trigger_message_id,
            permissions=frozenset({"send_messages", "reactions", "owner_escalation"}),
        )
        personality = self.store.personality() or {}
        prompt = AgentPrompt(
            settings.prompt_locale,
            context.assistant_name,
            settings.base_instructions,
            str(personality.get("profile") or ""),
            settings.owner_details,
        )
        agent = build_agent(
            self.models.build_model(),
            self.gateway,
            prompt,
            checkpointer=self.checkpointer,
            store=self.memory,
            extra_middleware=(LiveConversationMiddleware(self.events, settings.prompt_locale),),
            attachments=self.attachments,
            diagnostics=self._buffer_trace,
        )
        initial_messages = [message for event in claimed if (message := self._message(event)) is not None]
        state = {
            "messages": initial_messages,
            "logical_request_id": run_id,
            "claimed_event_ids": [event.id for event in claimed],
        }
        config = {
            "configurable": {"thread_id": thread_id},
            "run_id": uuid.UUID(run_id),
            "recursion_limit": 40,
        }
        self.store.start_agent_run(
            run_id=run_id,
            thread_id=thread_id,
            channel_id=channel_id,
            trace_id=trace_id,
        )
        self.store.record_agent_trace(
            run_id,
            "run_input",
            {"event_ids": [event.id for event in claimed], "force_reply": force},
        )
        try:
            result = await agent.ainvoke(state, config=config, context=context)
        except asyncio.CancelledError:
            self.events.release(channel_id, run_id)
            self._flush_trace(trace_id, run_id)
            self.store.finish_agent_run(run_id, "cancelled")
            raise
        except Exception as error:
            self.events.release(channel_id, run_id)
            self._flush_trace(trace_id, run_id)
            self.store.finish_agent_run(run_id, "failed", f"{type(error).__name__}: {error}")
            self.store.record_agent_trace(
                run_id,
                "run_error",
                {"type": type(error).__name__, "detail": str(error)[:4000]},
            )
            raise
        interrupted = bool(result.get("__interrupt__"))
        self._flush_trace(trace_id, run_id)
        self.events.complete(channel_id, run_id)
        self.store.finish_agent_run(run_id, "interrupted" if interrupted else "completed")
        self.store.record_agent_trace(
            run_id,
            "run_output",
            {
                "interrupted": interrupted,
                "message_count": len(result.get("messages", [])),
                "successful_written_sends": result.get("successful_written_sends", 0),
            },
        )

    def _message(self, event: QueuedDiscordEvent) -> BaseMessage | None:
        if event.kind == "force_reply":
            return None
        if event.kind == "delete":
            return RemoveMessage(id=str(event.payload["message_id"]))
        content = str(event.payload.get("content") or "")
        attachments = event.payload.get("attachments") or []
        text_bundle = runtime_context_text(self.store.app_settings().prompt_locale)
        if attachments:
            notes = []
            for attachment in attachments:
                if not isinstance(attachment, dict):
                    continue
                filename = str(attachment.get("filename") or "attachment")
                media_type = str(attachment.get("content_type") or "unknown")
                size = int(attachment.get("size") or 0)
                note = f"- {filename} ({media_type}, {size} bytes)"
                if text := attachment.get("text"):
                    note += (
                        f"\n<untrusted_attachment_text filename={filename!r}>\n"
                        f"{text}\n</untrusted_attachment_text>"
                    )
                notes.append(note)
            if notes:
                content = (content.strip() + "\n\n" if content.strip() else "") + (
                    text_bundle["attachments"] + "\n" + "\n".join(notes)
                )
        return HumanMessage(
            content=content,
            id=str(event.payload.get("message_id") or event.id),
            additional_kwargs={
                "diskovod_participant": {
                    "id": str(event.payload.get("author_id") or "unknown"),
                    "name": str(event.payload.get("author_name") or text_bundle["unknown_participant"]),
                    "role": str(event.payload.get("participant_role") or "peer"),
                    "discord_event_id": event.id,
                    "edited": event.kind == "edit",
                },
                "diskovod_attachments": attachments,
            },
        )

    def _account_id(self, channel_id: str) -> str:
        with self.store._lock:
            row = self.store._db.execute(
                "SELECT account_id FROM chat_threads WHERE channel_id=?", (channel_id,)
            ).fetchone()
        if row:
            return str(row["account_id"])
        client = getattr(self.transport, "client", None)
        if client is not None and getattr(client, "user", None) is not None:
            return str(client.user.id)
        credentials = self.store.chat_credentials()
        return str(credentials.account_id if credentials and credentials.account_id else "discord-owner")

    def _mode(self, channel_id: str) -> str:
        conversation = self.store.conversation(channel_id)
        return str(conversation["mode"] if conversation else "automatic")

    def _has_pending(self, channel_id: str) -> bool:
        with self.store._lock:
            row = self.store._db.execute(
                "SELECT 1 FROM chat_event_queue WHERE channel_id=? AND disposition='pending' LIMIT 1",
                (channel_id,),
            ).fetchone()
        return row is not None

    def _pending_channels(self) -> list[str]:
        with self.store._lock:
            rows = self.store._db.execute(
                "SELECT DISTINCT channel_id FROM chat_event_queue WHERE disposition='pending'"
            ).fetchall()
        return [str(row["channel_id"]) for row in rows]

    def _is_interrupted(self, channel_id: str) -> bool:
        with self.store._lock:
            row = self.store._db.execute(
                "SELECT 1 FROM escalation_interrupts WHERE channel_id=? AND state IN ('pending','claimed') LIMIT 1",
                (channel_id,),
            ).fetchone()
        return row is not None

    def _buffer_trace(self, trace_id: str, kind: str, payload: dict[str, Any]) -> None:
        self._trace_buffers.setdefault(trace_id, []).append((kind, payload))

    def _flush_trace(self, trace_id: str, run_id: str) -> None:
        for kind, payload in self._trace_buffers.pop(trace_id, []):
            self.store.record_agent_trace(run_id, kind, payload)

    @staticmethod
    def _capabilities(configuration) -> CapabilityProfile:
        return CapabilityProfile(
            native_tools=configuration.capabilities.native_tools,
            hosted_web_search=configuration.capabilities.hosted_web_search,
            image_input=configuration.capabilities.image_input,
            file_input=configuration.capabilities.file_input,
            prompt_cache=configuration.capabilities.prompt_cache,
            standard_content_blocks=configuration.capabilities.standard_content_blocks,
            details=configuration.capabilities.details,
        )
