from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from contextlib import AbstractAsyncContextManager
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from .agent import AgentPrompt, build_agent
from .agent_types import AgentRuntimeContext, CapabilityProfile, DiskovodAgentState
from .attachments import AttachmentRepository
from .agent_actions import DeliveryRecord
from .durable_actions import DiscordActionTransport, DurableActionGateway, SideEffectLedger
from .events import DiscordEventQueue, QueuedDiscordEvent
from .http_client import PublicHTTP
from .localization import assistant_name_for, runtime_context_text, summarization_prompt
from .persistence import SQLiteLangGraphStore, open_checkpointer
from .providers import ModelService
from .steering import LiveConversationMiddleware
from .store import Store

log = logging.getLogger(__name__)
ROLLOVER_MESSAGE_LIMIT = 400


class EmulatedActionGateway:
    """Capture replay actions without contacting Discord or changing durable action state."""

    def __init__(self):
        self.actions: list[dict[str, Any]] = []

    async def send_messages(self, context, messages, *, tool_call_id):
        records = [
            DeliveryRecord("accepted", index, f"emulated:{tool_call_id}:{index}")
            for index, _ in enumerate(messages)
        ]
        self.actions.append(
            {"action": "send_messages", "messages": list(messages), "tool_call_id": tool_call_id}
        )
        return records

    async def react_to_message(self, context, emoji, *, tool_call_id):
        del context
        self.actions.append({"action": "react_to_message", "emoji": emoji, "tool_call_id": tool_call_id})
        return DeliveryRecord("accepted", 0, f"emulated:{tool_call_id}:reaction")

    async def record_escalation(self, context, payload, *, tool_call_id):
        del context
        self.actions.append({"action": "escalate_to_owner", "payload": payload, "tool_call_id": tool_call_id})


class AgentService:
    """Per-chat LangGraph execution service and Discord ingress boundary."""

    def __init__(
        self,
        store: Store,
        models: ModelService,
        transport: DiscordActionTransport,
        checkpoint_secret: str,
        http: PublicHTTP,
    ):
        self.store = store
        self.models = models
        self.transport = transport
        self.checkpoint_secret = checkpoint_secret
        self.http = http
        self.events = DiscordEventQueue(store.database)
        self.attachments = AttachmentRepository(store.database, http)
        self.memory = SQLiteLangGraphStore(store.path, store.database)
        self.ledger = SideEffectLedger(store.database)
        self.gateway = DurableActionGateway(self.ledger, transport)
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self._trace_buffers: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._checkpoint_context: AbstractAsyncContextManager[AsyncSqliteSaver] | None = None
        self.checkpointer: AsyncSqliteSaver | None = None
        self._closing = False

    async def start(self) -> None:
        if self.checkpointer is not None:
            return
        self._closing = False
        await self.store.start()
        await self.memory.start()
        self._checkpoint_context = open_checkpointer(self.store.path, self.checkpoint_secret)
        self.checkpointer = await self._checkpoint_context.__aenter__()
        for channel_id in await self.store.apending_channels():
            self._ensure_task(channel_id)

    async def close(self) -> None:
        self._closing = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
        if self._checkpoint_context is not None:
            await self._checkpoint_context.__aexit__(None, None, None)
            self._checkpoint_context = None
            self.checkpointer = None

    @property
    def ready(self) -> bool:
        return self.checkpointer is not None and self.models.ready

    def cancel(self, channel_id: str) -> None:
        task = self.tasks.pop(channel_id, None)
        if task is not None:
            task.cancel()

    async def permanently_pause(self, channel_id: str) -> None:
        await self.store.aset_permanent_pause(channel_id, True)
        self.cancel(channel_id)

    async def human_activity(self, channel_id: str) -> float:
        snoozed_until, should_cancel = await self._record_human_activity(channel_id)
        if should_cancel:
            self.cancel(channel_id)
        return snoozed_until

    async def _record_human_activity(self, channel_id: str) -> tuple[float, bool]:
        conversation = await self.store.aconversation(channel_id)
        if conversation and conversation["mode"] == "inline" and not conversation["paused"]:
            return time.time(), False
        settings = self.store.automation_settings()
        quiet_minutes = random.uniform(
            settings.min_human_quiet_minutes,
            settings.max_human_quiet_minutes,
        )
        return await self.store.asnooze(channel_id, quiet_minutes * 60), True

    async def submit_message(
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
        inserted, automate = await self._ingest_message(
            message_id=message_id,
            channel_id=channel_id,
            account_id=account_id,
            author_id=author_id,
            author_name=author_name,
            participant_role=participant_role,
            content=content,
            attachments=attachments,
            observed_at=observed_at,
            edited=edited,
            force=force,
            agent_input=agent_input,
        )
        if inserted and automate:
            self._ensure_task(channel_id, force=force)
        return inserted

    async def _ingest_message(
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
        edited: bool,
        force: bool,
        agent_input: bool | None,
    ) -> tuple[bool, bool]:
        mode = await self._mode(channel_id)
        automate = force or (
            self.store.automation_settings().enabled
            and await self.store.acan_automate(channel_id)
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
        inserted = await self.events.ingest(
            event_id,
            channel_id,
            "edit" if edited else "message",
            payload,
            observed_at=observed_at,
            enqueue=automate,
        )
        if inserted and automate:
            await self.events.thread_id(account_id, channel_id)
        return inserted, automate

    async def submit_delete(
        self,
        *,
        message_id: str,
        channel_id: str,
        account_id: str,
        observed_at: float | None = None,
    ) -> bool:
        inserted, automate = await self._ingest_delete(
            message_id=message_id,
            channel_id=channel_id,
            account_id=account_id,
            observed_at=observed_at,
        )
        if inserted and automate:
            self._ensure_task(channel_id)
        return inserted

    async def _ingest_delete(
        self,
        *,
        message_id: str,
        channel_id: str,
        account_id: str,
        observed_at: float | None,
    ) -> tuple[bool, bool]:
        automate = self.store.automation_settings().enabled and await self.store.acan_automate(channel_id)
        timestamp = observed_at or time.time()
        inserted = await self.events.ingest(
            f"discord:delete:{message_id}:{int(timestamp * 1_000_000)}",
            channel_id,
            "delete",
            {"message_id": message_id, "account_id": account_id},
            observed_at=timestamp,
            enqueue=automate,
        )
        return inserted, automate

    async def force_reply(self, *, channel_id: str, account_id: str, trigger_message_id: str) -> None:
        await self._ingest_force_reply(
            channel_id=channel_id,
            account_id=account_id,
            trigger_message_id=trigger_message_id,
        )
        self._ensure_task(channel_id, force=True)

    async def _ingest_force_reply(self, *, channel_id: str, account_id: str, trigger_message_id: str) -> None:
        event_id = f"diskovod:force:{uuid.uuid4()}"
        await self.events.ingest(
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
        await self.events.thread_id(account_id, channel_id)

    async def set_live_steering(self, channel_id: str, enabled: bool) -> None:
        await self.events.set_live_steering(await self._account_id(channel_id), channel_id, enabled)

    async def checkpoint_views(self, *, limit_per_thread: int = 20) -> list[dict[str, Any]]:
        if self.checkpointer is None:
            return []
        rows = await self.store.achat_threads()
        result: list[dict[str, Any]] = []
        for row in rows:
            thread = dict(row)
            history = []
            async for item in self.checkpointer.alist(
                {"configurable": {"thread_id": thread["thread_id"]}},
                limit=limit_per_thread,
            ):
                values = item.checkpoint.get("channel_values", {})
                history.append(
                    {
                        "checkpoint_id": str(item.config["configurable"]["checkpoint_id"]),
                        "created_at": str(item.checkpoint.get("ts") or ""),
                        "step": item.metadata.get("step"),
                        "source": item.metadata.get("source"),
                        "message_count": len(values.get("messages", [])),
                    }
                )
            thread["checkpoints"] = history
            result.append(thread)
        return result

    async def replay_checkpoint(self, thread_id: str, checkpoint_id: str) -> str:
        """Replay a historical state with isolated memory and emulated Discord actions."""
        if self.checkpointer is None or not self.models.ready:
            raise RuntimeError("The agent runtime is not ready")
        snapshot = await self.checkpointer.aget_tuple(
            {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        )
        if snapshot is None:
            raise ValueError("Unknown checkpoint")
        thread = await self.store.achat_thread_by_id(thread_id)
        if thread is None:
            raise ValueError("Unknown checkpoint thread")
        channel_id = str(thread["channel_id"])
        configuration = self.models.configuration
        assert configuration is not None
        interface = self.store.interface_settings()
        profile = self.store.assistant_profile()
        conversation = await self.store.aconversation(channel_id)
        replay_id = str(uuid.uuid4())
        trace_id = f"replay:{replay_id}"
        context = AgentRuntimeContext(
            account_id=str(thread["account_id"]),
            channel_id=channel_id,
            participant_ids=(str(conversation["peer_id"]),) if conversation else (),
            owner_id=str(thread["account_id"]),
            ui_locale=interface.locale,
            prompt_locale=profile.prompt_locale,
            assistant_name=assistant_name_for(profile.prompt_locale, profile.assistant_name),
            automation_mode=str(conversation["mode"] if conversation else "automatic"),
            force_reply=False,
            provider_id=configuration.provider_id,
            model_id=configuration.model_id,
            transport_profile=configuration.transport_profile,
            capabilities=self._capabilities(configuration),
            trace_id=trace_id,
            thread_id=f"replay:{thread_id}:{checkpoint_id}",
            owner_timezone=profile.owner_timezone,
            trigger_message_id="historical-replay",
            permissions=frozenset({"send_messages", "reactions", "owner_escalation"}),
        )
        personality = self.store.personality() or {}
        gateway = EmulatedActionGateway()
        agent = build_agent(
            self.models.build_model(),
            gateway,
            AgentPrompt(
                profile.prompt_locale,
                context.assistant_name,
                profile.base_instructions,
                str(personality.get("profile") or ""),
                profile.owner_details,
            ),
            self.http,
            checkpointer=InMemorySaver(),
            store=InMemoryStore(),
            attachments=self.attachments,
            diagnostics=self._buffer_trace,
            hosted_web_search=context.capabilities.hosted_web_search,
        )
        await self.store.astart_agent_run(
            run_id=replay_id,
            thread_id=context.thread_id,
            channel_id=channel_id,
            trace_id=trace_id,
        )
        await self.store.arecord_agent_trace(
            replay_id,
            "historical_replay",
            {"source_thread_id": thread_id, "source_checkpoint_id": checkpoint_id},
        )
        try:
            await agent.ainvoke(
                {"messages": snapshot.checkpoint.get("channel_values", {}).get("messages", [])},
                config={"configurable": {"thread_id": context.thread_id}, "recursion_limit": 40},
                context=context,
            )
        except Exception as error:
            await self._flush_trace(trace_id, replay_id)
            await self.store.arecord_agent_trace(
                replay_id,
                "emulated_actions",
                {"actions": gateway.actions},
            )
            await self.store.afinish_agent_run(
                replay_id,
                "failed",
                f"{type(error).__name__}: {error}",
            )
            raise
        await self._flush_trace(trace_id, replay_id)
        await self.store.arecord_agent_trace(
            replay_id,
            "emulated_actions",
            {"actions": gateway.actions},
        )
        await self.store.afinish_agent_run(replay_id, "completed")
        return replay_id

    async def apply_configuration_transition(self, previous) -> int:
        """Roll provider-affine checkpoint histories at a completed-turn boundary."""
        current = self.models.configuration
        if previous is None or current is None or self._affinity(previous) == self._affinity(current):
            return 0
        if any(not task.done() for task in self.tasks.values()):
            raise RuntimeError("Model configuration cannot change while an agent run is active")
        if await self.store.aactive_interrupts():
            raise RuntimeError("Resolve active owner escalations before changing the model")
        threads = await self.store.achat_threads()
        rolled = 0
        for thread in threads:
            if await self._roll_thread(thread, reason="model_configuration_changed"):
                rolled += 1
        return rolled

    async def ensure_configuration_transition_allowed(self) -> None:
        if any(not task.done() for task in self.tasks.values()):
            raise RuntimeError("Model configuration cannot change while an agent run is active")
        if await self.store.aactive_interrupts():
            raise RuntimeError("Resolve active owner escalations before changing the model")

    async def _roll_if_needed(self, channel_id: str, thread_id: str) -> None:
        if self.checkpointer is None:
            return
        snapshot = await self.checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
        messages = snapshot.checkpoint.get("channel_values", {}).get("messages", []) if snapshot else []
        if len(messages) <= ROLLOVER_MESSAGE_LIMIT:
            return
        row = await self.store.achat_thread_for_channel(channel_id)
        if row:
            await self._roll_thread(row, reason="completed_turn_retention")

    async def _roll_thread(self, thread: dict[str, Any], *, reason: str) -> bool:
        if self.checkpointer is None:
            return False
        thread_id = str(thread["thread_id"])
        snapshot = await self.checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
        if snapshot is None:
            return False
        messages = snapshot.checkpoint.get("channel_values", {}).get("messages", [])
        if not messages:
            return False
        locale = self.store.assistant_profile().prompt_locale
        rendered = "\n".join(
            f"[{getattr(message, 'type', 'message')} {getattr(message, 'id', '')}] {message.text}"
            for message in messages
        )[-100_000:]
        try:
            response = await self.models.build_model().ainvoke(
                [SystemMessage(summarization_prompt(locale)), HumanMessage(rendered)]
            )
            summary = response.text.strip()
        except Exception as error:
            log.warning("Could not summarize checkpoint generation %s: %s", thread_id, error)
            summary = ""
        if not summary:
            summary = runtime_context_text(locale)["rollover_summary"].format(count=len(messages))
        new_thread_id = await self.events.roll_generation(
            str(thread["account_id"]),
            str(thread["channel_id"]),
        )

        async def seed(state: DiskovodAgentState) -> dict[str, Any]:
            del state
            return {}

        builder = StateGraph(DiskovodAgentState)
        builder.add_node("seed", seed)
        builder.add_edge(START, "seed")
        builder.add_edge("seed", END)
        graph = builder.compile(checkpointer=self.checkpointer, store=self.memory)
        await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        summary,
                        id=f"rollover:{thread_id}",
                        additional_kwargs={
                            "diskovod_generation_summary": {
                                "source_thread_id": thread_id,
                                "reason": reason,
                                "message_count": len(messages),
                            }
                        },
                    )
                ]
            },
            config={"configurable": {"thread_id": new_thread_id}},
        )
        return True

    @staticmethod
    def _affinity(configuration) -> tuple[str, str, str]:
        return (
            configuration.provider_id,
            configuration.model_id,
            configuration.transport_profile,
        )

    async def claim_escalation(self, escalation_id: str) -> bool:
        return await self.store.aset_interrupt_state(escalation_id, "claimed")

    async def resume_escalation(
        self,
        escalation_id: str,
        *,
        action: str,
        owner_message: str = "",
        owner_message_id: str = "",
        owner_author_id: str = "",
        owner_author_name: str = "",
    ) -> bool:
        if action not in {"resolved", "dismissed", "owner_reply"}:
            raise ValueError("Unknown escalation resolution")
        if self.checkpointer is None or not self.models.ready:
            raise RuntimeError("The agent runtime is not ready")
        escalation = await self.store.aescalation_interrupt(escalation_id)
        if escalation is None or escalation["state"] not in {"pending", "claimed"}:
            return False
        payload = escalation["payload"]
        tool_call_id = str(payload.get("tool_call_id") or "")
        suffix = f":{tool_call_id}"
        trace_id = escalation_id[: -len(suffix)] if tool_call_id and escalation_id.endswith(suffix) else ""
        run = await self.store.aagent_run_for_trace(trace_id)
        if run is None:
            raise RuntimeError("The interrupted agent run cannot be found")
        channel_id = str(escalation["channel_id"])
        thread_id = str(escalation["thread_id"])
        interface = self.store.interface_settings()
        profile = self.store.assistant_profile()
        configuration = self.models.configuration
        assert configuration is not None
        conversation = await self.store.aconversation(channel_id)
        account_id = await self._account_id(channel_id)
        context = AgentRuntimeContext(
            account_id=account_id,
            channel_id=channel_id,
            participant_ids=(str(conversation["peer_id"]),) if conversation else (),
            owner_id=account_id,
            ui_locale=interface.locale,
            prompt_locale=profile.prompt_locale,
            assistant_name=assistant_name_for(profile.prompt_locale, profile.assistant_name),
            automation_mode=str(conversation["mode"] if conversation else "automatic"),
            force_reply=False,
            provider_id=configuration.provider_id,
            model_id=configuration.model_id,
            transport_profile=configuration.transport_profile,
            capabilities=self._capabilities(configuration),
            trace_id=trace_id,
            thread_id=thread_id,
            owner_timezone=profile.owner_timezone,
            trigger_message_id=str(payload.get("trigger_message_id") or ""),
            permissions=frozenset({"send_messages", "reactions", "owner_escalation"}),
        )
        personality = self.store.personality() or {}
        agent = build_agent(
            self.models.build_model(),
            self.gateway,
            AgentPrompt(
                profile.prompt_locale,
                context.assistant_name,
                profile.base_instructions,
                str(personality.get("profile") or ""),
                profile.owner_details,
            ),
            self.http,
            checkpointer=self.checkpointer,
            store=self.memory,
            extra_middleware=(LiveConversationMiddleware(self.events, profile.prompt_locale),),
            attachments=self.attachments,
            diagnostics=self._buffer_trace,
            hosted_web_search=context.capabilities.hosted_web_search,
        )
        resume_payload = {
            "action": action,
            "owner_message": owner_message[:4000],
            "resolved_at": time.time(),
        }
        await self.store.arecord_agent_trace(
            str(run["id"]),
            "interrupt_resume",
            resume_payload,
        )
        invocation_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 40,
        }
        if action == "owner_reply" and owner_message.strip():
            await agent.aupdate_state(
                invocation_config,
                {
                    "messages": [
                        HumanMessage(
                            owner_message[:4000],
                            id=owner_message_id or f"owner-reply:{time.time_ns()}",
                            additional_kwargs={
                                "diskovod_participant": {
                                    "id": owner_author_id or account_id,
                                    "name": owner_author_name
                                    or runtime_context_text(profile.prompt_locale)["account_owner"],
                                    "role": "owner",
                                    "discord_event_id": (
                                        f"discord:message:{owner_message_id}" if owner_message_id else ""
                                    ),
                                }
                            },
                        )
                    ]
                },
            )
        try:
            result = await agent.ainvoke(
                Command(resume=resume_payload),
                config=invocation_config,
                context=context,
            )
        except Exception as error:
            await self._flush_trace(trace_id, str(run["id"]))
            await self.store.arecord_agent_trace(
                str(run["id"]),
                "interrupt_resume_error",
                {"type": type(error).__name__, "detail": str(error)[:4000]},
            )
            await self.store.afinish_agent_run(
                str(run["id"]),
                "failed",
                f"{type(error).__name__}: {error}",
            )
            raise
        await self._flush_trace(trace_id, str(run["id"]))
        state = "dismissed" if action == "dismissed" else "resolved"
        await self.store.aset_interrupt_state(escalation_id, state)
        await self.store.afinish_agent_run(
            str(run["id"]),
            "interrupted" if result.get("__interrupt__") else "completed",
        )
        if not result.get("__interrupt__") and await self.store.ahas_pending_events(channel_id):
            self._ensure_task(channel_id)
        return True

    async def resume_escalation_for_owner_reply(
        self,
        channel_id: str,
        owner_message: str,
        *,
        message_id: str = "",
        author_id: str = "",
        author_name: str = "",
    ) -> bool:
        escalation = await self.store.aactive_interrupt_for_channel(channel_id)
        if escalation is None:
            return False
        return await self.resume_escalation(
            str(escalation["id"]),
            action="owner_reply",
            owner_message=owner_message,
            owner_message_id=message_id,
            owner_author_id=author_id,
            owner_author_name=author_name,
        )

    def _ensure_task(self, channel_id: str, *, force: bool = False) -> None:
        if self._closing or self.checkpointer is None:
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
            log.error(
                "Agent run failed for %s: %s",
                channel_id,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
        if not self._closing:
            asyncio.create_task(self._resume_pending(channel_id), name=f"agent-resume-{channel_id}")

    async def _resume_pending(self, channel_id: str) -> None:
        if self._closing:
            return
        if not await self.store.ahas_active_interrupt(channel_id) and await self.store.ahas_pending_events(
            channel_id
        ):
            self._ensure_task(channel_id)

    async def _run(self, channel_id: str, *, force: bool) -> None:
        if self.checkpointer is None or not self.models.ready:
            return
        automation = self.store.automation_settings()
        interface = self.store.interface_settings()
        profile = self.store.assistant_profile()
        if not force:
            await asyncio.sleep(automation.debounce_seconds)
        account_id = await self._account_id(channel_id)
        thread_id = await self.events.thread_id(account_id, channel_id)
        run_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        claimed = await self.events.claim_ready(channel_id, run_id)
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
        conversation = await self.store.aconversation(channel_id)
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
            ui_locale=interface.locale,
            prompt_locale=profile.prompt_locale,
            assistant_name=assistant_name_for(profile.prompt_locale, profile.assistant_name),
            automation_mode=str(conversation["mode"] if conversation else "automatic"),
            force_reply=force,
            provider_id=configuration.provider_id,
            model_id=configuration.model_id,
            transport_profile=configuration.transport_profile,
            capabilities=self._capabilities(configuration),
            trace_id=trace_id,
            thread_id=thread_id,
            owner_timezone=profile.owner_timezone,
            trigger_message_id=trigger_message_id,
            permissions=frozenset({"send_messages", "reactions", "owner_escalation"}),
        )
        personality = self.store.personality() or {}
        prompt = AgentPrompt(
            profile.prompt_locale,
            context.assistant_name,
            profile.base_instructions,
            str(personality.get("profile") or ""),
            profile.owner_details,
        )
        agent = build_agent(
            self.models.build_model(),
            self.gateway,
            prompt,
            self.http,
            checkpointer=self.checkpointer,
            store=self.memory,
            extra_middleware=(LiveConversationMiddleware(self.events, profile.prompt_locale),),
            attachments=self.attachments,
            diagnostics=self._buffer_trace,
            hosted_web_search=context.capabilities.hosted_web_search,
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
        await self.store.astart_agent_run(
            run_id=run_id,
            thread_id=thread_id,
            channel_id=channel_id,
            trace_id=trace_id,
        )
        await self.store.arecord_agent_trace(
            run_id,
            "run_input",
            {"event_ids": [event.id for event in claimed], "force_reply": force},
        )
        try:
            result = await agent.ainvoke(state, config=config, context=context)
        except asyncio.CancelledError:
            await self.events.release(channel_id, run_id)
            await self._flush_trace(trace_id, run_id)
            await self.store.afinish_agent_run(run_id, "cancelled")
            raise
        except Exception as error:
            await self.events.release(channel_id, run_id)
            await self._flush_trace(trace_id, run_id)
            await self.store.afinish_agent_run(
                run_id,
                "failed",
                f"{type(error).__name__}: {error}",
            )
            await self.store.arecord_agent_trace(
                run_id,
                "run_error",
                {"type": type(error).__name__, "detail": str(error)[:4000]},
            )
            raise
        interrupted = bool(result.get("__interrupt__"))
        await self._flush_trace(trace_id, run_id)
        await self.events.complete(channel_id, run_id)
        await self.store.afinish_agent_run(
            run_id,
            "interrupted" if interrupted else "completed",
        )
        await self.store.arecord_agent_trace(
            run_id,
            "run_output",
            {
                "interrupted": interrupted,
                "message_count": len(result.get("messages", [])),
                "successful_written_sends": result.get("successful_written_sends", 0),
            },
        )
        if not interrupted:
            try:
                await self._roll_if_needed(channel_id, thread_id)
            except Exception as error:
                log.warning("Could not apply completed-turn retention for %s: %s", channel_id, error)

    def _message(self, event: QueuedDiscordEvent) -> BaseMessage | None:
        if event.kind == "force_reply":
            return None
        if event.kind == "delete":
            return RemoveMessage(id=str(event.payload["message_id"]))
        content = str(event.payload.get("content") or "")
        attachments = event.payload.get("attachments") or []
        text_bundle = runtime_context_text(self.store.assistant_profile().prompt_locale)
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
                    "observed_at": event.observed_at,
                    "edited": event.kind == "edit",
                },
                "diskovod_attachments": attachments,
            },
        )

    async def _account_id(self, channel_id: str) -> str:
        thread = await self.store.achat_thread_for_channel(channel_id)
        if thread:
            return str(thread["account_id"])
        client = getattr(self.transport, "client", None)
        if client is not None and getattr(client, "user", None) is not None:
            return str(client.user.id)
        # A model-provider account is unrelated to the Discord identity. Offline migration uses
        # one installation-scoped owner label, which remains stable after Discord connects.
        return "discord-owner"

    async def _mode(self, channel_id: str) -> str:
        conversation = await self.store.aconversation(channel_id)
        return str(conversation["mode"] if conversation else "automatic")

    def _buffer_trace(self, trace_id: str, kind: str, payload: dict[str, Any]) -> None:
        self._trace_buffers.setdefault(trace_id, []).append((kind, payload))

    async def _flush_trace(self, trace_id: str, run_id: str) -> None:
        for kind, payload in self._trace_buffers.pop(trace_id, []):
            await self.store.arecord_agent_trace(run_id, kind, payload)

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
