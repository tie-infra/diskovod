from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from contextlib import AbstractAsyncContextManager
from datetime import datetime
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
from .http_client import PublicHTTP
from .mailbox import ConversationEvent, ConversationMailbox
from .models import AssistantProfile
from .outbound import DeliveryRecord, DiscordActionTransport, OutboundPublisher
from .localization import assistant_name_for, runtime_context_text, summarization_prompt
from .persistence import SQLiteLangGraphStore, open_checkpointer
from .providers import ModelConfiguration, ModelService
from .store import Store
from .waits import ConversationWait, ConversationWaits

log = logging.getLogger(__name__)
ROLLOVER_MESSAGE_LIMIT = 400
PUBLIC_OUTPUT_GRAPH_CUTOVER_KEY = "migration.public_output_graph_v2"


class EmulatedOutboundActions:
    """Capture replay actions without contacting Discord or changing durable action state."""

    def __init__(self):
        self.actions: list[dict[str, Any]] = []

    async def publish_messages(self, context, messages, *, source_kind, source_id):
        del context
        records = [
            DeliveryRecord("accepted", index, f"emulated:{source_id}:{index}")
            for index, _ in enumerate(messages)
        ]
        self.actions.append(
            {
                "action": "discord_messages",
                "messages": list(messages),
                "source_kind": source_kind,
                "source_id": source_id,
            }
        )
        return records

    async def react(self, context, emoji, message_id, *, source_id):
        del context
        self.actions.append(
            {
                "action": "discord_reaction",
                "emoji": emoji,
                "message_id": message_id,
                "source_id": source_id,
            }
        )
        return DeliveryRecord("accepted", 0, f"emulated:{source_id}:reaction")

    async def record_escalation(self, context, *, source_id, payload):
        del context
        self.actions.append({"action": "escalate_to_owner", "payload": payload, "source_id": source_id})


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
        self.mailbox = ConversationMailbox(store.database)
        self.attachments = AttachmentRepository(store.database, http)
        self.memory = SQLiteLangGraphStore(store.path, store.database)
        self.publisher = OutboundPublisher(store.database, transport)
        self.waits = ConversationWaits(store.database)
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self._trace_buffers: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._checkpoint_context: AbstractAsyncContextManager[AsyncSqliteSaver] | None = None
        self.checkpointer: AsyncSqliteSaver | None = None
        self._wait_notifier: asyncio.Task[None] | None = None
        self._closing = False

    async def start(self) -> None:
        if self.checkpointer is not None:
            return
        self._closing = False
        await self.store.start()
        await self.memory.start()
        for action in await self.publisher.reconcile_abandoned():
            await self.store.arecord_agent_trace(
                action["run_id"],
                "outbound_action_reconciled",
                {"action_id": action["id"], "state": action["state"]},
            )
        await self._reconcile_abandoned_agent_runs()
        self._checkpoint_context = open_checkpointer(self.store.path, self.checkpoint_secret)
        self.checkpointer = await self._checkpoint_context.__aenter__()
        await self._backfill_checkpoint_index()
        await self._cutover_legacy_graph_generations()
        await self._reconcile_arming_waits()
        self._wait_notifier = asyncio.create_task(self._notify_due_waits(), name="conversation-wait-notifier")
        for channel_id in await self.mailbox.pending_channels():
            self._ensure_task(channel_id)

    async def close(self) -> None:
        self._closing = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
        if self._wait_notifier is not None:
            self._wait_notifier.cancel()
            await asyncio.gather(self._wait_notifier, return_exceptions=True)
            self._wait_notifier = None
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
        await self.cancel_followup(
            channel_id,
            reason="conversation_paused",
            resume_pending=False,
        )
        self.cancel(channel_id)

    async def cancel_followup(
        self,
        channel_id: str,
        *,
        reason: str = "cancelled_by_owner",
        resume_pending: bool = True,
    ) -> bool:
        wait = await self.waits.active(channel_id) or await self.waits.incomplete_resume(channel_id)
        if wait is None:
            return False
        task = self.tasks.pop(channel_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if not await self.waits.cancel(wait.id, reason):
            return False
        await self.mailbox.release(channel_id, wait.run_id)
        await self.store.arecord_agent_trace(
            wait.run_id,
            "followup_wait_cancelled",
            {"wait_id": wait.id, "reason": reason},
        )
        await self.store.afinish_agent_run(
            wait.run_id,
            "cancelled",
            f"Conversational follow-up cancelled: {reason}",
        )
        if resume_pending and await self.mailbox.has_pending(channel_id):
            self._ensure_task(channel_id)
        return True

    async def cancel_all_followups(self, reason: str) -> int:
        cancelled = 0
        for wait in await self.waits.resumable():
            if await self.cancel_followup(
                wait.channel_id,
                reason=reason,
                resume_pending=False,
            ):
                cancelled += 1
        return cancelled

    async def resolve_outbound_action(
        self,
        run_id: str,
        action_id: str,
        operation: str,
        *,
        remote_id: str = "",
    ) -> DeliveryRecord | None:
        action = await self.publisher.action(action_id)
        if action is None or str(action["run_id"]) != run_id:
            return None
        if operation == "retry":
            result = await self.publisher.retry(action_id)
        elif operation in {"confirmed_succeeded", "confirmed_failed"}:
            result = await self.publisher.resolve(
                action_id,
                operation,
                remote_id=remote_id,
            )
        else:
            raise ValueError("Unknown outbound action operation")
        if result is not None:
            await self.store.arecord_agent_trace(
                run_id,
                "outbound_action_operator_resolution",
                {
                    "action_id": action_id,
                    "operation": operation,
                    "result": result.to_dict(),
                },
            )
        return result

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
            await self._wake_followup_for_input(channel_id)
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
        inserted = await self.mailbox.ingest(
            event_id,
            channel_id,
            "edit" if edited else "message",
            payload,
            observed_at=observed_at,
            enqueue=automate,
        )
        if inserted and automate:
            await self.mailbox.thread_id(account_id, channel_id)
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
            await self._wake_followup_for_input(channel_id)
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
        inserted = await self.mailbox.ingest(
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
        await self._wake_followup_for_input(channel_id)
        self._ensure_task(channel_id, force=True)

    async def _wake_followup_for_input(self, channel_id: str) -> bool:
        wait_id = await self.waits.wake_for_input(channel_id)
        if wait_id is None:
            return False
        wait = await self.waits.active(channel_id)
        if wait is not None:
            await self.store.arecord_agent_trace(
                wait.run_id,
                "followup_wait_woken",
                {"wait_id": wait_id, "reason": "new_input"},
            )
        return True

    async def _ingest_force_reply(self, *, channel_id: str, account_id: str, trigger_message_id: str) -> None:
        event_id = f"diskovod:force:{uuid.uuid4()}"
        await self.mailbox.ingest(
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
        await self.mailbox.thread_id(account_id, channel_id)

    async def set_live_steering(self, channel_id: str, enabled: bool) -> None:
        await self.mailbox.set_live_steering(await self._account_id(channel_id), channel_id, enabled)

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

    async def _backfill_checkpoint_index(self) -> None:
        if self.checkpointer is None or self.store._get("admin.checkpoint_index_backfilled", False):
            return
        try:
            indexed: list[dict[str, Any]] = []
            async for item in self.checkpointer.alist(None):
                indexed.append(self._checkpoint_metadata(item))
                if len(indexed) >= 200:
                    await self.store.aindex_checkpoints(indexed)
                    indexed.clear()
            await self.store.aindex_checkpoints(indexed)
            await self.store._aset("admin.checkpoint_index_backfilled", True)
        except Exception:
            log.exception("Could not backfill the administrative checkpoint index")

    async def _cutover_legacy_graph_generations(self) -> None:
        progress = self.store._get(PUBLIC_OUTPUT_GRAPH_CUTOVER_KEY, {})
        processed = set(str(item) for item in progress.get("processed_threads", []))
        if not progress.get("completed"):
            for thread in await self.store.achat_threads():
                thread_id = str(thread["thread_id"])
                if thread_id in processed:
                    continue
                snapshot = await self.checkpointer.aget_tuple(
                    {"configurable": {"thread_id": thread_id}}
                )
                if snapshot is not None:
                    await self._roll_thread(dict(thread), reason="public_output_graph_cutover")
                processed.add(thread_id)
                await self.store._aset(
                    PUBLIC_OUTPUT_GRAPH_CUTOVER_KEY,
                    {"completed": False, "processed_threads": sorted(processed)},
                )
            await self.store._aset(
                PUBLIC_OUTPUT_GRAPH_CUTOVER_KEY,
                {
                    "completed": True,
                    "processed_threads": sorted(processed),
                    "completed_at": time.time(),
                },
            )
        await self._reconcile_public_output_cutover(processed)

    async def _reconcile_public_output_cutover(self, legacy_threads: set[str]) -> None:
        if not legacy_threads:
            return
        placeholders = ",".join("?" for _ in legacy_threads)
        thread_parameters = tuple(sorted(legacy_threads))
        audits: list[tuple[str, str, dict[str, object]]] = []
        now = time.time()
        async with self.store.database.transaction() as connection:
            claimed = await (
                await connection.execute(
                    f"""
                    SELECT m.*, r.thread_id AS run_thread_id
                    FROM conversation_mailbox AS m
                    JOIN agent_runs AS r ON r.id=m.run_id
                    WHERE m.state='claimed' AND r.thread_id IN ({placeholders})
                    ORDER BY m.run_id, m.sequence
                    """,
                    thread_parameters,
                )
            ).fetchall()
            by_run: dict[str, list[Any]] = {}
            for event in claimed:
                by_run.setdefault(str(event["run_id"]), []).append(event)
            for run_id, events in by_run.items():
                actions = await (
                    await connection.execute(
                        "SELECT state, result FROM outbound_actions WHERE run_id=?",
                        (run_id,),
                    )
                ).fetchall()
                states = [str(action["state"]) for action in actions]
                event_state, run_state, outcome = self._claimed_run_outcome(actions)
                failure = (
                    "public_output_cutover_delivery_requires_review"
                    if event_state == "failed"
                    else None
                )
                event_ids = [str(event["id"]) for event in events]
                markers = ",".join("?" for _ in event_ids)
                if event_state == "pending":
                    await connection.execute(
                        f"""
                        UPDATE conversation_mailbox
                        SET state='pending', run_id=NULL, injection_batch=NULL,
                            claimed_at=NULL, completed_at=NULL, failure=NULL
                        WHERE id IN ({markers}) AND state='claimed'
                        """,
                        tuple(event_ids),
                    )
                else:
                    await connection.execute(
                        f"""
                        UPDATE conversation_mailbox
                        SET state=?, completed_at=?, failure=?
                        WHERE id IN ({markers}) AND state='claimed'
                        """,
                        (event_state, now, failure, *event_ids),
                    )
                await connection.execute(
                    "UPDATE agent_runs SET status=?, completed_at=?, error=? WHERE id=?",
                    (run_state, now, failure, run_id),
                )
                audits.append(
                    (
                        run_id,
                        "public_output_cutover_claim_reconciled",
                        {
                            "event_ids": event_ids,
                            "outcome": outcome,
                            "delivery_states": states,
                        },
                    )
                )

            escalations = await (
                await connection.execute(
                    f"""
                    SELECT * FROM escalation_interrupts
                    WHERE state IN ('pending','claimed') AND thread_id IN ({placeholders})
                    """,
                    thread_parameters,
                )
            ).fetchall()
            for escalation in escalations:
                payload = json.loads(escalation["payload"])
                if payload.get("resume_strategy") == "mailbox":
                    continue
                payload["resume_strategy"] = "mailbox"
                payload["cutover_from_thread"] = str(escalation["thread_id"])
                await connection.execute(
                    "UPDATE escalation_interrupts SET payload=?, updated_at=? WHERE id=?",
                    (
                        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        now,
                        escalation["id"],
                    ),
                )
                trace_id = str(payload.get("trace_id") or "")
                run = await (
                    await connection.execute("SELECT id FROM agent_runs WHERE trace_id=?", (trace_id,))
                ).fetchone()
                if run is not None:
                    await connection.execute(
                        "UPDATE agent_runs SET status='cancelled', completed_at=?, error=? WHERE id=?",
                        (now, "Escalation checkpoint replaced by mailbox cutover", run["id"]),
                    )
                    audits.append(
                        (
                            str(run["id"]),
                            "public_output_cutover_escalation_reconciled",
                            {"escalation_id": str(escalation["id"]), "resolution": "mailbox"},
                        )
                    )
        for run_id, kind, payload in audits:
            await self.store.arecord_agent_trace(run_id, kind, payload)

    async def _reconcile_abandoned_agent_runs(self) -> None:
        """Resolve input claims owned by a process that stopped during an ordinary run."""
        audits: list[tuple[str, dict[str, object]]] = []
        now = time.time()
        async with self.store.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT m.* FROM conversation_mailbox AS m
                    JOIN agent_runs AS r ON r.id=m.run_id
                    WHERE m.state='claimed' AND r.status='running'
                    ORDER BY m.run_id, m.sequence
                    """
                )
            ).fetchall()
            by_run: dict[str, list[Any]] = {}
            for event in rows:
                by_run.setdefault(str(event["run_id"]), []).append(event)
            for run_id, events in by_run.items():
                actions = await (
                    await connection.execute(
                        "SELECT state, result FROM outbound_actions WHERE run_id=?",
                        (run_id,),
                    )
                ).fetchall()
                states = [str(action["state"]) for action in actions]
                event_state, run_state, outcome = self._claimed_run_outcome(actions)
                failure = "abandoned_run_delivery_requires_review" if event_state == "failed" else None
                event_ids = [str(event["id"]) for event in events]
                markers = ",".join("?" for _ in event_ids)
                if event_state == "pending":
                    await connection.execute(
                        f"""
                        UPDATE conversation_mailbox
                        SET state='pending', run_id=NULL, injection_batch=NULL,
                            claimed_at=NULL, completed_at=NULL, failure=NULL
                        WHERE id IN ({markers}) AND state='claimed'
                        """,
                        tuple(event_ids),
                    )
                else:
                    await connection.execute(
                        f"""
                        UPDATE conversation_mailbox
                        SET state=?, completed_at=?, failure=?
                        WHERE id IN ({markers}) AND state='claimed'
                        """,
                        (event_state, now, failure, *event_ids),
                    )
                await connection.execute(
                    "UPDATE agent_runs SET status=?, completed_at=?, error=? WHERE id=?",
                    (run_state, now, failure, run_id),
                )
                audits.append(
                    (
                        run_id,
                        {
                            "event_ids": event_ids,
                            "outcome": outcome,
                            "delivery_states": states,
                        },
                    )
                )
        for run_id, payload in audits:
            await self.store.arecord_agent_trace(run_id, "abandoned_run_reconciled", payload)

    @classmethod
    def _claimed_run_outcome(cls, actions) -> tuple[str, str, str]:
        states = [str(action["state"]) for action in actions]
        any_accepted = any(cls._delivery_result_has_accepted(action["result"]) for action in actions)
        if states and all(state == "succeeded" for state in states):
            return "completed", "completed", "completed_from_delivery_evidence"
        if not states or (set(states) == {"failed"} and not any_accepted):
            return "pending", "cancelled", "released_without_visible_effect"
        return "failed", "failed", "failed_for_delivery_review"

    @staticmethod
    def _delivery_result_has_accepted(value: object) -> bool:
        if not value:
            return False
        try:
            decoded = json.loads(str(value))
        except (TypeError, ValueError):
            return False
        if isinstance(decoded, dict):
            return decoded.get("status") == "accepted" or any(
                AgentService._delivery_result_has_accepted(json.dumps(item))
                for item in decoded.values()
                if isinstance(item, (dict, list))
            )
        if isinstance(decoded, list):
            return any(
                item.get("status") == "accepted"
                if isinstance(item, dict)
                else AgentService._delivery_result_has_accepted(json.dumps(item))
                for item in decoded
                if isinstance(item, (dict, list))
            )
        return False

    async def _index_run_checkpoints(
        self,
        thread_id: str,
        run_id: str,
        previous_checkpoint_id: str | None,
    ) -> tuple[str | None, str | None]:
        try:
            if self.checkpointer is None:
                return None, None
            newest_first: list[dict[str, Any]] = []
            async for item in self.checkpointer.alist(
                {"configurable": {"thread_id": thread_id}},
                limit=100,
            ):
                checkpoint_id = str(item.config["configurable"]["checkpoint_id"])
                if checkpoint_id == previous_checkpoint_id:
                    break
                metadata = self._checkpoint_metadata(item)
                metadata["run_id"] = run_id
                newest_first.append(metadata)
            await self.store.aindex_checkpoints(newest_first)
            first = str(newest_first[-1]["checkpoint_id"]) if newest_first else None
            final = str(newest_first[0]["checkpoint_id"]) if newest_first else previous_checkpoint_id
            await self.store.aupdate_run_checkpoints(run_id, first, final)
            return first, final
        except Exception:
            log.exception("Could not update the administrative checkpoint index for run %s", run_id)
            return None, None

    @staticmethod
    def _checkpoint_metadata(item) -> dict[str, Any]:
        created = str(item.checkpoint.get("ts") or "")
        try:
            created_at = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        except ValueError:
            created_at = time.time()
        values = item.checkpoint.get("channel_values", {})
        parent = item.parent_config or {}
        return {
            "thread_id": str(item.config["configurable"]["thread_id"]),
            "checkpoint_id": str(item.config["configurable"]["checkpoint_id"]),
            "parent_checkpoint_id": parent.get("configurable", {}).get("checkpoint_id"),
            "created_at": created_at,
            "step": item.metadata.get("step"),
            "source": item.metadata.get("source"),
            "message_count": len(values.get("messages", [])),
        }

    async def replay_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
        *,
        configuration_id: int | None = None,
    ) -> str:
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
        configuration = (
            await self.store.aagent_configuration(configuration_id)
            if configuration_id is not None
            else self.models.configuration
        )
        if configuration is None:
            raise ValueError("Unknown model configuration")
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
            run_id=replay_id,
            thread_id=f"replay:{thread_id}:{checkpoint_id}",
            owner_timezone=profile.owner_timezone,
            trigger_message_id="historical-replay",
            allow_conversational_followups=False,
            permissions=frozenset({"reactions", "owner_escalation"}),
        )
        personality = self.store.personality() or {}
        gateway = EmulatedOutboundActions()
        replay_model = (
            self.models.build_configuration(
                configuration,
                self.models.credentials_for(configuration),
            )
            if configuration_id is not None
            else self.models.build_model()
        )
        agent = build_agent(
            replay_model,
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
            native_tools=context.capabilities.native_tools,
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
        except asyncio.CancelledError:
            await self._flush_trace(trace_id, replay_id)
            await self.store.arecord_agent_trace(
                replay_id,
                "emulated_actions",
                {"actions": gateway.actions},
            )
            await self.store.afinish_agent_run(replay_id, "cancelled", "Replay cancelled by owner")
            raise
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
        new_thread_id = await self.mailbox.roll_generation(
            str(thread["account_id"]),
            str(thread["channel_id"]),
            reason=reason,
            summary=summary,
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
        escalation = await self.store.aescalation_interrupt(escalation_id)
        if escalation is None or escalation["state"] not in {"pending", "claimed"}:
            return False
        payload = escalation["payload"]
        if payload.get("migrated") or payload.get("resume_strategy") == "mailbox":
            channel_id = str(escalation["channel_id"])
            await self.store.aset_interrupt_state(
                escalation_id,
                "dismissed" if action == "dismissed" else "resolved",
            )
            if action == "owner_reply" and owner_message.strip():
                account_id = await self._account_id(channel_id)
                inserted, automate = await self._ingest_message(
                    message_id=owner_message_id or f"migrated-owner-reply:{time.time_ns()}",
                    channel_id=channel_id,
                    account_id=account_id,
                    author_id=owner_author_id or account_id,
                    author_name=owner_author_name
                    or runtime_context_text(self.store.assistant_profile().prompt_locale)["account_owner"],
                    participant_role="owner",
                    content=owner_message[:4000],
                    attachments=[],
                    observed_at=time.time(),
                    edited=False,
                    force=False,
                    agent_input=True,
                )
                if inserted and automate:
                    self._ensure_task(channel_id)
            return True
        if self.checkpointer is None or not self.models.ready:
            raise RuntimeError("The agent runtime is not ready")
        trace_id = str(payload.get("trace_id") or "")
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
            run_id=str(run["id"]),
            thread_id=thread_id,
            owner_timezone=profile.owner_timezone,
            trigger_message_id=str(payload.get("trigger_message_id") or ""),
            allow_conversational_followups=(
                profile.allow_conversational_followups and configuration.capabilities.native_tools
            ),
            permissions=frozenset({"reactions", "owner_escalation"}),
        )
        personality = self.store.personality() or {}
        agent = build_agent(
            self.models.build_model(),
            self.publisher,
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
            input_injector=self._inject_pending,
            attachments=self.attachments,
            diagnostics=self._buffer_trace,
            hosted_web_search=context.capabilities.hosted_web_search,
            followup_scheduler=(self if context.allow_conversational_followups else None),
            native_tools=context.capabilities.native_tools,
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
            await self.mailbox.fail(
                channel_id,
                str(run["id"]),
                f"{type(error).__name__}: {error}",
            )
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
        await self.mailbox.complete(channel_id, str(run["id"]))
        state = "dismissed" if action == "dismissed" else "resolved"
        await self.store.aset_interrupt_state(escalation_id, state)
        await self.store.afinish_agent_run(
            str(run["id"]),
            "interrupted" if result.get("__interrupt__") else "completed",
        )
        if not result.get("__interrupt__") and await self.mailbox.has_pending(channel_id):
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
        task = asyncio.create_task(self._drive(channel_id, force=force), name=f"agent-{channel_id}")
        self.tasks[channel_id] = task
        task.add_done_callback(lambda done: self._finished(channel_id, done))

    async def _drive(self, channel_id: str, *, force: bool) -> None:
        active_wait = await self.waits.active(channel_id)
        if active_wait is not None:
            if active_wait.state == "scheduled":
                wait = await self.waits.claim_ready(channel_id)
                if wait is not None:
                    await self._resume_followup(wait)
            elif active_wait.state == "resuming":
                await self._resume_followup(active_wait)
            return
        incomplete = await self.waits.incomplete_resume(channel_id)
        if incomplete is not None:
            await self._resume_followup(incomplete)
            return
        if await self.store.ahas_active_interrupt(channel_id):
            return
        await self._run(channel_id, force=force)

    async def _notify_due_waits(self) -> None:
        try:
            while not self._closing:
                for channel_id in await self.waits.due_channels():
                    self._ensure_task(channel_id)
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Conversation wait notifier failed")

    async def _reconcile_arming_waits(self) -> None:
        if self.checkpointer is None:
            return
        for wait in await self.waits.arming():
            snapshot = await self.checkpointer.aget_tuple({"configurable": {"thread_id": wait.thread_id}})
            writes = snapshot.pending_writes if snapshot is not None else None
            persisted = any(
                channel == "__interrupt__" and wait.id in str(value) for _, channel, value in (writes or [])
            )
            if persisted:
                await self.waits.schedule(wait.id)
                await self.store.arecord_agent_trace(
                    wait.run_id,
                    "followup_wait_reconciled",
                    {"wait_id": wait.id, "outcome": "scheduled"},
                )
            else:
                await self.waits.cancel(wait.id, "arming_without_persisted_interrupt")
                await self.store.arecord_agent_trace(
                    wait.run_id,
                    "followup_wait_reconciled",
                    {"wait_id": wait.id, "outcome": "cancelled_without_interrupt"},
                )

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
        if not await self.store.ahas_active_interrupt(channel_id) and await self.mailbox.has_pending(
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
        thread_id = await self.mailbox.thread_id(account_id, channel_id)
        run_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        claimed = await self.mailbox.claim_ready(channel_id, run_id)
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
        reaction_target_message_id = next(
            (
                str(event.payload.get("message_id") or "")
                for event in reversed(claimed)
                if event.kind in {"message", "edit"} and event.payload.get("message_id")
            ),
            trigger_message_id,
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
            run_id=run_id,
            thread_id=thread_id,
            owner_timezone=profile.owner_timezone,
            trigger_message_id=trigger_message_id,
            allow_conversational_followups=(
                profile.allow_conversational_followups and configuration.capabilities.native_tools
            ),
            permissions=frozenset({"reactions", "owner_escalation"}),
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
            self.publisher,
            prompt,
            self.http,
            checkpointer=self.checkpointer,
            store=self.memory,
            input_injector=self._inject_pending,
            attachments=self.attachments,
            diagnostics=self._buffer_trace,
            hosted_web_search=context.capabilities.hosted_web_search,
            followup_scheduler=(self if context.allow_conversational_followups else None),
            native_tools=context.capabilities.native_tools,
        )
        initial_messages = [message for event in claimed if (message := self._message(event)) is not None]
        state = {
            "messages": initial_messages,
            "logical_request_id": run_id,
            "claimed_event_ids": [event.id for event in claimed],
            "reaction_target_message_id": reaction_target_message_id,
        }
        config = {
            "configurable": {"thread_id": thread_id},
            "run_id": uuid.UUID(run_id),
            "recursion_limit": 40,
        }
        previous_snapshot = await self.checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
        previous_checkpoint_id = (
            str(previous_snapshot.config["configurable"]["checkpoint_id"])
            if previous_snapshot is not None
            else None
        )
        await self.store.astart_agent_run(
            run_id=run_id,
            thread_id=thread_id,
            channel_id=channel_id,
            trace_id=trace_id,
            trigger_kind="force_reply" if force else claimed[-1].kind,
            trigger_message_id=trigger_message_id,
        )
        await self.store.arecord_agent_trace(
            run_id,
            "run_input",
            {"event_ids": [event.id for event in claimed], "force_reply": force},
        )
        try:
            result = await agent.ainvoke(state, config=config, context=context)
        except asyncio.CancelledError:
            await self.mailbox.release(channel_id, run_id)
            await self._flush_trace(trace_id, run_id)
            await self._index_run_checkpoints(thread_id, run_id, previous_checkpoint_id)
            await self.store.afinish_agent_run(run_id, "cancelled")
            raise
        except Exception as error:
            await self.mailbox.fail(channel_id, run_id, f"{type(error).__name__}: {error}")
            await self._flush_trace(trace_id, run_id)
            await self._index_run_checkpoints(thread_id, run_id, previous_checkpoint_id)
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
        active_wait = await self.waits.active(channel_id)
        if interrupted and active_wait is not None and active_wait.run_id == run_id:
            await self.waits.schedule(active_wait.id)
            self._buffer_trace(
                trace_id,
                "followup_wait_scheduled",
                {"wait_id": active_wait.id, "resume_at": active_wait.resume_at},
            )
        await self._flush_trace(trace_id, run_id)
        await self._index_run_checkpoints(thread_id, run_id, previous_checkpoint_id)
        await self.mailbox.complete(channel_id, run_id)
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
                "outbound_delivery_count": result.get("outbound_delivery_count", 0),
            },
        )
        if not interrupted:
            try:
                await self._roll_if_needed(channel_id, thread_id)
            except Exception as error:
                log.warning("Could not apply completed-turn retention for %s: %s", channel_id, error)

    async def arm_followup(
        self,
        context: AgentRuntimeContext,
        state: DiskovodAgentState,
        *,
        tool_call_id: str,
        pause: str,
        max_duration: float,
    ) -> tuple[str, float, float]:
        ranges = {"brief": (1.0, 3.0), "short": (3.0, 8.0)}
        minimum, maximum = ranges[pause]
        duration = min(random.uniform(minimum, maximum), max_duration)
        profile = self.store.assistant_profile()
        configuration = self.models.configuration
        if configuration is None:
            raise RuntimeError("Cannot schedule a follow-up without a model configuration")
        payload = {
            "configuration_version_id": await self.store.aactive_configuration_id(),
            "model_configuration": configuration.to_dict(),
            "assistant_profile": profile.to_dict(),
            "personality": self.store.personality() or {},
            "account_id": context.account_id,
            "participant_ids": list(context.participant_ids),
            "automation_mode": context.automation_mode,
            "trigger_message_id": context.trigger_message_id,
            "duration": duration,
            "pause": pause,
        }
        wait = await self.waits.arm(
            context,
            run_id=str(state.get("logical_request_id") or context.trace_id),
            tool_call_id=tool_call_id,
            duration=duration,
            payload=payload,
        )
        stored_duration = float(wait.payload.get("duration", duration))
        self._buffer_trace(
            context.trace_id,
            "followup_wait_armed",
            {
                "wait_id": wait.id,
                "pause": wait.payload.get("pause", pause),
                "resume_at": wait.resume_at,
                "duration": stored_duration,
            },
        )
        return wait.id, wait.resume_at, stored_duration

    async def resolve_followup(self, wait_id: str) -> None:
        await self.waits.resolve(wait_id)

    async def _resume_followup(self, wait: ConversationWait) -> None:
        if self.checkpointer is None or not self.models.ready:
            await self.waits.fail(wait.id, "runtime_not_ready")
            return
        configuration_id = wait.payload.get("configuration_version_id")
        configuration = (
            await self.store.aagent_configuration(int(configuration_id))
            if configuration_id is not None
            else None
        )
        if configuration is None and wait.payload.get("model_configuration"):
            configuration = ModelConfiguration.from_dict(wait.payload["model_configuration"])
        if configuration is None:
            await self.waits.fail(wait.id, "pinned_model_configuration_unavailable")
            await self.store.afinish_agent_run(
                wait.run_id, "failed", "Pinned model configuration is unavailable"
            )
            return
        profile = AssistantProfile(**dict(wait.payload["assistant_profile"]))
        conversation = await self.store.aconversation(wait.channel_id)
        context = AgentRuntimeContext(
            account_id=str(wait.payload.get("account_id") or await self._account_id(wait.channel_id)),
            channel_id=wait.channel_id,
            participant_ids=tuple(str(item) for item in wait.payload.get("participant_ids", [])),
            owner_id=str(wait.payload.get("account_id") or "discord-owner"),
            ui_locale=self.store.interface_settings().locale,
            prompt_locale=profile.prompt_locale,
            assistant_name=assistant_name_for(profile.prompt_locale, profile.assistant_name),
            automation_mode=str(
                wait.payload.get("automation_mode") or (conversation["mode"] if conversation else "automatic")
            ),
            force_reply=False,
            provider_id=configuration.provider_id,
            model_id=configuration.model_id,
            transport_profile=configuration.transport_profile,
            capabilities=self._capabilities(configuration),
            trace_id=wait.trace_id,
            run_id=wait.run_id,
            thread_id=wait.thread_id,
            owner_timezone=profile.owner_timezone,
            trigger_message_id=str(wait.payload.get("trigger_message_id") or ""),
            allow_conversational_followups=(
                profile.allow_conversational_followups and configuration.capabilities.native_tools
            ),
            permissions=frozenset({"reactions", "owner_escalation"}),
        )
        personality = dict(wait.payload.get("personality") or {})
        model = (
            self.models.build_model()
            if self.models.configuration is not None
            and self.models.configuration.to_dict() == configuration.to_dict()
            else self.models.build_configuration(
                configuration,
                self.models.credentials_for(configuration),
            )
        )
        agent = build_agent(
            model,
            self.publisher,
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
            input_injector=self._inject_pending,
            attachments=self.attachments,
            diagnostics=self._buffer_trace,
            hosted_web_search=context.capabilities.hosted_web_search,
            followup_scheduler=self,
            native_tools=context.capabilities.native_tools,
        )
        config = {
            "configurable": {"thread_id": wait.thread_id},
            "run_id": uuid.UUID(wait.run_id),
            "recursion_limit": 40,
        }
        snapshot = await self.checkpointer.aget_tuple(
            {"configurable": {"thread_id": wait.thread_id}}
        )
        pending_writes = snapshot.pending_writes if snapshot is not None else None
        has_wait_interrupt = any(
            channel == "__interrupt__" and wait.id in str(value)
            for _, channel, value in (pending_writes or [])
        )
        graph_input = (
            Command(resume={"reason": wait.resume_reason, "wait_id": wait.id})
            if has_wait_interrupt
            else None
        )
        await self.store.arecord_agent_trace(
            wait.run_id,
            "followup_wait_recovery" if not has_wait_interrupt else "followup_wait_resume",
            {
                "wait_id": wait.id,
                "reason": wait.resume_reason,
                "checkpoint_interrupt": has_wait_interrupt,
                "wait_state": wait.state,
            },
        )
        try:
            result = await agent.ainvoke(
                graph_input,
                config=config,
                context=context,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.waits.fail(wait.id, f"{type(error).__name__}: {error}")
            await self.mailbox.fail(
                wait.channel_id,
                wait.run_id,
                f"{type(error).__name__}: {error}",
            )
            await self._flush_trace(wait.trace_id, wait.run_id)
            await self.store.afinish_agent_run(
                wait.run_id,
                "failed",
                f"{type(error).__name__}: {error}",
            )
            raise
        interrupted = bool(result.get("__interrupt__"))
        next_wait = await self.waits.active(wait.channel_id)
        if interrupted and next_wait is not None and next_wait.id != wait.id:
            await self.waits.schedule(next_wait.id)
        await self.waits.resolve(wait.id)
        await self.mailbox.complete(wait.channel_id, wait.run_id)
        await self._flush_trace(wait.trace_id, wait.run_id)
        await self.store.afinish_agent_run(
            wait.run_id,
            "interrupted" if interrupted else "completed",
        )
        await self.store.arecord_agent_trace(
            wait.run_id,
            "followup_wait_result",
            {
                "wait_id": wait.id,
                "interrupted": interrupted,
                "outbound_delivery_count": result.get("outbound_delivery_count", 0),
            },
        )
        if not interrupted and await self.mailbox.has_pending(wait.channel_id):
            self._ensure_task(wait.channel_id)

    async def _inject_pending(
        self,
        state: DiskovodAgentState,
        context: AgentRuntimeContext,
    ) -> dict[str, Any] | None:
        if not state.get("continuation_resume") and not await self.mailbox.live_steering(context.channel_id):
            return None
        run_id = str(state.get("logical_request_id") or "")
        if not run_id:
            return None
        batch = int(state.get("live_injection_batches", 0)) + 1
        known = set(state.get("claimed_event_ids", []))
        recovered = [
            event for event in await self.mailbox.claimed(context.channel_id, run_id) if event.id not in known
        ]
        newly_claimed = await self.mailbox.claim_ready(
            context.channel_id,
            run_id,
            injection_batch=batch,
        )
        events = recovered + newly_claimed
        messages = [message for event in events if (message := self._message(event)) is not None]
        if not events:
            return None
        target = next(
            (
                str(event.payload.get("message_id") or "")
                for event in reversed(events)
                if event.kind in {"message", "edit"}
            ),
            str(state.get("reaction_target_message_id") or context.trigger_message_id),
        )
        self._buffer_trace(
            context.trace_id,
            "mailbox_injection",
            {
                "event_ids": [event.id for event in events],
                "batch": batch,
                "reaction_target_message_id": target,
            },
        )
        return {
            "messages": messages,
            "claimed_event_ids": [event.id for event in events],
            "reaction_target_message_id": target,
            "live_injection_batches": 1,
        }

    def _message(self, event: ConversationEvent) -> BaseMessage | None:
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
                filename = str(attachment.get("filename") or text_bundle["attachment_filename"])
                media_type = str(attachment.get("content_type") or text_bundle["unknown_media_type"])
                size = int(attachment.get("size") or 0)
                note = text_bundle["attachment_summary"].format(
                    filename=filename,
                    media_type=media_type,
                    size=size,
                )
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
