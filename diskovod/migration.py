from __future__ import annotations

import asyncio
import json
import logging
import hashlib
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph

from .agent import AgentPrompt, build_agent
from .agent_actions import DeliveryRecord
from .agent_types import AgentRuntimeContext, CapabilityProfile, DiskovodAgentState
from .localization import assistant_name_for, runtime_context_text, summarization_prompt
from .runtime import AgentService
from .store import Store

log = logging.getLogger(__name__)
MIGRATION_KEY = "migration.langgraph_cutover_v1"


@dataclass(frozen=True, slots=True)
class MigrationReport:
    backup_path: Path | None
    conversations: int
    events: int
    checkpoints: int
    archived_records: int


class _MigrationEscalationModel(BaseChatModel):
    reason: str
    acknowledgement: str
    tool_call_id: str

    @property
    def _llm_type(self) -> str:
        return "diskovod-migration-escalation"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        del tools, tool_choice, kwargs
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del messages, stop, run_manager, kwargs
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "escalate_to_owner",
                                "args": {
                                    "reason": self.reason,
                                    "acknowledgement": self.acknowledgement,
                                },
                                "id": self.tool_call_id,
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )


class _MigrationEscalationGateway:
    def __init__(self, runtime: AgentService, legacy_message_id: str):
        self.runtime = runtime
        self.legacy_message_id = legacy_message_id

    async def send_messages(self, context, messages, *, tool_call_id):
        return [
            DeliveryRecord("accepted", index, f"legacy:{self.legacy_message_id}:{index}")
            for index, _ in enumerate(messages)
        ]

    async def react_to_message(self, context, emoji, *, tool_call_id):
        del context, emoji
        return DeliveryRecord("accepted", 0, f"legacy:{tool_call_id}:reaction")

    async def record_escalation(self, context, payload, *, tool_call_id):
        await self.runtime.ledger.record_escalation(
            f"{context.trace_id}:{tool_call_id}",
            context.thread_id,
            context.channel_id,
            payload,
        )


class LegacyMigrator:
    """One-time offline transformer; the target runtime never reads legacy model state."""

    def __init__(self, store: Store, runtime: AgentService):
        self.store = store
        self.runtime = runtime

    async def run(self) -> MigrationReport:
        if self.store._get(MIGRATION_KEY, None):
            return MigrationReport(None, 0, 0, 0, 0)
        backup = await asyncio.to_thread(self._backup)
        conversations = self.store.conversations()
        event_count = 0
        checkpoint_count = 0
        for conversation in conversations:
            channel_id = str(conversation["channel_id"])
            account_id = self.runtime._account_id(channel_id)
            thread_id = await self.runtime.events.thread_id(account_id, channel_id)
            history = self.store.history(channel_id, 100_000)
            messages = await self._messages(history, conversation, account_id)
            for item in history:
                payload = {
                    "message_id": str(item["id"]),
                    "account_id": account_id,
                    "author_id": str(item["author_id"]),
                    "author_name": str(item["author_name"]),
                    "participant_role": self._role(item),
                    "content": str(item["content"]),
                    "attachments": item.get("attachments") or [],
                    "legacy_source": str(item["source"]),
                }
                if await self.runtime.events.ingest(
                    f"legacy:message:{item['id']}",
                    channel_id,
                    "message",
                    payload,
                    observed_at=float(item["timestamp"]),
                    enqueue=False,
                ):
                    event_count += 1
            if messages and not await self.runtime.checkpointer.aget_tuple(
                {"configurable": {"thread_id": thread_id}}
            ):
                await self._seed(thread_id, messages)
                checkpoint_count += 1
        await self._migrate_active_escalations()
        archived = await self._archive_legacy_records()
        await self._migrate_owner_details()
        report = MigrationReport(backup, len(conversations), event_count, checkpoint_count, archived)
        await self._validate(report)
        await self._drop_legacy_tables()
        await self.store._aset(
            MIGRATION_KEY,
            {
                "completed_at": time.time(),
                "backup_path": str(backup),
                "conversations": report.conversations,
                "events": report.events,
                "checkpoints": report.checkpoints,
                "archived_records": report.archived_records,
            },
        )
        log.info(
            "LangGraph cutover migration completed: %d chats, %d events, %d checkpoints; backup %s",
            report.conversations,
            report.events,
            report.checkpoints,
            backup,
        )
        return report

    def _backup(self) -> Path:
        backup_dir = self.store.path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = backup_dir / f"pre-langgraph-{stamp}.sqlite3"
        suffix = 1
        while target.exists():
            target = backup_dir / f"pre-langgraph-{stamp}-{suffix}.sqlite3"
            suffix += 1
        destination = sqlite3.connect(target)
        destination.row_factory = sqlite3.Row
        try:
            with self.store._lock:
                self.store._db.backup(destination)
            integrity = destination.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"Migration backup integrity check failed: {integrity}")
            objects = [
                dict(row)
                for row in destination.execute(
                    "SELECT sha256, size, storage_path FROM attachment_objects ORDER BY sha256"
                ).fetchall()
            ]
        finally:
            destination.close()
        manifest = {
            "database": target.name,
            "created_at": time.time(),
            "attachments_root": str(self.runtime.attachments.object_root),
            "objects": objects,
        }
        for item in objects:
            path = self.runtime.attachments.object_root / str(item["storage_path"])
            if not path.is_file() or path.stat().st_size != int(item["size"]):
                raise RuntimeError(f"Attachment backup object is missing or has the wrong size: {path}")
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    hasher.update(block)
            digest = hasher.hexdigest()
            if digest != item["sha256"]:
                raise RuntimeError(f"Attachment backup object has the wrong content hash: {path}")
        target.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    async def _validate(self, report: MigrationReport) -> None:
        with self.store._lock:
            integrity = self.store._db.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"Migration integrity check failed: {integrity}")
            legacy_messages = int(self.store._db.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
            migrated_messages = int(
                self.store._db.execute(
                    "SELECT COUNT(*) FROM discord_events WHERE id LIKE 'legacy:message:%'"
                ).fetchone()[0]
            )
            invalid_deliveries = int(
                self.store._db.execute(
                    "SELECT COUNT(*) FROM side_effect_deliveries "
                    "WHERE state NOT IN ('claimed','completed','ambiguous')"
                ).fetchone()[0]
            )
        if migrated_messages != legacy_messages:
            raise RuntimeError(
                f"Migration event count mismatch: {migrated_messages} events for {legacy_messages} messages"
            )
        if invalid_deliveries:
            raise RuntimeError("Migration found invalid side-effect ledger states")
        for conversation in self.store.conversations():
            channel_id = str(conversation["channel_id"])
            thread_id = await self.runtime.events.thread_id(
                self.runtime._account_id(channel_id),
                channel_id,
            )
            history = self.store.history(channel_id, 1)
            checkpoint = await self.runtime.checkpointer.aget_tuple(
                {"configurable": {"thread_id": thread_id}}
            )
            if history and checkpoint is None:
                raise RuntimeError(f"Migration checkpoint is unreachable for channel {channel_id}")
        if report.backup_path is None or not report.backup_path.is_file():
            raise RuntimeError("Migration backup is unavailable")

    async def _messages(
        self,
        history: list[dict[str, Any]],
        conversation: dict[str, Any],
        account_id: str,
    ) -> list[Any]:
        selected = history
        prefix: list[Any] = []
        if len(history) > 200:
            older, selected = history[:-150], history[-150:]
            summary = await self._summarize(older)
            prefix.append(
                HumanMessage(
                    summary,
                    id=f"migration-summary:{conversation['channel_id']}",
                    additional_kwargs={
                        "diskovod_archive_summary": {
                            "message_ids": [str(item["id"]) for item in older],
                            "created_by": "langgraph_cutover_v1",
                        }
                    },
                )
            )
        messages = prefix
        for item in selected:
            if item["direction"] == "out" and item["source"] == "assistant":
                messages.append(
                    AIMessage(
                        str(item["content"]),
                        id=str(item["id"]),
                        additional_kwargs={"diskovod_delivered_discord_message": True},
                    )
                )
                continue
            role = self._role(item)
            messages.append(
                HumanMessage(
                    str(item["content"]),
                    id=str(item["id"]),
                    additional_kwargs={
                        "diskovod_participant": {
                            "id": str(item["author_id"]),
                            "name": str(item["author_name"]),
                            "role": role,
                            "migrated": True,
                            "observed_at": float(item["timestamp"]),
                        },
                        "diskovod_attachments": item.get("attachments") or [],
                        "diskovod_account_id": account_id,
                    },
                )
            )
        return messages

    async def _summarize(self, messages: list[dict[str, Any]]) -> str:
        rendered = "\n".join(f"[{item['id']} {self._role(item)}] {item['content']}" for item in messages)[
            :100_000
        ]
        locale = self.store.app_settings().prompt_locale
        if self.runtime.models.ready:
            try:
                response = await self.runtime.models.build_model().ainvoke(
                    [
                        SystemMessage(summarization_prompt(locale)),
                        HumanMessage(rendered),
                    ]
                )
                if response.text.strip():
                    return response.text.strip()
            except Exception as error:
                log.warning("Could not summarize legacy history during migration: %s", error)
        return runtime_context_text(locale)["migration_summary"].format(count=len(messages))

    async def _seed(self, thread_id: str, messages: list[Any]) -> None:
        if self.runtime.checkpointer is None:
            raise RuntimeError("The LangGraph checkpointer must be open during migration")

        async def seed(state: DiskovodAgentState) -> dict[str, Any]:
            del state
            return {}

        builder = StateGraph(DiskovodAgentState)
        builder.add_node("seed", seed)
        builder.add_edge(START, "seed")
        builder.add_edge("seed", END)
        graph = builder.compile(checkpointer=self.runtime.checkpointer, store=self.runtime.memory)
        await graph.ainvoke(
            {"messages": messages},
            config={"configurable": {"thread_id": thread_id}},
        )

    async def _archive_legacy_records(self) -> int:
        archived = 0
        async with self.store.database.transaction() as connection:
            tables = {
                str(row[0])
                for row in await (
                    await connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            }
            for table, kind, identity in (
                ("conversation_escalations", "owner_escalation", "id"),
                ("chatgpt_usage", "model_usage", "id"),
                ("model_request_logs", "model_request", "id"),
            ):
                if table not in tables:
                    continue
                rows = await (await connection.execute(f"SELECT * FROM {table}")).fetchall()
                for row in rows:
                    payload = {key: row[key] for key in row.keys()}
                    await connection.execute(
                        "INSERT OR IGNORE INTO legacy_import_records VALUES(?, ?, ?, ?)",
                        (
                            kind,
                            str(row[identity]),
                            json.dumps(payload, ensure_ascii=False, default=str),
                            time.time(),
                        ),
                    )
                    archived += 1
        return archived

    async def _migrate_active_escalations(self) -> None:
        with self.store._lock:
            exists = self.store._db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversation_escalations'"
            ).fetchone()
            rows = (
                self.store._db.execute(
                    "SELECT * FROM conversation_escalations WHERE state IN ('pending','claimed')"
                ).fetchall()
                if exists
                else []
            )
        settings = self.store.app_settings()
        localized = runtime_context_text(settings.prompt_locale)
        for row in rows:
            channel_id = str(row["channel_id"])
            account_id = self.runtime._account_id(channel_id)
            thread_id = await self.runtime.events.thread_id(account_id, channel_id)
            trace_id = f"migration-escalation:{row['id']}"
            tool_call_id = f"legacy-escalation-{row['id']}"
            escalation_id = f"{trace_id}:{tool_call_id}"
            if self.store.escalation_interrupt(escalation_id) is not None:
                continue
            run_id = f"migration-escalation-run:{row['id']}"
            context = AgentRuntimeContext(
                account_id=account_id,
                channel_id=channel_id,
                participant_ids=(),
                owner_id=account_id,
                ui_locale=settings.admin_locale,
                prompt_locale=settings.prompt_locale,
                assistant_name=assistant_name_for(settings.prompt_locale, settings.assistant_name),
                automation_mode="automatic",
                force_reply=False,
                provider_id="migration",
                model_id="deterministic-escalation",
                transport_profile="migration",
                capabilities=CapabilityProfile(),
                trace_id=trace_id,
                thread_id=thread_id,
                owner_timezone=settings.owner_timezone,
                trigger_message_id=str(row["trigger_message_id"]),
                permissions=frozenset({"owner_escalation"}),
            )
            model = _MigrationEscalationModel(
                reason=(
                    str(row["reason"])
                    if row["reason"]
                    in {"peer_requested_owner", "owner_only_information", "other_explicit_request"}
                    else "other_explicit_request"
                ),
                acknowledgement=localized["migration_escalation_ack"],
                tool_call_id=tool_call_id,
            )
            agent = build_agent(
                model,
                _MigrationEscalationGateway(self.runtime, str(row["trigger_message_id"])),
                AgentPrompt(
                    settings.prompt_locale,
                    context.assistant_name,
                    settings.base_instructions,
                ),
                self.runtime.http,
                checkpointer=self.runtime.checkpointer,
                store=self.runtime.memory,
            )
            await self.store.astart_agent_run(
                run_id=run_id,
                thread_id=thread_id,
                channel_id=channel_id,
                trace_id=trace_id,
            )
            result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            localized["migration_escalation_event"],
                            id=f"migration-escalation-event:{row['id']}",
                        )
                    ]
                },
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
                context=context,
            )
            if not result.get("__interrupt__"):
                raise RuntimeError(f"Legacy escalation {row['id']} did not create a graph interrupt")
            await self.store.afinish_agent_run(run_id, "interrupted")
            if row["state"] == "claimed":
                await self.store.aset_interrupt_state(escalation_id, "claimed")

    async def _migrate_owner_details(self) -> None:
        details = self.store.app_settings().owner_details.strip()
        if details:
            await self.runtime.memory.aput(
                ("account", self.runtime._account_id("migration"), "preferences"),
                "owner-provided-details",
                {
                    "value": details,
                    "source": "legacy_app_settings",
                    "migrated_at": time.time(),
                },
            )

    async def _drop_legacy_tables(self) -> None:
        async with self.store.database.transaction() as connection:
            for table in ("conversation_escalations", "chatgpt_usage", "model_request_logs"):
                await connection.execute(f"DROP TABLE IF EXISTS {table}")

    @staticmethod
    def _role(message: dict[str, Any]) -> str:
        if message["direction"] == "in":
            return "peer"
        return "assistant" if message["source"] == "assistant" else "owner"
