from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .agent_types import DiskovodAgentState
from .localization import summarization_prompt
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


class LegacyMigrator:
    """One-time offline transformer; the target runtime never reads legacy model state."""

    def __init__(self, store: Store, runtime: AgentService):
        self.store = store
        self.runtime = runtime

    async def run(self) -> MigrationReport:
        if self.store._get(MIGRATION_KEY, None):
            return MigrationReport(None, 0, 0, 0, 0)
        backup = self._backup()
        conversations = self.store.conversations()
        event_count = 0
        checkpoint_count = 0
        for conversation in conversations:
            channel_id = str(conversation["channel_id"])
            account_id = self.runtime._account_id(channel_id)
            thread_id = self.runtime.events.thread_id(account_id, channel_id)
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
                if self.runtime.events.ingest(
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
        archived = self._archive_legacy_records()
        self._migrate_owner_details()
        report = MigrationReport(backup, len(conversations), event_count, checkpoint_count, archived)
        self.store._set(
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
        try:
            with self.store._lock:
                self.store._db.backup(destination)
            integrity = destination.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"Migration backup integrity check failed: {integrity}")
        finally:
            destination.close()
        return target

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
        return (
            f"Archived {len(messages)} earlier Discord messages. Their full audit records remain "
            "in the local database; no semantic summary was available during migration."
        )

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

    def _archive_legacy_records(self) -> int:
        archived = 0
        with self.store._lock, self.store._db:
            tables = {
                str(row[0])
                for row in self.store._db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            for table, kind, identity in (
                ("conversation_escalations", "owner_escalation", "id"),
                ("chatgpt_usage", "model_usage", "id"),
                ("model_request_logs", "model_request", "id"),
            ):
                if table not in tables:
                    continue
                rows = self.store._db.execute(f"SELECT * FROM {table}").fetchall()
                for row in rows:
                    payload = {key: row[key] for key in row.keys()}
                    self.store._db.execute(
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

    def _migrate_owner_details(self) -> None:
        details = self.store.app_settings().owner_details.strip()
        if details:
            self.runtime.memory.put(
                ("account", self.runtime._account_id("migration"), "preferences"),
                "owner-provided-details",
                {
                    "value": details,
                    "source": "legacy_app_settings",
                    "migrated_at": time.time(),
                },
            )

    @staticmethod
    def _role(message: dict[str, Any]) -> str:
        if message["direction"] == "in":
            return "peer"
        return "assistant" if message["source"] == "assistant" else "owner"
