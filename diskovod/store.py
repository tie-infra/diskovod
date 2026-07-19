from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .localization import normalize_locale
from .models import (
    ADMIN_THEMES,
    DEFAULT_BASE_INSTRUCTIONS,
    AppSettings,
    ChatCredentials,
    CustomProvider,
)
from .persistence import AsyncSQLite, initialize_target_schema
from .security import SecretBox

LEGACY_BASE_INSTRUCTIONS_SHA256 = "ce9bd3d8ffbef462362269db68c7996d9ca0e3e93761d197fcf82f5e0f25502c"
DATABASE_TABLES = {
    "config": {"label": "Configuration", "primary_key": "key", "order_by": "updated_at", "read_only": True},
    "conversations": {
        "label": "Conversations",
        "primary_key": "channel_id",
        "order_by": "updated_at",
        "read_only": False,
    },
    "messages": {"label": "Messages", "primary_key": "id", "order_by": "timestamp", "read_only": False},
    "assistant_reactions": {
        "label": "Assistant reactions",
        "primary_key": "trigger_message_id",
        "order_by": "created_at",
        "read_only": False,
    },
    "bot_nonces": {
        "label": "Pending nonces",
        "primary_key": "nonce",
        "order_by": "created_at",
        "read_only": False,
    },
    "bot_message_ids": {
        "label": "Assistant message markers",
        "primary_key": "id",
        "order_by": "created_at",
        "read_only": False,
    },
    "agent_configuration_versions": {
        "label": "Agent configurations",
        "primary_key": "id",
        "order_by": "created_at",
        "read_only": True,
    },
    "chat_threads": {
        "label": "Graph threads",
        "primary_key": "channel_id",
        "order_by": "updated_at",
        "read_only": True,
    },
    "discord_events": {
        "label": "Discord event audit",
        "primary_key": "id",
        "order_by": "observed_at",
        "read_only": True,
    },
    "chat_event_queue": {
        "label": "Agent event queue",
        "primary_key": "event_id",
        "order_by": "claimed_at",
        "read_only": True,
    },
    "side_effect_deliveries": {
        "label": "Side-effect deliveries",
        "primary_key": "tool_call_id",
        "order_by": "claimed_at",
        "read_only": True,
    },
    "agent_runs": {
        "label": "Agent runs",
        "primary_key": "id",
        "order_by": "started_at",
        "read_only": True,
    },
    "agent_trace_events": {
        "label": "Agent trace events",
        "primary_key": "id",
        "order_by": "recorded_at",
        "read_only": True,
    },
    "provider_capability_probes": {
        "label": "Capability probes",
        "primary_key": "id",
        "order_by": "completed_at",
        "read_only": True,
    },
    "attachment_objects": {
        "label": "Attachment objects",
        "primary_key": "sha256",
        "order_by": "created_at",
        "read_only": True,
    },
    "attachment_references": {
        "label": "Attachment references",
        "primary_key": "id",
        "order_by": "created_at",
        "read_only": False,
    },
    "attachment_artifacts": {
        "label": "Attachment artifacts",
        "primary_key": "id",
        "order_by": "updated_at",
        "read_only": True,
    },
    "attachment_chunks": {
        "label": "Attachment chunks",
        "primary_key": "id",
        "order_by": "id",
        "read_only": True,
    },
    "escalation_interrupts": {
        "label": "Graph interrupts",
        "primary_key": "id",
        "order_by": "updated_at",
        "read_only": True,
    },
    "langgraph_store_items": {
        "label": "Long-term memories",
        "primary_key": "key",
        "order_by": "updated_at",
        "read_only": True,
    },
    "legacy_import_records": {
        "label": "Archived pre-LangGraph records",
        "primary_key": "source_id",
        "order_by": "imported_at",
        "read_only": True,
    },
}


class Store:
    def __init__(self, path: Path, secret: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.database = AsyncSQLite(path)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._box = SecretBox(secret)
        with self._db:
            self._db.execute("PRAGMA busy_timeout=5000")
            self._db.execute("PRAGMA foreign_keys=ON")
            initialize_target_schema(self._db)
            message_columns = {
                row["name"] for row in self._db.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "attachments" not in message_columns:
                self._db.execute("ALTER TABLE messages ADD COLUMN attachments TEXT NOT NULL DEFAULT '[]'")
            conversation_columns = {
                row["name"] for row in self._db.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "mode" not in conversation_columns:
                self._db.execute(
                    "ALTER TABLE conversations ADD COLUMN mode TEXT NOT NULL DEFAULT 'automatic'"
                )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    async def aclose(self) -> None:
        await self.database.close()
        self.close()

    async def _aset(self, key: str, value: Any, *, secret: bool = False) -> None:
        raw = json.dumps(value)
        if secret:
            raw = self._box.seal(raw)
        async with self.database.transaction() as connection:
            await connection.execute(
                "INSERT INTO config VALUES(?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "secret=excluded.secret, updated_at=excluded.updated_at",
                (key, raw, int(secret), time.time()),
            )

    async def _adelete(self, key: str) -> None:
        async with self.database.transaction() as connection:
            await connection.execute("DELETE FROM config WHERE key=?", (key,))

    def _get(self, key: str, default: Any) -> Any:
        with self._lock:
            row = self._db.execute("SELECT value, secret FROM config WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        raw = self._box.open(row["value"]) if row["secret"] else row["value"]
        return json.loads(raw)

    def _set(self, key: str, value: Any, *, secret: bool = False) -> None:
        raw = json.dumps(value)
        if secret:
            raw = self._box.seal(raw)
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO config VALUES(?,?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, secret=excluded.secret, updated_at=excluded.updated_at",
                (key, raw, int(secret), time.time()),
            )

    def _delete(self, key: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM config WHERE key=?", (key,))

    def app_settings(self) -> AppSettings:
        saved = self._get("app.settings", {})
        base_instructions = saved.get("base_instructions")
        if (
            isinstance(base_instructions, str)
            and hashlib.sha256(base_instructions.encode()).hexdigest() == LEGACY_BASE_INSTRUCTIONS_SHA256
        ):
            saved["base_instructions"] = DEFAULT_BASE_INSTRUCTIONS
        saved["admin_locale"] = normalize_locale(str(saved.get("admin_locale", "en")))
        saved["admin_theme"] = (
            str(saved.get("admin_theme", "system")) if saved.get("admin_theme") in ADMIN_THEMES else "system"
        )
        saved["prompt_locale"] = normalize_locale(str(saved.get("prompt_locale", "en")))
        defaults = AppSettings().to_dict()
        known = {key: value for key, value in saved.items() if key in defaults}
        return AppSettings(**(defaults | known))

    def set_app_settings(self, value: AppSettings) -> None:
        self._set("app.settings", value.to_dict())

    def discord_token(self) -> str | None:
        return self._get("discord.token", None)

    def set_discord_token(self, value: str) -> None:
        self._set("discord.token", value, secret=True)

    def clear_discord_token(self) -> None:
        self._delete("discord.token")

    def chat_credentials(self) -> ChatCredentials | None:
        value = self._get("chatgpt.credentials", None)
        return ChatCredentials(**value) if value else None

    def set_chat_credentials(self, value: ChatCredentials) -> None:
        self._set("chatgpt.credentials", value.to_dict(), secret=True)

    def clear_chat_credentials(self) -> None:
        self._delete("chatgpt.credentials")

    def custom_provider(self) -> CustomProvider | None:
        value = self._get("openai_compatible.provider", None)
        if not value:
            return None
        # Providers saved before protocol selection existed used Chat Completions.
        value.setdefault("protocol", "chat_completions")
        # Providers saved before this capability existed always received the limit.
        value.setdefault("capabilities", {}).setdefault("output_token_limit", True)
        return CustomProvider(**value)

    def set_custom_provider(self, value: CustomProvider) -> None:
        self._set("openai_compatible.provider", value.to_dict(), secret=True)

    def clear_custom_provider(self) -> None:
        self._delete("openai_compatible.provider")

    def active_agent_configuration(self):
        from .providers import ModelConfiguration

        with self._lock:
            row = self._db.execute(
                "SELECT configuration FROM agent_configuration_versions WHERE active=1"
            ).fetchone()
        return ModelConfiguration.from_dict(json.loads(row["configuration"])) if row else None

    def save_agent_configuration(self, configuration) -> int:
        payload = json.dumps(configuration.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._db:
            self._db.execute("UPDATE agent_configuration_versions SET active=0 WHERE active=1")
            cursor = self._db.execute(
                "INSERT INTO agent_configuration_versions(created_at, configuration, active) VALUES(?, ?, 1)",
                (time.time(), payload),
            )
            return int(cursor.lastrowid)

    def set_provider_credentials(self, profile: str, value: dict[str, Any]) -> None:
        if not profile or not profile.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid credential profile")
        self._set(f"provider.credentials.{profile}", value, secret=True)

    def provider_credentials(self, profile: str) -> dict[str, Any] | None:
        return self._get(f"provider.credentials.{profile}", None)

    def clear_provider_credentials(self, profile: str) -> None:
        self._delete(f"provider.credentials.{profile}")

    def personality(self) -> dict[str, Any] | None:
        return self._get("personality", None)

    def set_personality(self, profile: str, source_hash: str, source: str = "inferred") -> None:
        self._set(
            "personality",
            {
                "profile": profile,
                "source_hash": source_hash,
                "source": source,
                "updated_at": time.time(),
            },
        )

    def start_agent_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        channel_id: str,
        trace_id: str,
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO agent_runs(id, thread_id, channel_id, trace_id, status, started_at) "
                "VALUES(?, ?, ?, ?, 'running', ?)",
                (run_id, thread_id, channel_id, trace_id, time.time()),
            )

    def finish_agent_run(self, run_id: str, status: str, error: str | None = None) -> None:
        if status not in {"completed", "interrupted", "failed", "cancelled"}:
            raise ValueError("Invalid agent run status")
        with self._lock, self._db:
            self._db.execute(
                "UPDATE agent_runs SET status=?, completed_at=?, error=? WHERE id=?",
                (status, time.time(), error, run_id),
            )

    def record_agent_trace(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        with self._lock, self._db:
            sequence = int(
                self._db.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM agent_trace_events WHERE run_id=?",
                    (run_id,),
                ).fetchone()[0]
            )
            self._db.execute(
                "INSERT INTO agent_trace_events(run_id, sequence, kind, payload, recorded_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (
                    run_id,
                    sequence,
                    kind,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    time.time(),
                ),
            )

    def upsert_conversation(self, channel_id: str, peer_id: str, peer_name: str) -> None:
        now = time.time()
        default_paused = not self.app_settings().default_conversation_enabled
        with self._lock, self._db:
            self._db.execute(
                """INSERT INTO conversations
                   (channel_id, peer_id, peer_name, paused, paused_at, updated_at, snoozed_until, mode)
                   VALUES(?,?,?,?,?,?,NULL,'automatic')
              ON CONFLICT(channel_id) DO UPDATE SET peer_id=excluded.peer_id, peer_name=excluded.peer_name, updated_at=excluded.updated_at""",
                (
                    channel_id,
                    peer_id,
                    peer_name,
                    int(default_paused),
                    now if default_paused else None,
                    now,
                ),
            )

    def set_permanent_pause(self, channel_id: str, paused: bool) -> None:
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "UPDATE conversations SET paused=?, paused_at=?, updated_at=? WHERE channel_id=?",
                (int(paused), now if paused else None, now, channel_id),
            )

    def set_conversation_mode(self, channel_id: str, mode: str) -> bool:
        if mode not in {"automatic", "inline", "paused"}:
            raise ValueError("Unknown conversation mode")
        now = time.time()
        paused = mode == "paused"
        with self._lock, self._db:
            cursor = self._db.execute(
                """UPDATE conversations
                   SET mode=CASE WHEN ?='paused' THEN mode ELSE ? END,
                       paused=?, paused_at=?, snoozed_until=NULL, updated_at=?
                   WHERE channel_id=?""",
                (mode, mode, int(paused), now if paused else None, now, channel_id),
            )
        return cursor.rowcount > 0

    def snooze(self, channel_id: str, seconds: float) -> float:
        until = time.time() + max(0.0, seconds)
        with self._lock, self._db:
            self._db.execute(
                "UPDATE conversations SET snoozed_until=?, updated_at=? WHERE channel_id=?",
                (until, time.time(), channel_id),
            )
        return until

    def clear_snooze(self, channel_id: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE conversations SET snoozed_until=NULL, updated_at=? WHERE channel_id=?",
                (time.time(), channel_id),
            )

    def active_interrupts(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT i.*, c.peer_name FROM escalation_interrupts i
                LEFT JOIN conversations c ON c.channel_id=i.channel_id
                WHERE i.state IN ('pending', 'claimed')
                ORDER BY i.created_at DESC
                """
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result

    def escalation_interrupt(self, escalation_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM escalation_interrupts WHERE id=?", (escalation_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = json.loads(item["payload"])
        return item

    def set_interrupt_state(self, escalation_id: str, state: str) -> bool:
        if state not in {"claimed", "resolved", "dismissed"}:
            raise ValueError("Unknown interrupt state")
        with self._lock, self._db:
            cursor = self._db.execute(
                """
                UPDATE escalation_interrupts SET state=?, updated_at=?
                WHERE id=? AND state IN ('pending', 'claimed')
                """,
                (state, time.time(), escalation_id),
            )
        return cursor.rowcount > 0

    def active_interrupt_for_channel(self, channel_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT * FROM escalation_interrupts
                WHERE channel_id=? AND state IN ('pending', 'claimed')
                ORDER BY created_at DESC LIMIT 1
                """,
                (channel_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = json.loads(item["payload"])
        return item

    def can_automate(self, channel_id: str, now: float | None = None) -> bool:
        conversation = self.conversation(channel_id)
        if not conversation or conversation["paused"]:
            return False
        current_time = time.time() if now is None else now
        return not conversation["snoozed_until"] or conversation["snoozed_until"] <= current_time

    def conversation(self, channel_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM conversations WHERE channel_id=?", (channel_id,)).fetchone()
        return self._conversation_dict(row) if row else None

    def conversations(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT 200"
            ).fetchall()
        return [self._conversation_dict(row) for row in rows]

    @staticmethod
    def _conversation_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "channel_id": row["channel_id"],
            "peer_id": row["peer_id"],
            "peer_name": row["peer_name"],
            "paused": bool(row["paused"]),
            "paused_at": row["paused_at"],
            "mode": row["mode"],
            "snoozed_until": row["snoozed_until"],
            "updated_at": row["updated_at"],
        }

    def save_message(
        self,
        *,
        id: str,
        channel_id: str,
        author_id: str,
        author_name: str,
        direction: str,
        source: str,
        content: str,
        timestamp: float,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                """INSERT OR IGNORE INTO messages
                   (id, channel_id, author_id, author_name, direction, source, content,
                    timestamp, attachments) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    id,
                    channel_id,
                    author_id,
                    author_name,
                    direction,
                    source,
                    content,
                    timestamp,
                    json.dumps(attachments or [], ensure_ascii=False),
                ),
            )
            self._db.execute(
                "UPDATE conversations SET updated_at=? WHERE channel_id=?", (timestamp, channel_id)
            )

    def history(self, channel_id: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM messages WHERE channel_id=? ORDER BY timestamp DESC LIMIT ?",
                (channel_id, limit),
            ).fetchall()
        result = []
        for row in reversed(rows):
            item = dict(row)
            try:
                attachments = json.loads(item.get("attachments") or "[]")
            except (TypeError, json.JSONDecodeError):
                attachments = []
            item["attachments"] = attachments if isinstance(attachments, list) else []
            result.append(item)
        return result

    def latest_incoming_message(self, channel_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                """SELECT * FROM messages
                   WHERE channel_id=? AND direction='in'
                   ORDER BY timestamp DESC LIMIT 1""",
                (channel_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        try:
            attachments = json.loads(item.get("attachments") or "[]")
        except (TypeError, json.JSONDecodeError):
            attachments = []
        item["attachments"] = attachments if isinstance(attachments, list) else []
        return item

    def update_message_content(
        self, message_id: str, content: str, *, source: str | None = None
    ) -> dict[str, Any] | None:
        with self._lock, self._db:
            row = self._db.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
            if row is None:
                return None
            new_source = source or row["source"]
            changed = row["content"] != content or row["source"] != new_source
            if changed:
                self._db.execute(
                    "UPDATE messages SET content=?, source=? WHERE id=?",
                    (content, new_source, message_id),
                )
        result = dict(row)
        result.update(content=content, source=new_source, changed=changed)
        return result

    def remember_nonce(self, nonce: str) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO bot_nonces VALUES(?,?)", (nonce, time.time()))

    def consume_nonce(self, nonce: str) -> bool:
        with self._lock, self._db:
            found = (
                self._db.execute("SELECT 1 FROM bot_nonces WHERE nonce=?", (nonce,)).fetchone() is not None
            )
            if found:
                self._db.execute("DELETE FROM bot_nonces WHERE nonce=?", (nonce,))
        return found

    def remember_bot_message(self, message_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO bot_message_ids VALUES(?,?)", (message_id, time.time()))

    def is_bot_message(self, message_id: str) -> bool:
        with self._lock:
            return (
                self._db.execute("SELECT 1 FROM bot_message_ids WHERE id=?", (message_id,)).fetchone()
                is not None
            )

    def is_assistant_message(self, message_id: str) -> bool:
        with self._lock:
            return (
                self._db.execute(
                    "SELECT 1 FROM messages WHERE id=? AND source='assistant'", (message_id,)
                ).fetchone()
                is not None
            )

    def reaction_allowed(
        self,
        channel_id: str,
        *,
        now: float | None = None,
        channel_cooldown_seconds: float = 6 * 60 * 60,
        recent_action_limit: int = 12,
    ) -> bool:
        current_time = time.time() if now is None else now
        with self._lock:
            recent_in_channel = self._db.execute(
                """SELECT 1 FROM assistant_reactions
                   WHERE channel_id=? AND created_at>=? LIMIT 1""",
                (channel_id, current_time - channel_cooldown_seconds),
            ).fetchone()
            if recent_in_channel:
                return False
            recent_actions = self._db.execute(
                """SELECT kind FROM (
                     SELECT timestamp AS action_time, 'message' AS kind
                       FROM messages WHERE source='assistant'
                     UNION ALL
                     SELECT created_at AS action_time, 'reaction' AS kind
                       FROM assistant_reactions
                   ) ORDER BY action_time DESC LIMIT ?""",
                (max(1, recent_action_limit),),
            ).fetchall()
        return all(row["kind"] != "reaction" for row in recent_actions)

    def record_assistant_reaction(
        self,
        *,
        trigger_message_id: str,
        channel_id: str,
        emoji: str,
        created_at: float | None = None,
    ) -> None:
        timestamp = time.time() if created_at is None else created_at
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO assistant_reactions VALUES(?,?,?,?)",
                (trigger_message_id, channel_id, emoji, timestamp),
            )
            self._db.execute(
                "UPDATE conversations SET updated_at=? WHERE channel_id=?",
                (timestamp, channel_id),
            )

    def prune(self) -> None:
        cutoff = time.time() - 86400
        with self._lock, self._db:
            self._db.execute("DELETE FROM bot_nonces WHERE created_at<?", (cutoff,))
            self._db.execute("DELETE FROM bot_message_ids WHERE created_at<?", (cutoff,))

    def database_tables(self) -> list[dict[str, Any]]:
        with self._lock:
            result = []
            for name, spec in DATABASE_TABLES.items():
                count = self._db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                result.append({"name": name, "count": count, **spec})
        return result

    def database_rows(
        self,
        table: str,
        *,
        limit: int = 50,
        offset: int = 0,
        query: str = "",
    ) -> dict[str, Any]:
        spec = DATABASE_TABLES.get(table)
        if spec is None:
            raise ValueError("Unknown database table")
        page_limit = max(1, min(limit, 100))
        page_offset = max(0, offset)
        with self._lock:
            columns = [row["name"] for row in self._db.execute(f'PRAGMA table_info("{table}")').fetchall()]
            searchable = ["key"] if table == "config" else columns
            parameters: list[Any] = []
            where = ""
            if query:
                where = " WHERE " + " OR ".join(f'CAST("{column}" AS TEXT) LIKE ?' for column in searchable)
                parameters.extend([f"%{query}%"] * len(searchable))
            total = self._db.execute(f'SELECT COUNT(*) FROM "{table}"{where}', parameters).fetchone()[0]
            rows = [
                dict(row)
                for row in self._db.execute(
                    f'SELECT * FROM "{table}"{where} ORDER BY "{spec["order_by"]}" DESC LIMIT ? OFFSET ?',
                    (*parameters, page_limit, page_offset),
                ).fetchall()
            ]
        if table == "config":
            for row in rows:
                if row["secret"]:
                    row["value"] = "[encrypted value redacted]"
        return {
            "name": table,
            "label": spec["label"],
            "primary_key": spec["primary_key"],
            "read_only": spec["read_only"],
            "columns": columns,
            "rows": rows,
            "total": total,
            "limit": page_limit,
            "offset": page_offset,
            "query": query,
        }

    def delete_database_row(self, table: str, row_key: str) -> bool:
        spec = DATABASE_TABLES.get(table)
        if spec is None:
            raise ValueError("Unknown database table")
        if spec["read_only"]:
            raise ValueError("This database table is read-only")
        with self._lock, self._db:
            cursor = self._db.execute(
                f'DELETE FROM "{table}" WHERE "{spec["primary_key"]}"=?',
                (row_key,),
            )
        return cursor.rowcount > 0

    async def aset_app_settings(self, value: AppSettings) -> None:
        await self._aset("app.settings", value.to_dict())

    async def aset_discord_token(self, value: str) -> None:
        await self._aset("discord.token", value, secret=True)

    async def aclear_discord_token(self) -> None:
        await self._adelete("discord.token")

    async def aset_chat_credentials(self, value: ChatCredentials) -> None:
        await self._aset("chatgpt.credentials", value.to_dict(), secret=True)

    async def aclear_chat_credentials(self) -> None:
        await self._adelete("chatgpt.credentials")

    async def aset_custom_provider(self, value: CustomProvider) -> None:
        await self._aset("openai_compatible.provider", value.to_dict(), secret=True)

    async def aclear_custom_provider(self) -> None:
        await self._adelete("openai_compatible.provider")

    async def aset_provider_credentials(self, profile: str, value: dict[str, Any]) -> None:
        if not profile or not profile.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid credential profile")
        await self._aset(f"provider.credentials.{profile}", value, secret=True)

    async def aclear_provider_credentials(self, profile: str) -> None:
        await self._adelete(f"provider.credentials.{profile}")

    async def aset_personality(self, profile: str, source_hash: str, source: str = "inferred") -> None:
        await self._aset(
            "personality",
            {
                "profile": profile,
                "source_hash": source_hash,
                "source": source,
                "updated_at": time.time(),
            },
        )

    async def asave_agent_configuration(self, configuration) -> int:
        payload = json.dumps(configuration.to_dict(), ensure_ascii=False, separators=(",", ":"))
        async with self.database.transaction() as connection:
            await connection.execute("UPDATE agent_configuration_versions SET active=0 WHERE active=1")
            cursor = await connection.execute(
                "INSERT INTO agent_configuration_versions(created_at, configuration, active) VALUES(?, ?, 1)",
                (time.time(), payload),
            )
            return int(cursor.lastrowid)

    async def astart_agent_run(self, *, run_id: str, thread_id: str, channel_id: str, trace_id: str) -> None:
        async with self.database.transaction() as connection:
            await connection.execute(
                "INSERT INTO agent_runs(id, thread_id, channel_id, trace_id, status, started_at) "
                "VALUES(?, ?, ?, ?, 'running', ?)",
                (run_id, thread_id, channel_id, trace_id, time.time()),
            )

    async def afinish_agent_run(self, run_id: str, status: str, error: str | None = None) -> None:
        if status not in {"completed", "interrupted", "failed", "cancelled"}:
            raise ValueError("Invalid agent run status")
        async with self.database.transaction() as connection:
            await connection.execute(
                "UPDATE agent_runs SET status=?, completed_at=?, error=? WHERE id=?",
                (status, time.time(), error, run_id),
            )

    async def arecord_agent_trace(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM agent_trace_events WHERE run_id=?",
                    (run_id,),
                )
            ).fetchone()
            await connection.execute(
                "INSERT INTO agent_trace_events(run_id, sequence, kind, payload, recorded_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (
                    run_id,
                    int(row[0]),
                    kind,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    time.time(),
                ),
            )

    async def aupsert_conversation(self, channel_id: str, peer_id: str, peer_name: str) -> None:
        now = time.time()
        default_paused = not self.app_settings().default_conversation_enabled
        async with self.database.transaction() as connection:
            await connection.execute(
                """INSERT INTO conversations
                   (channel_id, peer_id, peer_name, paused, paused_at, updated_at, snoozed_until, mode)
                   VALUES(?,?,?,?,?,?,NULL,'automatic')
                   ON CONFLICT(channel_id) DO UPDATE SET
                     peer_id=excluded.peer_id, peer_name=excluded.peer_name,
                     updated_at=excluded.updated_at""",
                (
                    channel_id,
                    peer_id,
                    peer_name,
                    int(default_paused),
                    now if default_paused else None,
                    now,
                ),
            )

    async def aset_permanent_pause(self, channel_id: str, paused: bool) -> None:
        now = time.time()
        async with self.database.transaction() as connection:
            await connection.execute(
                "UPDATE conversations SET paused=?, paused_at=?, updated_at=? WHERE channel_id=?",
                (int(paused), now if paused else None, now, channel_id),
            )

    async def aset_conversation_mode(self, channel_id: str, mode: str) -> bool:
        if mode not in {"automatic", "inline", "paused"}:
            raise ValueError("Unknown conversation mode")
        now = time.time()
        paused = mode == "paused"
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """UPDATE conversations
                   SET mode=CASE WHEN ?='paused' THEN mode ELSE ? END,
                       paused=?, paused_at=?, snoozed_until=NULL, updated_at=?
                   WHERE channel_id=?""",
                (mode, mode, int(paused), now if paused else None, now, channel_id),
            )
            return cursor.rowcount > 0

    async def asnooze(self, channel_id: str, seconds: float) -> float:
        until = time.time() + max(0.0, seconds)
        async with self.database.transaction() as connection:
            await connection.execute(
                "UPDATE conversations SET snoozed_until=?, updated_at=? WHERE channel_id=?",
                (until, time.time(), channel_id),
            )
        return until

    async def aclear_snooze(self, channel_id: str) -> None:
        async with self.database.transaction() as connection:
            await connection.execute(
                "UPDATE conversations SET snoozed_until=NULL, updated_at=? WHERE channel_id=?",
                (time.time(), channel_id),
            )

    async def aset_interrupt_state(self, escalation_id: str, state: str) -> bool:
        if state not in {"claimed", "resolved", "dismissed"}:
            raise ValueError("Unknown interrupt state")
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                "UPDATE escalation_interrupts SET state=?, updated_at=? "
                "WHERE id=? AND state IN ('pending', 'claimed')",
                (state, time.time(), escalation_id),
            )
            return cursor.rowcount > 0

    async def asave_message(
        self,
        *,
        id: str,
        channel_id: str,
        author_id: str,
        author_name: str,
        direction: str,
        source: str,
        content: str,
        timestamp: float,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        async with self.database.transaction() as connection:
            await connection.execute(
                """INSERT OR IGNORE INTO messages
                   (id, channel_id, author_id, author_name, direction, source, content,
                    timestamp, attachments) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    id,
                    channel_id,
                    author_id,
                    author_name,
                    direction,
                    source,
                    content,
                    timestamp,
                    json.dumps(attachments or [], ensure_ascii=False),
                ),
            )
            await connection.execute(
                "UPDATE conversations SET updated_at=? WHERE channel_id=?", (timestamp, channel_id)
            )

    async def aupdate_message_content(
        self, message_id: str, content: str, *, source: str | None = None
    ) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT * FROM messages WHERE id=?", (message_id,))
            ).fetchone()
            if row is None:
                return None
            new_source = source or row["source"]
            changed = row["content"] != content or row["source"] != new_source
            if changed:
                await connection.execute(
                    "UPDATE messages SET content=?, source=? WHERE id=?",
                    (content, new_source, message_id),
                )
        result = dict(row)
        result.update(content=content, source=new_source, changed=changed)
        return result

    async def aremember_nonce(self, nonce: str) -> None:
        async with self.database.transaction() as connection:
            await connection.execute("INSERT OR REPLACE INTO bot_nonces VALUES(?,?)", (nonce, time.time()))

    async def aconsume_nonce(self, nonce: str) -> bool:
        async with self.database.transaction() as connection:
            found = await (
                await connection.execute("SELECT 1 FROM bot_nonces WHERE nonce=?", (nonce,))
            ).fetchone()
            if found:
                await connection.execute("DELETE FROM bot_nonces WHERE nonce=?", (nonce,))
        return found is not None

    async def ais_bot_message(self, message_id: str) -> bool:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT 1 FROM bot_message_ids WHERE id=?", (message_id,))
            ).fetchone()
            return row is not None

    async def aremember_bot_message(self, message_id: str) -> None:
        async with self.database.transaction() as connection:
            await connection.execute(
                "INSERT OR REPLACE INTO bot_message_ids VALUES(?,?)", (message_id, time.time())
            )

    async def arecord_assistant_reaction(
        self,
        *,
        trigger_message_id: str,
        channel_id: str,
        emoji: str,
        created_at: float | None = None,
    ) -> None:
        timestamp = time.time() if created_at is None else created_at
        async with self.database.transaction() as connection:
            await connection.execute(
                "INSERT OR IGNORE INTO assistant_reactions VALUES(?,?,?,?)",
                (trigger_message_id, channel_id, emoji, timestamp),
            )
            await connection.execute(
                "UPDATE conversations SET updated_at=? WHERE channel_id=?",
                (timestamp, channel_id),
            )

    async def aprune(self) -> None:
        cutoff = time.time() - 86400
        async with self.database.transaction() as connection:
            await connection.execute("DELETE FROM bot_nonces WHERE created_at<?", (cutoff,))
            await connection.execute("DELETE FROM bot_message_ids WHERE created_at<?", (cutoff,))

    async def adelete_database_row(self, table: str, row_key: str) -> bool:
        spec = DATABASE_TABLES.get(table)
        if spec is None:
            raise ValueError("Unknown database table")
        if spec["read_only"]:
            raise ValueError("This database table is read-only")
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                f'DELETE FROM "{table}" WHERE "{spec["primary_key"]}"=?',
                (row_key,),
            )
            return cursor.rowcount > 0
