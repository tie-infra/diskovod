from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .models import (
    DEFAULT_BASE_INSTRUCTIONS,
    AppSettings,
    ChatCredentials,
    CustomProvider,
)
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
    "chatgpt_usage": {
        "label": "Model usage",
        "primary_key": "id",
        "order_by": "recorded_at",
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
}


class Store:
    def __init__(self, path: Path, secret: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._box = SecretBox(secret)
        with self._db:
            self._db.executescript("""
              PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, secret INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
              );
              CREATE TABLE IF NOT EXISTS conversations (
                channel_id TEXT PRIMARY KEY, peer_id TEXT NOT NULL, peer_name TEXT NOT NULL,
                paused INTEGER NOT NULL DEFAULT 0, paused_at REAL, updated_at REAL NOT NULL,
                snoozed_until REAL
              );
              CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, channel_id TEXT NOT NULL, author_id TEXT NOT NULL,
                author_name TEXT NOT NULL, direction TEXT NOT NULL, source TEXT NOT NULL,
                content TEXT NOT NULL, timestamp REAL NOT NULL,
                attachments TEXT NOT NULL DEFAULT '[]'
              );
              CREATE INDEX IF NOT EXISTS messages_channel_time ON messages(channel_id, timestamp DESC);
              CREATE TABLE IF NOT EXISTS bot_nonces (nonce TEXT PRIMARY KEY, created_at REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS bot_message_ids (id TEXT PRIMARY KEY, created_at REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS assistant_reactions (
                trigger_message_id TEXT PRIMARY KEY, channel_id TEXT NOT NULL,
                emoji TEXT NOT NULL, created_at REAL NOT NULL
              );
              CREATE INDEX IF NOT EXISTS assistant_reactions_channel_time
                ON assistant_reactions(channel_id, created_at DESC);
              CREATE TABLE IF NOT EXISTS chatgpt_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id TEXT UNIQUE,
                recorded_at REAL NOT NULL,
                model TEXT NOT NULL,
                purpose TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                cached_input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                reasoning_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL
              );
              CREATE INDEX IF NOT EXISTS chatgpt_usage_recorded_at
                ON chatgpt_usage(recorded_at DESC);
            """)
            message_columns = {
                row["name"] for row in self._db.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "attachments" not in message_columns:
                self._db.execute("ALTER TABLE messages ADD COLUMN attachments TEXT NOT NULL DEFAULT '[]'")

    def close(self) -> None:
        with self._lock:
            self._db.close()

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
        return AppSettings(**(AppSettings().to_dict() | saved))

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
        return CustomProvider(**value) if value else None

    def set_custom_provider(self, value: CustomProvider) -> None:
        self._set("openai_compatible.provider", value.to_dict(), secret=True)

    def clear_custom_provider(self) -> None:
        self._delete("openai_compatible.provider")

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

    def record_chatgpt_usage(
        self,
        *,
        response_id: str | None,
        model: str,
        purpose: str,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        total_tokens: int,
        recorded_at: float | None = None,
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                """INSERT OR IGNORE INTO chatgpt_usage
                   (response_id, recorded_at, model, purpose, input_tokens,
                    cached_input_tokens, output_tokens, reasoning_tokens, total_tokens)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    response_id,
                    time.time() if recorded_at is None else recorded_at,
                    model,
                    purpose,
                    max(0, input_tokens),
                    max(0, cached_input_tokens),
                    max(0, output_tokens),
                    max(0, reasoning_tokens),
                    max(0, total_tokens),
                ),
            )

    def chatgpt_usage_stats(self, now: float | None = None) -> dict[str, Any]:
        current_time = time.time() if now is None else now
        windows = [
            ("Last 24 hours", current_time - 86400),
            ("Last 7 days", current_time - 7 * 86400),
            ("Last 30 days", current_time - 30 * 86400),
            ("All time", None),
        ]
        with self._lock:
            window_stats = [{"label": label, **self._usage_summary(cutoff)} for label, cutoff in windows]
            by_model = self._usage_groups("model")
            by_purpose = self._usage_groups("purpose")
            recent = [
                dict(row)
                for row in self._db.execute(
                    """SELECT recorded_at, model, purpose, input_tokens,
                              cached_input_tokens, output_tokens, reasoning_tokens, total_tokens
                       FROM chatgpt_usage ORDER BY recorded_at DESC LIMIT 50"""
                ).fetchall()
            ]
        return {
            "windows": window_stats,
            "all_time": window_stats[-1],
            "by_model": by_model,
            "by_purpose": by_purpose,
            "recent": recent,
        }

    def _usage_summary(self, cutoff: float | None = None) -> dict[str, Any]:
        where = " WHERE recorded_at >= ?" if cutoff is not None else ""
        parameters = (cutoff,) if cutoff is not None else ()
        row = self._db.execute(
            """SELECT COUNT(*) AS requests,
                      COALESCE(SUM(input_tokens), 0) AS input_tokens,
                      COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                      COALESCE(SUM(output_tokens), 0) AS output_tokens,
                      COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                      COALESCE(SUM(total_tokens), 0) AS total_tokens
               FROM chatgpt_usage"""
            + where,
            parameters,
        ).fetchone()
        result = dict(row)
        requests = result["requests"]
        input_tokens = result["input_tokens"]
        result["average_tokens"] = round(result["total_tokens"] / requests) if requests else 0
        result["cache_rate"] = (
            round(result["cached_input_tokens"] * 100 / input_tokens, 1) if input_tokens else 0.0
        )
        return result

    def _usage_groups(self, column: str) -> list[dict[str, Any]]:
        if column not in {"model", "purpose"}:
            raise ValueError("Unsupported usage grouping")
        rows = self._db.execute(
            f"""SELECT {column} AS name, COUNT(*) AS requests,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens
                FROM chatgpt_usage GROUP BY {column}
                ORDER BY total_tokens DESC, name ASC"""
        ).fetchall()
        return [dict(row) for row in rows]

    def upsert_conversation(self, channel_id: str, peer_id: str, peer_name: str) -> None:
        now = time.time()
        default_paused = not self.app_settings().default_conversation_enabled
        with self._lock, self._db:
            self._db.execute(
                """INSERT INTO conversations
                   (channel_id, peer_id, peer_name, paused, paused_at, updated_at, snoozed_until)
                   VALUES(?,?,?,?,?,?,NULL)
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
