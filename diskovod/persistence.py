from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import (
    BaseStore,
    GetOp,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)


SQLITE_BUSY_TIMEOUT_MS = 5_000
TARGET_SCHEMA_VERSION = 5


TARGET_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
      version INTEGER PRIMARY KEY,
      applied_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS agent_configuration_versions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at REAL NOT NULL,
      configuration TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 0
    );
    CREATE UNIQUE INDEX IF NOT EXISTS one_active_agent_configuration
      ON agent_configuration_versions(active) WHERE active = 1;
    CREATE TABLE IF NOT EXISTS chat_threads (
      channel_id TEXT PRIMARY KEY,
      account_id TEXT NOT NULL,
      generation INTEGER NOT NULL DEFAULT 1,
      thread_id TEXT NOT NULL UNIQUE,
      live_steering INTEGER NOT NULL DEFAULT 1,
      queue_cursor INTEGER NOT NULL DEFAULT 0,
      updated_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS discord_events (
      id TEXT PRIMARY KEY,
      channel_id TEXT NOT NULL,
      sequence INTEGER NOT NULL,
      kind TEXT NOT NULL,
      payload TEXT NOT NULL,
      observed_at REAL NOT NULL,
      UNIQUE(channel_id, sequence)
    );
    CREATE TABLE IF NOT EXISTS chat_event_queue (
      event_id TEXT PRIMARY KEY REFERENCES discord_events(id) ON DELETE CASCADE,
      channel_id TEXT NOT NULL,
      disposition TEXT NOT NULL DEFAULT 'pending',
      logical_request_id TEXT,
      injection_batch INTEGER,
      claimed_at REAL,
      completed_at REAL
    );
    CREATE INDEX IF NOT EXISTS chat_event_queue_ready
      ON chat_event_queue(channel_id, disposition, event_id);
    CREATE TABLE IF NOT EXISTS side_effect_deliveries (
      run_id TEXT NOT NULL,
      tool_call_id TEXT NOT NULL,
      action TEXT NOT NULL,
      state TEXT NOT NULL,
      request TEXT NOT NULL,
      result TEXT,
      claimed_at REAL NOT NULL,
      completed_at REAL,
      PRIMARY KEY(run_id, tool_call_id)
    );
    CREATE TABLE IF NOT EXISTS agent_runs (
      id TEXT PRIMARY KEY,
      thread_id TEXT NOT NULL,
      channel_id TEXT NOT NULL,
      trace_id TEXT NOT NULL UNIQUE,
      status TEXT NOT NULL,
      started_at REAL NOT NULL,
      completed_at REAL,
      error TEXT
    );
    CREATE TABLE IF NOT EXISTS agent_trace_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
      sequence INTEGER NOT NULL,
      kind TEXT NOT NULL,
      payload TEXT NOT NULL,
      recorded_at REAL NOT NULL,
      UNIQUE(run_id, sequence)
    );
    CREATE TABLE IF NOT EXISTS attachment_objects (
      sha256 TEXT PRIMARY KEY,
      size INTEGER NOT NULL,
      media_type TEXT,
      storage_path TEXT NOT NULL,
      created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS attachment_artifacts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      object_sha256 TEXT NOT NULL REFERENCES attachment_objects(sha256) ON DELETE CASCADE,
      kind TEXT NOT NULL,
      state TEXT NOT NULL,
      content TEXT,
      metadata TEXT NOT NULL DEFAULT '{}',
      created_at REAL NOT NULL,
      updated_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS attachment_chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      artifact_id INTEGER NOT NULL REFERENCES attachment_artifacts(id) ON DELETE CASCADE,
      chunk_index INTEGER NOT NULL,
      content TEXT NOT NULL,
      metadata TEXT NOT NULL DEFAULT '{}',
      UNIQUE(artifact_id, chunk_index)
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS attachment_chunks_fts USING fts5(
      content, content='attachment_chunks', content_rowid='id'
    );
    CREATE TABLE IF NOT EXISTS escalation_interrupts (
      id TEXT PRIMARY KEY,
      thread_id TEXT NOT NULL,
      channel_id TEXT NOT NULL,
      state TEXT NOT NULL,
      payload TEXT NOT NULL,
      created_at REAL NOT NULL,
      updated_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS langgraph_store_items (
      namespace TEXT NOT NULL,
      key TEXT NOT NULL,
      value TEXT NOT NULL,
      index_text TEXT,
      created_at REAL NOT NULL,
      updated_at REAL NOT NULL,
      PRIMARY KEY(namespace, key)
    );
    CREATE INDEX IF NOT EXISTS langgraph_store_namespace
      ON langgraph_store_items(namespace, updated_at DESC);
    CREATE VIRTUAL TABLE IF NOT EXISTS langgraph_store_fts USING fts5(
      namespace UNINDEXED, key UNINDEXED, body, tokenize='unicode61'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS provider_capability_probes (
      id TEXT PRIMARY KEY,
      configuration TEXT NOT NULL,
      capability TEXT NOT NULL,
      status TEXT NOT NULL,
      request_payload TEXT NOT NULL,
      response_payload TEXT,
      conclusion TEXT NOT NULL,
      started_at REAL NOT NULL,
      completed_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS provider_capability_probes_time
      ON provider_capability_probes(completed_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS attachment_references (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      channel_id TEXT NOT NULL,
      message_id TEXT NOT NULL,
      attachment_id TEXT NOT NULL,
      filename TEXT NOT NULL,
      object_sha256 TEXT NOT NULL REFERENCES attachment_objects(sha256) ON DELETE CASCADE,
      metadata TEXT NOT NULL DEFAULT '{}',
      created_at REAL NOT NULL,
      UNIQUE(message_id, attachment_id)
    );
    CREATE INDEX IF NOT EXISTS attachment_references_chat
      ON attachment_references(channel_id, created_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS legacy_import_records (
      kind TEXT NOT NULL,
      source_id TEXT NOT NULL,
      payload TEXT NOT NULL,
      imported_at REAL NOT NULL,
      PRIMARY KEY(kind, source_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS config (
      key TEXT PRIMARY KEY, value TEXT NOT NULL, secret INTEGER NOT NULL DEFAULT 0,
      updated_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS conversations (
      channel_id TEXT PRIMARY KEY, peer_id TEXT NOT NULL, peer_name TEXT NOT NULL,
      paused INTEGER NOT NULL DEFAULT 0, paused_at REAL, updated_at REAL NOT NULL,
      snoozed_until REAL, mode TEXT NOT NULL DEFAULT 'automatic'
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
    """,
)


class AsyncSQLite:
    """One serialized async connection for Diskovod's application repositories."""

    def __init__(self, path: Path):
        self.path = path
        self._connection: aiosqlite.Connection | None = None
        self._connection_lock = asyncio.Lock()
        self._transaction_lock = asyncio.Lock()

    async def start(self) -> None:
        await self._get_connection()

    async def close(self) -> None:
        async with self._transaction_lock:
            async with self._connection_lock:
                connection, self._connection = self._connection, None
            if connection is not None:
                await connection.close()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Serialize a complete application transaction on the shared connection."""
        async with self._transaction_lock:
            connection = await self._get_connection()
            try:
                yield connection
            except BaseException:
                await connection.rollback()
                raise
            else:
                await connection.commit()

    async def _get_connection(self) -> aiosqlite.Connection:
        if self._connection is not None:
            return self._connection
        async with self._connection_lock:
            if self._connection is None:
                connection = await aiosqlite.connect(self.path)
                connection.row_factory = aiosqlite.Row
                await connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
                await connection.execute("PRAGMA foreign_keys=ON")
                self._connection = connection
        return self._connection


def initialize_target_schema(connection: sqlite3.Connection) -> None:
    """Apply target-schema migrations to Diskovod's single relational database."""
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
    )
    applied = {int(row[0]) for row in connection.execute("SELECT version FROM schema_migrations").fetchall()}
    for version, migration in enumerate(TARGET_MIGRATIONS, start=1):
        if version in applied:
            continue
        connection.executescript(migration)
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
            (version, datetime.now(UTC).timestamp()),
        )
    current = connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0]
    if current != TARGET_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported Diskovod schema version {current}")


class CheckpointCipher:
    """AES-GCM checkpoint cipher derived from Diskovod's existing secret."""

    name = "diskovod-aesgcm-v1"

    def __init__(self, secret: str):
        if len(secret) < 32:
            raise ValueError("The secret key file must contain at least 32 characters")
        self._cipher = AESGCM(hashlib.sha256(b"diskovod-checkpoints\0" + secret.encode()).digest())

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        nonce = os.urandom(12)
        ciphertext = self._cipher.encrypt(nonce, plaintext, self.name.encode())
        return self.name, nonce + ciphertext

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        if ciphername != self.name:
            raise ValueError(f"Unsupported checkpoint cipher {ciphername!r}")
        if len(ciphertext) < 13:
            raise ValueError("Invalid encrypted checkpoint")
        return self._cipher.decrypt(ciphertext[:12], ciphertext[12:], self.name.encode())


def checkpoint_serializer(secret: str) -> EncryptedSerializer:
    serde = JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=None)
    return EncryptedSerializer(CheckpointCipher(secret), serde=serde)


@asynccontextmanager
async def open_checkpointer(path: Path, secret: str) -> AsyncIterator[AsyncSqliteSaver]:
    """Open an encrypted LangGraph checkpointer on the shared Diskovod database."""
    async with aiosqlite.connect(path) as connection:
        await connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        await connection.execute("PRAGMA foreign_keys=ON")
        saver = AsyncSqliteSaver(connection, serde=checkpoint_serializer(secret))
        await saver.setup()
        yield saver


class SQLiteLangGraphStore(BaseStore):
    """Persistent local LangGraph Store with JSON filtering and lexical FTS search."""

    supports_ttl = False

    def __init__(self, path: Path, database: AsyncSQLite | None = None):
        self.database = database or AsyncSQLite(path)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._connection:
            initialize_target_schema(self._connection)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        operations = list(ops)
        results: list[Result] = []
        with self._lock, self._connection:
            for operation in operations:
                if isinstance(operation, GetOp):
                    results.append(self._get(operation))
                elif isinstance(operation, PutOp):
                    results.append(self._put(operation))
                elif isinstance(operation, SearchOp):
                    results.append(self._search(operation))
                elif isinstance(operation, ListNamespacesOp):
                    results.append(self._list_namespaces(operation))
                else:
                    raise TypeError(f"Unsupported Store operation: {type(operation).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        operations = list(ops)
        results: list[Result] = []
        async with self.database.transaction() as connection:
            for operation in operations:
                if isinstance(operation, GetOp):
                    results.append(await self._aget(connection, operation))
                elif isinstance(operation, PutOp):
                    results.append(await self._aput(connection, operation))
                elif isinstance(operation, SearchOp):
                    results.append(await self._asearch(connection, operation))
                elif isinstance(operation, ListNamespacesOp):
                    results.append(await self._alist_namespaces(connection, operation))
                else:
                    raise TypeError(f"Unsupported Store operation: {type(operation).__name__}")
        return results

    async def _aget(self, connection: aiosqlite.Connection, operation: GetOp) -> Item | None:
        namespace = self._namespace(operation.namespace)
        row = await (
            await connection.execute(
                "SELECT * FROM langgraph_store_items WHERE namespace=? AND key=?",
                (namespace, operation.key),
            )
        ).fetchone()
        return self._item(row) if row else None

    async def _aput(self, connection: aiosqlite.Connection, operation: PutOp) -> None:
        namespace = self._namespace(operation.namespace)
        if not operation.key:
            raise ValueError("Store keys cannot be empty")
        await connection.execute(
            "DELETE FROM langgraph_store_fts WHERE namespace=? AND key=?",
            (namespace, operation.key),
        )
        if operation.value is None:
            await connection.execute(
                "DELETE FROM langgraph_store_items WHERE namespace=? AND key=?",
                (namespace, operation.key),
            )
            return None
        value = self._json(operation.value)
        index_text = self._index_text(operation.value, operation.index)
        now = datetime.now(UTC).timestamp()
        await connection.execute(
            """
            INSERT INTO langgraph_store_items(namespace, key, value, index_text, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET
              value=excluded.value, index_text=excluded.index_text, updated_at=excluded.updated_at
            """,
            (namespace, operation.key, value, index_text, now, now),
        )
        if index_text:
            await connection.execute(
                "INSERT INTO langgraph_store_fts(namespace, key, body) VALUES(?, ?, ?)",
                (namespace, operation.key, index_text),
            )
        return None

    async def _asearch(self, connection: aiosqlite.Connection, operation: SearchOp) -> list[SearchItem]:
        rows = await (
            await connection.execute(
                "SELECT * FROM langgraph_store_items ORDER BY updated_at DESC, namespace, key"
            )
        ).fetchall()
        prefix = operation.namespace_prefix
        candidates = [
            row for row in rows if self._decode_namespace(row["namespace"])[: len(prefix)] == prefix
        ]
        if operation.filter:
            candidates = [
                row for row in candidates if self._matches_filter(json.loads(row["value"]), operation.filter)
            ]
        scores: dict[tuple[str, str], float] = {}
        if operation.query:
            query = self._fts_query(operation.query)
            if not query:
                return []
            matches = await (
                await connection.execute(
                    "SELECT namespace, key, bm25(langgraph_store_fts) AS rank "
                    "FROM langgraph_store_fts WHERE body MATCH ?",
                    (query,),
                )
            ).fetchall()
            scores = {(row["namespace"], row["key"]): -float(row["rank"]) for row in matches}
            candidates = [row for row in candidates if (row["namespace"], row["key"]) in scores]
            candidates.sort(
                key=lambda row: (scores[(row["namespace"], row["key"])], row["updated_at"]),
                reverse=True,
            )
        selected = candidates[operation.offset : operation.offset + operation.limit]
        return [
            SearchItem(
                namespace=self._decode_namespace(row["namespace"]),
                key=row["key"],
                value=json.loads(row["value"]),
                created_at=self._datetime(row["created_at"]),
                updated_at=self._datetime(row["updated_at"]),
                score=scores.get((row["namespace"], row["key"])),
            )
            for row in selected
        ]

    async def _alist_namespaces(
        self, connection: aiosqlite.Connection, operation: ListNamespacesOp
    ) -> list[tuple[str, ...]]:
        rows = await (
            await connection.execute("SELECT DISTINCT namespace FROM langgraph_store_items")
        ).fetchall()
        namespaces = [self._decode_namespace(row["namespace"]) for row in rows]
        if operation.match_conditions:
            namespaces = [
                namespace
                for namespace in namespaces
                if all(
                    self._matches_namespace(namespace, condition.match_type, condition.path)
                    for condition in operation.match_conditions
                )
            ]
        if operation.max_depth is not None:
            namespaces = list({namespace[: operation.max_depth] for namespace in namespaces})
        namespaces.sort()
        return namespaces[operation.offset : operation.offset + operation.limit]

    def _get(self, operation: GetOp) -> Item | None:
        namespace = self._namespace(operation.namespace)
        row = self._connection.execute(
            "SELECT * FROM langgraph_store_items WHERE namespace=? AND key=?",
            (namespace, operation.key),
        ).fetchone()
        return self._item(row) if row else None

    def _put(self, operation: PutOp) -> None:
        namespace = self._namespace(operation.namespace)
        if not operation.key:
            raise ValueError("Store keys cannot be empty")
        self._connection.execute(
            "DELETE FROM langgraph_store_fts WHERE namespace=? AND key=?",
            (namespace, operation.key),
        )
        if operation.value is None:
            self._connection.execute(
                "DELETE FROM langgraph_store_items WHERE namespace=? AND key=?",
                (namespace, operation.key),
            )
            return None
        value = self._json(operation.value)
        index_text = self._index_text(operation.value, operation.index)
        now = datetime.now(UTC).timestamp()
        self._connection.execute(
            """
            INSERT INTO langgraph_store_items(namespace, key, value, index_text, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET
              value=excluded.value, index_text=excluded.index_text, updated_at=excluded.updated_at
            """,
            (namespace, operation.key, value, index_text, now, now),
        )
        if index_text:
            self._connection.execute(
                "INSERT INTO langgraph_store_fts(namespace, key, body) VALUES(?, ?, ?)",
                (namespace, operation.key, index_text),
            )
        return None

    def _search(self, operation: SearchOp) -> list[SearchItem]:
        rows = self._connection.execute(
            "SELECT * FROM langgraph_store_items ORDER BY updated_at DESC, namespace, key"
        ).fetchall()
        prefix = operation.namespace_prefix
        candidates = [
            row for row in rows if self._decode_namespace(row["namespace"])[: len(prefix)] == prefix
        ]
        if operation.filter:
            candidates = [
                row for row in candidates if self._matches_filter(json.loads(row["value"]), operation.filter)
            ]
        scores: dict[tuple[str, str], float] = {}
        if operation.query:
            query = self._fts_query(operation.query)
            if not query:
                return []
            matches = self._connection.execute(
                "SELECT namespace, key, bm25(langgraph_store_fts) AS rank FROM langgraph_store_fts WHERE body MATCH ?",
                (query,),
            ).fetchall()
            scores = {(row["namespace"], row["key"]): -float(row["rank"]) for row in matches}
            candidates = [row for row in candidates if (row["namespace"], row["key"]) in scores]
            candidates.sort(
                key=lambda row: (scores[(row["namespace"], row["key"])], row["updated_at"]),
                reverse=True,
            )
        selected = candidates[operation.offset : operation.offset + operation.limit]
        return [
            SearchItem(
                namespace=self._decode_namespace(row["namespace"]),
                key=row["key"],
                value=json.loads(row["value"]),
                created_at=self._datetime(row["created_at"]),
                updated_at=self._datetime(row["updated_at"]),
                score=scores.get((row["namespace"], row["key"])),
            )
            for row in selected
        ]

    def _list_namespaces(self, operation: ListNamespacesOp) -> list[tuple[str, ...]]:
        rows = self._connection.execute("SELECT DISTINCT namespace FROM langgraph_store_items").fetchall()
        namespaces = [self._decode_namespace(row["namespace"]) for row in rows]
        if operation.match_conditions:
            namespaces = [
                namespace
                for namespace in namespaces
                if all(
                    self._matches_namespace(namespace, condition.match_type, condition.path)
                    for condition in operation.match_conditions
                )
            ]
        if operation.max_depth is not None:
            namespaces = list({namespace[: operation.max_depth] for namespace in namespaces})
        namespaces.sort()
        return namespaces[operation.offset : operation.offset + operation.limit]

    @classmethod
    def _item(cls, row: sqlite3.Row) -> Item:
        return Item(
            namespace=cls._decode_namespace(row["namespace"]),
            key=row["key"],
            value=json.loads(row["value"]),
            created_at=cls._datetime(row["created_at"]),
            updated_at=cls._datetime(row["updated_at"]),
        )

    @staticmethod
    def _namespace(namespace: tuple[str, ...]) -> str:
        if not namespace:
            raise InvalidNamespaceError("Namespace cannot be empty")
        if namespace[0] == "langgraph":
            raise InvalidNamespaceError('Root namespace label cannot be "langgraph"')
        if any(not isinstance(label, str) or not label or "." in label for label in namespace):
            raise InvalidNamespaceError("Namespace labels must be non-empty strings without periods")
        return json.dumps(namespace, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode_namespace(value: str) -> tuple[str, ...]:
        return tuple(json.loads(value))

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def _index_text(cls, value: dict[str, Any], index: bool | list[str] | None) -> str | None:
        if index is False:
            return None
        selected: Any = value
        if isinstance(index, list):
            selected = [cls._path_value(value, path) for path in index]
        parts: list[str] = []

        def visit(item: Any) -> None:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)
            elif item is not None:
                parts.append(str(item))

        visit(selected)
        return "\n".join(parts) or None

    @staticmethod
    def _path_value(value: dict[str, Any], path: str) -> Any:
        current: Any = value
        normalized = path[2:] if path.startswith("$.") else path
        for component in normalized.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(component)
        return current

    @classmethod
    def _matches_filter(cls, value: Any, expected: Any) -> bool:
        if isinstance(expected, dict):
            if any(str(key).startswith("$") for key in expected):
                return all(
                    cls._matches_operator(value, operator, operand) for operator, operand in expected.items()
                )
            if not isinstance(value, dict):
                return False
            return all(
                key in value and cls._matches_filter(value[key], child) for key, child in expected.items()
            )
        if isinstance(expected, (list, tuple)):
            return (
                isinstance(value, (list, tuple))
                and len(value) == len(expected)
                and all(cls._matches_filter(item, child) for item, child in zip(value, expected, strict=True))
            )
        return value == expected

    @staticmethod
    def _matches_operator(value: Any, operator: str, operand: Any) -> bool:
        if operator == "$eq":
            return value == operand
        if operator == "$ne":
            return value != operand
        if operator in {"$gt", "$gte", "$lt", "$lte"}:
            left = float(value)
            right = float(operand)
            return {
                "$gt": left > right,
                "$gte": left >= right,
                "$lt": left < right,
                "$lte": left <= right,
            }[operator]
        raise ValueError(f"Unsupported filter operator: {operator}")

    @staticmethod
    def _matches_namespace(namespace: tuple[str, ...], match_type: str, path: tuple[str, ...]) -> bool:
        if len(namespace) < len(path):
            return False
        candidate = namespace[: len(path)] if match_type == "prefix" else namespace[-len(path) :]
        return all(pattern == "*" or pattern == value for value, pattern in zip(candidate, path, strict=True))

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = [token.replace('"', '""') for token in query.split() if token]
        return " AND ".join(f'"{token}"' for token in tokens)

    @staticmethod
    def _datetime(timestamp: float) -> datetime:
        return datetime.fromtimestamp(timestamp, UTC)
