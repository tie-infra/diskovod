from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .persistence import SQLITE_BUSY_TIMEOUT_MS, initialize_target_schema


@dataclass(frozen=True, slots=True)
class QueuedDiscordEvent:
    id: str
    channel_id: str
    sequence: int
    kind: str
    payload: dict[str, Any]
    observed_at: float


class DiscordEventQueue:
    """Ordered, crash-safe ingress queue and chat-to-checkpoint mapping."""

    def __init__(self, path: Path):
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._connection:
            initialize_target_schema(self._connection)
            self._connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def thread_id(self, account_id: str, channel_id: str) -> str:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT thread_id FROM chat_threads WHERE channel_id=?", (channel_id,)
            ).fetchone()
            if row:
                return str(row["thread_id"])
            thread_id = f"discord:{account_id}:{channel_id}:g1"
            self._connection.execute(
                """
                INSERT INTO chat_threads(
                  channel_id, account_id, generation, thread_id, updated_at
                ) VALUES(?, ?, 1, ?, ?)
                """,
                (channel_id, account_id, thread_id, time.time()),
            )
            return thread_id

    def roll_generation(self, account_id: str, channel_id: str) -> str:
        with self._lock, self._connection:
            self.thread_id(account_id, channel_id)
            row = self._connection.execute(
                "SELECT generation FROM chat_threads WHERE channel_id=?", (channel_id,)
            ).fetchone()
            generation = int(row["generation"]) + 1
            thread_id = f"discord:{account_id}:{channel_id}:g{generation}"
            self._connection.execute(
                """
                UPDATE chat_threads
                SET generation=?, thread_id=?, queue_cursor=0, updated_at=?
                WHERE channel_id=?
                """,
                (generation, thread_id, time.time(), channel_id),
            )
            return thread_id

    def set_live_steering(self, account_id: str, channel_id: str, enabled: bool) -> None:
        with self._lock, self._connection:
            self.thread_id(account_id, channel_id)
            self._connection.execute(
                "UPDATE chat_threads SET live_steering=?, updated_at=? WHERE channel_id=?",
                (int(enabled), time.time(), channel_id),
            )

    def live_steering(self, channel_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT live_steering FROM chat_threads WHERE channel_id=?", (channel_id,)
            ).fetchone()
        return bool(row and row["live_steering"])

    def ingest(
        self,
        event_id: str,
        channel_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        observed_at: float | None = None,
    ) -> bool:
        with self._lock, self._connection:
            if self._connection.execute("SELECT 1 FROM discord_events WHERE id=?", (event_id,)).fetchone():
                return False
            sequence = int(
                self._connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM discord_events WHERE channel_id=?",
                    (channel_id,),
                ).fetchone()[0]
            )
            self._connection.execute(
                """
                INSERT INTO discord_events(id, channel_id, sequence, kind, payload, observed_at)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    channel_id,
                    sequence,
                    kind,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    observed_at or time.time(),
                ),
            )
            self._connection.execute(
                "INSERT INTO chat_event_queue(event_id, channel_id) VALUES(?, ?)",
                (event_id, channel_id),
            )
            return True

    def claim_ready(
        self,
        channel_id: str,
        logical_request_id: str,
        *,
        limit: int = 20,
        injection_batch: int = 0,
    ) -> list[QueuedDiscordEvent]:
        with self._lock, self._connection:
            rows = self._connection.execute(
                """
                SELECT e.* FROM discord_events e
                JOIN chat_event_queue q ON q.event_id=e.id
                WHERE q.channel_id=? AND q.disposition='pending'
                ORDER BY e.sequence, e.id LIMIT ?
                """,
                (channel_id, limit),
            ).fetchall()
            if rows:
                placeholders = ",".join("?" for _ in rows)
                self._connection.execute(
                    f"""
                    UPDATE chat_event_queue
                    SET disposition='claimed', logical_request_id=?, injection_batch=?, claimed_at=?
                    WHERE event_id IN ({placeholders}) AND disposition='pending'
                    """,
                    (logical_request_id, injection_batch, time.time(), *(row["id"] for row in rows)),
                )
            return [self._event(row) for row in rows]

    def claimed(self, channel_id: str, logical_request_id: str) -> list[QueuedDiscordEvent]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT e.* FROM discord_events e
                JOIN chat_event_queue q ON q.event_id=e.id
                WHERE q.channel_id=? AND q.logical_request_id=? AND q.disposition='claimed'
                ORDER BY e.sequence, e.id
                """,
                (channel_id, logical_request_id),
            ).fetchall()
        return [self._event(row) for row in rows]

    def complete(self, channel_id: str, logical_request_id: str) -> int:
        with self._lock, self._connection:
            return self._connection.execute(
                """
                UPDATE chat_event_queue SET disposition='completed', completed_at=?
                WHERE channel_id=? AND logical_request_id=? AND disposition='claimed'
                """,
                (time.time(), channel_id, logical_request_id),
            ).rowcount

    @staticmethod
    def _event(row: sqlite3.Row) -> QueuedDiscordEvent:
        return QueuedDiscordEvent(
            id=row["id"],
            channel_id=row["channel_id"],
            sequence=row["sequence"],
            kind=row["kind"],
            payload=json.loads(row["payload"]),
            observed_at=row["observed_at"],
        )
