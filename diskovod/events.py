from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import aiosqlite

from .persistence import AsyncSQLite


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

    def __init__(self, database: AsyncSQLite):
        self.database = database

    async def thread_id(self, account_id: str, channel_id: str) -> str:
        async with self.database.transaction() as connection:
            return await self._thread_id(connection, account_id, channel_id)

    @staticmethod
    async def _thread_id(connection: aiosqlite.Connection, account_id: str, channel_id: str) -> str:
        row = await (
            await connection.execute("SELECT thread_id FROM chat_threads WHERE channel_id=?", (channel_id,))
        ).fetchone()
        if row:
            return str(row["thread_id"])
        thread_id = f"discord:{account_id}:{channel_id}:g1"
        await connection.execute(
            """
                INSERT INTO chat_threads(
                  channel_id, account_id, generation, thread_id, updated_at
                ) VALUES(?, ?, 1, ?, ?)
                """,
            (channel_id, account_id, thread_id, time.time()),
        )
        await connection.execute(
            """
            INSERT INTO chat_thread_generations(
              thread_id, channel_id, account_id, generation,
              configuration_version_id, created_at
            ) VALUES(?, ?, ?, 1,
              (SELECT id FROM agent_configuration_versions WHERE active=1), ?)
            """,
            (thread_id, channel_id, account_id, time.time()),
        )
        return thread_id

    async def roll_generation(
        self,
        account_id: str,
        channel_id: str,
        *,
        reason: str | None = None,
        summary: str | None = None,
    ) -> str:
        async with self.database.transaction() as connection:
            await self._thread_id(connection, account_id, channel_id)
            row = await (
                await connection.execute(
                    "SELECT generation FROM chat_threads WHERE channel_id=?", (channel_id,)
                )
            ).fetchone()
            generation = int(row["generation"]) + 1
            thread_id = f"discord:{account_id}:{channel_id}:g{generation}"
            now = time.time()
            await connection.execute(
                """
                UPDATE chat_thread_generations
                SET closed_at=?, close_reason=?, summary=?
                WHERE channel_id=? AND generation=? AND closed_at IS NULL
                """,
                (now, reason, summary, channel_id, generation - 1),
            )
            await connection.execute(
                """
                UPDATE chat_threads
                SET generation=?, thread_id=?, queue_cursor=0, updated_at=?
                WHERE channel_id=?
                """,
                (generation, thread_id, now, channel_id),
            )
            await connection.execute(
                """
                INSERT INTO chat_thread_generations(
                  thread_id, channel_id, account_id, generation,
                  configuration_version_id, created_at
                ) VALUES(?, ?, ?, ?,
                  (SELECT id FROM agent_configuration_versions WHERE active=1), ?)
                """,
                (thread_id, channel_id, account_id, generation, now),
            )
            return thread_id

    async def set_live_steering(self, account_id: str, channel_id: str, enabled: bool) -> None:
        async with self.database.transaction() as connection:
            await self._thread_id(connection, account_id, channel_id)
            await connection.execute(
                "UPDATE chat_threads SET live_steering=?, updated_at=? WHERE channel_id=?",
                (int(enabled), time.time(), channel_id),
            )

    async def live_steering(self, channel_id: str) -> bool:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT live_steering FROM chat_threads WHERE channel_id=?", (channel_id,)
                )
            ).fetchone()
        return bool(row and row["live_steering"])

    async def ingest(
        self,
        event_id: str,
        channel_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        observed_at: float | None = None,
        enqueue: bool = True,
    ) -> bool:
        async with self.database.transaction() as connection:
            if await (
                await connection.execute("SELECT 1 FROM discord_events WHERE id=?", (event_id,))
            ).fetchone():
                return False
            sequence = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM discord_events WHERE channel_id=?",
                            (channel_id,),
                        )
                    ).fetchone()
                )[0]
            )
            await connection.execute(
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
            if enqueue:
                await connection.execute(
                    "INSERT INTO chat_event_queue(event_id, channel_id) VALUES(?, ?)",
                    (event_id, channel_id),
                )
            return True

    async def claim_ready(
        self,
        channel_id: str,
        logical_request_id: str,
        *,
        limit: int = 20,
        injection_batch: int = 0,
    ) -> list[QueuedDiscordEvent]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                SELECT e.* FROM discord_events e
                JOIN chat_event_queue q ON q.event_id=e.id
                WHERE q.channel_id=? AND q.disposition='pending'
                ORDER BY e.sequence, e.id LIMIT ?
                """,
                    (channel_id, limit),
                )
            ).fetchall()
            if rows:
                placeholders = ",".join("?" for _ in rows)
                await connection.execute(
                    f"""
                    UPDATE chat_event_queue
                    SET disposition='claimed', logical_request_id=?, injection_batch=?, claimed_at=?
                    WHERE event_id IN ({placeholders}) AND disposition='pending'
                    """,
                    (logical_request_id, injection_batch, time.time(), *(row["id"] for row in rows)),
                )
            return [self._event(row) for row in rows]

    async def claimed(self, channel_id: str, logical_request_id: str) -> list[QueuedDiscordEvent]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                SELECT e.* FROM discord_events e
                JOIN chat_event_queue q ON q.event_id=e.id
                WHERE q.channel_id=? AND q.logical_request_id=? AND q.disposition='claimed'
                ORDER BY e.sequence, e.id
                """,
                    (channel_id, logical_request_id),
                )
            ).fetchall()
        return [self._event(row) for row in rows]

    async def complete(self, channel_id: str, logical_request_id: str) -> int:
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE chat_event_queue SET disposition='completed', completed_at=?
                WHERE channel_id=? AND logical_request_id=? AND disposition='claimed'
                """,
                (time.time(), channel_id, logical_request_id),
            )
            return cursor.rowcount

    async def release(self, channel_id: str, logical_request_id: str) -> int:
        """Return an unfinished invocation's claims to the ordered ready queue."""
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE chat_event_queue
                SET disposition='pending', logical_request_id=NULL, injection_batch=NULL, claimed_at=NULL
                WHERE channel_id=? AND logical_request_id=? AND disposition='claimed'
                """,
                (channel_id, logical_request_id),
            )
            return cursor.rowcount

    @staticmethod
    def _event(row: aiosqlite.Row) -> QueuedDiscordEvent:
        return QueuedDiscordEvent(
            id=row["id"],
            channel_id=row["channel_id"],
            sequence=row["sequence"],
            kind=row["kind"],
            payload=json.loads(row["payload"]),
            observed_at=row["observed_at"],
        )
