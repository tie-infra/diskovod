from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import aiosqlite

from .persistence import AsyncSQLite


@dataclass(frozen=True, slots=True)
class ConversationEvent:
    id: str
    channel_id: str
    sequence: int
    kind: str
    payload: dict[str, Any]
    observed_at: float
    available_at: float


class ConversationMailbox:
    """One durable, ordered source of conversation input and scheduled wake-ups."""

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
        now = time.time()
        await connection.execute(
            """
            INSERT INTO chat_threads(channel_id, account_id, generation, thread_id, updated_at)
            VALUES(?, ?, 1, ?, ?)
            """,
            (channel_id, account_id, thread_id, now),
        )
        await connection.execute(
            """
            INSERT INTO chat_thread_generations(
              thread_id, channel_id, account_id, generation,
              configuration_version_id, created_at
            ) VALUES(?, ?, ?, 1,
              (SELECT id FROM agent_configuration_versions WHERE active=1), ?)
            """,
            (thread_id, channel_id, account_id, now),
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
        available_at: float | None = None,
        enqueue: bool = True,
    ) -> bool:
        observed = observed_at or time.time()
        available = available_at if available_at is not None else observed
        async with self.database.transaction() as connection:
            if await (
                await connection.execute("SELECT 1 FROM conversation_mailbox WHERE id=?", (event_id,))
            ).fetchone():
                return False
            sequence = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COALESCE(MAX(sequence), 0) + 1 "
                            "FROM conversation_mailbox WHERE channel_id=?",
                            (channel_id,),
                        )
                    ).fetchone()
                )[0]
            )
            await connection.execute(
                """
                INSERT INTO conversation_mailbox(
                  id, channel_id, sequence, kind, available_at, observed_at,
                  payload, state, completed_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    channel_id,
                    sequence,
                    kind,
                    available,
                    observed,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    "pending" if enqueue else "completed",
                    None if enqueue else observed,
                ),
            )
            return True

    async def claim_ready(
        self,
        channel_id: str,
        run_id: str,
        *,
        limit: int = 20,
        injection_batch: int = 0,
    ) -> list[ConversationEvent]:
        now = time.time()
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT * FROM conversation_mailbox
                    WHERE channel_id=? AND state='pending' AND available_at<=?
                    ORDER BY sequence, id LIMIT ?
                    """,
                    (channel_id, now, limit),
                )
            ).fetchall()
            if rows:
                placeholders = ",".join("?" for _ in rows)
                await connection.execute(
                    f"""
                    UPDATE conversation_mailbox
                    SET state='claimed', run_id=?, injection_batch=?, claimed_at=?
                    WHERE id IN ({placeholders}) AND state='pending'
                    """,
                    (run_id, injection_batch, now, *(row["id"] for row in rows)),
                )
            return [self._event(row) for row in rows]

    async def claimed(self, channel_id: str, run_id: str) -> list[ConversationEvent]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT * FROM conversation_mailbox
                    WHERE channel_id=? AND run_id=? AND state='claimed'
                    ORDER BY sequence, id
                    """,
                    (channel_id, run_id),
                )
            ).fetchall()
        return [self._event(row) for row in rows]

    async def complete(self, channel_id: str, run_id: str) -> int:
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE conversation_mailbox SET state='completed', completed_at=?
                WHERE channel_id=? AND run_id=? AND state='claimed'
                """,
                (time.time(), channel_id, run_id),
            )
            return cursor.rowcount

    async def fail(self, channel_id: str, run_id: str, failure: str) -> int:
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE conversation_mailbox SET state='failed', completed_at=?, failure=?
                WHERE channel_id=? AND run_id=? AND state='claimed'
                """,
                (time.time(), failure[:4000], channel_id, run_id),
            )
            return cursor.rowcount

    async def release(self, channel_id: str, run_id: str) -> int:
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE conversation_mailbox
                SET state='pending', run_id=NULL, injection_batch=NULL, claimed_at=NULL
                WHERE channel_id=? AND run_id=? AND state='claimed'
                """,
                (channel_id, run_id),
            )
            return cursor.rowcount

    async def pending_channels(self) -> list[str]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    "SELECT DISTINCT channel_id FROM conversation_mailbox "
                    "WHERE state='pending' AND available_at<=?",
                    (time.time(),),
                )
            ).fetchall()
        return [str(row["channel_id"]) for row in rows]

    async def has_pending(self, channel_id: str, *, include_future: bool = False) -> bool:
        sql = "SELECT 1 FROM conversation_mailbox WHERE channel_id=? AND state='pending'"
        parameters: tuple[Any, ...] = (channel_id,)
        if not include_future:
            sql += " AND available_at<=?"
            parameters += (time.time(),)
        sql += " LIMIT 1"
        async with self.database.transaction() as connection:
            row = await (await connection.execute(sql, parameters)).fetchone()
        return row is not None

    @staticmethod
    def _event(row: aiosqlite.Row) -> ConversationEvent:
        return ConversationEvent(
            id=str(row["id"]),
            channel_id=str(row["channel_id"]),
            sequence=int(row["sequence"]),
            kind=str(row["kind"]),
            payload=json.loads(row["payload"]),
            observed_at=float(row["observed_at"]),
            available_at=float(row["available_at"]),
        )
