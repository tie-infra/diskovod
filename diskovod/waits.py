from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from .agent_types import AgentRuntimeContext
from .persistence import AsyncSQLite


@dataclass(frozen=True, slots=True)
class ConversationWait:
    id: str
    thread_id: str
    channel_id: str
    run_id: str
    trace_id: str
    tool_call_id: str
    wake_event_id: str
    state: str
    resume_at: float
    payload: dict[str, Any]
    resume_reason: str = "deadline"


class ConversationWaits:
    """Durable follow-up continuations and their mailbox wake events."""

    def __init__(self, database: AsyncSQLite):
        self.database = database

    async def arm(
        self,
        context: AgentRuntimeContext,
        *,
        run_id: str,
        tool_call_id: str,
        duration: float,
        payload: dict[str, Any],
    ) -> ConversationWait:
        wait_id = self.wait_id(context.thread_id, run_id, tool_call_id)
        wake_event_id = f"continuation:{wait_id}"
        now = time.time()
        resume_at = now + duration
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        async with self.database.transaction() as connection:
            existing = await (
                await connection.execute("SELECT * FROM conversation_waits WHERE id=?", (wait_id,))
            ).fetchone()
            if existing is not None:
                return self._record(existing)
            sequence = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM conversation_mailbox "
                            "WHERE channel_id=?",
                            (context.channel_id,),
                        )
                    ).fetchone()
                )[0]
            )
            await connection.execute(
                """
                INSERT INTO conversation_mailbox(
                  id, channel_id, sequence, kind, available_at, observed_at, payload, state
                ) VALUES(?, ?, ?, 'continuation_due', ?, ?, ?, 'pending')
                """,
                (
                    wake_event_id,
                    context.channel_id,
                    sequence,
                    resume_at,
                    now,
                    json.dumps(
                        {"wait_id": wait_id, "reason": "deadline"},
                        separators=(",", ":"),
                    ),
                ),
            )
            await connection.execute(
                """
                INSERT INTO conversation_waits(
                  id, thread_id, channel_id, run_id, trace_id, tool_call_id,
                  wake_event_id, state, resume_at, created_at, updated_at, payload
                ) VALUES(?, ?, ?, ?, ?, ?, ?, 'arming', ?, ?, ?, ?)
                """,
                (
                    wait_id,
                    context.thread_id,
                    context.channel_id,
                    run_id,
                    context.trace_id,
                    tool_call_id,
                    wake_event_id,
                    resume_at,
                    now,
                    now,
                    encoded,
                ),
            )
            row = await (
                await connection.execute("SELECT * FROM conversation_waits WHERE id=?", (wait_id,))
            ).fetchone()
        return self._record(row)

    async def schedule(self, wait_id: str) -> bool:
        return await self._transition(wait_id, "arming", "scheduled")

    async def resolve(self, wait_id: str) -> bool:
        return await self._transition(wait_id, "resuming", "completed")

    async def fail(self, wait_id: str, failure: str) -> bool:
        async with self.database.transaction() as connection:
            changed = (
                await connection.execute(
                    """
                    UPDATE conversation_waits SET state='failed', failure=?, updated_at=?
                    WHERE id=? AND state IN ('arming','scheduled','resuming','completed')
                    """,
                    (failure[:4000], time.time(), wait_id),
                )
            ).rowcount
        return changed == 1

    async def cancel(self, wait_id: str, reason: str) -> bool:
        now = time.time()
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT wake_event_id FROM conversation_waits WHERE id=?", (wait_id,)
                )
            ).fetchone()
            if row is None:
                return False
            changed = (
                await connection.execute(
                    """
                    UPDATE conversation_waits SET state='cancelled', failure=?, updated_at=?
                    WHERE id=? AND (
                      state IN ('arming','scheduled','resuming') OR (
                        state='completed'
                        AND EXISTS(
                          SELECT 1 FROM conversation_mailbox AS m
                          WHERE m.id=conversation_waits.wake_event_id
                            AND m.state IN ('pending','claimed')
                        )
                        AND EXISTS(
                          SELECT 1 FROM agent_runs AS r
                          WHERE r.id=conversation_waits.run_id AND r.status='interrupted'
                        )
                      )
                    )
                    """,
                    (reason[:4000], now, wait_id),
                )
            ).rowcount
            if changed:
                await connection.execute(
                    """
                    UPDATE conversation_mailbox SET state='cancelled', completed_at=?
                    WHERE id=? AND state IN ('pending','claimed')
                    """,
                    (now, row["wake_event_id"]),
                )
        return changed == 1

    async def wake_for_input(self, channel_id: str) -> bool:
        now = time.time()
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT id, wake_event_id FROM conversation_waits
                    WHERE channel_id=? AND state='scheduled' LIMIT 1
                    """,
                    (channel_id,),
                )
            ).fetchone()
            if row is None:
                return False
            await connection.execute(
                """
                UPDATE conversation_mailbox SET available_at=?, payload=?
                WHERE id=? AND state='pending'
                """,
                (
                    now,
                    json.dumps(
                        {"wait_id": str(row["id"]), "reason": "new_input"},
                        separators=(",", ":"),
                    ),
                    row["wake_event_id"],
                ),
            )
            await connection.execute(
                "UPDATE conversation_waits SET resume_at=?, updated_at=? WHERE id=?",
                (now, now, row["id"]),
            )
        return True

    async def claim_ready(self, channel_id: str) -> ConversationWait | None:
        now = time.time()
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT w.*, m.payload AS wake_payload FROM conversation_waits AS w
                    JOIN conversation_mailbox AS m ON m.id=w.wake_event_id
                    WHERE w.channel_id=? AND w.state='scheduled'
                      AND m.state='pending' AND m.available_at<=?
                    LIMIT 1
                    """,
                    (channel_id, now),
                )
            ).fetchone()
            if row is None:
                return None
            changed = (
                await connection.execute(
                    "UPDATE conversation_waits SET state='resuming', updated_at=? "
                    "WHERE id=? AND state='scheduled'",
                    (now, row["id"]),
                )
            ).rowcount
            if changed != 1:
                return None
        return self._record(row, state="resuming")

    async def active(self, channel_id: str) -> ConversationWait | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT * FROM conversation_waits WHERE channel_id=?
                    AND state IN ('arming','scheduled','resuming') LIMIT 1
                    """,
                    (channel_id,),
                )
            ).fetchone()
        return self._record(row) if row is not None else None

    async def incomplete_resume(self, channel_id: str) -> ConversationWait | None:
        """Return a resumed interrupt whose runtime finalization did not finish."""
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT w.*, m.payload AS wake_payload
                    FROM conversation_waits AS w
                    JOIN conversation_mailbox AS m ON m.id=w.wake_event_id
                    JOIN agent_runs AS r ON r.id=w.run_id
                    WHERE w.channel_id=? AND w.state='completed'
                      AND m.state IN ('pending','claimed') AND r.status='interrupted'
                    ORDER BY w.updated_at DESC LIMIT 1
                    """,
                    (channel_id,),
                )
            ).fetchone()
        return self._record(row) if row is not None else None

    async def resumable(self) -> list[ConversationWait]:
        """List active and incompletely finalized waits for lifecycle cancellation."""
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT w.*, m.payload AS wake_payload
                    FROM conversation_waits AS w
                    JOIN conversation_mailbox AS m ON m.id=w.wake_event_id
                    LEFT JOIN agent_runs AS r ON r.id=w.run_id
                    WHERE w.state IN ('arming','scheduled','resuming')
                       OR (w.state='completed' AND m.state IN ('pending','claimed')
                           AND r.status='interrupted')
                    ORDER BY w.updated_at
                    """
                )
            ).fetchall()
        return [self._record(row) for row in rows]

    async def due_channels(self) -> list[str]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT channel_id FROM conversation_waits
                    WHERE (state='scheduled' AND resume_at<=?) OR state='resuming'
                    UNION
                    SELECT w.channel_id FROM conversation_waits AS w
                    JOIN conversation_mailbox AS m ON m.id=w.wake_event_id
                    JOIN agent_runs AS r ON r.id=w.run_id
                    WHERE w.state='completed' AND m.state IN ('pending','claimed')
                      AND r.status='interrupted'
                    """,
                    (time.time(),),
                )
            ).fetchall()
        return [str(row["channel_id"]) for row in rows]

    async def arming(self) -> list[ConversationWait]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute("SELECT * FROM conversation_waits WHERE state='arming'")
            ).fetchall()
        return [self._record(row) for row in rows]

    async def _transition(self, wait_id: str, before: str, after: str) -> bool:
        async with self.database.transaction() as connection:
            changed = (
                await connection.execute(
                    "UPDATE conversation_waits SET state=?, updated_at=? WHERE id=? AND state=?",
                    (after, time.time(), wait_id, before),
                )
            ).rowcount
        return changed == 1

    @staticmethod
    def wait_id(thread_id: str, run_id: str, tool_call_id: str) -> str:
        value = "\0".join((thread_id, run_id, tool_call_id))
        return f"wait:{hashlib.sha256(value.encode()).hexdigest()}"

    @staticmethod
    def _record(row, *, state: str | None = None) -> ConversationWait:
        keys = set(row.keys())
        wake_payload = json.loads(row["wake_payload"]) if "wake_payload" in keys else {}
        return ConversationWait(
            id=str(row["id"]),
            thread_id=str(row["thread_id"]),
            channel_id=str(row["channel_id"]),
            run_id=str(row["run_id"]),
            trace_id=str(row["trace_id"]),
            tool_call_id=str(row["tool_call_id"]),
            wake_event_id=str(row["wake_event_id"]),
            state=state or str(row["state"]),
            resume_at=float(row["resume_at"]),
            payload=json.loads(row["payload"]),
            resume_reason=str(wake_payload.get("reason") or "deadline"),
        )
