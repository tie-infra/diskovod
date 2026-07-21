from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import aiosqlite

from .interaction import InteractionPolicy, TriggerDecision
from .persistence import AsyncSQLite


@dataclass(frozen=True, slots=True)
class ConversationEvent:
    id: str
    channel_id: str
    sequence: int
    kind: str
    payload: dict[str, Any]
    observed_at: float


@dataclass(frozen=True, slots=True)
class AgentWork:
    id: str
    channel_id: str
    kind: str
    source_event_id: str | None
    trigger_kind: str
    trigger_participant: str | None
    policy_version: int
    policy_snapshot: dict[str, Any]
    available_at: float
    captured_through_sequence: int | None
    decision: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CapturedBatch:
    work: AgentWork
    events: tuple[ConversationEvent, ...]


class ConversationJournal:
    """Canonical chat events and the independent durable work queue."""

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
            INSERT INTO chat_threads(
              channel_id, account_id, generation, thread_id, applied_event_sequence, updated_at
            ) VALUES(?, ?, 1, ?, 0, ?)
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
                "UPDATE chat_threads SET generation=?, thread_id=?, updated_at=? WHERE channel_id=?",
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

    async def admit(
        self,
        event_id: str,
        channel_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        observed_at: float,
        schedule: bool,
        trigger_kind: str,
        trigger_participant: str | None,
        policy: InteractionPolicy,
        policy_version: int,
        decision: TriggerDecision,
        applied: bool = False,
    ) -> bool:
        encoded_payload = _json(payload)
        now = time.time()
        async with self.database.transaction() as connection:
            if await (
                await connection.execute("SELECT 1 FROM conversation_events WHERE id=?", (event_id,))
            ).fetchone():
                return False
            sequence = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM conversation_events "
                            "WHERE channel_id=?",
                            (channel_id,),
                        )
                    ).fetchone()
                )[0]
            )
            await connection.execute(
                """
                INSERT INTO conversation_events(
                  id, channel_id, sequence, kind, payload, observed_at,
                  admission_decision, context_state
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    channel_id,
                    sequence,
                    kind,
                    encoded_payload,
                    observed_at,
                    _json(decision.to_dict()),
                    "applied" if applied else "unapplied",
                ),
            )
            if schedule:
                await connection.execute(
                    """
                    INSERT INTO agent_work(
                      id, channel_id, kind, source_event_id, trigger_kind, trigger_participant,
                      policy_version, policy_snapshot, available_at, state, decision, created_at
                    ) VALUES(?, ?, 'turn', ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        f"work:{event_id}",
                        channel_id,
                        event_id,
                        trigger_kind,
                        trigger_participant,
                        policy_version,
                        policy.encoded(),
                        observed_at,
                        _json(decision.to_dict()),
                        now,
                    ),
                )
            if applied:
                await self._advance_cursor(connection, channel_id)
            return True

    async def schedule_force(
        self,
        channel_id: str,
        *,
        trigger_message_id: str,
        policy: InteractionPolicy,
        policy_version: int,
    ) -> str:
        import uuid

        work_id = f"force:{uuid.uuid4()}"
        now = time.time()
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                INSERT INTO agent_work(
                  id, channel_id, kind, trigger_kind, trigger_participant,
                  policy_version, policy_snapshot, available_at, state, decision, created_at
                ) VALUES(?, ?, 'force', 'force_reply', 'control', ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    work_id,
                    channel_id,
                    policy_version,
                    policy.encoded(),
                    now,
                    _json({"reason": "dashboard_force", "message_id": trigger_message_id}),
                    now,
                ),
            )
        return work_id

    async def message_has_trigger_work(self, channel_id: str, message_id: str) -> bool:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT 1 FROM agent_work AS work
                    JOIN conversation_events AS event ON event.id=work.source_event_id
                    WHERE work.channel_id=? AND json_extract(event.payload, '$.message_id')=?
                      AND work.trigger_kind NOT IN ('delete','legacy_history')
                    LIMIT 1
                    """,
                    (channel_id, message_id),
                )
            ).fetchone()
        return row is not None

    async def claim_ready(self, channel_id: str, run_id: str) -> CapturedBatch | None:
        now = time.time()
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT * FROM agent_work
                    WHERE channel_id=? AND state='pending' AND kind IN ('turn','force')
                      AND available_at<=?
                    ORDER BY available_at, created_at, id LIMIT 1
                    """,
                    (channel_id, now),
                )
            ).fetchone()
            if row is None:
                return None
            cutoff = row["captured_through_sequence"]
            if cutoff is None:
                cutoff_row = await (
                    await connection.execute(
                        "SELECT COALESCE(MAX(sequence), 0) FROM conversation_events WHERE channel_id=?",
                        (channel_id,),
                    )
                ).fetchone()
                cutoff = int(cutoff_row[0])
            changed = (
                await connection.execute(
                    """
                    UPDATE agent_work
                    SET state='claimed', run_id=?, captured_through_sequence=?, claimed_at=?
                    WHERE id=? AND state='pending'
                    """,
                    (run_id, cutoff, now, row["id"]),
                )
            ).rowcount
            if changed != 1:
                return None
            await connection.execute(
                """
                UPDATE conversation_events
                SET admission_decision=json_set(
                  admission_decision, '$.matched', json('false'), '$.reason', 'coalesced',
                  '$.coalesced_into', ?
                )
                WHERE id IN (
                  SELECT source_event_id FROM agent_work
                  WHERE channel_id=? AND state='pending' AND kind='turn'
                    AND source_event_id IN (
                      SELECT id FROM conversation_events WHERE channel_id=? AND sequence<=?
                    )
                )
                """,
                (str(row["id"]), channel_id, channel_id, cutoff),
            )
            await connection.execute(
                """
                UPDATE agent_work SET state='cancelled', completed_at=?,
                  decision=json_set(decision, '$.coalesced_into', ?)
                WHERE channel_id=? AND state='pending' AND kind='turn'
                  AND source_event_id IN (
                    SELECT id FROM conversation_events WHERE channel_id=? AND sequence<=?
                  )
                """,
                (now, str(row["id"]), channel_id, channel_id, cutoff),
            )
            events = await self._claim_events(connection, channel_id, run_id, int(cutoff), 0, now)
            claimed = dict(row)
            claimed["captured_through_sequence"] = cutoff
            return CapturedBatch(self._work(claimed), tuple(self._event(item) for item in events))

    async def claim_injection(
        self,
        channel_id: str,
        run_id: str,
        *,
        injection_batch: int,
        participants: frozenset[str],
    ) -> list[ConversationEvent]:
        now = time.time()
        async with self.database.transaction() as connection:
            if participants:
                placeholders = ",".join("?" for _ in participants)
                eligible = await (
                    await connection.execute(
                        f"""
                        SELECT 1 FROM conversation_events
                        WHERE channel_id=? AND context_state='unapplied'
                          AND (
                            kind='delete'
                            OR json_extract(payload, '$.participant_role') IN ({placeholders})
                          )
                        LIMIT 1
                        """,
                        (channel_id, *sorted(participants)),
                    )
                ).fetchone()
            else:
                eligible = await (
                    await connection.execute(
                        """
                        SELECT 1 FROM conversation_events
                        WHERE channel_id=? AND context_state='unapplied' AND kind='delete'
                        LIMIT 1
                        """,
                        (channel_id,),
                    )
                ).fetchone()
            if eligible is None:
                return []
            cutoff = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COALESCE(MAX(sequence), 0) FROM conversation_events WHERE channel_id=?",
                            (channel_id,),
                        )
                    ).fetchone()
                )[0]
            )
            rows = await self._claim_events(
                connection,
                channel_id,
                run_id,
                cutoff,
                injection_batch,
                now,
            )
        return [self._event(row) for row in rows]

    @staticmethod
    async def _claim_events(
        connection: aiosqlite.Connection,
        channel_id: str,
        run_id: str,
        cutoff: int,
        injection_batch: int,
        now: float,
    ) -> list[aiosqlite.Row]:
        rows = await (
            await connection.execute(
                """
                SELECT * FROM conversation_events
                WHERE channel_id=? AND context_state='unapplied' AND sequence<=?
                ORDER BY sequence, id
                """,
                (channel_id, cutoff),
            )
        ).fetchall()
        if rows:
            placeholders = ",".join("?" for _ in rows)
            await connection.execute(
                f"""
                UPDATE conversation_events
                SET context_state='claimed', run_id=?, injection_batch=?, claimed_at=?
                WHERE id IN ({placeholders}) AND context_state='unapplied'
                """,
                (run_id, injection_batch, now, *(row["id"] for row in rows)),
            )
        return list(rows)

    async def claimed(self, channel_id: str, run_id: str) -> list[ConversationEvent]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT * FROM conversation_events
                    WHERE channel_id=? AND run_id=? AND context_state='claimed'
                    ORDER BY sequence, id
                    """,
                    (channel_id, run_id),
                )
            ).fetchall()
        return [self._event(row) for row in rows]

    async def complete(self, channel_id: str, run_id: str) -> int:
        now = time.time()
        async with self.database.transaction() as connection:
            events = await connection.execute(
                """
                UPDATE conversation_events SET context_state='applied', applied_at=?
                WHERE channel_id=? AND run_id=? AND context_state='claimed'
                """,
                (now, channel_id, run_id),
            )
            await connection.execute(
                """
                UPDATE agent_work SET state='completed', completed_at=?
                WHERE channel_id=? AND run_id=? AND state='claimed'
                """,
                (now, channel_id, run_id),
            )
            await self._advance_cursor(connection, channel_id)
            return events.rowcount

    async def fail(self, channel_id: str, run_id: str, failure: str) -> int:
        now = time.time()
        async with self.database.transaction() as connection:
            events = await connection.execute(
                """
                UPDATE conversation_events
                SET context_state='unapplied', run_id=NULL, injection_batch=NULL,
                    claimed_at=NULL, failure=?
                WHERE channel_id=? AND run_id=? AND context_state='claimed'
                """,
                (failure[:4000], channel_id, run_id),
            )
            await connection.execute(
                """
                UPDATE agent_work SET state='failed', completed_at=?, failure=?
                WHERE channel_id=? AND run_id=? AND state='claimed'
                """,
                (now, failure[:4000], channel_id, run_id),
            )
            return events.rowcount

    async def release(self, channel_id: str, run_id: str) -> int:
        async with self.database.transaction() as connection:
            events = await connection.execute(
                """
                UPDATE conversation_events
                SET context_state='unapplied', run_id=NULL, injection_batch=NULL, claimed_at=NULL
                WHERE channel_id=? AND run_id=? AND context_state='claimed'
                """,
                (channel_id, run_id),
            )
            await connection.execute(
                """
                UPDATE agent_work SET state='pending', run_id=NULL, claimed_at=NULL
                WHERE channel_id=? AND run_id=? AND state='claimed'
                """,
                (channel_id, run_id),
            )
            return events.rowcount

    async def cancel_claimed(self, channel_id: str, run_id: str, reason: str) -> int:
        now = time.time()
        async with self.database.transaction() as connection:
            events = await connection.execute(
                """
                UPDATE conversation_events
                SET context_state='unapplied', run_id=NULL, injection_batch=NULL, claimed_at=NULL
                WHERE channel_id=? AND run_id=? AND context_state='claimed'
                """,
                (channel_id, run_id),
            )
            await connection.execute(
                """
                UPDATE agent_work SET state='cancelled', completed_at=?, failure=?
                WHERE channel_id=? AND run_id=? AND state='claimed'
                """,
                (now, reason[:4000], channel_id, run_id),
            )
            return events.rowcount

    async def cancel_pending_turns(self, channel_id: str, reason: str) -> int:
        """Cancel unclaimed ordinary turns while preserving their context events."""
        async with self.database.transaction() as connection:
            work = await connection.execute(
                """
                UPDATE agent_work SET state='cancelled', completed_at=?, failure=?
                WHERE channel_id=? AND state='pending' AND kind='turn'
                """,
                (time.time(), reason[:4000], channel_id),
            )
            return work.rowcount

    async def claimed_run_ids(self, channel_id: str) -> tuple[str, ...]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT DISTINCT run_id FROM agent_work
                    WHERE channel_id=? AND state='claimed' AND run_id IS NOT NULL
                    """,
                    (channel_id,),
                )
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    async def policy_for_active_turn(
        self,
        channel_id: str,
        *,
        run_id: str | None = None,
    ) -> tuple[InteractionPolicy, int] | None:
        clauses = ["channel_id=?", "policy_snapshot!='{}'"]
        parameters: list[Any] = [channel_id]
        if run_id is not None:
            clauses.append("run_id=?")
            parameters.append(run_id)
        else:
            clauses.append("state IN ('pending','claimed')")
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    f"""
                    SELECT policy_snapshot, policy_version FROM agent_work
                    WHERE {" AND ".join(clauses)}
                    ORDER BY CASE state WHEN 'claimed' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
                      created_at, id
                    LIMIT 1
                    """,
                    tuple(parameters),
                )
            ).fetchone()
        if row is None:
            return None
        return InteractionPolicy.from_dict(json.loads(row["policy_snapshot"])), int(row["policy_version"])

    async def pending_channels(self) -> list[str]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT DISTINCT channel_id FROM agent_work
                    WHERE state='pending' AND kind IN ('turn','force') AND available_at<=?
                    """,
                    (time.time(),),
                )
            ).fetchall()
        return [str(row["channel_id"]) for row in rows]

    async def next_ready_kind(self, channel_id: str) -> str | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT kind FROM agent_work
                    WHERE channel_id=? AND state='pending' AND kind IN ('turn','force')
                      AND available_at<=?
                    ORDER BY available_at, created_at, id LIMIT 1
                    """,
                    (channel_id, time.time()),
                )
            ).fetchone()
        return str(row["kind"]) if row else None

    async def has_pending(self, channel_id: str, *, include_future: bool = False) -> bool:
        sql = "SELECT 1 FROM agent_work WHERE channel_id=? AND state='pending' AND kind IN ('turn','force')"
        parameters: tuple[Any, ...] = (channel_id,)
        if not include_future:
            sql += " AND available_at<=?"
            parameters += (time.time(),)
        sql += " LIMIT 1"
        async with self.database.transaction() as connection:
            row = await (await connection.execute(sql, parameters)).fetchone()
        return row is not None

    @staticmethod
    async def _advance_cursor(connection: aiosqlite.Connection, channel_id: str) -> None:
        row = await (
            await connection.execute(
                """
                SELECT COALESCE(MIN(sequence) - 1,
                  (SELECT COALESCE(MAX(sequence), 0) FROM conversation_events WHERE channel_id=?))
                FROM conversation_events
                WHERE channel_id=? AND context_state!='applied'
                """,
                (channel_id, channel_id),
            )
        ).fetchone()
        await connection.execute(
            "UPDATE chat_threads SET applied_event_sequence=?, updated_at=? WHERE channel_id=?",
            (int(row[0]), time.time(), channel_id),
        )

    @staticmethod
    def _event(row: aiosqlite.Row | dict[str, Any]) -> ConversationEvent:
        return ConversationEvent(
            id=str(row["id"]),
            channel_id=str(row["channel_id"]),
            sequence=int(row["sequence"]),
            kind=str(row["kind"]),
            payload=json.loads(row["payload"]),
            observed_at=float(row["observed_at"]),
        )

    @staticmethod
    def _work(row: aiosqlite.Row | dict[str, Any]) -> AgentWork:
        return AgentWork(
            id=str(row["id"]),
            channel_id=str(row["channel_id"]),
            kind=str(row["kind"]),
            source_event_id=str(row["source_event_id"]) if row["source_event_id"] else None,
            trigger_kind=str(row["trigger_kind"]),
            trigger_participant=(str(row["trigger_participant"]) if row["trigger_participant"] else None),
            policy_version=int(row["policy_version"]),
            policy_snapshot=json.loads(row["policy_snapshot"]),
            available_at=float(row["available_at"]),
            captured_through_sequence=(
                int(row["captured_through_sequence"])
                if row["captured_through_sequence"] is not None
                else None
            ),
            decision=json.loads(row["decision"]),
        )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
