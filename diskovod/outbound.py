from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Protocol

from .agent_types import AgentRuntimeContext, CapabilityProfile
from .persistence import AsyncSQLite


@dataclass(frozen=True, slots=True)
class DeliveryRecord:
    status: str
    message_index: int
    discord_message_id: str | None = None
    error_code: str | None = None
    error_detail: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and self.discord_message_id is not None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DiscordActionTransport(Protocol):
    async def send_messages(
        self,
        context: AgentRuntimeContext,
        messages: tuple[str, ...],
    ) -> list[DeliveryRecord]: ...

    async def react_to_message(
        self,
        context: AgentRuntimeContext,
        message_id: str,
        emoji: str,
    ) -> DeliveryRecord: ...


class OutboundActions(Protocol):
    async def publish_messages(
        self,
        context: AgentRuntimeContext,
        messages: tuple[str, ...],
        *,
        source_kind: str,
        source_id: str,
    ) -> list[DeliveryRecord]: ...

    async def react(
        self,
        context: AgentRuntimeContext,
        emoji: str,
        message_id: str,
        *,
        source_id: str,
    ) -> DeliveryRecord: ...

    async def record_escalation(
        self,
        context: AgentRuntimeContext,
        *,
        source_id: str,
        payload: dict[str, object],
    ) -> None: ...


class OutboundPublisher:
    """Materialize and dispatch externally visible Discord actions idempotently."""

    def __init__(self, database: AsyncSQLite, transport: DiscordActionTransport):
        self.database = database
        self.transport = transport
        self.owner = f"embedded:{uuid.uuid4()}"

    async def publish_messages(
        self,
        context: AgentRuntimeContext,
        messages: tuple[str, ...],
        *,
        source_kind: str,
        source_id: str,
    ) -> list[DeliveryRecord]:
        batch_id = self._action_id(context.thread_id, source_kind, source_id, "message-batch", 0)
        segments = [
            (message_index, segment)
            for message_index, message in enumerate(messages)
            for segment in _discord_segments(message)
        ]
        records: list[DeliveryRecord] = []
        for ordinal, (message_index, message) in enumerate(segments):
            action_id = self._action_id(
                context.thread_id,
                source_kind,
                source_id,
                "discord_message",
                ordinal,
            )
            payload = {"channel_id": context.channel_id, "message": message}
            if len(segments) != len(messages):
                payload.update(
                    logical_message_index=message_index,
                    transport_segment=ordinal,
                    transport_segment_count=len(segments),
                )
            claimed, existing = await self._claim(
                action_id=action_id,
                batch_id=batch_id,
                ordinal=ordinal,
                context=context,
                source_kind=source_kind,
                source_id=source_id,
                kind="discord_message",
                payload=payload,
            )
            if not claimed:
                records.append(existing or self._ambiguous(ordinal, "incomplete_prior_attempt"))
                continue
            try:
                delivered = await self.transport.send_messages(context, (message,))
                record = (
                    delivered[0]
                    if delivered
                    else DeliveryRecord("failed", 0, error_code="missing_transport_result")
                )
                record = DeliveryRecord(
                    record.status,
                    ordinal,
                    record.discord_message_id,
                    record.error_code,
                    record.error_detail,
                )
            except Exception as error:
                record = self._ambiguous(
                    ordinal,
                    "transport_exception",
                    f"{type(error).__name__}: {error}"[:1000],
                )
            await self._finish(action_id, record)
            records.append(record)
        return records

    async def reconcile_abandoned(self) -> list[dict[str, str]]:
        """Make foreign in-flight attempts explicitly ambiguous after process restart."""
        reconciled: list[dict[str, str]] = []
        now = time.time()
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    "SELECT * FROM outbound_actions "
                    "WHERE state='dispatching' AND COALESCE(lease_owner, '')!=?",
                    (self.owner,),
                )
            ).fetchall()
            for row in rows:
                record = self._ambiguous(int(row["ordinal"]), "abandoned_dispatch")
                await connection.execute(
                    """
                    UPDATE outbound_actions
                    SET state='ambiguous', result=?, error_code=?, completed_at=?,
                        lease_owner=NULL, lease_expires_at=NULL
                    WHERE id=? AND state='dispatching' AND COALESCE(lease_owner, '')!=?
                    """,
                    (
                        _json(record.to_dict()),
                        record.error_code,
                        now,
                        row["id"],
                        self.owner,
                    ),
                )
                reconciled.append(
                    {"id": str(row["id"]), "run_id": str(row["run_id"]), "state": "ambiguous"}
                )
        return reconciled

    async def action(self, action_id: str) -> dict[str, object] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT * FROM outbound_actions WHERE id=?", (action_id,))
            ).fetchone()
        return dict(row) if row is not None else None

    async def resolve(
        self,
        action_id: str,
        resolution: str,
        *,
        remote_id: str = "",
    ) -> DeliveryRecord | None:
        if resolution not in {"confirmed_succeeded", "confirmed_failed"}:
            raise ValueError("Unknown outbound resolution")
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT * FROM outbound_actions WHERE id=?", (action_id,))
            ).fetchone()
            if row is None or row["state"] not in {"ambiguous", "dispatching"}:
                return None
            if resolution == "confirmed_succeeded":
                confirmed_remote_id = remote_id.strip()[:200] or f"owner-confirmed:{action_id}"
                record = DeliveryRecord(
                    "accepted",
                    int(row["ordinal"]),
                    discord_message_id=confirmed_remote_id,
                )
                state = "succeeded"
                error_code = None
            else:
                record = DeliveryRecord(
                    "failed",
                    int(row["ordinal"]),
                    error_code="owner_confirmed_not_delivered",
                )
                state = "failed"
                error_code = record.error_code
            changed = (
                await connection.execute(
                    """
                    UPDATE outbound_actions
                    SET state=?, result=?, remote_id=?, error_code=?, completed_at=?,
                        lease_owner=NULL, lease_expires_at=NULL
                    WHERE id=? AND state IN ('ambiguous','dispatching')
                    """,
                    (
                        state,
                        _json(record.to_dict()),
                        record.discord_message_id,
                        error_code,
                        time.time(),
                        action_id,
                    ),
                )
            ).rowcount
        return record if changed == 1 else None

    async def retry(self, action_id: str) -> DeliveryRecord | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT * FROM outbound_actions WHERE id=?", (action_id,))
            ).fetchone()
            if row is None or row["state"] not in {"ambiguous", "failed"}:
                return None
            await connection.execute(
                """
                UPDATE outbound_actions
                SET state='pending', result=NULL, remote_id=NULL, error_code=NULL,
                    completed_at=NULL, lease_owner=NULL, lease_expires_at=NULL
                WHERE id=? AND state IN ('ambiguous','failed')
                """,
                (action_id,),
            )
        payload = json.loads(row["payload"])
        context = self._stored_context(row)
        claimed, existing = await self._claim(
            action_id=str(row["id"]),
            batch_id=str(row["batch_id"]),
            ordinal=int(row["ordinal"]),
            context=context,
            source_kind=str(row["source_kind"]),
            source_id=str(row["source_id"]),
            kind=str(row["kind"]),
            payload=payload,
        )
        if not claimed:
            return existing
        try:
            if row["kind"] == "discord_message":
                delivered = await self.transport.send_messages(context, (str(payload["message"]),))
                record = (
                    delivered[0]
                    if delivered
                    else DeliveryRecord("failed", 0, error_code="missing_transport_result")
                )
            else:
                record = await self.transport.react_to_message(
                    context,
                    str(payload["message_id"]),
                    str(payload["emoji"]),
                )
            record = DeliveryRecord(
                record.status,
                int(row["ordinal"]),
                record.discord_message_id,
                record.error_code,
                record.error_detail,
            )
        except Exception as error:
            record = self._ambiguous(
                int(row["ordinal"]),
                "transport_exception",
                type(error).__name__,
            )
        await self._finish(action_id, record)
        return record

    async def react(
        self,
        context: AgentRuntimeContext,
        emoji: str,
        message_id: str,
        *,
        source_id: str,
    ) -> DeliveryRecord:
        action_id = self._action_id(
            context.thread_id,
            "tool",
            source_id,
            "discord_reaction",
            0,
        )
        payload = {
            "channel_id": context.channel_id,
            "message_id": message_id,
            "emoji": emoji,
        }
        claimed, existing = await self._claim(
            action_id=action_id,
            batch_id=action_id,
            ordinal=0,
            context=context,
            source_kind="tool",
            source_id=source_id,
            kind="discord_reaction",
            payload=payload,
        )
        if not claimed:
            return existing or self._ambiguous(0, "incomplete_prior_attempt")
        try:
            record = await self.transport.react_to_message(context, message_id, emoji)
        except Exception as error:
            record = self._ambiguous(
                0,
                "transport_exception",
                f"{type(error).__name__}: {error}"[:1000],
            )
        await self._finish(action_id, record)
        return record

    async def record_escalation(
        self,
        context: AgentRuntimeContext,
        *,
        source_id: str,
        payload: dict[str, object],
    ) -> None:
        escalation_id = self._action_id(
            context.thread_id,
            "tool",
            source_id,
            "owner_escalation",
            0,
        )
        now = time.time()
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                INSERT INTO escalation_interrupts(
                  id, thread_id, channel_id, state, payload, created_at, updated_at
                ) VALUES(?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (escalation_id, context.thread_id, context.channel_id, _json(payload), now, now),
            )

    async def _claim(
        self,
        *,
        action_id: str,
        batch_id: str,
        ordinal: int,
        context: AgentRuntimeContext,
        source_kind: str,
        source_id: str,
        kind: str,
        payload: dict[str, object],
    ) -> tuple[bool, DeliveryRecord | None]:
        encoded = _json(payload)
        now = time.time()
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                INSERT OR IGNORE INTO outbound_actions(
                  id, batch_id, ordinal, thread_id, channel_id, run_id,
                  source_kind, source_id, kind, payload, state, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    action_id,
                    batch_id,
                    ordinal,
                    context.thread_id,
                    context.channel_id,
                    context.run_id or context.trace_id,
                    source_kind,
                    source_id,
                    kind,
                    encoded,
                    now,
                ),
            )
            row = await (
                await connection.execute("SELECT * FROM outbound_actions WHERE id=?", (action_id,))
            ).fetchone()
            if row is None:
                raise RuntimeError("Outbound action disappeared after materialization")
            if row["kind"] != kind or row["payload"] != encoded:
                raise RuntimeError("An outbound action ID was reused for different content")
            if row["state"] in {"succeeded", "failed", "ambiguous"}:
                return False, _record(row["result"], ordinal, str(row["state"]))
            if row["state"] == "dispatching":
                return False, self._ambiguous(ordinal, "incomplete_prior_attempt")
            changed = (
                await connection.execute(
                    """
                    UPDATE outbound_actions
                    SET state='dispatching', lease_owner=?, lease_expires_at=?
                    WHERE id=? AND state='pending'
                    """,
                    (self.owner, now + 30, action_id),
                )
            ).rowcount
            return changed == 1, None

    async def _finish(self, action_id: str, record: DeliveryRecord) -> None:
        state = (
            "succeeded" if record.accepted else ("ambiguous" if record.status == "ambiguous" else "failed")
        )
        async with self.database.transaction() as connection:
            changed = (
                await connection.execute(
                    """
                    UPDATE outbound_actions
                    SET state=?, result=?, remote_id=?, error_code=?, completed_at=?,
                        lease_owner=NULL, lease_expires_at=NULL
                    WHERE id=? AND state='dispatching' AND lease_owner=?
                    """,
                    (
                        state,
                        _json(record.to_dict()),
                        record.discord_message_id,
                        record.error_code,
                        time.time(),
                        action_id,
                        self.owner,
                    ),
                )
            ).rowcount
            if changed != 1:
                raise RuntimeError("Outbound action lease was lost before completion")

    @staticmethod
    def _action_id(
        thread_id: str,
        source_kind: str,
        source_id: str,
        action_kind: str,
        ordinal: int,
    ) -> str:
        value = "\0".join((thread_id, source_kind, source_id, action_kind, str(ordinal)))
        return f"action:{hashlib.sha256(value.encode()).hexdigest()}"

    @staticmethod
    def _ambiguous(index: int, code: str, detail: str | None = None) -> DeliveryRecord:
        return DeliveryRecord(
            "ambiguous",
            index,
            error_code=code,
            error_detail=detail,
        )

    @staticmethod
    def _stored_context(row) -> AgentRuntimeContext:
        return AgentRuntimeContext(
            account_id="",
            channel_id=str(row["channel_id"]),
            participant_ids=(),
            owner_id="",
            ui_locale="en",
            prompt_locale="en",
            assistant_name="Diskovod",
            automation_mode="automatic",
            force_reply=False,
            provider_id="",
            model_id="",
            transport_profile="",
            capabilities=CapabilityProfile(),
            trace_id=str(row["run_id"]),
            run_id=str(row["run_id"]),
            thread_id=str(row["thread_id"]),
        )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _discord_segments(message: str, limit: int = 1900) -> tuple[str, ...]:
    """Split one logical reply deterministically without implying model-chosen follow-ups."""
    remaining = message.strip()
    if not remaining:
        return ()
    segments: list[str] = []
    while len(remaining) > limit:
        window = remaining[: limit + 1]
        boundaries = (
            window.rfind("\n\n", 0, limit + 1),
            window.rfind("\n", 0, limit + 1),
            window.rfind(" ", 0, limit + 1),
        )
        cut = next((boundary for boundary in boundaries if boundary >= limit // 2), limit)
        segment = remaining[:cut].rstrip()
        if not segment:
            segment = remaining[:limit]
            cut = limit
        segments.append(segment)
        remaining = remaining[cut:].lstrip()
    if remaining:
        segments.append(remaining)
    return tuple(segments)


def _record(value: str | None, ordinal: int, state: str) -> DeliveryRecord:
    if value:
        return DeliveryRecord(**json.loads(value))
    status = "ambiguous" if state == "ambiguous" else "failed"
    return DeliveryRecord(status, ordinal, error_code="missing_recorded_result")
