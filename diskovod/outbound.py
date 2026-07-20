from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Protocol

from .agent_types import AgentRuntimeContext
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
        records: list[DeliveryRecord] = []
        for ordinal, message in enumerate(messages):
            action_id = self._action_id(
                context.thread_id,
                source_kind,
                source_id,
                "discord_message",
                ordinal,
            )
            payload = {"channel_id": context.channel_id, "message": message}
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
                record = delivered[0] if delivered else DeliveryRecord(
                    "failed", 0, error_code="missing_transport_result"
                )
                record = DeliveryRecord(
                    record.status,
                    ordinal,
                    record.discord_message_id,
                    record.error_code,
                    record.error_detail,
                )
            except Exception as error:
                record = self._ambiguous(ordinal, "transport_exception", type(error).__name__)
            await self._finish(action_id, record)
            records.append(record)
        return records

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
            record = self._ambiguous(0, "transport_exception", type(error).__name__)
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
                    context.trace_id,
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
        state = "succeeded" if record.accepted else (
            "ambiguous" if record.status == "ambiguous" else "failed"
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


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _record(value: str | None, ordinal: int, state: str) -> DeliveryRecord:
    if value:
        return DeliveryRecord(**json.loads(value))
    status = "ambiguous" if state == "ambiguous" else "failed"
    return DeliveryRecord(status, ordinal, error_code="missing_recorded_result")
