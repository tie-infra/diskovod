from __future__ import annotations

import json
import time
from typing import Protocol

from .agent_actions import AgentActionGateway, DeliveryRecord
from .agent_types import AgentRuntimeContext
from .persistence import AsyncSQLite


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


class SideEffectLedger:
    """At-most-once claim and result storage for externally visible actions."""

    def __init__(self, database: AsyncSQLite):
        self.database = database

    async def claim(
        self,
        run_id: str,
        tool_call_id: str,
        action: str,
        request: dict,
    ) -> tuple[str, list[DeliveryRecord] | None]:
        serialized = _json(request)
        async with self.database.transaction() as connection:
            inserted = await connection.execute(
                """
                INSERT OR IGNORE INTO side_effect_deliveries(
                  run_id, tool_call_id, action, state, request, claimed_at
                ) VALUES(?, ?, ?, 'claimed', ?, ?)
                """,
                (run_id, tool_call_id, action, serialized, time.time()),
            )
            if inserted.rowcount == 1:
                return "new", None
            row = await (
                await connection.execute(
                    "SELECT * FROM side_effect_deliveries WHERE run_id=? AND tool_call_id=?",
                    (run_id, tool_call_id),
                )
            ).fetchone()
            if row is None:
                raise RuntimeError("Side-effect claim disappeared after a uniqueness conflict")
            if row["action"] != action or row["request"] != serialized:
                raise RuntimeError("A tool-call ID was reused for a different side effect")
            result = _delivery_records(row["result"]) if row["result"] else None
            return str(row["state"]), result

    async def finish(
        self,
        run_id: str,
        tool_call_id: str,
        state: str,
        records: list[DeliveryRecord],
    ) -> None:
        if state not in {"completed", "ambiguous"}:
            raise ValueError(f"Invalid side-effect terminal state {state!r}")
        async with self.database.transaction() as connection:
            changed = (
                await connection.execute(
                    """
                UPDATE side_effect_deliveries
                SET state=?, result=?, completed_at=?
                WHERE run_id=? AND tool_call_id=? AND state='claimed'
                """,
                    (
                        state,
                        _json([record.to_dict() for record in records]),
                        time.time(),
                        run_id,
                        tool_call_id,
                    ),
                )
            ).rowcount
            if changed != 1:
                raise RuntimeError("Side-effect claim is missing or already terminal")

    async def record_escalation(
        self,
        escalation_id: str,
        thread_id: str,
        channel_id: str,
        payload: dict[str, object],
    ) -> None:
        now = time.time()
        async with self.database.transaction() as connection:
            if await (
                await connection.execute("SELECT 1 FROM escalation_interrupts WHERE id=?", (escalation_id,))
            ).fetchone():
                return
            await connection.execute(
                """
                INSERT INTO escalation_interrupts(
                  id, thread_id, channel_id, state, payload, created_at, updated_at
                ) VALUES(?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (escalation_id, thread_id, channel_id, _json(payload), now, now),
            )


class DurableActionGateway(AgentActionGateway):
    def __init__(self, ledger: SideEffectLedger, transport: DiscordActionTransport):
        self.ledger = ledger
        self.transport = transport

    async def send_messages(
        self,
        context: AgentRuntimeContext,
        messages: tuple[str, ...],
        *,
        tool_call_id: str,
    ) -> list[DeliveryRecord]:
        state, recorded = await self.ledger.claim(
            context.trace_id,
            tool_call_id,
            "send_messages",
            {"channel_id": context.channel_id, "messages": messages},
        )
        if state in {"completed", "ambiguous"} and recorded is not None:
            return recorded
        if state == "claimed":
            return [
                DeliveryRecord(
                    status="ambiguous",
                    message_index=index,
                    error_code="incomplete_prior_attempt",
                )
                for index, _ in enumerate(messages)
            ]
        try:
            records = await self.transport.send_messages(context, messages)
        except Exception as error:
            records = [
                DeliveryRecord(
                    status="ambiguous",
                    message_index=index,
                    error_code="transport_exception",
                    error_detail=type(error).__name__,
                )
                for index, _ in enumerate(messages)
            ]
            await self.ledger.finish(
                context.trace_id,
                tool_call_id,
                "ambiguous",
                records,
            )
            return records
        terminal_state = (
            "completed" if all(record.status != "ambiguous" for record in records) else "ambiguous"
        )
        await self.ledger.finish(
            context.trace_id,
            tool_call_id,
            terminal_state,
            records,
        )
        return records

    async def react_to_message(
        self,
        context: AgentRuntimeContext,
        emoji: str,
        *,
        tool_call_id: str,
    ) -> DeliveryRecord:
        request = {
            "channel_id": context.channel_id,
            "message_id": context.trigger_message_id,
            "emoji": emoji,
        }
        state, recorded = await self.ledger.claim(
            context.trace_id,
            tool_call_id,
            "react_to_message",
            request,
        )
        if state in {"completed", "ambiguous"} and recorded:
            return recorded[0]
        if state == "claimed":
            return DeliveryRecord("ambiguous", 0, error_code="incomplete_prior_attempt")
        try:
            result = await self.transport.react_to_message(context, context.trigger_message_id, emoji)
        except Exception as error:
            result = DeliveryRecord(
                "ambiguous",
                0,
                error_code="transport_exception",
                error_detail=type(error).__name__,
            )
        await self.ledger.finish(
            context.trace_id,
            tool_call_id,
            "ambiguous" if result.status == "ambiguous" else "completed",
            [result],
        )
        return result

    async def record_escalation(
        self,
        context: AgentRuntimeContext,
        payload: dict[str, object],
        *,
        tool_call_id: str,
    ) -> None:
        await self.ledger.record_escalation(
            f"{context.trace_id}:{tool_call_id}",
            context.thread_id,
            context.channel_id,
            payload,
        )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _delivery_records(value: str) -> list[DeliveryRecord]:
    return [DeliveryRecord(**item) for item in json.loads(value)]
