from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Protocol

from .agent_actions import AgentActionGateway, DeliveryRecord
from .agent_types import AgentRuntimeContext
from .persistence import SQLITE_BUSY_TIMEOUT_MS, initialize_target_schema


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

    def claim(
        self,
        run_id: str,
        tool_call_id: str,
        action: str,
        request: dict,
    ) -> tuple[str, list[DeliveryRecord] | None]:
        serialized = _json(request)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM side_effect_deliveries WHERE run_id=? AND tool_call_id=?",
                (run_id, tool_call_id),
            ).fetchone()
            if row is None:
                self._connection.execute(
                    """
                    INSERT INTO side_effect_deliveries(
                      run_id, tool_call_id, action, state, request, claimed_at
                    ) VALUES(?, ?, ?, 'claimed', ?, ?)
                    """,
                    (run_id, tool_call_id, action, serialized, time.time()),
                )
                return "new", None
            if row["action"] != action or row["request"] != serialized:
                raise RuntimeError("A tool-call ID was reused for a different side effect")
            result = _delivery_records(row["result"]) if row["result"] else None
            return str(row["state"]), result

    def finish(
        self,
        run_id: str,
        tool_call_id: str,
        state: str,
        records: list[DeliveryRecord],
    ) -> None:
        if state not in {"completed", "ambiguous"}:
            raise ValueError(f"Invalid side-effect terminal state {state!r}")
        with self._lock, self._connection:
            changed = self._connection.execute(
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
            ).rowcount
            if changed != 1:
                raise RuntimeError("Side-effect claim is missing or already terminal")

    def record_escalation(
        self,
        escalation_id: str,
        thread_id: str,
        channel_id: str,
        payload: dict[str, object],
    ) -> None:
        now = time.time()
        with self._lock, self._connection:
            if self._connection.execute(
                "SELECT 1 FROM escalation_interrupts WHERE id=?", (escalation_id,)
            ).fetchone():
                return
            self._connection.execute(
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
        state, recorded = await asyncio.to_thread(
            self.ledger.claim,
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
            await asyncio.to_thread(
                self.ledger.finish,
                context.trace_id,
                tool_call_id,
                "ambiguous",
                records,
            )
            return records
        terminal_state = (
            "completed" if all(record.status != "ambiguous" for record in records) else "ambiguous"
        )
        await asyncio.to_thread(
            self.ledger.finish,
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
        state, recorded = await asyncio.to_thread(
            self.ledger.claim,
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
        await asyncio.to_thread(
            self.ledger.finish,
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
        await asyncio.to_thread(
            self.ledger.record_escalation,
            f"{context.trace_id}:{tool_call_id}",
            context.thread_id,
            context.channel_id,
            payload,
        )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _delivery_records(value: str) -> list[DeliveryRecord]:
    return [DeliveryRecord(**item) for item in json.loads(value)]
