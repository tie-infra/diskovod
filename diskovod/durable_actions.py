from __future__ import annotations

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
        state, recorded = self.ledger.claim(
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
            self.ledger.finish(context.trace_id, tool_call_id, "ambiguous", records)
            return records
        terminal_state = (
            "completed" if all(record.status != "ambiguous" for record in records) else "ambiguous"
        )
        self.ledger.finish(context.trace_id, tool_call_id, terminal_state, records)
        return records


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _delivery_records(value: str) -> list[DeliveryRecord]:
    return [DeliveryRecord(**item) for item in json.loads(value)]
