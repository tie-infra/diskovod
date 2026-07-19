from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from .store import Store


class AdminQueryService:
    """Task-oriented, bounded projections for the administrative interface."""

    def __init__(self, store: Store):
        self.store = store

    async def overview(self) -> dict[str, Any]:
        async with self.store.database.transaction() as connection:
            counts = {}
            for key, query in {
                "chats": "SELECT COUNT(*) FROM conversations",
                "pending_escalations": (
                    "SELECT COUNT(*) FROM escalation_interrupts WHERE state IN ('pending','claimed')"
                ),
                "failed_runs": "SELECT COUNT(*) FROM agent_runs WHERE status='failed'",
                "active_jobs": (
                    "SELECT COUNT(*) FROM admin_jobs "
                    "WHERE status IN ('queued','running','cancellation_requested')"
                ),
            }.items():
                counts[key] = int((await (await connection.execute(query)).fetchone())[0])
            runs = await (
                await connection.execute(
                    "SELECT id, channel_id, status, started_at, completed_at, error "
                    "FROM agent_runs ORDER BY started_at DESC LIMIT 5"
                )
            ).fetchall()
        return {**counts, "recent_runs": [self._run_summary(row) for row in runs]}

    async def chats(
        self,
        *,
        query: str = "",
        mode: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if query:
            clauses.append("(peer_name LIKE ? OR channel_id LIKE ?)")
            parameters.extend((f"%{query[:200]}%", f"%{query[:200]}%"))
        if mode in {"automatic", "inline", "paused"}:
            clauses.append("mode=?")
            parameters.append(mode)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        page_limit = max(1, min(limit, 100))
        page_offset = max(0, offset)
        async with self.store.database.transaction() as connection:
            total = int(
                (
                    await (
                        await connection.execute(f"SELECT COUNT(*) FROM conversations{where}", parameters)
                    ).fetchone()
                )[0]
            )
            rows = await (
                await connection.execute(
                    f"""
                    SELECT c.*, t.thread_id, t.generation, t.live_steering,
                      (SELECT content FROM messages m WHERE m.channel_id=c.channel_id
                       ORDER BY timestamp DESC LIMIT 1) AS latest_content,
                      (SELECT timestamp FROM messages m WHERE m.channel_id=c.channel_id
                       ORDER BY timestamp DESC LIMIT 1) AS latest_message_at,
                      (SELECT COUNT(*) FROM escalation_interrupts e
                       WHERE e.channel_id=c.channel_id AND e.state IN ('pending','claimed'))
                       AS escalation_count
                    FROM conversations c LEFT JOIN chat_threads t ON t.channel_id=c.channel_id
                    {where} ORDER BY COALESCE(latest_message_at, c.updated_at) DESC
                    LIMIT ? OFFSET ?
                    """,
                    (*parameters, page_limit, page_offset),
                )
            ).fetchall()
        return {
            "items": [dict(row) for row in rows],
            "total": total,
            "limit": page_limit,
            "offset": page_offset,
            "query": query,
            "mode": mode,
        }

    async def chat(self, channel_id: str, *, generation: int | None = None) -> dict[str, Any] | None:
        conversation = await self.store.aconversation(channel_id)
        if conversation is None:
            return None
        generations = await self.store.achat_thread_generations(channel_id)
        selected = next(
            (item for item in generations if generation is None or item["generation"] == generation),
            None,
        )
        if generation is not None and selected is None:
            return None
        async with self.store.database.transaction() as connection:
            messages = await (
                await connection.execute(
                    "SELECT * FROM messages WHERE channel_id=? ORDER BY timestamp DESC LIMIT 200",
                    (channel_id,),
                )
            ).fetchall()
            runs = await (
                await connection.execute(
                    "SELECT * FROM agent_runs WHERE channel_id=? ORDER BY started_at DESC LIMIT 20",
                    (channel_id,),
                )
            ).fetchall()
            checkpoints = (
                await (
                    await connection.execute(
                        "SELECT * FROM checkpoint_index WHERE thread_id=? ORDER BY created_at DESC LIMIT 100",
                        (selected["thread_id"],),
                    )
                ).fetchall()
                if selected
                else []
            )
        transcript = [self._message(row) for row in reversed(messages)]
        return {
            "conversation": conversation,
            "generations": generations,
            "selected_generation": selected,
            "messages": transcript,
            "runs": [self._run_summary(row) for row in runs],
            "checkpoints": [dict(row) for row in checkpoints],
        }

    async def runs(
        self, *, status: str = "", channel_id: str = "", limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if status:
            clauses.append("status=?")
            parameters.append(status)
        if channel_id:
            clauses.append("channel_id=?")
            parameters.append(channel_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        page_limit = max(1, min(limit, 100))
        async with self.store.database.transaction() as connection:
            total = int(
                (
                    await (
                        await connection.execute(f"SELECT COUNT(*) FROM agent_runs{where}", parameters)
                    ).fetchone()
                )[0]
            )
            rows = await (
                await connection.execute(
                    f"SELECT * FROM agent_runs{where} ORDER BY started_at DESC LIMIT ? OFFSET ?",
                    (*parameters, page_limit, max(0, offset)),
                )
            ).fetchall()
        return {
            "items": [self._run_summary(row) for row in rows],
            "total": total,
            "limit": page_limit,
            "offset": max(0, offset),
            "status": status,
            "channel_id": channel_id,
        }

    async def run(self, run_id: str) -> dict[str, Any] | None:
        async with self.store.database.transaction() as connection:
            run = await (
                await connection.execute("SELECT * FROM agent_runs WHERE id=?", (run_id,))
            ).fetchone()
            if run is None:
                return None
            traces = await (
                await connection.execute(
                    "SELECT * FROM agent_trace_events WHERE run_id=? ORDER BY sequence", (run_id,)
                )
            ).fetchall()
            deliveries = await (
                await connection.execute(
                    "SELECT * FROM side_effect_deliveries WHERE run_id=? ORDER BY claimed_at", (run_id,)
                )
            ).fetchall()
            checkpoints = await (
                await connection.execute(
                    "SELECT * FROM checkpoint_index WHERE run_id=? ORDER BY created_at", (run_id,)
                )
            ).fetchall()
        return {
            "run": self._run_summary(run, include_error=True),
            "timeline": [self._trace(row) for row in traces],
            "deliveries": [self._delivery(row) for row in deliveries],
            "checkpoints": [dict(row) for row in checkpoints],
        }

    async def inbox(self) -> list[dict[str, Any]]:
        rows = await self.store.aactive_interrupts()
        result = []
        for item in rows:
            payload = item["payload"]
            item["reason"] = str(payload.get("reason") or "other_explicit_request")
            item["acknowledgement"] = str(payload.get("acknowledgement") or "")
            result.append(item)
        return result

    async def memories(self, *, limit: int = 100) -> list[dict[str, Any]]:
        async with self.store.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    "SELECT namespace, key, value, created_at, updated_at "
                    "FROM langgraph_store_items ORDER BY updated_at DESC LIMIT ?",
                    (max(1, min(limit, 500)),),
                )
            ).fetchall()
        return [{**dict(row), "value_pretty": self._pretty(row["value"])} for row in rows]

    async def attachments(self, *, limit: int = 100) -> list[dict[str, Any]]:
        async with self.store.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT r.*, o.size, o.media_type, o.created_at AS object_created_at
                    FROM attachment_references r JOIN attachment_objects o ON o.sha256=r.object_sha256
                    ORDER BY r.created_at DESC LIMIT ?
                    """,
                    (max(1, min(limit, 500)),),
                )
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _message(row) -> dict[str, Any]:
        item = dict(row)
        try:
            item["attachments"] = json.loads(item.get("attachments") or "[]")
        except json.JSONDecodeError:
            item["attachments"] = []
        return item

    @classmethod
    def _run_summary(cls, row, *, include_error: bool = False) -> dict[str, Any]:
        item = dict(row)
        started = float(item["started_at"])
        completed = float(item["completed_at"]) if item.get("completed_at") else None
        item["started_at_label"] = cls._time(started)
        item["duration_ms"] = round(((completed or time.time()) - started) * 1000)
        if not include_error and item.get("error"):
            item["error"] = str(item["error"])[:240]
        return item

    @classmethod
    def _trace(cls, row) -> dict[str, Any]:
        item = dict(row)
        try:
            payload = json.loads(item["payload"])
        except (TypeError, json.JSONDecodeError):
            payload = {"value": str(item["payload"])}
        item["payload_value"] = payload
        item["payload_pretty"] = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        item["recorded_at_label"] = cls._time(float(item["recorded_at"]))
        return item

    @classmethod
    def _delivery(cls, row) -> dict[str, Any]:
        item = dict(row)
        item["request_pretty"] = cls._pretty(item.get("request"))
        item["result_pretty"] = cls._pretty(item.get("result"))
        return item

    @staticmethod
    def _pretty(value: Any) -> str:
        try:
            return json.dumps(json.loads(value), ensure_ascii=False, indent=2, sort_keys=True)
        except (TypeError, json.JSONDecodeError):
            return str(value or "")

    @staticmethod
    def _time(value: float) -> str:
        return datetime.fromtimestamp(value).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
