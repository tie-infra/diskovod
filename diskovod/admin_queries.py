from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from typing import Any

from .redaction import redact_sensitive
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
            mode_rows = await (
                await connection.execute(
                    "SELECT mode, COUNT(*) AS count FROM conversations GROUP BY mode"
                )
            ).fetchall()
            chats = await (
                await connection.execute(
                    """
                    SELECT c.*,
                      (SELECT content FROM messages m WHERE m.channel_id=c.channel_id
                       ORDER BY timestamp DESC LIMIT 1) AS latest_content,
                      (SELECT timestamp FROM messages m WHERE m.channel_id=c.channel_id
                       ORDER BY timestamp DESC LIMIT 1) AS latest_message_at
                    FROM conversations c
                    ORDER BY COALESCE(latest_message_at, c.updated_at) DESC LIMIT 5
                    """
                )
            ).fetchall()
            jobs = await (
                await connection.execute(
                    "SELECT id, type, status, progress_stage, requested_at, completed_at "
                    "FROM admin_jobs ORDER BY requested_at DESC LIMIT 5"
                )
            ).fetchall()
            last_event = await (
                await connection.execute("SELECT MAX(observed_at) FROM discord_events")
            ).fetchone()
        return {
            **counts,
            "conversation_modes": {str(row["mode"]): int(row["count"]) for row in mode_rows},
            "recent_runs": [self._run_summary(row) for row in runs],
            "recent_chats": [dict(row) for row in chats],
            "recent_jobs": [dict(row) for row in jobs],
            "last_discord_event_at": float(last_event[0]) if last_event[0] is not None else None,
        }

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
            "previous_offset": max(0, page_offset - page_limit) if page_offset else None,
            "next_offset": page_offset + page_limit if page_offset + page_limit < total else None,
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
                    "SELECT * FROM messages WHERE channel_id=? ORDER BY timestamp DESC, id DESC LIMIT 51",
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
        has_older_messages = len(messages) > 50
        bounded_messages = messages[:50]
        transcript = [self._message(row) for row in reversed(bounded_messages)]
        return {
            "conversation": conversation,
            "generations": generations,
            "selected_generation": selected,
            "messages": transcript,
            "older_messages_before": (
                float(bounded_messages[-1]["timestamp"]) if has_older_messages else None
            ),
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
            "previous_offset": max(0, max(0, offset) - page_limit) if offset > 0 else None,
            "next_offset": max(0, offset) + page_limit if max(0, offset) + page_limit < total else None,
        }

    async def messages(
        self,
        channel_id: str,
        *,
        before: float | None = None,
        limit: int = 50,
    ) -> dict[str, Any] | None:
        if await self.store.aconversation(channel_id) is None:
            return None
        page_limit = max(1, min(limit, 100))
        where = "channel_id=?"
        parameters: list[Any] = [channel_id]
        if before is not None:
            where += " AND timestamp<?"
            parameters.append(before)
        parameters.append(page_limit)
        async with self.store.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    f"SELECT * FROM messages WHERE {where} ORDER BY timestamp DESC, id DESC LIMIT ?",
                    parameters,
                )
            ).fetchall()
        items = [self._message(row) for row in reversed(rows)]
        return {
            "items": items,
            "next_before": float(rows[-1]["timestamp"]) if len(rows) == page_limit else None,
        }

    async def chat_timeline(
        self,
        channel_id: str,
        *,
        after: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        async with self.store.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT e.* FROM agent_trace_events e
                    JOIN agent_runs r ON r.id=e.run_id
                    WHERE r.channel_id=? AND e.id>?
                    ORDER BY e.id LIMIT ?
                    """,
                    (channel_id, max(0, after), max(1, min(limit, 200))),
                )
            ).fetchall()
        items = [self._trace(row) for row in rows]
        return {"items": items, "next_cursor": int(rows[-1]["id"]) if rows else after}

    async def run_events(self, run_id: str, *, after: int = 0, limit: int = 100) -> dict[str, Any] | None:
        async with self.store.database.transaction() as connection:
            exists = await (
                await connection.execute("SELECT 1 FROM agent_runs WHERE id=?", (run_id,))
            ).fetchone()
            if exists is None:
                return None
            rows = await (
                await connection.execute(
                    "SELECT * FROM agent_trace_events WHERE run_id=? AND sequence>? "
                    "ORDER BY sequence LIMIT ?",
                    (run_id, max(0, after), max(1, min(limit, 200))),
                )
            ).fetchall()
        items = [self._trace(row) for row in rows]
        return {"items": items, "next_cursor": int(rows[-1]["sequence"]) if rows else after}

    async def search(self, query: str, *, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
        text = query.strip()[:200]
        if len(text) < 2:
            return {"chats": [], "messages": [], "runs": [], "memories": [], "attachments": []}
        pattern = f"%{text}%"
        bounded = max(1, min(limit, 20))
        queries = {
            "chats": (
                "SELECT channel_id, peer_name, mode, updated_at FROM conversations "
                "WHERE peer_name LIKE ? OR channel_id LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (pattern, pattern, bounded),
            ),
            "messages": (
                "SELECT id, channel_id, author_name, content, timestamp FROM messages "
                "WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (pattern, bounded),
            ),
            "runs": (
                "SELECT id, channel_id, status, started_at FROM agent_runs "
                "WHERE id LIKE ? OR trace_id LIKE ? OR channel_id LIKE ? "
                "ORDER BY started_at DESC LIMIT ?",
                (pattern, pattern, pattern, bounded),
            ),
            "memories": (
                "SELECT namespace, key, value, updated_at FROM langgraph_store_items "
                "WHERE key LIKE ? OR value LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (pattern, pattern, bounded),
            ),
            "attachments": (
                "SELECT id, channel_id, message_id, filename, created_at FROM attachment_references "
                "WHERE filename LIKE ? ORDER BY created_at DESC LIMIT ?",
                (pattern, bounded),
            ),
        }
        result: dict[str, list[dict[str, Any]]] = {}
        async with self.store.database.transaction() as connection:
            for name, (statement, parameters) in queries.items():
                rows = await (await connection.execute(statement, parameters)).fetchall()
                result[name] = [dict(row) for row in rows]
        return result

    async def resource_versions(self, topics: set[str]) -> dict[str, str]:
        versions: dict[str, str] = {}
        async with self.store.database.transaction() as connection:
            for topic in sorted(topics):
                if topic == "jobs":
                    row = await (
                        await connection.execute(
                            "SELECT COALESCE(MAX(id), 0) FROM admin_job_events"
                        )
                    ).fetchone()
                elif topic == "inbox":
                    row = await (
                        await connection.execute(
                            "SELECT COALESCE(MAX(updated_at), 0), COUNT(*) FROM escalation_interrupts "
                            "WHERE state IN ('pending','claimed')"
                        )
                    ).fetchone()
                elif topic.startswith("chat:"):
                    channel_id = topic[5:]
                    row = await (
                        await connection.execute(
                            """
                            SELECT
                              COALESCE((SELECT MAX(timestamp + COALESCE(edited_at, 0) +
                                COALESCE(deleted_at, 0)) FROM messages WHERE channel_id=?), 0),
                              COALESCE((SELECT MAX(updated_at) FROM conversations WHERE channel_id=?), 0),
                              COALESCE((SELECT MAX(started_at + COALESCE(completed_at, 0))
                                FROM agent_runs WHERE channel_id=?), 0),
                              COALESCE((SELECT MAX(c.created_at) FROM checkpoint_index c
                                JOIN chat_thread_generations g ON g.thread_id=c.thread_id
                                WHERE g.channel_id=?), 0)
                            """,
                            (channel_id, channel_id, channel_id, channel_id),
                        )
                    ).fetchone()
                elif topic.startswith("run:"):
                    run_id = topic[4:]
                    row = await (
                        await connection.execute(
                            """
                            SELECT status, COALESCE(completed_at, started_at),
                              COALESCE((SELECT MAX(sequence) FROM agent_trace_events WHERE run_id=?), 0)
                            FROM agent_runs WHERE id=?
                            """,
                            (run_id, run_id),
                        )
                    ).fetchone()
                elif topic.startswith("job:"):
                    job_id = topic[4:]
                    row = await (
                        await connection.execute(
                            """
                            SELECT status, COALESCE(completed_at, started_at, requested_at), progress_stage,
                              progress_current, progress_total,
                              COALESCE((SELECT MAX(sequence) FROM admin_job_events WHERE job_id=?), 0)
                            FROM admin_jobs WHERE id=?
                            """,
                            (job_id, job_id),
                        )
                    ).fetchone()
                else:
                    continue
                values = tuple(row) if row is not None else ("missing",)
                versions[topic] = hashlib.sha256(repr(values).encode()).hexdigest()[:16]
        return versions

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

    async def inbox(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        page_limit = max(1, min(limit, 100))
        page_offset = max(0, offset)
        async with self.store.database.transaction() as connection:
            total = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COUNT(*) FROM escalation_interrupts "
                            "WHERE state IN ('pending','claimed')"
                        )
                    ).fetchone()
                )[0]
            )
            rows = await (
                await connection.execute(
                    "SELECT e.*, c.peer_name FROM escalation_interrupts e "
                    "LEFT JOIN conversations c ON c.channel_id=e.channel_id "
                    "WHERE e.state IN ('pending','claimed') "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (page_limit, page_offset),
                )
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                payload = json.loads(item["payload"])
            except (TypeError, json.JSONDecodeError):
                payload = {}
            item["payload"] = redact_sensitive(payload)
            item["reason"] = str(payload.get("reason") or "other_explicit_request")
            item["acknowledgement"] = str(payload.get("acknowledgement") or "")
            result.append(item)
        return {
            "items": result,
            "total": total,
            "limit": page_limit,
            "offset": page_offset,
            "previous_offset": max(0, page_offset - page_limit) if page_offset else None,
            "next_offset": page_offset + page_limit if page_offset + page_limit < total else None,
        }

    async def escalation(self, escalation_id: str) -> dict[str, Any] | None:
        escalation = await self.store.aescalation_interrupt(escalation_id)
        if escalation is None:
            return None
        channel_id = str(escalation["channel_id"])
        async with self.store.database.transaction() as connection:
            conversation = await (
                await connection.execute(
                    "SELECT * FROM conversations WHERE channel_id=?", (channel_id,)
                )
            ).fetchone()
            rows = await (
                await connection.execute(
                    "SELECT * FROM messages WHERE channel_id=? AND timestamp<=? "
                    "ORDER BY timestamp DESC, id DESC LIMIT 30",
                    (channel_id, float(escalation["created_at"])),
                )
            ).fetchall()
        payload = redact_sensitive(escalation.get("payload") or {})
        escalation["payload"] = payload
        escalation["reason"] = str(payload.get("reason") or "other_explicit_request")
        escalation["acknowledgement"] = str(payload.get("acknowledgement") or "")
        return {
            "escalation": escalation,
            "conversation": dict(conversation) if conversation else None,
            "messages": [self._message(row) for row in reversed(rows)],
        }

    async def actionable_runs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        async with self.store.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    "SELECT * FROM agent_runs WHERE status IN ('failed','interrupted') "
                    "ORDER BY started_at DESC LIMIT ?",
                    (max(1, min(limit, 50)),),
                )
            ).fetchall()
        return [self._run_summary(row, include_error=True) for row in rows]

    async def inbox_count(self) -> int:
        async with self.store.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT COUNT(*) FROM escalation_interrupts WHERE state IN ('pending','claimed')"
                )
            ).fetchone()
        return int(row[0])

    async def memories(
        self,
        *,
        query: str = "",
        scope: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if query:
            clauses.append("(key LIKE ? OR value LIKE ?)")
            parameters.extend((f"%{query[:200]}%", f"%{query[:200]}%"))
        if scope:
            clauses.append("namespace LIKE ?")
            parameters.append(f"%{scope[:200]}%")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        page_limit = max(1, min(limit, 100))
        page_offset = max(0, offset)
        async with self.store.database.transaction() as connection:
            total = int(
                (
                    await (
                        await connection.execute(
                            f"SELECT COUNT(*) FROM langgraph_store_items{where}", parameters
                        )
                    ).fetchone()
                )[0]
            )
            rows = await (
                await connection.execute(
                    "SELECT namespace, key, value, created_at, updated_at "
                    f"FROM langgraph_store_items{where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                    (*parameters, page_limit, page_offset),
                )
            ).fetchall()
        return {
            "items": [{**dict(row), "value_pretty": self._pretty(row["value"])} for row in rows],
            "query": query,
            "scope": scope,
            "total": total,
            "limit": page_limit,
            "offset": page_offset,
            "previous_offset": max(0, page_offset - page_limit) if page_offset else None,
            "next_offset": page_offset + page_limit if page_offset + page_limit < total else None,
        }

    async def attachments(
        self,
        *,
        query: str = "",
        channel_id: str = "",
        media_type: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if query:
            clauses.append("r.filename LIKE ?")
            parameters.append(f"%{query[:200]}%")
        if channel_id:
            clauses.append("r.channel_id=?")
            parameters.append(channel_id[:200])
        if media_type:
            clauses.append("o.media_type LIKE ?")
            parameters.append(f"{media_type[:100]}%")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        page_limit = max(1, min(limit, 100))
        page_offset = max(0, offset)
        async with self.store.database.transaction() as connection:
            total = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COUNT(*) FROM attachment_references r "
                            f"JOIN attachment_objects o ON o.sha256=r.object_sha256{where}",
                            parameters,
                        )
                    ).fetchone()
                )[0]
            )
            rows = await (
                await connection.execute(
                    f"""
                    SELECT r.*, o.size, o.media_type, o.created_at AS object_created_at
                    FROM attachment_references r JOIN attachment_objects o ON o.sha256=r.object_sha256
                    {where} ORDER BY r.created_at DESC LIMIT ? OFFSET ?
                    """,
                    (*parameters, page_limit, page_offset),
                )
            ).fetchall()
        return {
            "items": [dict(row) for row in rows],
            "query": query,
            "channel_id": channel_id,
            "media_type": media_type,
            "total": total,
            "limit": page_limit,
            "offset": page_offset,
            "previous_offset": max(0, page_offset - page_limit) if page_offset else None,
            "next_offset": page_offset + page_limit if page_offset + page_limit < total else None,
        }

    @staticmethod
    def _message(row) -> dict[str, Any]:
        item = dict(row)
        try:
            item["attachments"] = json.loads(item.get("attachments") or "[]")
        except json.JSONDecodeError:
            item["attachments"] = []
        item["role"] = (
            "peer"
            if item["direction"] == "in"
            else ("assistant" if item.get("source") == "assistant" else "owner")
        )
        item["timestamp_label"] = AdminQueryService._time(float(item["timestamp"]))
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
        payload = redact_sensitive(payload)
        item.pop("payload", None)
        item["payload_value"] = payload
        item["payload_pretty"] = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        item["recorded_at_label"] = cls._time(float(item["recorded_at"]))
        return item

    @classmethod
    def _delivery(cls, row) -> dict[str, Any]:
        item = dict(row)
        item["request_pretty"] = cls._pretty(item.get("request"), redact=True)
        item["result_pretty"] = cls._pretty(item.get("result"), redact=True)
        item.pop("request", None)
        item.pop("result", None)
        return item

    @staticmethod
    def _pretty(value: Any, *, redact: bool = False) -> str:
        try:
            parsed = json.loads(value)
            return json.dumps(
                redact_sensitive(parsed) if redact else parsed,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        except (TypeError, json.JSONDecodeError):
            return str(value or "")

    @staticmethod
    def _time(value: float) -> str:
        return datetime.fromtimestamp(value).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
