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
                await connection.execute("SELECT mode, COUNT(*) AS count FROM conversations GROUP BY mode")
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
        return {
            **counts,
            "conversation_modes": {str(row["mode"]): int(row["count"]) for row in mode_rows},
            "recent_runs": [self._run_summary(row) for row in runs],
            "recent_chats": [dict(row) for row in chats],
            "recent_jobs": [dict(row) for row in jobs],
        }

    async def chats(
        self,
        *,
        query: str = "",
        state: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if query:
            clauses.append(
                "(c.peer_name LIKE ? OR c.channel_id LIKE ? OR EXISTS ("
                "SELECT 1 FROM messages search_message "
                "WHERE search_message.channel_id=c.channel_id AND search_message.content LIKE ?))"
            )
            pattern = f"%{query[:200]}%"
            parameters.extend((pattern, pattern, pattern))
        if state in {"automatic", "inline"}:
            clauses.append("c.mode=? AND c.paused=0")
            parameters.append(state)
        elif state == "paused":
            clauses.append("c.paused=1")
        elif state == "snoozed":
            clauses.append("c.snoozed_until>?")
            parameters.append(time.time())
        elif state == "escalated":
            clauses.append(
                "EXISTS (SELECT 1 FROM escalation_interrupts filter_escalation "
                "WHERE filter_escalation.channel_id=c.channel_id "
                "AND filter_escalation.state IN ('pending','claimed'))"
            )
        elif state == "failed":
            clauses.append(
                "EXISTS (SELECT 1 FROM agent_runs filter_run "
                "WHERE filter_run.channel_id=c.channel_id AND filter_run.status='failed')"
            )
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        page_limit = max(1, min(limit, 100))
        page_offset = max(0, offset)
        async with self.store.database.transaction() as connection:
            total = int(
                (
                    await (
                        await connection.execute(f"SELECT COUNT(*) FROM conversations c{where}", parameters)
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
                       AS escalation_count,
                      (SELECT COUNT(*) FROM attachment_references a
                       WHERE a.channel_id=c.channel_id) AS attachment_count,
                      (SELECT COUNT(*) FROM agent_runs active_run
                       WHERE active_run.channel_id=c.channel_id AND active_run.status='running')
                       AS active_run_count,
                      (SELECT COUNT(*) FROM agent_runs failed_run
                       WHERE failed_run.channel_id=c.channel_id AND failed_run.status='failed')
                       AS failed_run_count
                    FROM conversations c LEFT JOIN chat_threads t ON t.channel_id=c.channel_id
                    {where} ORDER BY COALESCE(latest_message_at, c.updated_at) DESC
                    LIMIT ? OFFSET ?
                    """,
                    (*parameters, page_limit, page_offset),
                )
            ).fetchall()
        items = [dict(row) for row in rows]
        now = time.time()
        for item in items:
            item["effective_mode"] = (
                "paused"
                if item["paused"]
                else "snoozed"
                if item.get("snoozed_until") and float(item["snoozed_until"]) > now
                else item["mode"]
            )
        return {
            "items": items,
            "total": total,
            "limit": page_limit,
            "offset": page_offset,
            "query": query,
            "state": state,
            "previous_offset": max(0, page_offset - page_limit) if page_offset else None,
            "next_offset": page_offset + page_limit if page_offset + page_limit < total else None,
        }

    async def chat(self, channel_id: str, *, generation: int | None = None) -> dict[str, Any] | None:
        conversation = await self.store.aconversation(channel_id)
        if conversation is None:
            return None
        active_thread = await self.store.achat_thread_for_channel(channel_id)
        conversation["effective_mode"] = "paused" if conversation["paused"] else conversation["mode"]
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
            reactions = await (
                await connection.execute(
                    "SELECT trigger_message_id, emoji FROM assistant_reactions WHERE channel_id=?",
                    (channel_id,),
                )
            ).fetchall()
            attachments = await (
                await connection.execute(
                    """
                    SELECT r.*, o.size, o.media_type,
                      (SELECT state FROM attachment_artifacts artifact
                       WHERE artifact.object_sha256=r.object_sha256
                       ORDER BY updated_at DESC LIMIT 1) AS extraction_state
                    FROM attachment_references r
                    JOIN attachment_objects o ON o.sha256=r.object_sha256
                    WHERE r.channel_id=? ORDER BY r.created_at DESC LIMIT 10
                    """,
                    (channel_id,),
                )
            ).fetchall()
            memory_rows = []
            if selected:
                namespace = json.dumps(
                    ["chat", selected["account_id"], channel_id, "memory"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                memory_rows = await (
                    await connection.execute(
                        "SELECT key, value, updated_at FROM langgraph_store_items "
                        "WHERE namespace=? ORDER BY updated_at DESC LIMIT 10",
                        (namespace,),
                    )
                ).fetchall()
            configuration = (
                await (
                    await connection.execute(
                        "SELECT configuration FROM agent_configuration_versions WHERE id=?",
                        (selected["configuration_version_id"],),
                    )
                ).fetchone()
                if selected and selected.get("configuration_version_id") is not None
                else None
            )
            active_wait = await (
                await connection.execute(
                    """
                    SELECT id, state, resume_at, created_at, run_id
                    FROM conversation_waits WHERE channel_id=?
                      AND state IN ('arming','scheduled','resuming') LIMIT 1
                    """,
                    (channel_id,),
                )
            ).fetchone()
        has_older_messages = len(messages) > 50
        bounded_messages = messages[:50]
        reaction_map = {str(row["trigger_message_id"]): str(row["emoji"]) for row in reactions}
        transcript = [self._message(row) for row in reversed(bounded_messages)]
        for message in transcript:
            message["assistant_reaction"] = reaction_map.get(str(message["id"]))
        memories = []
        for row in memory_rows:
            item = dict(row)
            item["value"] = redact_sensitive(self._payload(item["value"]))
            memories.append(item)
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
            "attachments": [dict(row) for row in attachments],
            "memories": memories,
            "configuration": redact_sensitive(
                self._payload(configuration["configuration"]) if configuration else None
            ),
            "historical": bool(
                selected and generations and selected["generation"] != generations[0]["generation"]
            ),
            "live_steering": bool(active_thread.get("live_steering", 1)) if active_thread else True,
            "active_wait": dict(active_wait) if active_wait else None,
        }

    async def checkpoint(
        self,
        channel_id: str,
        generation: int,
        checkpoint_id: str,
    ) -> dict[str, Any] | None:
        async with self.store.database.transaction() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT checkpoint.*, generation.generation, generation.channel_id,
                      generation.configuration_version_id
                    FROM checkpoint_index checkpoint
                    JOIN chat_thread_generations generation
                      ON generation.thread_id=checkpoint.thread_id
                    WHERE generation.channel_id=? AND generation.generation=?
                      AND checkpoint.checkpoint_id=?
                    """,
                    (channel_id, generation, checkpoint_id),
                )
            ).fetchone()
        return dict(row) if row else None

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
                        await connection.execute("SELECT COALESCE(MAX(id), 0) FROM admin_job_events")
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

    async def run(
        self,
        run_id: str,
        *,
        event_limit: int = 200,
        event_offset: int = 0,
    ) -> dict[str, Any] | None:
        bounded_limit = max(1, min(event_limit, 500))
        bounded_offset = max(0, event_offset)
        async with self.store.database.transaction() as connection:
            run = await (
                await connection.execute(
                    """
                    SELECT r.*, v.configuration
                    FROM agent_runs r
                    LEFT JOIN agent_configuration_versions v ON v.id=r.configuration_version_id
                    WHERE r.id=?
                    """,
                    (run_id,),
                )
            ).fetchone()
            if run is None:
                return None
            event_count = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COUNT(*) FROM agent_trace_events WHERE run_id=?", (run_id,)
                        )
                    ).fetchone()
                )[0]
            )
            traces = await (
                await connection.execute(
                    "SELECT * FROM agent_trace_events WHERE run_id=? ORDER BY sequence LIMIT ? OFFSET ?",
                    (run_id, bounded_limit, bounded_offset),
                )
            ).fetchall()
            deliveries = await (
                await connection.execute(
                    "SELECT * FROM outbound_actions WHERE run_id=? ORDER BY created_at, ordinal",
                    (run_id,),
                )
            ).fetchall()
            checkpoints = await (
                await connection.execute(
                    "SELECT * FROM checkpoint_index WHERE run_id=? ORDER BY created_at", (run_id,)
                )
            ).fetchall()
            waits = await (
                await connection.execute(
                    "SELECT * FROM conversation_waits WHERE run_id=? ORDER BY created_at",
                    (run_id,),
                )
            ).fetchall()
            messages = await (
                await connection.execute(
                    "SELECT * FROM messages WHERE channel_id=? AND timestamp<=? "
                    "ORDER BY timestamp DESC, id DESC LIMIT 50",
                    (run["channel_id"], float(run["started_at"])),
                )
            ).fetchall()
        parsed_traces = [(row, self._payload(row["payload"])) for row in traces]
        run_summary = self._run_summary(run, include_error=True)
        run_summary["configuration"] = redact_sensitive(self._payload(run_summary.pop("configuration", None)))
        return {
            "run": run_summary,
            "timeline": [self._trace(row, payload) for row, payload in parsed_traces],
            "deliveries": [self._delivery(row) for row in deliveries],
            "checkpoints": [dict(row) for row in checkpoints],
            "waits": [self._wait(row) for row in waits],
            "conversation": [self._message(row) for row in reversed(messages)],
            "model_exchanges": self._model_exchanges(parsed_traces),
            "event_count": event_count,
            "event_offset": bounded_offset,
            "event_limit": bounded_limit,
            "previous_event_offset": (max(0, bounded_offset - bounded_limit) if bounded_offset else None),
            "next_event_offset": (
                bounded_offset + bounded_limit if bounded_offset + bounded_limit < event_count else None
            ),
        }

    async def run_event(self, run_id: str, sequence: int) -> dict[str, Any] | None:
        async with self.store.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT * FROM agent_trace_events WHERE run_id=? AND sequence=?",
                    (run_id, sequence),
                )
            ).fetchone()
        return self._trace(row, self._payload(row["payload"]), include_payload=True) if row else None

    async def run_delivery(self, run_id: str, action_id: str) -> dict[str, Any] | None:
        async with self.store.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT * FROM outbound_actions WHERE run_id=? AND id=?",
                    (run_id, action_id),
                )
            ).fetchone()
        return self._delivery(row, include_payload=True) if row else None

    async def run_diagnostic(self, run_id: str) -> dict[str, Any] | None:
        view = await self.run(run_id, event_limit=500)
        if view is None:
            return None
        async with self.store.database.transaction() as connection:
            event_rows = await (
                await connection.execute(
                    "SELECT * FROM agent_trace_events WHERE run_id=? ORDER BY sequence LIMIT 500",
                    (run_id,),
                )
            ).fetchall()
            delivery_rows = await (
                await connection.execute(
                    "SELECT * FROM outbound_actions WHERE run_id=? ORDER BY created_at, ordinal",
                    (run_id,),
                )
            ).fetchall()
        events = [self._trace(row, self._payload(row["payload"]), include_payload=True) for row in event_rows]
        deliveries = [self._delivery(row, include_payload=True) for row in delivery_rows]
        return {
            "schema": "diskovod.run-diagnostic.v1",
            "run": view["run"],
            "events": events,
            "deliveries": deliveries,
            "checkpoints": view["checkpoints"],
            "waits": view["waits"],
            "truncated": view["event_count"] > 500,
        }

    async def inbox(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        page_limit = max(1, min(limit, 100))
        page_offset = max(0, offset)
        async with self.store.database.transaction() as connection:
            total = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COUNT(*) FROM escalation_interrupts WHERE state IN ('pending','claimed')"
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
                await connection.execute("SELECT * FROM conversations WHERE channel_id=?", (channel_id,))
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

    async def capability_probes(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        bounded_limit = max(1, min(limit, 100))
        bounded_offset = max(0, offset)
        async with self.store.database.transaction() as connection:
            total = int(
                (
                    await (
                        await connection.execute("SELECT COUNT(*) FROM provider_capability_probes")
                    ).fetchone()
                )[0]
            )
            rows = await (
                await connection.execute(
                    "SELECT id, configuration, capability, status, conclusion, started_at, completed_at "
                    "FROM provider_capability_probes ORDER BY completed_at DESC LIMIT ? OFFSET ?",
                    (bounded_limit, bounded_offset),
                )
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            configuration = self._payload(item.pop("configuration"))
            item["configuration"] = redact_sensitive(configuration)
            item["completed_at_label"] = self._time(float(item["completed_at"]))
            items.append(item)
        return {
            "items": items,
            "total": total,
            "limit": bounded_limit,
            "offset": bounded_offset,
            "previous_offset": max(0, bounded_offset - bounded_limit) if bounded_offset else None,
            "next_offset": (
                bounded_offset + bounded_limit if bounded_offset + bounded_limit < total else None
            ),
        }

    async def capability_probe(self, probe_id: str) -> dict[str, Any] | None:
        async with self.store.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT * FROM provider_capability_probes WHERE id=?", (probe_id,))
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        for field in ("configuration", "request_payload", "response_payload"):
            item[field] = redact_sensitive(self._payload(item[field]))
        return item

    async def diagnostic_counts(self) -> dict[str, Any]:
        queries = {
            "conversations": "SELECT COUNT(*) FROM conversations",
            "pending_events": "SELECT COUNT(*) FROM conversation_mailbox WHERE state='pending'",
            "failed_runs": "SELECT COUNT(*) FROM agent_runs WHERE status='failed'",
            "active_jobs": (
                "SELECT COUNT(*) FROM admin_jobs "
                "WHERE status IN ('queued','running','cancellation_requested')"
            ),
            "attachments": "SELECT COUNT(*) FROM attachment_references",
            "memories": "SELECT COUNT(*) FROM langgraph_store_items",
        }
        result: dict[str, Any] = {}
        async with self.store.database.transaction() as connection:
            for name, statement in queries.items():
                result[name] = int((await (await connection.execute(statement)).fetchone())[0])
            schema = await (await connection.execute("SELECT MAX(version) FROM schema_migrations")).fetchone()
            result["schema_version"] = int(schema[0] or 0)
            sqlite_version = await (await connection.execute("SELECT sqlite_version()")).fetchone()
            result["sqlite_version"] = str(sqlite_version[0])
        return result

    async def configuration_versions(self, *, limit: int = 10) -> list[dict[str, Any]]:
        async with self.store.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    "SELECT * FROM agent_configuration_versions ORDER BY created_at DESC, id DESC LIMIT ?",
                    (max(1, min(limit, 50)),),
                )
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["configuration"] = redact_sensitive(self._payload(item["configuration"]))
            item["created_at_label"] = self._time(float(item["created_at"]))
            item["active"] = bool(item["active"])
            result.append(item)
        return result

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
    def _trace(
        cls,
        row,
        payload: Any | None = None,
        *,
        include_payload: bool = False,
    ) -> dict[str, Any]:
        item = dict(row)
        payload = redact_sensitive(cls._payload(item["payload"]) if payload is None else payload)
        item.pop("payload", None)
        item.update(cls._trace_summary(str(item["kind"]), payload))
        if include_payload:
            item["payload"] = payload
        item["recorded_at_label"] = cls._time(float(item["recorded_at"]))
        return item

    @classmethod
    def _delivery(cls, row, *, include_payload: bool = False) -> dict[str, Any]:
        item = dict(row)
        request = redact_sensitive(cls._payload(item.get("payload")))
        result = redact_sensitive(cls._payload(item.get("result")))
        item.pop("payload", None)
        item.pop("result", None)
        item["action"] = str(item.get("kind") or "")
        item["action_id"] = str(item["id"])
        item["summary"] = cls._value_preview(result or request)
        if include_payload:
            item["request"] = request
            item["result"] = result
        return item

    @classmethod
    def _wait(cls, row) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = redact_sensitive(cls._payload(item.get("payload")))
        item["created_at_label"] = cls._time(float(item["created_at"]))
        item["resume_at_label"] = cls._time(float(item["resume_at"]))
        return item

    @classmethod
    def _model_exchanges(cls, traces: list[tuple[Any, Any]]) -> list[dict[str, Any]]:
        exchanges: list[dict[str, Any]] = []
        pending: dict[str, Any] | None = None
        for row, payload in traces:
            kind = str(row["kind"])
            if kind == "model_request":
                if pending is not None:
                    exchanges.append(pending)
                pending = {
                    "request_sequence": int(row["sequence"]),
                    "model": str(payload.get("model_class") or "—") if isinstance(payload, dict) else "—",
                    "message_count": len(payload.get("messages") or []) if isinstance(payload, dict) else 0,
                    "tools": payload.get("tools") or [] if isinstance(payload, dict) else [],
                    "response_sequence": None,
                    "response_summary": "",
                    "failed": False,
                }
            elif kind in {"model_response", "model_error"} and pending is not None:
                pending["response_sequence"] = int(row["sequence"])
                pending["response_summary"] = cls._value_preview(payload)
                pending["failed"] = kind == "model_error"
                exchanges.append(pending)
                pending = None
        if pending is not None:
            exchanges.append(pending)
        return exchanges

    @classmethod
    def _trace_summary(cls, kind: str, payload: Any) -> dict[str, Any]:
        category = (
            "model"
            if kind.startswith("model_")
            else "tool"
            if kind.startswith("tool_")
            or kind == "emulated_actions"
            or kind.startswith("outbound_action_")
            else "state"
            if "checkpoint" in kind
            or "interrupt" in kind
            or kind.startswith("followup_wait_")
            or kind.startswith("public_output_cutover_")
            or kind in {"mailbox_injection", "abandoned_run_reconciled", "public_text_extracted"}
            else "run"
        )
        failed = kind.endswith("error") or kind in {"run_error", "tool_error", "model_error"}
        summary_key = ""
        summary_values: dict[str, object] = {}
        if isinstance(payload, dict):
            if kind == "model_request":
                summary_key = "trace_summary_model_request"
                summary_values = {
                    "messages": len(payload.get("messages") or []),
                    "tools": len(payload.get("tools") or []),
                }
                summary = ""
            elif kind == "run_input":
                summary_key = "trace_summary_run_input"
                summary_values = {"events": len(payload.get("event_ids") or [])}
                summary = ""
            elif kind == "run_output":
                summary_key = "trace_summary_run_output"
                summary_values = {"messages": payload.get("outbound_delivery_count", 0)}
                summary = ""
            elif kind in {"model_error", "tool_error", "run_error", "interrupt_resume_error"}:
                summary = str(payload.get("detail") or payload.get("type") or "")[:240]
            elif kind in {"tool_request", "tool_response"}:
                call = payload.get("tool_call") if isinstance(payload.get("tool_call"), dict) else payload
                summary = str(call.get("name") or cls._value_preview(payload))[:240]
            else:
                summary = cls._value_preview(payload)
        else:
            summary = cls._value_preview(payload)
        return {
            "category": category,
            "failed": failed,
            "summary": summary,
            "summary_key": summary_key,
            "summary_values": summary_values,
        }

    @staticmethod
    def _value_preview(value: Any) -> str:
        if value in (None, "", [], {}):
            return "—"
        if isinstance(value, dict):
            for key in ("detail", "content", "text", "output", "status", "type"):
                candidate = value.get(key)
                if isinstance(candidate, (str, int, float, bool)) and str(candidate):
                    return str(candidate)[:240]
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        return rendered[:237] + "…" if len(rendered) > 240 else rendered

    @staticmethod
    def _payload(value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {"value": value}

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
