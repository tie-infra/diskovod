from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import aiosqlite
from pydantic import BaseModel, ConfigDict

from .persistence import AsyncSQLite

JOB_STATES = frozenset({"queued", "running", "cancellation_requested", "succeeded", "failed", "cancelled"})
ACTIVE_JOB_STATES = frozenset({"queued", "running", "cancellation_requested"})


@dataclass(frozen=True, slots=True)
class JobResult:
    kind: str | None = None
    id: str | None = None


@dataclass(frozen=True, slots=True)
class JobDefinition:
    payload_model: type[BaseModel]
    handler: Callable[[AdminJobContext, BaseModel], Awaitable[JobResult | None]]
    retryable: bool = True
    cancellable: bool = True
    max_concurrency: int = 1


class EmptyJobPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdminJobRepository:
    def __init__(self, database: AsyncSQLite):
        self.database = database

    async def enqueue(
        self,
        *,
        job_type: str,
        schema_version: int,
        payload: Mapping[str, Any],
        idempotency_key: str | None = None,
        target_kind: str | None = None,
        target_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        job_id = str(uuid.uuid4())
        now = time.time()
        encoded = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
        async with self.database.transaction() as connection:
            if idempotency_key:
                existing = await self._active_by_idempotency(connection, idempotency_key)
                if existing is not None:
                    return self._job(existing), False
            try:
                await connection.execute(
                    """
                    INSERT INTO admin_jobs(
                      id, type, schema_version, status, idempotency_key, requested_at,
                      input_payload, target_kind, target_id
                    ) VALUES(?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        job_type,
                        schema_version,
                        idempotency_key,
                        now,
                        encoded,
                        target_kind,
                        target_id,
                    ),
                )
            except aiosqlite.IntegrityError:
                if not idempotency_key:
                    raise
                existing = await self._active_by_idempotency(connection, idempotency_key)
                if existing is None:
                    raise
                return self._job(existing), False
            await self._append_event(connection, job_id, "queued", {})
            row = await self._get(connection, job_id)
            assert row is not None
            return self._job(row), True

    async def claim(
        self, owner: str, lease_seconds: float, *, allowed_types: set[str] | None = None
    ) -> dict[str, Any] | None:
        now = time.time()
        if allowed_types is not None and not allowed_types:
            return None
        where = "status='queued'"
        parameters: list[Any] = []
        if allowed_types is not None:
            placeholders = ",".join("?" for _ in allowed_types)
            where += f" AND type IN ({placeholders})"
            parameters.extend(sorted(allowed_types))
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    f"SELECT * FROM admin_jobs WHERE {where} ORDER BY requested_at, id LIMIT 1",
                    parameters,
                )
            ).fetchone()
            if row is None:
                return None
            cursor = await connection.execute(
                """
                UPDATE admin_jobs
                SET status='running', started_at=COALESCE(started_at, ?),
                    lease_owner=?, lease_expires_at=?, attempt_count=attempt_count+1
                WHERE id=? AND status='queued'
                """,
                (now, owner, now + lease_seconds, row["id"]),
            )
            if cursor.rowcount != 1:
                return None
            await self._append_event(
                connection,
                str(row["id"]),
                "started",
                {"attempt": int(row["attempt_count"]) + 1},
            )
            claimed = await self._get(connection, str(row["id"]))
            assert claimed is not None
            return self._job(claimed)

    async def renew(self, job_id: str, owner: str, lease_seconds: float) -> bool:
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE admin_jobs SET lease_expires_at=?
                WHERE id=? AND lease_owner=? AND status IN ('running','cancellation_requested')
                """,
                (time.time() + lease_seconds, job_id, owner),
            )
            return cursor.rowcount == 1

    async def progress(
        self,
        job_id: str,
        owner: str,
        stage: str,
        *,
        current: int | None = None,
        total: int | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> bool:
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE admin_jobs
                SET progress_stage=?, progress_current=?, progress_total=?
                WHERE id=? AND lease_owner=? AND status IN ('running','cancellation_requested')
                """,
                (stage, current, total, job_id, owner),
            )
            if cursor.rowcount != 1:
                return False
            await self._append_event(
                connection,
                job_id,
                "progress",
                {"stage": stage, "current": current, "total": total, **dict(detail or {})},
            )
            return True

    async def succeed(self, job_id: str, owner: str, result: JobResult | None) -> bool:
        now = time.time()
        result = result or JobResult()
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE admin_jobs
                SET status='succeeded', completed_at=?, lease_owner=NULL, lease_expires_at=NULL,
                    result_kind=?, result_id=?, error_code=NULL, error_summary=NULL
                WHERE id=? AND lease_owner=? AND status IN ('running','cancellation_requested')
                """,
                (now, result.kind, result.id, job_id, owner),
            )
            if cursor.rowcount != 1:
                return False
            await self._append_event(
                connection,
                job_id,
                "succeeded",
                {"result_kind": result.kind, "result_id": result.id},
            )
            return True

    async def fail(self, job_id: str, owner: str, code: str, summary: str) -> bool:
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE admin_jobs
                SET status='failed', completed_at=?, lease_owner=NULL, lease_expires_at=NULL,
                    error_code=?, error_summary=?
                WHERE id=? AND lease_owner=? AND status IN ('running','cancellation_requested')
                """,
                (time.time(), code[:200], summary[:4000], job_id, owner),
            )
            if cursor.rowcount != 1:
                return False
            await self._append_event(
                connection,
                job_id,
                "failed",
                {"error_code": code[:200], "error_summary": summary[:4000]},
            )
            return True

    async def mark_cancelled(self, job_id: str, owner: str) -> bool:
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE admin_jobs
                SET status='cancelled', completed_at=?, lease_owner=NULL, lease_expires_at=NULL
                WHERE id=? AND lease_owner=? AND status='cancellation_requested'
                """,
                (time.time(), job_id, owner),
            )
            if cursor.rowcount != 1:
                return False
            await self._append_event(connection, job_id, "cancelled", {})
            return True

    async def request_cancel(self, job_id: str, *, cancellable: bool) -> dict[str, Any] | None:
        now = time.time()
        async with self.database.transaction() as connection:
            row = await self._get(connection, job_id)
            if row is None:
                return None
            status = str(row["status"])
            if status == "queued" and cancellable:
                await connection.execute(
                    "UPDATE admin_jobs SET status='cancelled', completed_at=?, "
                    "cancellation_requested_at=? WHERE id=? AND status='queued'",
                    (now, now, job_id),
                )
                await self._append_event(connection, job_id, "cancelled", {"before_start": True})
            elif status == "running" and cancellable:
                await connection.execute(
                    "UPDATE admin_jobs SET status='cancellation_requested', "
                    "cancellation_requested_at=? WHERE id=? AND status='running'",
                    (now, job_id),
                )
                await self._append_event(connection, job_id, "cancellation_requested", {})
            updated = await self._get(connection, job_id)
            return self._job(updated) if updated is not None else None

    async def recover_expired(self, retryable_types: set[str]) -> int:
        now = time.time()
        recovered = 0
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT id, type FROM admin_jobs
                    WHERE status IN ('running','cancellation_requested')
                      AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                    """,
                    (now,),
                )
            ).fetchall()
            for row in rows:
                job_id = str(row["id"])
                if str(row["type"]) in retryable_types:
                    await connection.execute(
                        """
                        UPDATE admin_jobs SET status='queued', lease_owner=NULL, lease_expires_at=NULL,
                          progress_stage='recovered_after_restart'
                        WHERE id=?
                        """,
                        (job_id,),
                    )
                    await self._append_event(connection, job_id, "recovered", {})
                else:
                    await connection.execute(
                        """
                        UPDATE admin_jobs SET status='failed', completed_at=?, lease_owner=NULL,
                          lease_expires_at=NULL, error_code='lease_expired',
                          error_summary='The worker stopped before this non-retryable job completed.'
                        WHERE id=?
                        """,
                        (now, job_id),
                    )
                    await self._append_event(connection, job_id, "failed", {"error_code": "lease_expired"})
                recovered += 1
        return recovered

    async def get(self, job_id: str) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await self._get(connection, job_id)
        return self._job(row) if row is not None else None

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        parameters: list[Any] = []
        where = ""
        if status:
            if status not in JOB_STATES:
                raise ValueError("Unknown administrative job state")
            where = " WHERE status=?"
            parameters.append(status)
        parameters.extend((max(1, min(limit, 500)), max(0, offset)))
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    f"SELECT * FROM admin_jobs{where} ORDER BY requested_at DESC, id DESC LIMIT ? OFFSET ?",
                    parameters,
                )
            ).fetchall()
        return [self._job(row) for row in rows]

    async def count(self, *, status: str | None = None) -> int:
        parameters: list[Any] = []
        where = ""
        if status:
            if status not in JOB_STATES:
                raise ValueError("Unknown administrative job state")
            where = " WHERE status=?"
            parameters.append(status)
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(f"SELECT COUNT(*) FROM admin_jobs{where}", parameters)
            ).fetchone()
        return int(row[0])

    async def events(self, job_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    "SELECT * FROM admin_job_events WHERE job_id=? AND sequence>? ORDER BY sequence",
                    (job_id, max(0, after)),
                )
            ).fetchall()
        return [self._event(row) for row in rows]

    async def active_count(self) -> int:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT COUNT(*) FROM admin_jobs "
                    "WHERE status IN ('queued','running','cancellation_requested')"
                )
            ).fetchone()
        return int(row[0])

    async def cancellation_requested(self, job_id: str, owner: str) -> bool:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT status FROM admin_jobs WHERE id=? AND lease_owner=?",
                    (job_id, owner),
                )
            ).fetchone()
        return bool(row and row["status"] == "cancellation_requested")

    @staticmethod
    async def _active_by_idempotency(connection: aiosqlite.Connection, key: str) -> aiosqlite.Row | None:
        return await (
            await connection.execute(
                "SELECT * FROM admin_jobs WHERE idempotency_key=? "
                "AND status IN ('queued','running','cancellation_requested') LIMIT 1",
                (key,),
            )
        ).fetchone()

    @staticmethod
    async def _get(connection: aiosqlite.Connection, job_id: str) -> aiosqlite.Row | None:
        return await (await connection.execute("SELECT * FROM admin_jobs WHERE id=?", (job_id,))).fetchone()

    @staticmethod
    async def _append_event(
        connection: aiosqlite.Connection,
        job_id: str,
        kind: str,
        payload: Mapping[str, Any],
    ) -> None:
        row = await (
            await connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM admin_job_events WHERE job_id=?",
                (job_id,),
            )
        ).fetchone()
        await connection.execute(
            "INSERT INTO admin_job_events(job_id, sequence, occurred_at, kind, payload) "
            "VALUES(?, ?, ?, ?, ?)",
            (
                job_id,
                int(row[0]),
                time.time(),
                kind,
                json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")),
            ),
        )

    @staticmethod
    def _job(row: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["input_payload"] = json.loads(str(result["input_payload"]))
        return result

    @staticmethod
    def _event(row: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(str(result["payload"]))
        return result


class AdminJobContext:
    def __init__(self, repository: AdminJobRepository, job_id: str, owner: str, lease_seconds: float):
        self.repository = repository
        self.job_id = job_id
        self.owner = owner
        self.lease_seconds = lease_seconds

    async def progress(
        self,
        stage: str,
        *,
        current: int | None = None,
        total: int | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if await self.cancelled():
            raise asyncio.CancelledError
        if not await self.repository.progress(
            self.job_id,
            self.owner,
            stage,
            current=current,
            total=total,
            detail=detail,
        ):
            raise RuntimeError("Administrative job lease was lost")
        await self.repository.renew(self.job_id, self.owner, self.lease_seconds)

    async def cancelled(self) -> bool:
        return await self.repository.cancellation_requested(self.job_id, self.owner)

    async def checkpoint(self) -> None:
        if await self.cancelled():
            raise asyncio.CancelledError
        if not await self.repository.renew(self.job_id, self.owner, self.lease_seconds):
            raise RuntimeError("Administrative job lease was lost")


class AdminJobService:
    def __init__(self, repository: AdminJobRepository):
        self.repository = repository
        self.definitions: dict[str, JobDefinition] = {}
        self._wake: Callable[[], None] | None = None

    def register(self, job_type: str, definition: JobDefinition) -> None:
        if not job_type or job_type in self.definitions:
            raise ValueError(f"Administrative job type is already registered: {job_type}")
        if definition.max_concurrency < 1:
            raise ValueError("Job concurrency must be positive")
        self.definitions[job_type] = definition

    async def enqueue(
        self,
        job_type: str,
        payload: BaseModel | Mapping[str, Any],
        *,
        schema_version: int = 1,
        idempotency_key: str | None = None,
        target_kind: str | None = None,
        target_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        definition = self._definition(job_type)
        validated = definition.payload_model.model_validate(payload)
        job, created = await self.repository.enqueue(
            job_type=job_type,
            schema_version=schema_version,
            payload=validated.model_dump(mode="json"),
            idempotency_key=idempotency_key,
            target_kind=target_kind,
            target_id=target_id,
        )
        if created and self._wake is not None:
            self._wake()
        return job, created

    async def cancel(self, job_id: str) -> dict[str, Any] | None:
        job = await self.repository.get(job_id)
        if job is None:
            return None
        definition = self._definition(str(job["type"]))
        updated = await self.repository.request_cancel(job_id, cancellable=definition.cancellable)
        if self._wake is not None:
            self._wake()
        return updated

    async def get(self, job_id: str) -> dict[str, Any] | None:
        return await self.repository.get(job_id)

    def _definition(self, job_type: str) -> JobDefinition:
        try:
            return self.definitions[job_type]
        except KeyError as error:
            raise ValueError(f"Unknown administrative job type: {job_type}") from error


class AdminJobWorker:
    def __init__(
        self,
        service: AdminJobService,
        *,
        concurrency: int = 2,
        lease_seconds: float = 30.0,
        idle_poll_seconds: float = 2.0,
    ):
        if concurrency < 1:
            raise ValueError("Worker concurrency must be positive")
        self.service = service
        self.repository = service.repository
        self.concurrency = concurrency
        self.lease_seconds = lease_seconds
        self.idle_poll_seconds = idle_poll_seconds
        self.owner = f"embedded:{uuid.uuid4()}"
        self._wake_event = asyncio.Event()
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        self._loop_task: asyncio.Task[None] | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._task_types: dict[str, str] = {}
        self._stopping = False

    async def start(self) -> None:
        if self._loop_task is not None:
            return
        self._stopping = False
        retryable = {name for name, value in self.service.definitions.items() if value.retryable}
        await self.repository.recover_expired(retryable)
        self.service._wake = self.wake
        self._loop_task = asyncio.create_task(self._run_loop(), name="admin-job-worker")
        self.wake()

    async def close(self) -> None:
        self._stopping = True
        self.service._wake = None
        self.wake()
        if self._loop_task is not None:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
            self._loop_task = None
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._task_types.clear()

    def wake(self) -> None:
        self._idle_event.clear()
        self._wake_event.set()

    async def wait_idle(self) -> None:
        await self._idle_event.wait()

    async def _run_loop(self) -> None:
        while not self._stopping:
            await self._reap_and_cancel()
            claimed_any = False
            while len(self._tasks) < self.concurrency:
                type_counts = {
                    job_type: sum(1 for value in self._task_types.values() if value == job_type)
                    for job_type in self.service.definitions
                }
                allowed_types = {
                    job_type
                    for job_type, definition in self.service.definitions.items()
                    if type_counts[job_type] < definition.max_concurrency
                }
                job = await self.repository.claim(
                    self.owner,
                    self.lease_seconds,
                    allowed_types=allowed_types,
                )
                if job is None:
                    break
                job_type = str(job["type"])
                definition = self.service._definition(job_type)
                task = asyncio.create_task(self._execute(job, definition), name=f"admin-job-{job['id']}")
                self._tasks[str(job["id"])] = task
                self._task_types[str(job["id"])] = job_type
                claimed_any = True
            if claimed_any:
                await asyncio.sleep(0)
                continue
            if not self._tasks and await self.repository.active_count() == 0:
                self._idle_event.set()
            else:
                self._idle_event.clear()
            self._wake_event.clear()
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=self.idle_poll_seconds)
            except TimeoutError:
                pass

    async def _reap_and_cancel(self) -> None:
        for job_id, task in list(self._tasks.items()):
            if task.done():
                self._tasks.pop(job_id, None)
                self._task_types.pop(job_id, None)
                await asyncio.gather(task, return_exceptions=True)
                continue
            if await self.repository.cancellation_requested(job_id, self.owner):
                task.cancel()

    async def _execute(self, job: dict[str, Any], definition: JobDefinition) -> None:
        job_id = str(job["id"])
        context = AdminJobContext(self.repository, job_id, self.owner, self.lease_seconds)
        heartbeat = asyncio.create_task(self._heartbeat(job_id), name=f"admin-job-heartbeat-{job_id}")
        try:
            payload = definition.payload_model.model_validate(job["input_payload"])
            result = await definition.handler(context, payload)
        except asyncio.CancelledError:
            if not self._stopping:
                await self.repository.mark_cancelled(job_id, self.owner)
            raise
        except Exception as error:
            await self.repository.fail(
                job_id,
                self.owner,
                type(error).__name__,
                f"{type(error).__name__}: {error}",
            )
        else:
            await self.repository.succeed(job_id, self.owner, result)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            self.wake()

    async def _heartbeat(self, job_id: str) -> None:
        interval = max(0.1, self.lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            if not await self.repository.renew(job_id, self.owner, self.lease_seconds):
                return
