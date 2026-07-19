from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from diskovod.admin_jobs import (
    AdminJobRepository,
    AdminJobService,
    AdminJobWorker,
    JobDefinition,
    JobResult,
)
from diskovod.store import Store


class ProbePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


async def test_jobs_are_durable_idempotent_and_record_progress(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    repository = AdminJobRepository(store.database)
    service = AdminJobService(repository)

    async def handler(context, payload):
        assert payload.value == "ready"
        await context.progress("testing", current=1, total=1)
        return JobResult("probe", "probe-1")

    service.register("test.probe", JobDefinition(ProbePayload, handler))
    worker = AdminJobWorker(service, idle_poll_seconds=0.05)
    await worker.start()
    first, created = await service.enqueue(
        "test.probe",
        {"value": "ready"},
        idempotency_key="same-probe",
        target_kind="model",
        target_id="configuration-1",
    )
    duplicate, duplicate_created = await service.enqueue(
        "test.probe", {"value": "ready"}, idempotency_key="same-probe"
    )
    assert created is True
    assert duplicate_created is False
    assert duplicate["id"] == first["id"]

    await asyncio.wait_for(worker.wait_idle(), timeout=2)
    completed = await service.get(str(first["id"]))
    assert completed["status"] == "succeeded"
    assert completed["result_kind"] == "probe"
    assert completed["result_id"] == "probe-1"
    assert [event["kind"] for event in await repository.events(str(first["id"]))] == [
        "queued",
        "started",
        "progress",
        "succeeded",
    ]
    await worker.close()
    await store.aclose()


async def test_running_job_can_be_cancelled_without_request_lifetime(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    repository = AdminJobRepository(store.database)
    service = AdminJobService(repository)
    started = asyncio.Event()

    async def handler(context, payload):
        del context, payload
        started.set()
        await asyncio.Future()

    service.register("test.wait", JobDefinition(ProbePayload, handler))
    worker = AdminJobWorker(service, idle_poll_seconds=0.05)
    await worker.start()
    job, _ = await service.enqueue("test.wait", {"value": "wait"})
    await asyncio.wait_for(started.wait(), timeout=2)
    requested = await service.cancel(str(job["id"]))
    assert requested["status"] == "cancellation_requested"

    await asyncio.wait_for(worker.wait_idle(), timeout=2)
    cancelled = await service.get(str(job["id"]))
    assert cancelled["status"] == "cancelled"
    await worker.close()
    await store.aclose()


async def test_expired_leases_requeue_only_retryable_work(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    repository = AdminJobRepository(store.database)
    retryable, _ = await repository.enqueue(job_type="retryable", schema_version=1, payload={"value": "one"})
    unsafe, _ = await repository.enqueue(job_type="unsafe", schema_version=1, payload={"value": "two"})
    await repository.claim("dead-worker", -1)
    await repository.claim("dead-worker", -1)

    assert await repository.recover_expired({"retryable"}) == 2
    assert (await repository.get(str(retryable["id"])))["status"] == "queued"
    failed = await repository.get(str(unsafe["id"]))
    assert failed["status"] == "failed"
    assert failed["error_code"] == "lease_expired"
    await store.aclose()
