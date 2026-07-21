from __future__ import annotations

import time
from pathlib import Path

import pytest

from diskovod.migration import LegacyMigrator
from diskovod.runtime import AgentService
from diskovod.store import Store

from test_agent import ScriptedChatModel, UnusedPublicHTTP
from test_runtime import FakeModels, RecordingTransport


@pytest.mark.asyncio
async def test_cutover_migration_backs_up_audits_and_seeds_each_chat_once(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aupsert_conversation("channel", "peer", "Peer")
    await store.asave_message(
        id="peer-message",
        channel_id="channel",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="Do you remember this?",
        timestamp=time.time(),
    )
    await store.asave_message(
        id="owner-message",
        channel_id="channel",
        author_id="owner",
        author_name="Owner",
        direction="out",
        source="human",
        content="Yes.",
        timestamp=time.time() + 1,
    )
    runtime = AgentService(
        store,
        FakeModels(ScriptedChatModel(responses=[])),
        RecordingTransport(),
        "x" * 32,
        UnusedPublicHTTP(),
    )
    await runtime.start()
    migrator = LegacyMigrator(store, runtime)

    report = await migrator.run()

    assert report.backup_path is not None and report.backup_path.exists()
    assert report.backup_path.with_suffix(".manifest.json").exists()
    assert report.conversations == 1
    assert report.events == 2
    assert report.checkpoints == 1
    async with store.database.transaction() as connection:
        event_count = (
            await (
                await connection.execute(
                    "SELECT COUNT(*) FROM conversation_events WHERE id LIKE 'legacy:message:%'"
                )
            ).fetchone()
        )[0]
        checkpoint_count = (await (await connection.execute("SELECT COUNT(*) FROM checkpoints")).fetchone())[
            0
        ]
        table_rows = await (
            await connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
        cursor = await (
            await connection.execute(
                "SELECT applied_event_sequence FROM chat_threads WHERE channel_id='channel'"
            )
        ).fetchone()
    assert event_count == 2
    assert checkpoint_count > 0
    assert cursor[0] == 2
    tables = {row[0] for row in table_rows}
    assert "model_request_logs" not in tables
    assert "chatgpt_usage" not in tables
    assert "conversation_escalations" not in tables
    thread_id = await runtime.journal.thread_id("discord-owner", "channel")
    checkpoint = await runtime.checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    assert [message.content for message in checkpoint.checkpoint["channel_values"]["messages"]] == [
        "Do you remember this?",
        "Yes.",
    ]

    second = await migrator.run()
    assert second.backup_path is None
    assert len(list((tmp_path / "backups").glob("*.sqlite3"))) == 1
    await runtime.close()
    await store.aclose()


@pytest.mark.asyncio
async def test_cutover_converts_active_legacy_escalation_to_real_interrupt(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aupsert_conversation("channel", "peer", "Peer")
    await store.asave_message(
        id="trigger",
        channel_id="channel",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="Please get the owner",
        timestamp=time.time(),
    )
    async with store.database.transaction() as connection:
        await connection.executescript(
            """
            CREATE TABLE conversation_escalations (
              id INTEGER PRIMARY KEY AUTOINCREMENT, channel_id TEXT NOT NULL,
              trigger_message_id TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
              reason TEXT NOT NULL, requested_at REAL NOT NULL,
              acknowledged_at REAL, resolved_at REAL, delivery_error TEXT
            );
            """
        )
        await connection.execute(
            "INSERT INTO conversation_escalations(channel_id, trigger_message_id, state, reason, requested_at, acknowledged_at) VALUES('channel', 'trigger', 'claimed', 'peer_requested_owner', ?, ?)",
            (time.time(), time.time()),
        )
    runtime = AgentService(
        store,
        FakeModels(ScriptedChatModel(responses=[])),
        RecordingTransport(),
        "x" * 32,
        UnusedPublicHTTP(),
    )
    await runtime.start()

    report = await LegacyMigrator(store, runtime).run()

    assert report.archived_records == 1
    interrupt = (await store.aactive_interrupts())[0]
    assert interrupt["state"] == "claimed"
    assert interrupt["payload"]["trigger_message_id"] == "trigger"
    assert interrupt["payload"]["migrated"] is True
    await runtime.close()
    await store.aclose()
