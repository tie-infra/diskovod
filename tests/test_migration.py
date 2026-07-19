from __future__ import annotations

import time
from pathlib import Path

import pytest

from diskovod.migration import LegacyMigrator
from diskovod.runtime import AgentService
from diskovod.store import Store

from test_agent import ScriptedChatModel
from test_runtime import FakeModels, RecordingTransport


@pytest.mark.asyncio
async def test_cutover_migration_backs_up_audits_and_seeds_each_chat_once(tmp_path: Path):
    store = Store(tmp_path / "diskovod.sqlite3", "x" * 32)
    store.upsert_conversation("channel", "peer", "Peer")
    store.save_message(
        id="peer-message",
        channel_id="channel",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="Do you remember this?",
        timestamp=time.time(),
    )
    store.save_message(
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
    )
    await runtime.start()
    migrator = LegacyMigrator(store, runtime)

    report = await migrator.run()

    assert report.backup_path is not None and report.backup_path.exists()
    assert report.backup_path.with_suffix(".manifest.json").exists()
    assert report.conversations == 1
    assert report.events == 2
    assert report.checkpoints == 1
    assert store._db.execute("SELECT COUNT(*) FROM discord_events").fetchone()[0] == 2
    assert store._db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0
    tables = {
        row[0] for row in store._db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "model_request_logs" not in tables
    assert "chatgpt_usage" not in tables
    assert "conversation_escalations" not in tables
    thread_id = runtime.events.thread_id("discord-owner", "channel")
    checkpoint = await runtime.checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    assert [message.content for message in checkpoint.checkpoint["channel_values"]["messages"]] == [
        "Do you remember this?",
        "Yes.",
    ]

    second = await migrator.run()
    assert second.backup_path is None
    assert len(list((tmp_path / "backups").glob("*.sqlite3"))) == 1
    await runtime.close()
    store.close()


@pytest.mark.asyncio
async def test_cutover_converts_active_legacy_escalation_to_real_interrupt(tmp_path: Path):
    store = Store(tmp_path / "diskovod.sqlite3", "x" * 32)
    store.upsert_conversation("channel", "peer", "Peer")
    store.save_message(
        id="trigger",
        channel_id="channel",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="Please get the owner",
        timestamp=time.time(),
    )
    with store._db:
        store._db.executescript(
            """
            CREATE TABLE conversation_escalations (
              id INTEGER PRIMARY KEY AUTOINCREMENT, channel_id TEXT NOT NULL,
              trigger_message_id TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
              reason TEXT NOT NULL, requested_at REAL NOT NULL,
              acknowledged_at REAL, resolved_at REAL, delivery_error TEXT
            );
            """
        )
        store._db.execute(
            "INSERT INTO conversation_escalations(channel_id, trigger_message_id, state, reason, requested_at, acknowledged_at) VALUES('channel', 'trigger', 'claimed', 'peer_requested_owner', ?, ?)",
            (time.time(), time.time()),
        )
    runtime = AgentService(
        store,
        FakeModels(ScriptedChatModel(responses=[])),
        RecordingTransport(),
        "x" * 32,
    )
    await runtime.start()

    report = await LegacyMigrator(store, runtime).run()

    assert report.archived_records == 1
    interrupt = store.active_interrupts()[0]
    assert interrupt["state"] == "claimed"
    assert interrupt["payload"]["trigger_message_id"] == "trigger"
    assert (
        store._db.execute("SELECT status FROM agent_runs WHERE id='migration-escalation-run:1'").fetchone()[
            "status"
        ]
        == "interrupted"
    )
    await runtime.close()
    store.close()
