import json
from pathlib import Path

import aiosqlite
import pytest
from langgraph.checkpoint.base import empty_checkpoint

from diskovod.persistence import (
    TARGET_MIGRATIONS,
    CheckpointCipher,
    SQLiteLangGraphStore,
    initialize_target_schema,
    open_checkpointer,
)


SECRET = "x" * 32


async def test_target_schema_is_idempotent_and_uses_one_database(tmp_path: Path):
    path = tmp_path / "diskovod.sqlite3"
    async with aiosqlite.connect(path) as connection:
        await initialize_target_schema(connection)
        await initialize_target_schema(connection)

        tables = {
            row[0]
            for row in await (
                await connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
            ).fetchall()
        }
        assert {
            "schema_migrations",
            "chat_threads",
            "agent_runs",
            "langgraph_store_items",
            "attachment_objects",
            "escalation_interrupts",
            "chat_thread_generations",
            "checkpoint_index",
            "admin_jobs",
            "admin_job_events",
            "admin_job_inputs",
            "provider_setup_drafts",
            "conversation_events",
            "agent_work",
            "chat_interaction_policies",
            "outbound_actions",
            "conversation_waits",
        } <= tables
        assert {
            "discord_events",
            "chat_event_queue",
            "side_effect_deliveries",
        }.isdisjoint(tables)
        assert await (await connection.execute("SELECT version FROM schema_migrations")).fetchall() == [
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            (7,),
            (8,),
            (9,),
            (10,),
            (11,),
            (12,),
        ]
        assert (await (await connection.execute("PRAGMA journal_mode")).fetchone())[0] == "wal"


async def test_v11_interaction_state_is_migrated_once_without_legacy_runtime_tables(tmp_path: Path):
    path = tmp_path / "diskovod.sqlite3"
    async with aiosqlite.connect(path) as connection:
        for version, migration in enumerate(TARGET_MIGRATIONS[:11], 1):
            await connection.executescript(migration)
            await connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES(?, 0)", (version,)
            )
        await connection.execute(
            "INSERT INTO conversations VALUES('auto','p1','Peer',0,NULL,1,50,'automatic')"
        )
        await connection.execute(
            "INSERT INTO conversations VALUES('inline','p2','Peer 2',1,2,3,NULL,'inline')"
        )
        await connection.execute("INSERT INTO chat_threads VALUES('inline','owner',1,'thread',0,0,3)")
        await connection.execute(
            """
            INSERT INTO conversation_mailbox(
              id, channel_id, sequence, kind, available_at, observed_at, payload, state, completed_at
            ) VALUES('passive','auto',1,'message',1,1,'{}','completed',1)
            """
        )
        await connection.execute(
            """
            INSERT INTO conversation_mailbox(
              id, channel_id, sequence, kind, available_at, observed_at, payload, state
            ) VALUES('trigger','auto',2,'message',2,2,'{}','pending')
            """
        )
        await connection.commit()

        await initialize_target_schema(connection)
        connection.row_factory = aiosqlite.Row
        tables = {
            row[0]
            for row in await (
                await connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
        assert "conversation_mailbox" not in tables
        assert {"conversation_events", "agent_work", "chat_interaction_policies"} <= tables
        auto = await (
            await connection.execute("SELECT * FROM conversations WHERE channel_id='auto'")
        ).fetchone()
        inline = await (
            await connection.execute("SELECT * FROM conversations WHERE channel_id='inline'")
        ).fetchone()
        assert auto["availability"] == "active" and auto["snoozed_until"] == 50
        assert inline["availability"] == "paused"
        policies = {
            row["channel_id"]: (row["preset"], json.loads(row["active_turn_input"])["participants"])
            for row in await (
                await connection.execute(
                    "SELECT channel_id, preset, active_turn_input FROM chat_interaction_policies"
                )
            ).fetchall()
        }
        assert policies == {
            "auto": ("autonomous", ["peer"]),
            "inline": ("shared", ["owner", "peer"]),
        }
        events = {
            row["id"]: row["context_state"]
            for row in await (
                await connection.execute("SELECT id, context_state FROM conversation_events")
            ).fetchall()
        }
        assert events == {"passive": "unapplied", "trigger": "unapplied"}
        assert (await (await connection.execute("SELECT COUNT(*) FROM agent_work")).fetchone())[0] == 1


async def test_schema_migration_removes_unsupported_subscription_token_limit(tmp_path: Path):
    path = tmp_path / "diskovod.sqlite3"
    async with aiosqlite.connect(path) as connection:
        await initialize_target_schema(connection)
        await connection.execute("DELETE FROM schema_migrations WHERE version=6")
        await connection.execute(
            "INSERT INTO agent_configuration_versions(created_at, configuration, active) VALUES(0, ?, 1)",
            (
                json.dumps(
                    {
                        "provider_id": "chatgpt_subscription",
                        "model_id": "test-model",
                        "transport_profile": "responses",
                        "credential_profile": "chatgpt_subscription",
                        "options": {"reasoning_effort": "low", "max_completion_tokens": 256},
                        "capabilities": {"native_tools": True},
                    }
                ),
            ),
        )
        await connection.commit()

        await initialize_target_schema(connection)

        saved = json.loads(
            (
                await (
                    await connection.execute("SELECT configuration FROM agent_configuration_versions")
                ).fetchone()
            )[0]
        )
        assert saved["options"] == {"reasoning_effort": "low"}
        assert saved["capabilities"]["output_token_limit"] is False


async def test_sqlite_langgraph_store_supports_the_async_api(tmp_path: Path):
    store = SQLiteLangGraphStore(tmp_path / "diskovod.sqlite3")
    namespace = ("chat", "account-1", "channel-1", "memory")

    await store.aput(namespace, "preference", {"text": "Prefers decaf coffee", "kind": "preference"})
    await store.aput(
        namespace,
        "fact",
        {"text": "Lives in Berlin", "kind": "fact", "confidence": 0.9},
        index=["$.text"],
    )
    await store.aput(("chat", "account-1", "channel-2", "memory"), "other", {"text": "Other chat"})

    assert (await store.aget(namespace, "preference")).value["text"] == "Prefers decaf coffee"
    assert [item.key for item in await store.asearch(namespace, filter={"kind": "fact"})] == ["fact"]
    assert [item.key for item in await store.asearch(namespace, filter={"confidence": {"$gte": 0.8}})] == [
        "fact"
    ]
    assert [item.key for item in await store.asearch(namespace, query="decaf coffee")] == ["preference"]
    assert await store.alist_namespaces(prefix=("chat", "account-1"), max_depth=3) == [
        ("chat", "account-1", "channel-1"),
        ("chat", "account-1", "channel-2"),
    ]

    await store.adelete(namespace, "fact")
    assert await store.aget(namespace, "fact") is None
    await store.database.close()


async def test_async_langgraph_store_initializes_only_the_async_database(tmp_path: Path):
    store = SQLiteLangGraphStore(tmp_path / "diskovod.sqlite3")
    namespace = ("chat", "account", "channel", "memory")

    assert store._schema_ready is False
    await store.aput(namespace, "key", {"text": "async only"})
    assert (await store.aget(namespace, "key")).value == {"text": "async only"}
    assert store._schema_ready is True

    await store.database.close()


def test_langgraph_store_explicitly_rejects_the_sync_api(tmp_path: Path):
    store = SQLiteLangGraphStore(tmp_path / "diskovod.sqlite3")

    with pytest.raises(NotImplementedError, match="only the asynchronous API"):
        store.batch([])


def test_checkpoint_cipher_rejects_tampering_and_wrong_context():
    cipher = CheckpointCipher(SECRET)
    name, encrypted = cipher.encrypt(b"private checkpoint")

    assert b"private checkpoint" not in encrypted
    assert cipher.decrypt(name, encrypted) == b"private checkpoint"
    with pytest.raises(Exception):
        cipher.decrypt(name, encrypted[:-1] + bytes([encrypted[-1] ^ 1]))
    with pytest.raises(ValueError, match="Unsupported checkpoint cipher"):
        cipher.decrypt("other", encrypted)


@pytest.mark.asyncio
async def test_async_checkpointer_encrypts_payloads_in_shared_database(tmp_path: Path):
    path = tmp_path / "diskovod.sqlite3"
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"secret": "private conversation"}
    config = {"configurable": {"thread_id": "discord:account:channel:g1", "checkpoint_ns": ""}}

    async with open_checkpointer(path, SECRET) as saver:
        saved = await saver.aput(config, checkpoint, {"source": "input", "step": 0, "parents": {}}, {})
        restored = await saver.aget(saved)

    assert restored["channel_values"]["secret"] == "private conversation"
    assert b"private conversation" not in path.read_bytes()
