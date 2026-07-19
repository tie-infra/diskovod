import json
import sqlite3
from pathlib import Path

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from diskovod.persistence import (
    CheckpointCipher,
    SQLiteLangGraphStore,
    initialize_target_schema,
    open_checkpointer,
)


SECRET = "x" * 32


def test_target_schema_is_idempotent_and_uses_one_database(tmp_path: Path):
    path = tmp_path / "diskovod.sqlite3"
    connection = sqlite3.connect(path)

    with connection:
        initialize_target_schema(connection)
        initialize_target_schema(connection)

    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
    }
    assert {
        "schema_migrations",
        "chat_threads",
        "discord_events",
        "chat_event_queue",
        "side_effect_deliveries",
        "agent_runs",
        "langgraph_store_items",
        "attachment_objects",
        "escalation_interrupts",
    } <= tables
    assert connection.execute("SELECT version FROM schema_migrations").fetchall() == [
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
    ]
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    connection.close()


def test_schema_migration_removes_unsupported_subscription_token_limit(tmp_path: Path):
    path = tmp_path / "diskovod.sqlite3"
    connection = sqlite3.connect(path)
    with connection:
        initialize_target_schema(connection)
        connection.execute("DELETE FROM schema_migrations WHERE version=6")
        connection.execute(
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

    with connection:
        initialize_target_schema(connection)

    saved = json.loads(
        connection.execute("SELECT configuration FROM agent_configuration_versions").fetchone()[0]
    )
    assert saved["options"] == {"reasoning_effort": "low"}
    assert saved["capabilities"]["output_token_limit"] is False
    connection.close()


def test_sqlite_langgraph_store_conforms_to_sync_and_async_api(tmp_path: Path):
    store = SQLiteLangGraphStore(tmp_path / "diskovod.sqlite3")
    namespace = ("chat", "account-1", "channel-1", "memory")

    store.put(namespace, "preference", {"text": "Prefers decaf coffee", "kind": "preference"})
    store.put(
        namespace,
        "fact",
        {"text": "Lives in Berlin", "kind": "fact", "confidence": 0.9},
        index=["$.text"],
    )
    store.put(("chat", "account-1", "channel-2", "memory"), "other", {"text": "Other chat"})

    assert store.get(namespace, "preference").value["text"] == "Prefers decaf coffee"
    assert [item.key for item in store.search(namespace, filter={"kind": "fact"})] == ["fact"]
    assert [item.key for item in store.search(namespace, filter={"confidence": {"$gte": 0.8}})] == ["fact"]
    assert [item.key for item in store.search(namespace, query="decaf coffee")] == ["preference"]
    assert store.list_namespaces(prefix=("chat", "account-1"), max_depth=3) == [
        ("chat", "account-1", "channel-1"),
        ("chat", "account-1", "channel-2"),
    ]

    async def exercise_async_api():
        await store.aput(namespace, "async", {"text": "Stored asynchronously"})
        assert (await store.aget(namespace, "async")).value["text"] == "Stored asynchronously"
        await store.adelete(namespace, "async")
        assert await store.aget(namespace, "async") is None
        await store.database.close()

    import asyncio

    asyncio.run(exercise_async_api())
    store.delete(namespace, "fact")
    assert store.get(namespace, "fact") is None
    store.close()


async def test_async_langgraph_store_does_not_open_sync_compatibility_connection(tmp_path: Path):
    store = SQLiteLangGraphStore(tmp_path / "diskovod.sqlite3")
    namespace = ("chat", "account", "channel", "memory")

    assert store._connection is None
    await store.aput(namespace, "key", {"text": "async only"})
    assert (await store.aget(namespace, "key")).value == {"text": "async only"}
    assert store._connection is None

    await store.database.close()
    store.close()


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
