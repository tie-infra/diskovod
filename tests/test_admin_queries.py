from __future__ import annotations

from pathlib import Path
import json

from diskovod.admin_queries import AdminQueryService
from diskovod.redaction import REDACTED, redact_sensitive
from diskovod.store import Store


async def test_chat_projection_is_bounded_and_supports_loading_older_messages(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    await store.aupsert_conversation("channel", "peer", "Peer")
    for index in range(1, 61):
        await store.asave_message(
            id=f"message-{index:02}",
            channel_id="channel",
            author_id="peer" if index % 2 else "owner",
            author_name="Peer" if index % 2 else "Owner",
            direction="in" if index % 2 else "out",
            source="remote" if index % 2 else "human",
            content=f"message {index}",
            timestamp=float(index),
        )
    queries = AdminQueryService(store)

    chat = await queries.chat("channel")
    assert [item["id"] for item in chat["messages"]] == [
        f"message-{index:02}" for index in range(11, 61)
    ]
    assert chat["messages"][0]["role"] == "peer"
    assert chat["messages"][1]["role"] == "owner"
    assert chat["older_messages_before"] == 11

    older = await queries.messages("channel", before=chat["older_messages_before"], limit=50)
    assert [item["id"] for item in older["items"]] == [
        f"message-{index:02}" for index in range(1, 11)
    ]
    assert older["next_before"] is None
    await store.aclose()


async def test_run_projection_redacts_nested_credentials_but_preserves_usage_counts(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    await store.astart_agent_run(
        run_id="run",
        thread_id="thread",
        channel_id="channel",
        trace_id="trace",
    )
    await store.arecord_agent_trace(
        "run",
        "model_request",
        {
            "headers": {"Authorization": "Bearer private", "x-request-id": "visible"},
            "api_key": "private-key",
            "usage": {"input_tokens": 12, "output_tokens": 3},
        },
    )

    queries = AdminQueryService(store)
    view = await queries.run("run")

    assert "private" not in repr(view)
    assert "payload" not in view["timeline"][0]
    detail = await queries.run_event("run", 1)
    payload = detail["payload"]
    assert payload["headers"]["Authorization"] == REDACTED
    assert payload["api_key"] == REDACTED
    assert payload["usage"] == {"input_tokens": 12, "output_tokens": 3}
    diagnostic = await queries.run_diagnostic("run")
    assert "private" not in repr(diagnostic)
    assert diagnostic["events"][0]["payload"]["api_key"] == REDACTED
    await store.aclose()


async def test_live_resource_version_changes_with_authoritative_chat_state(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    await store.aupsert_conversation("channel", "peer", "Peer")
    queries = AdminQueryService(store)
    before = await queries.resource_versions({"chat:channel", "jobs", "inbox"})

    await store.asave_message(
        id="message",
        channel_id="channel",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="Hello",
        timestamp=100,
    )
    after = await queries.resource_versions({"chat:channel", "jobs", "inbox"})

    assert before["chat:channel"] != after["chat:channel"]
    assert before["jobs"] == after["jobs"]
    assert before["inbox"] == after["inbox"]
    await store.aclose()


async def test_escalation_projection_includes_bounded_conversation_context(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    await store.aupsert_conversation("channel", "peer", "Peer")
    for index in range(35):
        await store.asave_message(
            id=f"message-{index}",
            channel_id="channel",
            author_id="peer",
            author_name="Peer",
            direction="in",
            source="remote",
            content=f"context {index}",
            timestamp=float(index),
        )
    async with store.database.transaction() as connection:
        await connection.execute(
            "INSERT INTO escalation_interrupts VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                "escalation",
                "thread",
                "channel",
                "pending",
                json.dumps(
                    {
                        "reason": "peer_requested_owner",
                        "acknowledgement": "I marked this for the owner.",
                    }
                ),
                40.0,
                40.0,
            ),
        )

    detail = await AdminQueryService(store).escalation("escalation")

    assert detail is not None
    assert detail["conversation"]["peer_name"] == "Peer"
    assert detail["escalation"]["reason"] == "peer_requested_owner"
    assert len(detail["messages"]) == 30
    assert detail["messages"][0]["content"] == "context 5"
    assert detail["messages"][-1]["content"] == "context 34"
    await store.aclose()


async def test_chat_list_searches_messages_and_filters_actionable_states(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    await store.aupsert_conversation("alpha", "peer-a", "Alice")
    await store.aupsert_conversation("beta", "peer-b", "Bob")
    await store.asave_message(
        id="message",
        channel_id="beta",
        author_id="peer-b",
        author_name="Bob",
        direction="in",
        source="remote",
        content="The unusual invoice reference",
        timestamp=100,
    )
    await store.asnooze("alpha", 600)
    queries = AdminQueryService(store)

    search = await queries.chats(query="unusual invoice")
    snoozed = await queries.chats(state="snoozed")

    assert [item["channel_id"] for item in search["items"]] == ["beta"]
    assert [item["channel_id"] for item in snoozed["items"]] == ["alpha"]
    await store.aclose()


async def test_checkpoint_lookup_supports_historical_thread_generations(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    async with store.database.transaction() as connection:
        await connection.execute(
            "INSERT INTO chat_thread_generations("
            "thread_id, channel_id, account_id, generation, created_at, closed_at, close_reason"
            ") VALUES(?, ?, ?, ?, ?, ?, ?)",
            ("thread:g1", "channel", "owner", 1, 10.0, 20.0, "model_changed"),
        )
        await connection.execute(
            "INSERT INTO checkpoint_index VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            ("thread:g1", "checkpoint", None, None, 15.0, 2, "loop", 4),
        )

    checkpoint = await AdminQueryService(store).checkpoint("channel", 1, "checkpoint")
    thread = await store.achat_thread_by_id("thread:g1")

    assert checkpoint["message_count"] == 4
    assert checkpoint["generation"] == 1
    assert thread["channel_id"] == "channel"
    await store.aclose()


def test_recursive_redaction_does_not_hide_non_secret_token_metrics():
    assert redact_sensitive(
        {
            "access_token": "secret",
            "password": "secret",
            "input_tokens": 10,
            "nested": [{"cookie": "secret", "token_count": 4}],
        }
    ) == {
        "access_token": REDACTED,
        "password": REDACTED,
        "input_tokens": 10,
        "nested": [{"cookie": REDACTED, "token_count": 4}],
    }
