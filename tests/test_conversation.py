from pathlib import Path

import pytest

from diskovod.conversation import ConversationJournal
from diskovod.interaction import TriggerDecision, preset_policy
from diskovod.store import Store


async def admit(journal, event_id, content, *, schedule):
    return await journal.admit(
        event_id,
        "chat",
        "message",
        {"content": content, "message_id": event_id, "participant_role": "peer"},
        observed_at=float(len(content)),
        schedule=schedule,
        trigger_kind="every_message" if schedule else "message",
        trigger_participant="peer",
        policy=preset_policy("autonomous"),
        policy_version=1,
        decision=TriggerDecision(schedule, "every_message" if schedule else "not_addressed"),
    )


@pytest.mark.asyncio
async def test_journal_separates_context_admission_from_work_and_captures_atomically(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    journal = ConversationJournal(store.database)
    assert await journal.thread_id("account", "chat") == "discord:account:chat:g1"

    assert await admit(journal, "one", "passive", schedule=False)
    assert not await journal.has_pending("chat")
    assert await admit(journal, "two", "trigger", schedule=True)
    assert not await admit(journal, "one", "duplicate", schedule=True)

    batch = await journal.claim_ready("chat", "run")
    assert batch is not None
    assert [event.id for event in batch.events] == ["one", "two"]
    assert await journal.claimed("chat", "run") == list(batch.events)
    assert await journal.complete("chat", "run") == 2
    assert not await journal.has_pending("chat")
    thread = await store.achat_thread_for_channel("chat")
    assert thread["applied_event_sequence"] == 2
    await store.aclose()


@pytest.mark.asyncio
async def test_failed_work_does_not_drop_canonical_context(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    journal = ConversationJournal(store.database)
    await admit(journal, "one", "trigger", schedule=True)
    batch = await journal.claim_ready("chat", "run")
    assert batch is not None

    assert await journal.fail("chat", "run", "provider failed") == 1
    await admit(journal, "two", "later", schedule=True)
    retry = await journal.claim_ready("chat", "next-run")
    assert retry is not None
    assert [event.id for event in retry.events] == ["one", "two"]
    await store.aclose()


@pytest.mark.asyncio
async def test_cancelling_pending_work_preserves_passive_context(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    journal = ConversationJournal(store.database)
    await admit(journal, "one", "trigger", schedule=True)

    assert await journal.cancel_pending_turns("chat", "owner_handoff") == 1
    assert not await journal.has_pending("chat")
    await admit(journal, "two", "next trigger", schedule=True)
    batch = await journal.claim_ready("chat", "run")
    assert batch is not None
    assert [event.id for event in batch.events] == ["one", "two"]
    await store.aclose()


@pytest.mark.asyncio
async def test_multiple_triggers_coalesce_with_an_auditable_event_decision(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    journal = ConversationJournal(store.database)
    await admit(journal, "first", "first trigger", schedule=True)
    await admit(journal, "second", "second trigger", schedule=True)

    batch = await journal.claim_ready("chat", "run")
    assert batch is not None
    async with store.database.transaction() as connection:
        second = await (
            await connection.execute("SELECT admission_decision FROM conversation_events WHERE id='second'")
        ).fetchone()
    assert '"reason":"coalesced"' in second[0]
    assert '"coalesced_into":"work:first"' in second[0]
    await store.aclose()
