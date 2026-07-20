from pathlib import Path

import pytest

from diskovod.mailbox import ConversationMailbox
from diskovod.store import Store


@pytest.mark.asyncio
async def test_mailbox_orders_deduplicates_and_completes_events(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    mailbox = ConversationMailbox(store.database)
    assert await mailbox.thread_id("account", "chat") == "discord:account:chat:g1"

    assert await mailbox.ingest("one", "chat", "message", {"content": "one"})
    assert await mailbox.ingest("two", "chat", "message", {"content": "two"})
    assert not await mailbox.ingest("one", "chat", "message", {"content": "duplicate"})
    assert await mailbox.ingest("ignored", "chat", "message", {}, enqueue=False)

    claimed = await mailbox.claim_ready("chat", "run")
    assert [event.id for event in claimed] == ["one", "two"]
    assert await mailbox.claimed("chat", "run") == claimed
    assert await mailbox.complete("chat", "run") == 2
    assert not await mailbox.has_pending("chat")
    await store.aclose()


@pytest.mark.asyncio
async def test_mailbox_does_not_claim_future_wakeups(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    mailbox = ConversationMailbox(store.database)
    await mailbox.ingest(
        "wake",
        "chat",
        "continuation_due",
        {"wait_id": "wait"},
        available_at=9_999_999_999,
    )

    assert await mailbox.claim_ready("chat", "run") == []
    assert await mailbox.has_pending("chat", include_future=True)
    assert not await mailbox.has_pending("chat")
    await store.aclose()
