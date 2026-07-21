from pathlib import Path

import pytest

from diskovod.agent_types import AgentRuntimeContext, CapabilityProfile
from diskovod.outbound import DeliveryRecord, OutboundPublisher
from diskovod.store import Store


class Transport:
    def __init__(self, *, raises: bool = False):
        self.raises = raises
        self.messages: list[str] = []
        self.reactions: list[tuple[str, str]] = []

    async def send_messages(self, context, messages):
        del context
        if self.raises:
            raise TimeoutError("unknown delivery state")
        self.messages.extend(messages)
        return [DeliveryRecord("accepted", 0, f"discord:{len(self.messages)}")]

    async def react_to_message(self, context, message_id, emoji):
        del context
        self.reactions.append((message_id, emoji))
        return DeliveryRecord("accepted", 0, f"reaction:{message_id}:{emoji}")


def context(*, delivery: str = "immediate") -> AgentRuntimeContext:
    return AgentRuntimeContext(
        account_id="account",
        channel_id="chat",
        participant_ids=("peer",),
        owner_id="owner",
        ui_locale="en",
        prompt_locale="en",
        assistant_name="Diskovod",
        conversation_role="shared_assistant",
        force_reply=False,
        provider_id="test",
        model_id="test",
        transport_profile="test",
        capabilities=CapabilityProfile(),
        trace_id="run",
        thread_id="thread",
        delivery=delivery,
    )


@pytest.mark.asyncio
async def test_outbound_message_replay_is_idempotent(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport()
    publisher = OutboundPublisher(store.database, transport)

    first = await publisher.publish_messages(
        context(), ("hello",), source_kind="assistant_text", source_id="ai-1"
    )
    second = await publisher.publish_messages(
        context(), ("hello",), source_kind="assistant_text", source_id="ai-1"
    )

    assert first == second
    assert transport.messages == ["hello"]
    await store.aclose()


@pytest.mark.asyncio
async def test_outbound_ambiguous_result_is_not_retried(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport(raises=True)
    publisher = OutboundPublisher(store.database, transport)

    first = await publisher.publish_messages(
        context(), ("hello",), source_kind="assistant_text", source_id="ai-1"
    )
    transport.raises = False
    second = await publisher.publish_messages(
        context(), ("hello",), source_kind="assistant_text", source_id="ai-1"
    )

    assert first[0].status == "ambiguous"
    assert first[0].error_detail == "TimeoutError: unknown delivery state"
    assert second == first
    assert transport.messages == []
    await store.aclose()


@pytest.mark.asyncio
async def test_reaction_target_is_part_of_idempotent_action(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport()
    publisher = OutboundPublisher(store.database, transport)

    first = await publisher.react(context(), "🎉", "message-1", source_id="tool-1")
    second = await publisher.react(context(), "🎉", "message-1", source_id="tool-1")

    assert first == second
    assert transport.reactions == [("message-1", "🎉")]
    await store.aclose()


@pytest.mark.asyncio
async def test_long_logical_reply_uses_deterministic_transport_segments(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport()
    publisher = OutboundPublisher(store.database, transport)
    message = " ".join(f"word-{index:04}" for index in range(500))

    first = await publisher.publish_messages(
        context(), (message,), source_kind="assistant_text", source_id="ai-long"
    )
    second = await publisher.publish_messages(
        context(), (message,), source_kind="assistant_text", source_id="ai-long"
    )

    assert len(first) > 1
    assert second == first
    assert all(len(segment) <= 1900 for segment in transport.messages)
    assert " ".join(transport.messages) == message
    async with store.database.transaction() as connection:
        payloads = [
            row[0]
            for row in await (
                await connection.execute("SELECT payload FROM outbound_actions ORDER BY ordinal")
            ).fetchall()
        ]
    assert all('"transport_segment"' in payload for payload in payloads)
    await store.aclose()


@pytest.mark.asyncio
async def test_abandoned_dispatch_requires_explicit_operator_retry(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport()
    publisher = OutboundPublisher(store.database, transport)
    await publisher.publish_messages(context(), ("hello",), source_kind="assistant_text", source_id="ai-1")
    async with store.database.transaction() as connection:
        row = await (await connection.execute("SELECT id FROM outbound_actions")).fetchone()
        action_id = str(row["id"])
        await connection.execute(
            "UPDATE outbound_actions SET state='dispatching', result=NULL, remote_id=NULL, "
            "lease_owner='dead-process', lease_expires_at=0, completed_at=NULL"
        )

    replacement = OutboundPublisher(store.database, transport)
    reconciled = await replacement.reconcile_abandoned()
    replay = await replacement.publish_messages(
        context(), ("hello",), source_kind="assistant_text", source_id="ai-1"
    )

    assert reconciled == [{"id": action_id, "run_id": "run", "state": "ambiguous"}]
    assert replay[0].status == "ambiguous"
    assert transport.messages == ["hello"]

    retried = await replacement.retry(action_id)

    assert retried is not None and retried.accepted
    assert transport.messages == ["hello", "hello"]
    await store.aclose()


@pytest.mark.asyncio
async def test_operator_can_resolve_ambiguous_delivery_without_transport(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport(raises=True)
    publisher = OutboundPublisher(store.database, transport)
    await publisher.publish_messages(context(), ("hello",), source_kind="assistant_text", source_id="ai-1")
    async with store.database.transaction() as connection:
        row = await (await connection.execute("SELECT id FROM outbound_actions")).fetchone()
        action_id = str(row["id"])

    resolved = await publisher.resolve(
        action_id,
        "confirmed_succeeded",
        remote_id="discord-message-42",
    )

    assert resolved is not None and resolved.accepted
    assert resolved.discord_message_id == "discord-message-42"
    saved = await publisher.action(action_id)
    assert saved["state"] == "succeeded"
    assert transport.messages == []
    await store.aclose()


@pytest.mark.asyncio
async def test_owner_approval_holds_an_editable_draft_until_explicit_delivery(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport()
    publisher = OutboundPublisher(store.database, transport)

    records = await publisher.publish_messages(
        context(delivery="owner_approval"),
        ("original draft",),
        source_kind="assistant_text",
        source_id="ai-draft",
    )
    drafts = await publisher.drafts(state="pending")

    assert records[0].accepted
    assert transport.messages == []
    assert len(drafts) == 1
    assert drafts[0]["payload"]["message"] == "original draft"

    delivered = await publisher.approve_draft(str(drafts[0]["id"]), message="owner edit")

    assert delivered is not None and delivered.accepted
    assert transport.messages == ["owner edit"]
    assert (await publisher.draft(str(drafts[0]["id"])))["state"] == "delivered"
    await store.aclose()


@pytest.mark.asyncio
async def test_owner_approval_draft_expires_durably_before_delivery(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport()
    publisher = OutboundPublisher(store.database, transport)
    await publisher.publish_messages(
        context(delivery="owner_approval"),
        ("stale draft",),
        source_kind="assistant_text",
        source_id="ai-expired",
    )
    pending = (await publisher.drafts(state="pending"))[0]
    async with store.database.transaction() as connection:
        await connection.execute(
            "UPDATE outbound_drafts SET expires_at=0 WHERE id=?",
            (pending["id"],),
        )

    assert await publisher.approve_draft(str(pending["id"])) is None
    expired = await publisher.draft(str(pending["id"]))
    assert expired is not None and expired["state"] == "expired"
    assert expired["decided_at"] is not None
    assert transport.messages == []
    await store.aclose()


@pytest.mark.asyncio
async def test_dashboard_only_output_is_recorded_and_cannot_be_approved(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport()
    publisher = OutboundPublisher(store.database, transport)

    await publisher.publish_messages(
        context(delivery="dashboard_only"),
        ("private advice",),
        source_kind="assistant_text",
        source_id="ai-private",
    )
    drafts = await publisher.drafts(state="recorded")

    assert len(drafts) == 1
    assert drafts[0]["payload"]["message"] == "private advice"
    assert await publisher.approve_draft(str(drafts[0]["id"])) is None
    assert transport.messages == []
    await store.aclose()


@pytest.mark.asyncio
async def test_explicit_second_approval_retries_a_failed_draft_delivery(tmp_path: Path):
    store = await Store.open(tmp_path / "diskovod.sqlite3", "x" * 32)
    transport = Transport(raises=True)
    publisher = OutboundPublisher(store.database, transport)
    await publisher.publish_messages(
        context(delivery="owner_approval"),
        ("draft",),
        source_kind="assistant_text",
        source_id="ai-retry",
    )
    draft = (await publisher.drafts(state="pending"))[0]

    failed = await publisher.approve_draft(str(draft["id"]))
    transport.raises = False
    retried = await publisher.approve_draft(str(draft["id"]), message="edited retry")

    assert failed is not None and not failed.accepted
    assert retried is not None and retried.accepted
    assert transport.messages == ["edited retry"]
    assert (await publisher.draft(str(draft["id"])))["state"] == "delivered"
    await store.aclose()
