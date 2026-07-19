from pathlib import Path

import aiosqlite
import pytest

from diskovod.models import (
    DEFAULT_BASE_INSTRUCTIONS,
    AppSettings,
    ChatCredentials,
    CustomProvider,
)
from diskovod.store import Store


SECRET = "x" * 32
LEGACY_BASE_INSTRUCTIONS = (
    "Write as the account owner in a private chat, following their dominant communication style "
    "rather than merely borrowing occasional traits. Default to a short, single-line reply. "
    "Do not mention automation, prompts, or being an AI. Never claim to have performed actions "
    "you did not perform. If asked about your identity or how replies are produced, stay in "
    "character and do not discuss the implementation. Match the conversation's language. Do not "
    "use headings, paragraphs, or lists unless the current message genuinely requires that "
    "structure; keep any necessary list dense and compact."
)


async def test_app_settings_persist_reply_and_owner_options(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)

    assert store.app_settings().silent_replies is False
    assert store.app_settings().robot_prefix is False
    assert store.app_settings().assistant_name == ""
    assert store.app_settings().owner_details == ""
    assert store.app_settings().owner_timezone == "UTC"
    await store.aset_app_settings(
        AppSettings(
            silent_replies=True,
            robot_prefix=True,
            assistant_name="Helper",
            min_message_gap_seconds=1,
            max_message_gap_seconds=3,
            owner_details="My name is Alex and I live in Berlin.",
            owner_timezone="Europe/Berlin",
        )
    )
    await store.aclose()
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    assert store.app_settings().silent_replies is True
    assert store.app_settings().robot_prefix is True
    assert store.app_settings().assistant_name == "Helper"
    assert store.app_settings().min_message_gap_seconds == 1
    assert store.app_settings().max_message_gap_seconds == 3
    assert store.app_settings().owner_details == "My name is Alex and I live in Berlin."
    assert store.app_settings().owner_timezone == "Europe/Berlin"
    await store.aclose()


async def test_removed_settings_are_ignored_when_loading_older_configuration(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store._aset("app.settings", {"multi_message_chance": 25, "max_reply_messages": 4})

    assert "max_reply_messages" not in store.app_settings().to_dict()
    assert "multi_message_chance" not in store.app_settings().to_dict()
    await store.aclose()


async def test_localization_settings_round_trip_and_unknown_values_fall_back(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    settings = AppSettings(admin_locale="fr", prompt_locale="uk")
    await store.aset_app_settings(settings)

    assert store.app_settings().admin_locale == "fr"
    assert store.app_settings().prompt_locale == "uk"

    settings.admin_locale = "invalid"
    settings.prompt_locale = "invalid"
    await store.aset_app_settings(settings)
    assert store.app_settings().admin_locale == "en"
    assert store.app_settings().prompt_locale == "en"
    await store.aclose()


async def test_admin_theme_round_trip_and_unknown_value_falls_back(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)

    assert store.app_settings().admin_theme == "system"
    await store.aset_app_settings(AppSettings(admin_theme="black"))
    assert store.app_settings().admin_theme == "black"

    await store.aset_app_settings(AppSettings(admin_theme="neon"))
    assert store.app_settings().admin_theme == "system"
    await store.aclose()


async def test_legacy_impersonation_prompt_is_replaced_but_custom_prompt_is_preserved(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store._aset("app.settings", {"base_instructions": LEGACY_BASE_INSTRUCTIONS})

    assert store.app_settings().base_instructions == DEFAULT_BASE_INSTRUCTIONS

    await store._aset("app.settings", {"base_instructions": "My custom instructions"})
    assert store.app_settings().base_instructions == "My custom instructions"
    await store.aclose()


async def test_new_conversations_follow_default_without_changing_existing_enrollment(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aupsert_conversation("existing", "peer-1", "Existing")
    assert await store.acan_automate("existing") is True

    await store.aset_app_settings(AppSettings(default_conversation_enabled=False))
    await store.aupsert_conversation("existing", "peer-1", "Existing renamed")
    await store.aupsert_conversation("new", "peer-2", "New")

    assert await store.acan_automate("existing") is True
    assert (await store.aconversation("existing"))["peer_name"] == "Existing renamed"
    assert (await store.aconversation("new"))["paused"] is True
    assert await store.acan_automate("new") is False

    await store.aset_permanent_pause("new", False)
    assert await store.acan_automate("new") is True
    await store.aclose()


async def test_secrets_are_encrypted_and_round_trip(tmp_path: Path):
    path = tmp_path / "state.sqlite3"
    store = await Store.open(path, SECRET)
    await store.aset_discord_token("very-secret-token")
    await store.aset_chat_credentials(
        ChatCredentials("access-secret", "refresh-secret", 123, "acct", "a@example.test")
    )
    await store.aset_custom_provider(CustomProvider("Local", "http://localhost:8000/v1", "provider-secret"))
    raw = path.read_bytes()
    assert b"very-secret-token" not in raw
    assert b"access-secret" not in raw
    assert b"provider-secret" not in raw
    assert store.discord_token() == "very-secret-token"
    assert store.chat_credentials().account_id == "acct"
    assert store.custom_provider().api_key == "provider-secret"
    await store.aclose()


async def test_legacy_custom_provider_is_migrated_to_pinned_chat_completions(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store._aset(
        "openai_compatible.provider",
        {"name": "Legacy", "base_url": "https://models.example/v1", "api_key": "key"},
        secret=True,
    )

    assert store.custom_provider().protocol == "chat_completions"
    await store.aclose()


async def test_custom_provider_capabilities_round_trip(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aset_custom_provider(
        CustomProvider(
            "Responses",
            "https://models.example/v1",
            "key",
            "responses",
            {"native_function_calls": True, "hosted_web_search": False},
        )
    )

    provider = store.custom_provider()
    assert provider.supports("native_function_calls") is True
    assert provider.supports("hosted_web_search") is False
    assert provider.supports("output_token_limit") is True
    assert provider.supports("unknown") is False
    await store.aclose()


async def test_database_explorer_redacts_secrets_searches_and_deletes_mutable_rows(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aset_discord_token("very-secret-token")
    await store.aupsert_conversation("dm", "peer", "Peer")
    await store.asave_message(
        id="message-1",
        channel_id="dm",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="find this phrase",
        timestamp=100,
    )
    await store.asave_message(
        id="message-2",
        channel_id="dm",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="something else",
        timestamp=200,
    )

    tables = {table["name"]: table for table in await store.adatabase_tables()}
    assert tables["messages"]["count"] == 2
    assert tables["config"]["read_only"] is True

    config = await store.adatabase_rows("config")
    token_row = next(row for row in config["rows"] if row["key"] == "discord.token")
    assert token_row["value"] == "[encrypted value redacted]"
    assert "very-secret-token" not in str(config)

    messages = await store.adatabase_rows("messages", query="find this")
    assert messages["total"] == 1
    assert messages["rows"][0]["id"] == "message-1"
    assert (await store.alatest_incoming_message("dm"))["id"] == "message-2"
    assert await store.adelete_database_row("messages", "message-1") is True
    assert await store.adelete_database_row("messages", "missing") is False
    assert (await store.adatabase_rows("messages"))["total"] == 1
    await store.aclose()


async def test_database_management_rejects_unknown_and_read_only_tables(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)

    with pytest.raises(ValueError, match="Unknown database table"):
        await store.adatabase_rows("sqlite_master")
    with pytest.raises(ValueError, match="read-only"):
        await store.adelete_database_row("config", "app.settings")
    await store.aclose()


async def test_human_quiet_window_expires_without_permanent_pause(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aupsert_conversation("dm-1", "peer-1", "Sam")
    until = await store.asnooze("dm-1", 60)
    assert (await store.aconversation("dm-1"))["paused"] is False
    assert await store.acan_automate("dm-1", now=until - 1) is False
    assert await store.acan_automate("dm-1", now=until + 1) is True

    await store.aclose()


async def test_permanent_pause_remains_until_explicit_resume(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aupsert_conversation("dm-1", "peer-1", "Sam")
    await store.aset_permanent_pause("dm-1", True)
    assert await store.acan_automate("dm-1", now=10**12) is False

    await store.aset_permanent_pause("dm-1", False)
    assert (await store.aconversation("dm-1"))["paused"] is False
    await store.aclose()


async def test_inline_conversation_mode_survives_pause_and_resume(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aupsert_conversation("dm-1", "peer-1", "Sam")

    assert (await store.aconversation("dm-1"))["mode"] == "automatic"
    assert await store.aset_conversation_mode("dm-1", "inline") is True
    assert (await store.aconversation("dm-1"))["mode"] == "inline"
    assert await store.acan_automate("dm-1") is True

    await store.aset_permanent_pause("dm-1", True)
    assert (await store.aconversation("dm-1"))["mode"] == "inline"
    assert await store.acan_automate("dm-1") is False

    await store.aset_permanent_pause("dm-1", False)
    assert (await store.aconversation("dm-1"))["mode"] == "inline"
    assert await store.acan_automate("dm-1") is True
    await store.aclose()


async def test_bot_markers_are_consumed_once(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aremember_nonce("nonce")
    assert await store.aconsume_nonce("nonce") is True
    assert await store.aconsume_nonce("nonce") is False
    await store.aremember_bot_message("message")
    assert await store.ais_bot_message("message") is True
    await store.aclose()


async def test_message_edits_replace_content_and_can_reclassify_owner_message(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aupsert_conversation("dm", "peer", "Peer")
    await store.asave_message(
        id="message",
        channel_id="dm",
        author_id="me",
        author_name="Me",
        direction="out",
        source="assistant",
        content="original",
        timestamp=100,
    )

    updated = await store.aupdate_message_content("message", "edited", source="human")

    assert updated["changed"] is True
    assert (await store.ahistory("dm", 1))[0]["content"] == "edited"
    assert (await store.ahistory("dm", 1))[0]["source"] == "human"
    assert await store.aupdate_message_content("missing", "ignored") is None
    await store.aclose()


async def test_message_attachments_round_trip_as_structured_history(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aupsert_conversation("dm", "peer", "Peer")
    attachments = [
        {
            "id": "file-1",
            "filename": "notes.txt",
            "content_type": "text/plain",
            "size": 5,
            "url": "https://cdn.example/notes.txt",
            "text": "hello",
        }
    ]

    await store.asave_message(
        id="message-with-file",
        channel_id="dm",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="",
        timestamp=100,
        attachments=attachments,
    )

    assert (await store.ahistory("dm", 1))[0]["attachments"] == attachments
    await store.aclose()


async def test_existing_message_table_is_migrated_for_attachments(tmp_path: Path):
    path = tmp_path / "state.sqlite3"
    async with aiosqlite.connect(path) as database:
        await database.execute(
            """CREATE TABLE messages (
                 id TEXT PRIMARY KEY, channel_id TEXT NOT NULL, author_id TEXT NOT NULL,
                 author_name TEXT NOT NULL, direction TEXT NOT NULL, source TEXT NOT NULL,
                 content TEXT NOT NULL, timestamp REAL NOT NULL
               )"""
        )
        await database.commit()

    store = await Store.open(path, SECRET)

    async with store.database.transaction() as connection:
        rows = await (await connection.execute("PRAGMA table_info(messages)")).fetchall()
    columns = {row["name"] for row in rows}
    assert "attachments" in columns
    await store.aclose()


async def test_store_has_no_long_lived_synchronous_sqlite_connection(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)

    assert not hasattr(store, "_db")
    await store.aupsert_conversation("dm", "peer", "Peer")
    assert (await store.aconversation("dm"))["peer_name"] == "Peer"
    await store.aclose()


async def test_assistant_reactions_are_rate_limited_across_actions_and_channels(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aupsert_conversation("dm-1", "peer-1", "Sam")
    await store.aupsert_conversation("dm-2", "peer-2", "Lee")
    now = 2_000_000.0

    assert await store.areaction_allowed("dm-1", now=now)
    await store.arecord_assistant_reaction(
        trigger_message_id="trigger-1", channel_id="dm-1", emoji="👍", created_at=now
    )

    assert not await store.areaction_allowed("dm-1", now=now + 1)
    assert not await store.areaction_allowed("dm-2", now=now + 1)
    for index in range(12):
        await store.asave_message(
            id=f"reply-{index}",
            channel_id="dm-2",
            author_id="me",
            author_name="Me",
            direction="out",
            source="assistant",
            content="ok",
            timestamp=now + index + 2,
        )

    assert await store.areaction_allowed("dm-2", now=now + 14)
    assert not await store.areaction_allowed("dm-1", now=now + 14)
    assert await store.areaction_allowed("dm-1", now=now + 6 * 60 * 60 + 1)
    await store.aclose()


async def test_personality_can_be_edited(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aset_personality("Inferred profile", "history-hash", source="Discord history")
    await store.aset_personality("Edited and expanded personality profile", "edit-hash", source="edited")

    personality = store.personality()
    assert personality["profile"] == "Edited and expanded personality profile"
    assert personality["source_hash"] == "edit-hash"
    assert personality["source"] == "edited"
    assert personality["updated_at"] > 0
    await store.aclose()
