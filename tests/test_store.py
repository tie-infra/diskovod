import sqlite3
from pathlib import Path

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


def test_app_settings_persist_reply_and_owner_options(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)

    assert store.app_settings().silent_replies is False
    assert store.app_settings().robot_prefix is False
    assert store.app_settings().owner_details == ""
    assert store.app_settings().owner_timezone == "UTC"
    store.set_app_settings(
        AppSettings(
            silent_replies=True,
            robot_prefix=True,
            multi_message_replies=True,
            multi_message_chance=20,
            max_reply_messages=4,
            min_message_gap_seconds=1,
            max_message_gap_seconds=3,
            owner_details="My name is Alex and I live in Berlin.",
            owner_timezone="Europe/Berlin",
        )
    )
    assert store.app_settings().silent_replies is True
    assert store.app_settings().robot_prefix is True
    assert store.app_settings().multi_message_replies is True
    assert store.app_settings().multi_message_chance == 20
    assert store.app_settings().max_reply_messages == 4
    assert store.app_settings().min_message_gap_seconds == 1
    assert store.app_settings().max_message_gap_seconds == 3
    assert store.app_settings().owner_details == "My name is Alex and I live in Berlin."
    assert store.app_settings().owner_timezone == "Europe/Berlin"
    store.close()


def test_localization_settings_round_trip_and_unknown_values_fall_back(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    settings = AppSettings(admin_locale="fr", prompt_locale="uk")
    store.set_app_settings(settings)

    assert store.app_settings().admin_locale == "fr"
    assert store.app_settings().prompt_locale == "uk"

    settings.admin_locale = "invalid"
    settings.prompt_locale = "invalid"
    store.set_app_settings(settings)
    assert store.app_settings().admin_locale == "en"
    assert store.app_settings().prompt_locale == "en"
    store.close()


def test_legacy_impersonation_prompt_is_replaced_but_custom_prompt_is_preserved(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    store._set("app.settings", {"base_instructions": LEGACY_BASE_INSTRUCTIONS})

    assert store.app_settings().base_instructions == DEFAULT_BASE_INSTRUCTIONS

    store._set("app.settings", {"base_instructions": "My custom instructions"})
    assert store.app_settings().base_instructions == "My custom instructions"
    store.close()


def test_new_conversations_follow_default_without_changing_existing_enrollment(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    store.upsert_conversation("existing", "peer-1", "Existing")
    assert store.can_automate("existing") is True

    store.set_app_settings(AppSettings(default_conversation_enabled=False))
    store.upsert_conversation("existing", "peer-1", "Existing renamed")
    store.upsert_conversation("new", "peer-2", "New")

    assert store.can_automate("existing") is True
    assert store.conversation("existing")["peer_name"] == "Existing renamed"
    assert store.conversation("new")["paused"] is True
    assert store.can_automate("new") is False

    store.set_permanent_pause("new", False)
    assert store.can_automate("new") is True
    store.close()


def test_secrets_are_encrypted_and_round_trip(tmp_path: Path):
    path = tmp_path / "state.sqlite3"
    store = Store(path, SECRET)
    store.set_discord_token("very-secret-token")
    store.set_chat_credentials(
        ChatCredentials("access-secret", "refresh-secret", 123, "acct", "a@example.test")
    )
    store.set_custom_provider(CustomProvider("Local", "http://localhost:8000/v1", "provider-secret"))
    raw = path.read_bytes()
    assert b"very-secret-token" not in raw
    assert b"access-secret" not in raw
    assert b"provider-secret" not in raw
    assert store.discord_token() == "very-secret-token"
    assert store.chat_credentials().account_id == "acct"
    assert store.custom_provider().api_key == "provider-secret"
    store.close()


def test_legacy_custom_provider_is_migrated_to_pinned_chat_completions(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    store._set(
        "openai_compatible.provider",
        {"name": "Legacy", "base_url": "https://models.example/v1", "api_key": "key"},
        secret=True,
    )

    assert store.custom_provider().protocol == "chat_completions"
    store.close()


def test_custom_provider_capabilities_round_trip(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    store.set_custom_provider(
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
    assert provider.supports("unknown") is False
    store.close()


def test_database_explorer_redacts_secrets_searches_and_deletes_mutable_rows(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    store.set_discord_token("very-secret-token")
    store.upsert_conversation("dm", "peer", "Peer")
    store.save_message(
        id="message-1",
        channel_id="dm",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="find this phrase",
        timestamp=100,
    )
    store.save_message(
        id="message-2",
        channel_id="dm",
        author_id="peer",
        author_name="Peer",
        direction="in",
        source="remote",
        content="something else",
        timestamp=200,
    )

    tables = {table["name"]: table for table in store.database_tables()}
    assert tables["messages"]["count"] == 2
    assert tables["config"]["read_only"] is True

    config = store.database_rows("config")
    token_row = next(row for row in config["rows"] if row["key"] == "discord.token")
    assert token_row["value"] == "[encrypted value redacted]"
    assert "very-secret-token" not in str(config)

    messages = store.database_rows("messages", query="find this")
    assert messages["total"] == 1
    assert messages["rows"][0]["id"] == "message-1"
    assert store.latest_incoming_message("dm")["id"] == "message-2"
    assert store.delete_database_row("messages", "message-1") is True
    assert store.delete_database_row("messages", "missing") is False
    assert store.database_rows("messages")["total"] == 1
    store.close()


def test_database_management_rejects_unknown_and_read_only_tables(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)

    with pytest.raises(ValueError, match="Unknown database table"):
        store.database_rows("sqlite_master")
    with pytest.raises(ValueError, match="read-only"):
        store.delete_database_row("config", "app.settings")
    store.close()


def test_human_quiet_window_expires_without_permanent_pause(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    store.upsert_conversation("dm-1", "peer-1", "Sam")
    until = store.snooze("dm-1", 60)
    assert store.conversation("dm-1")["paused"] is False
    assert store.can_automate("dm-1", now=until - 1) is False
    assert store.can_automate("dm-1", now=until + 1) is True

    store.close()


def test_permanent_pause_remains_until_explicit_resume(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    store.upsert_conversation("dm-1", "peer-1", "Sam")
    store.set_permanent_pause("dm-1", True)
    assert store.can_automate("dm-1", now=10**12) is False

    store.set_permanent_pause("dm-1", False)
    assert store.conversation("dm-1")["paused"] is False
    store.close()


def test_bot_markers_are_consumed_once(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    store.remember_nonce("nonce")
    assert store.consume_nonce("nonce") is True
    assert store.consume_nonce("nonce") is False
    store.remember_bot_message("message")
    assert store.is_bot_message("message") is True
    store.close()


def test_message_edits_replace_content_and_can_reclassify_owner_message(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    store.upsert_conversation("dm", "peer", "Peer")
    store.save_message(
        id="message",
        channel_id="dm",
        author_id="me",
        author_name="Me",
        direction="out",
        source="assistant",
        content="original",
        timestamp=100,
    )

    updated = store.update_message_content("message", "edited", source="human")

    assert updated["changed"] is True
    assert store.history("dm", 1)[0]["content"] == "edited"
    assert store.history("dm", 1)[0]["source"] == "human"
    assert store.update_message_content("missing", "ignored") is None
    store.close()


def test_message_attachments_round_trip_as_structured_history(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    store.upsert_conversation("dm", "peer", "Peer")
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

    store.save_message(
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

    assert store.history("dm", 1)[0]["attachments"] == attachments
    store.close()


def test_existing_message_table_is_migrated_for_attachments(tmp_path: Path):
    path = tmp_path / "state.sqlite3"
    database = sqlite3.connect(path)
    database.execute(
        """CREATE TABLE messages (
             id TEXT PRIMARY KEY, channel_id TEXT NOT NULL, author_id TEXT NOT NULL,
             author_name TEXT NOT NULL, direction TEXT NOT NULL, source TEXT NOT NULL,
             content TEXT NOT NULL, timestamp REAL NOT NULL
           )"""
    )
    database.commit()
    database.close()

    store = Store(path, SECRET)

    columns = {row["name"] for row in store._db.execute("PRAGMA table_info(messages)").fetchall()}
    assert "attachments" in columns
    store.close()


def test_assistant_reactions_are_rate_limited_across_actions_and_channels(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    store.upsert_conversation("dm-1", "peer-1", "Sam")
    store.upsert_conversation("dm-2", "peer-2", "Lee")
    now = 2_000_000.0

    assert store.reaction_allowed("dm-1", now=now)
    store.record_assistant_reaction(
        trigger_message_id="trigger-1", channel_id="dm-1", emoji="👍", created_at=now
    )

    assert not store.reaction_allowed("dm-1", now=now + 1)
    assert not store.reaction_allowed("dm-2", now=now + 1)
    for index in range(12):
        store.save_message(
            id=f"reply-{index}",
            channel_id="dm-2",
            author_id="me",
            author_name="Me",
            direction="out",
            source="assistant",
            content="ok",
            timestamp=now + index + 2,
        )

    assert store.reaction_allowed("dm-2", now=now + 14)
    assert not store.reaction_allowed("dm-1", now=now + 14)
    assert store.reaction_allowed("dm-1", now=now + 6 * 60 * 60 + 1)
    store.close()


def test_personality_can_be_edited(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    store.set_personality("Inferred profile", "history-hash", source="Discord history")
    store.set_personality("Edited and expanded personality profile", "edit-hash", source="edited")

    personality = store.personality()
    assert personality["profile"] == "Edited and expanded personality profile"
    assert personality["source_hash"] == "edit-hash"
    assert personality["source"] == "edited"
    assert personality["updated_at"] > 0
    store.close()


def test_chatgpt_usage_is_aggregated_by_window_model_and_operation(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)
    now = 2_000_000.0
    store.record_chatgpt_usage(
        response_id="resp-recent",
        recorded_at=now - 60,
        model="gpt-5",
        purpose="dm_reply",
        input_tokens=100,
        cached_input_tokens=40,
        output_tokens=30,
        reasoning_tokens=10,
        total_tokens=130,
    )
    store.record_chatgpt_usage(
        response_id="resp-old",
        recorded_at=now - 10 * 86400,
        model="gpt-5-mini",
        purpose="personality_inference",
        input_tokens=900,
        cached_input_tokens=0,
        output_tokens=100,
        reasoning_tokens=25,
        total_tokens=1000,
    )
    store.record_chatgpt_usage(
        response_id="resp-recent",
        recorded_at=now,
        model="duplicate",
        purpose="dm_reply",
        input_tokens=999,
        cached_input_tokens=999,
        output_tokens=999,
        reasoning_tokens=999,
        total_tokens=999,
    )

    stats = store.chatgpt_usage_stats(now=now)

    assert stats["windows"][0] == {
        "label": "Last 24 hours",
        "requests": 1,
        "input_tokens": 100,
        "cached_input_tokens": 40,
        "output_tokens": 30,
        "reasoning_tokens": 10,
        "total_tokens": 130,
        "average_tokens": 130,
        "cache_rate": 40.0,
    }
    assert stats["all_time"]["requests"] == 2
    assert stats["all_time"]["total_tokens"] == 1130
    assert [group["name"] for group in stats["by_model"]] == ["gpt-5-mini", "gpt-5"]
    assert {group["name"] for group in stats["by_purpose"]} == {
        "dm_reply",
        "personality_inference",
    }
    assert stats["recent"][0]["model"] == "gpt-5"
    store.close()
