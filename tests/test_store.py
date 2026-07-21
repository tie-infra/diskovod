import json
from dataclasses import replace
from pathlib import Path

import aiosqlite
import pytest
from diskovod.interaction import AvailabilitySchedule, EngagementWindow, preset_policy

from diskovod.models import (
    DEFAULT_BASE_INSTRUCTIONS,
    AssistantProfile,
    AutomationSettings,
    ChatCredentials,
    CustomProvider,
    InterfaceSettings,
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


async def test_owned_settings_domains_persist_independently(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)

    assert store.automation_settings().silent_replies is False
    assert store.automation_settings().robot_prefix is False
    assert store.assistant_profile().assistant_name == ""
    assert store.assistant_profile().owner_details == ""
    assert store.assistant_profile().owner_timezone == "UTC"
    await store.aset_automation_settings(
        AutomationSettings(
            silent_replies=True,
            robot_prefix=True,
            min_message_gap_seconds=1,
            max_message_gap_seconds=3,
        )
    )
    await store.aset_assistant_profile(
        AssistantProfile(
            assistant_name="Helper",
            owner_details="My name is Alex and I live in Berlin.",
            owner_timezone="Europe/Berlin",
        )
    )
    await store.aclose()
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    assert store.automation_settings().silent_replies is True
    assert store.automation_settings().robot_prefix is True
    assert store.assistant_profile().assistant_name == "Helper"
    assert store.automation_settings().min_message_gap_seconds == 1
    assert store.automation_settings().max_message_gap_seconds == 3
    assert store.assistant_profile().owner_details == "My name is Alex and I live in Berlin."
    assert store.assistant_profile().owner_timezone == "Europe/Berlin"
    await store.aclose()


async def test_legacy_mixed_settings_are_atomically_split(tmp_path: Path):
    path = tmp_path / "state.sqlite3"
    store = await Store.open(path, SECRET)
    await store.aclose()
    legacy = {
        "admin_locale": "fr",
        "admin_theme": "black",
        "prompt_locale": "ja",
        "assistant_name": "Helper",
        "enabled": True,
        "silent_replies": True,
        "model": "legacy-model",
        "reasoning_effort": "high",
    }
    async with aiosqlite.connect(path) as connection:
        await connection.execute(
            "DELETE FROM config WHERE key IN "
            "('admin.interface','assistant.profile','automation.settings','legacy.model_selection')"
        )
        await connection.execute(
            "INSERT INTO config(key, value, secret, updated_at) VALUES('app.settings', ?, 0, 1)",
            (json.dumps(legacy),),
        )
        await connection.commit()

    store = await Store.open(path, SECRET)
    assert store.interface_settings() == InterfaceSettings(locale="fr", theme="black")
    assert store.assistant_profile().prompt_locale == "ja"
    assert store.assistant_profile().assistant_name == "Helper"
    assert store.automation_settings().enabled is True
    assert store.automation_settings().silent_replies is True
    assert store._get("legacy.model_selection", {}) == {
        "model": "legacy-model",
        "reasoning_effort": "high",
    }
    async with store.database.transaction() as connection:
        old = await (await connection.execute("SELECT 1 FROM config WHERE key='app.settings'")).fetchone()
    assert old is None
    await store.aclose()


async def test_removed_settings_are_ignored_when_loading_older_configuration(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store._aset("automation.settings", {"multi_message_chance": 25, "max_reply_messages": 4})

    assert "max_reply_messages" not in store.automation_settings().to_dict()
    assert "multi_message_chance" not in store.automation_settings().to_dict()
    await store.aclose()


async def test_localization_settings_round_trip_and_unknown_values_fall_back(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    interface = InterfaceSettings(locale="fr")
    profile = AssistantProfile(prompt_locale="uk")
    await store.aset_interface_settings(interface)
    await store.aset_assistant_profile(profile)

    assert store.interface_settings().locale == "fr"
    assert store.assistant_profile().prompt_locale == "uk"

    interface.locale = "invalid"
    profile.prompt_locale = "invalid"
    await store.aset_interface_settings(interface)
    await store.aset_assistant_profile(profile)
    assert store.interface_settings().locale == "en"
    assert store.assistant_profile().prompt_locale == "en"
    await store.aclose()


async def test_admin_theme_round_trip_and_unknown_value_falls_back(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)

    assert store.interface_settings().theme == "system"
    await store.aset_interface_settings(InterfaceSettings(theme="black"))
    assert store.interface_settings().theme == "black"

    await store.aset_interface_settings(InterfaceSettings(theme="neon"))
    assert store.interface_settings().theme == "system"
    await store.aclose()


async def test_legacy_impersonation_prompt_is_replaced_but_custom_prompt_is_preserved(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store._aset("assistant.profile", {"base_instructions": LEGACY_BASE_INSTRUCTIONS})

    assert store.assistant_profile().base_instructions == DEFAULT_BASE_INSTRUCTIONS

    await store._aset("assistant.profile", {"base_instructions": "My custom instructions"})
    assert store.assistant_profile().base_instructions == "My custom instructions"
    await store.aclose()


async def test_new_conversations_follow_default_without_changing_existing_enrollment(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aupsert_conversation("existing", "peer-1", "Existing")
    assert await store.acan_automate("existing") is True

    await store.aset_automation_settings(AutomationSettings(default_conversation_enabled=False))
    await store.aupsert_conversation("existing", "peer-1", "Existing renamed")
    await store.aupsert_conversation("new", "peer-2", "New")

    assert await store.acan_automate("existing") is True
    assert (await store.aconversation("existing"))["peer_name"] == "Existing renamed"
    assert (await store.aconversation("new"))["availability"] == "paused"
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
    assert (await store.aconversation("dm-1"))["availability"] == "active"
    assert await store.acan_automate("dm-1", now=until - 1) is False
    assert await store.acan_automate("dm-1", now=until + 1) is True

    await store.aclose()


async def test_permanent_pause_remains_until_explicit_resume(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aupsert_conversation("dm-1", "peer-1", "Sam")
    await store.aset_permanent_pause("dm-1", True)
    assert await store.acan_automate("dm-1", now=10**12) is False

    await store.aset_permanent_pause("dm-1", False)
    assert (await store.aconversation("dm-1"))["availability"] == "active"
    await store.aclose()


async def test_interaction_policy_survives_pause_and_resume(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aupsert_conversation("dm-1", "peer-1", "Sam")

    assert (await store.ainteraction_policy("dm-1"))[0].preset == "autonomous"
    assert await store.aset_interaction_policy("dm-1", preset_policy("shared")) is True
    assert (await store.ainteraction_policy("dm-1"))[0].preset == "shared"
    assert await store.acan_automate("dm-1") is True

    await store.aset_permanent_pause("dm-1", True)
    assert (await store.ainteraction_policy("dm-1"))[0].preset == "shared"
    assert await store.acan_automate("dm-1") is False

    await store.aset_permanent_pause("dm-1", False)
    assert (await store.ainteraction_policy("dm-1"))[0].preset == "shared"
    assert await store.acan_automate("dm-1") is True
    await store.aclose()


async def test_interaction_policy_persists_an_availability_schedule(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aupsert_conversation("dm-1", "peer-1", "Sam")
    schedule = AvailabilitySchedule(
        enabled=True,
        weekdays=frozenset({0, 2, 4}),
        start_minute=8 * 60 + 30,
        end_minute=18 * 60,
        timezone="Europe/Paris",
    )
    policy = replace(preset_policy("autonomous"), availability_schedule=schedule)

    assert await store.aset_interaction_policy("dm-1", policy)
    assert (await store.ainteraction_policy("dm-1"))[0].availability_schedule == schedule
    await store.aclose()


async def test_engagement_window_state_survives_restart_and_policy_change_closes_it(tmp_path: Path):
    path = tmp_path / "state.sqlite3"
    store = await Store.open(path, SECRET)
    await store.aupsert_conversation("dm-1", "peer-1", "Sam")
    engagement = EngagementWindow(duration_seconds=600, max_followup_turns=4)
    policy = replace(
        preset_policy("on_invocation"),
        invocation_turn_lifetime="engagement_window",
        engagement_window=engagement,
    )
    assert await store.aset_interaction_policy("dm-1", policy)
    _, policy_version, _ = await store.ainteraction_policy("dm-1")
    await store.aactivate_engagement(
        "dm-1",
        duration_seconds=engagement.duration_seconds,
        max_followup_turns=engagement.max_followup_turns,
        policy_version=policy_version,
    )
    await store.aclose()

    store = await Store.open(path, SECRET)
    active = await store.aengagement("dm-1")
    assert active is not None
    assert active["remaining_turns"] == 4
    assert active["policy_version"] == policy_version
    assert await store.aset_interaction_policy("dm-1", preset_policy("shared"))
    assert await store.aengagement("dm-1") is None
    await store.aclose()


async def test_inherited_policy_and_dynamic_name_changes_have_distinct_effective_versions(
    tmp_path: Path,
):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aset_automation_settings(
        replace(store.automation_settings(), default_interaction_preset="on_invocation")
    )
    await store.aupsert_conversation("dm-1", "peer-1", "Sam")

    policy, original_version, inherited = await store.ainteraction_policy("dm-1")
    assert policy.preset == "on_invocation"
    assert inherited is True
    await store.aset_assistant_profile(replace(store.assistant_profile(), assistant_name="Nova"))
    renamed_policy, renamed_version, renamed_inherited = await store.ainteraction_policy("dm-1")

    assert renamed_policy == policy
    assert renamed_inherited is True
    assert renamed_version != original_version
    assert await store.aset_interaction_policy("dm-1", renamed_policy)
    assert (await store.ainteraction_policy("dm-1"))[2] is False
    assert await store.areset_interaction_policy("dm-1")
    assert (await store.ainteraction_policy("dm-1"))[2] is True
    await store.aclose()


async def test_complete_global_policy_is_inherited_and_reset_restores_inheritance(tmp_path: Path):
    store = await Store.open(tmp_path / "state.sqlite3", SECRET)
    await store.aupsert_conversation("dm-1", "peer-1", "Sam")
    global_policy = replace(
        preset_policy("on_invocation"),
        trigger_participants=frozenset({"owner"}),
        active_turn_input=replace(
            preset_policy("on_invocation").active_turn_input,
            participants=frozenset({"owner"}),
        ),
    )

    await store.aset_default_interaction_policy(global_policy)
    inherited, _, is_inherited = await store.ainteraction_policy("dm-1")
    assert inherited == global_policy
    assert is_inherited

    assert await store.aset_interaction_policy("dm-1", preset_policy("shared"))
    assert (await store.ainteraction_policy("dm-1"))[0].preset == "shared"
    assert await store.areset_interaction_policy("dm-1")
    assert (await store.ainteraction_policy("dm-1"))[0] == global_policy
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
    saved = (await store.ahistory("dm", 1))[0]
    assert saved["content"] == "edited"
    assert saved["source"] == "human"
    assert saved["edited_at"] is not None
    assert saved["deleted_at"] is None
    assert await store.amark_message_deleted("message", deleted_at=200) is True
    assert await store.amark_message_deleted("message", deleted_at=300) is False
    assert (await store.ahistory("dm", 1))[0]["deleted_at"] == 200
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
