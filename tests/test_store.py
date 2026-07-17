from pathlib import Path

from diskovod.models import AppSettings, ChatCredentials, CustomProvider
from diskovod.store import Store


SECRET = "x" * 32


def test_app_settings_persist_silent_replies(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", SECRET)

    assert store.app_settings().silent_replies is False
    store.set_app_settings(AppSettings(silent_replies=True))
    assert store.app_settings().silent_replies is True
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
