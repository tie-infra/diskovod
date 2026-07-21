from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiosqlite

from .localization import assistant_name_for, invocation_attention_words, normalize_locale
from .interaction import InteractionPolicy, preset_policy, validate_policy
from .models import (
    ADMIN_DENSITIES,
    ADMIN_THEMES,
    DEFAULT_BASE_INSTRUCTIONS,
    AssistantProfile,
    AutomationSettings,
    ChatCredentials,
    CustomProvider,
    InterfaceSettings,
)
from .persistence import AsyncSQLite, initialize_target_schema
from .security import SecretBox

LEGACY_BASE_INSTRUCTIONS_SHA256 = "ce9bd3d8ffbef462362269db68c7996d9ca0e3e93761d197fcf82f5e0f25502c"
DATABASE_TABLES = {
    "config": {"label": "Configuration", "primary_key": "key", "order_by": "updated_at", "read_only": True},
    "conversations": {
        "label": "Conversations",
        "primary_key": "channel_id",
        "order_by": "updated_at",
        "read_only": False,
    },
    "messages": {"label": "Messages", "primary_key": "id", "order_by": "timestamp", "read_only": False},
    "assistant_reactions": {
        "label": "Assistant reactions",
        "primary_key": "trigger_message_id",
        "order_by": "created_at",
        "read_only": False,
    },
    "bot_nonces": {
        "label": "Pending nonces",
        "primary_key": "nonce",
        "order_by": "created_at",
        "read_only": False,
    },
    "bot_message_ids": {
        "label": "Assistant message markers",
        "primary_key": "id",
        "order_by": "created_at",
        "read_only": False,
    },
    "agent_configuration_versions": {
        "label": "Agent configurations",
        "primary_key": "id",
        "order_by": "created_at",
        "read_only": True,
    },
    "chat_threads": {
        "label": "Graph threads",
        "primary_key": "channel_id",
        "order_by": "updated_at",
        "read_only": True,
    },
    "conversation_events": {
        "label": "Conversation events",
        "primary_key": "id",
        "order_by": "observed_at",
        "read_only": True,
    },
    "agent_work": {
        "label": "Agent work",
        "primary_key": "id",
        "order_by": "created_at",
        "read_only": True,
    },
    "chat_interaction_policies": {
        "label": "Chat interaction policies",
        "primary_key": "channel_id",
        "order_by": "updated_at",
        "read_only": True,
    },
    "outbound_actions": {
        "label": "Outbound actions",
        "primary_key": "id",
        "order_by": "created_at",
        "read_only": True,
    },
    "conversation_waits": {
        "label": "Conversation waits",
        "primary_key": "id",
        "order_by": "created_at",
        "read_only": True,
    },
    "agent_runs": {
        "label": "Agent runs",
        "primary_key": "id",
        "order_by": "started_at",
        "read_only": True,
    },
    "agent_trace_events": {
        "label": "Agent trace events",
        "primary_key": "id",
        "order_by": "recorded_at",
        "read_only": True,
    },
    "provider_capability_probes": {
        "label": "Capability probes",
        "primary_key": "id",
        "order_by": "completed_at",
        "read_only": True,
    },
    "attachment_objects": {
        "label": "Attachment objects",
        "primary_key": "sha256",
        "order_by": "created_at",
        "read_only": True,
    },
    "attachment_references": {
        "label": "Attachment references",
        "primary_key": "id",
        "order_by": "created_at",
        "read_only": False,
    },
    "attachment_artifacts": {
        "label": "Attachment artifacts",
        "primary_key": "id",
        "order_by": "updated_at",
        "read_only": True,
    },
    "attachment_chunks": {
        "label": "Attachment chunks",
        "primary_key": "id",
        "order_by": "id",
        "read_only": True,
    },
    "escalation_interrupts": {
        "label": "Graph interrupts",
        "primary_key": "id",
        "order_by": "updated_at",
        "read_only": True,
    },
    "langgraph_store_items": {
        "label": "Long-term memories",
        "primary_key": "key",
        "order_by": "updated_at",
        "read_only": True,
    },
    "legacy_import_records": {
        "label": "Archived pre-LangGraph records",
        "primary_key": "source_id",
        "order_by": "imported_at",
        "read_only": True,
    },
    "chat_thread_generations": {
        "label": "Chat thread generations",
        "primary_key": "thread_id",
        "order_by": "created_at",
        "read_only": True,
    },
    "checkpoint_index": {
        "label": "Checkpoint index",
        "primary_key": "checkpoint_id",
        "order_by": "created_at",
        "read_only": True,
    },
    "admin_jobs": {
        "label": "Administrative jobs",
        "primary_key": "id",
        "order_by": "requested_at",
        "read_only": True,
    },
    "admin_job_events": {
        "label": "Administrative job events",
        "primary_key": "id",
        "order_by": "occurred_at",
        "read_only": True,
    },
    "provider_setup_drafts": {
        "label": "Provider setup drafts",
        "primary_key": "id",
        "order_by": "created_at",
        "read_only": True,
    },
    "admin_job_inputs": {
        "label": "Encrypted administrative job inputs",
        "primary_key": "id",
        "order_by": "created_at",
        "read_only": True,
    },
}


class Store:
    def __init__(self, path: Path, secret: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.database = AsyncSQLite(path)
        self._cache_lock = threading.RLock()
        self._cache_write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._started = False
        self._box = SecretBox(secret)
        self._config: dict[str, Any] = {}
        self._active_configuration: dict[str, Any] | None = None

    @classmethod
    async def open(cls, path: Path, secret: str) -> Store:
        store = cls(path, secret)
        await store.start()
        return store

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            async with self.database.transaction() as connection:
                await initialize_target_schema(connection)
                await self._migrate_app_settings(connection)
                await self._load(connection)
            self._started = True

    async def _migrate_app_settings(self, connection: aiosqlite.Connection) -> None:
        """Atomically split the former mixed settings object into owned domains."""
        row = await (
            await connection.execute("SELECT value, secret FROM config WHERE key='app.settings'")
        ).fetchone()
        if row is None:
            return
        saved = self._decode_config(str(row["value"]), bool(row["secret"]))
        if not isinstance(saved, dict):
            saved = {}

        base_instructions = saved.get("base_instructions")
        if (
            isinstance(base_instructions, str)
            and hashlib.sha256(base_instructions.encode()).hexdigest() == LEGACY_BASE_INSTRUCTIONS_SHA256
        ):
            base_instructions = DEFAULT_BASE_INSTRUCTIONS

        interface = InterfaceSettings(
            locale=normalize_locale(str(saved.get("admin_locale", "en"))),
            theme=(
                str(saved.get("admin_theme", "system"))
                if saved.get("admin_theme") in ADMIN_THEMES
                else "system"
            ),
        )
        assistant = AssistantProfile(
            prompt_locale=normalize_locale(str(saved.get("prompt_locale", "en"))),
            assistant_name=str(saved.get("assistant_name", "")),
            owner_timezone=str(saved.get("owner_timezone", "UTC")),
            owner_details=str(saved.get("owner_details", "")),
            base_instructions=(
                base_instructions if isinstance(base_instructions, str) else DEFAULT_BASE_INSTRUCTIONS
            ),
        )
        automation_defaults = AutomationSettings().to_dict()
        automation = AutomationSettings(
            **{key: saved.get(key, default) for key, default in automation_defaults.items()}
        )
        now = time.time()
        for key, value in (
            ("admin.interface", interface.to_dict()),
            ("assistant.profile", assistant.to_dict()),
            ("automation.settings", automation.to_dict()),
            (
                "legacy.model_selection",
                {
                    field: saved[field]
                    for field in ("provider", "model", "reasoning_effort", "max_reply_tokens")
                    if field in saved
                },
            ),
        ):
            await connection.execute(
                "INSERT INTO config(key, value, secret, updated_at) VALUES(?, ?, 0, ?) "
                "ON CONFLICT(key) DO NOTHING",
                (key, json.dumps(value), now),
            )
        await connection.execute("DELETE FROM config WHERE key='app.settings'")

    async def _load(self, connection: aiosqlite.Connection) -> None:
        message_columns = {
            row["name"] for row in await (await connection.execute("PRAGMA table_info(messages)")).fetchall()
        }
        if "attachments" not in message_columns:
            await connection.execute("ALTER TABLE messages ADD COLUMN attachments TEXT NOT NULL DEFAULT '[]'")
        config = {
            str(row["key"]): self._decode_config(row["value"], bool(row["secret"]))
            for row in await (await connection.execute("SELECT key, value, secret FROM config")).fetchall()
        }
        active = await (
            await connection.execute("SELECT configuration FROM agent_configuration_versions WHERE active=1")
        ).fetchone()
        with self._cache_lock:
            self._config = config
            self._active_configuration = json.loads(active["configuration"]) if active else None

    async def aclose(self) -> None:
        await self.database.close()

    def _decode_config(self, raw: str, secret: bool) -> Any:
        value = self._box.open(raw) if secret else raw
        return json.loads(value)

    async def _aset(self, key: str, value: Any, *, secret: bool = False) -> None:
        await self.start()
        raw = json.dumps(value)
        if secret:
            raw = self._box.seal(raw)
        async with self._cache_write_lock:
            async with self.database.transaction() as connection:
                await connection.execute(
                    "INSERT INTO config VALUES(?,?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                    "secret=excluded.secret, updated_at=excluded.updated_at",
                    (key, raw, int(secret), time.time()),
                )
            with self._cache_lock:
                self._config[key] = deepcopy(value)

    async def _adelete(self, key: str) -> None:
        await self.start()
        async with self._cache_write_lock:
            async with self.database.transaction() as connection:
                await connection.execute("DELETE FROM config WHERE key=?", (key,))
            with self._cache_lock:
                self._config.pop(key, None)

    def _get(self, key: str, default: Any) -> Any:
        with self._cache_lock:
            return deepcopy(self._config.get(key, default))

    def interface_settings(self) -> InterfaceSettings:
        saved = self._get("admin.interface", {})
        if not isinstance(saved, dict):
            saved = {}
        saved["locale"] = normalize_locale(str(saved.get("locale", "en")))
        saved["theme"] = str(saved.get("theme", "system")) if saved.get("theme") in ADMIN_THEMES else "system"
        saved["density"] = (
            str(saved.get("density", "comfortable"))
            if saved.get("density") in ADMIN_DENSITIES
            else "comfortable"
        )
        defaults = InterfaceSettings().to_dict()
        known = {key: value for key, value in saved.items() if key in defaults}
        return InterfaceSettings(**(defaults | known))

    def assistant_profile(self) -> AssistantProfile:
        saved = self._get("assistant.profile", {})
        if not isinstance(saved, dict):
            saved = {}
        base_instructions = saved.get("base_instructions")
        if (
            isinstance(base_instructions, str)
            and hashlib.sha256(base_instructions.encode()).hexdigest() == LEGACY_BASE_INSTRUCTIONS_SHA256
        ):
            saved["base_instructions"] = DEFAULT_BASE_INSTRUCTIONS
        saved["prompt_locale"] = normalize_locale(str(saved.get("prompt_locale", "en")))
        defaults = AssistantProfile().to_dict()
        known = {key: value for key, value in saved.items() if key in defaults}
        return AssistantProfile(**(defaults | known))

    def automation_settings(self) -> AutomationSettings:
        saved = self._get("automation.settings", {})
        if not isinstance(saved, dict):
            saved = {}
        defaults = AutomationSettings().to_dict()
        known = {key: value for key, value in saved.items() if key in defaults}
        result = AutomationSettings(**(defaults | known))
        if result.default_interaction_preset not in {
            "autonomous",
            "shared",
            "on_invocation",
            "manual",
            "draft",
        }:
            result.default_interaction_preset = "autonomous"
        return result

    def default_interaction_policy(self) -> InteractionPolicy:
        settings = self.automation_settings()
        saved = self._get("interaction.default_policy", None)
        if isinstance(saved, dict):
            try:
                policy = InteractionPolicy.from_dict(saved)
                profile = self.assistant_profile()
                validate_policy(
                    policy,
                    assistant_name=assistant_name_for(profile.prompt_locale, profile.assistant_name),
                    supported_attention_locales=frozenset(invocation_attention_words()),
                )
                return policy
            except (KeyError, TypeError, ValueError):
                pass
        return preset_policy(
            settings.default_interaction_preset,  # type: ignore[arg-type]
            prompt_locale=self.assistant_profile().prompt_locale,
        )

    def discord_token(self) -> str | None:
        return self._get("discord.token", None)

    def chat_credentials(self) -> ChatCredentials | None:
        value = self._get("chatgpt.credentials", None)
        return ChatCredentials(**value) if value else None

    def custom_provider(self) -> CustomProvider | None:
        value = self._get("openai_compatible.provider", None)
        if not value:
            return None
        # Providers saved before protocol selection existed used Chat Completions.
        value.setdefault("protocol", "chat_completions")
        # Providers saved before this capability existed always received the limit.
        value.setdefault("capabilities", {}).setdefault("output_token_limit", True)
        return CustomProvider(**value)

    def active_agent_configuration(self):
        from .providers import ModelConfiguration

        with self._cache_lock:
            value = deepcopy(self._active_configuration)
        return ModelConfiguration.from_dict(value) if value else None

    def provider_credentials(self, profile: str) -> dict[str, Any] | None:
        return self._get(f"provider.credentials.{profile}", None)

    def personality(self) -> dict[str, Any] | None:
        return self._get("personality", None)

    @staticmethod
    def _conversation_dict(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "channel_id": row["channel_id"],
            "peer_id": row["peer_id"],
            "peer_name": row["peer_name"],
            "availability": row["availability"],
            "paused_at": row["paused_at"],
            "snoozed_until": row["snoozed_until"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _message_dict(row: Mapping[str, Any]) -> dict[str, Any]:
        item = dict(row)
        try:
            attachments = json.loads(item.get("attachments") or "[]")
        except (TypeError, json.JSONDecodeError):
            attachments = []
        item["attachments"] = attachments if isinstance(attachments, list) else []
        return item

    async def aset_interface_settings(self, value: InterfaceSettings) -> None:
        await self._aset("admin.interface", value.to_dict())

    async def aset_assistant_profile(self, value: AssistantProfile) -> None:
        await self._aset("assistant.profile", value.to_dict())

    async def aset_automation_settings(self, value: AutomationSettings) -> None:
        await self._aset("automation.settings", value.to_dict())

    async def aset_default_interaction_policy(self, policy: InteractionPolicy) -> None:
        profile = self.assistant_profile()
        validate_policy(
            policy,
            assistant_name=assistant_name_for(profile.prompt_locale, profile.assistant_name),
            supported_attention_locales=frozenset(invocation_attention_words()),
        )
        await self._aset("interaction.default_policy", policy.to_dict())
        settings = self.automation_settings()
        if settings.default_interaction_preset != policy.preset:
            await self.aset_automation_settings(
                AutomationSettings(**(settings.to_dict() | {"default_interaction_preset": policy.preset}))
            )

    async def areset_default_interaction_policy(self) -> None:
        await self._adelete("interaction.default_policy")

    async def aset_discord_token(self, value: str) -> None:
        await self._aset("discord.token", value, secret=True)

    async def aclear_discord_token(self) -> None:
        await self._adelete("discord.token")

    async def aset_chat_credentials(self, value: ChatCredentials) -> None:
        await self._aset("chatgpt.credentials", value.to_dict(), secret=True)

    async def aclear_chat_credentials(self) -> None:
        await self._adelete("chatgpt.credentials")

    async def aset_custom_provider(self, value: CustomProvider) -> None:
        await self._aset("openai_compatible.provider", value.to_dict(), secret=True)

    async def aclear_custom_provider(self) -> None:
        await self._adelete("openai_compatible.provider")

    async def aset_provider_credentials(self, profile: str, value: dict[str, Any]) -> None:
        if not profile or not profile.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid credential profile")
        await self._aset(f"provider.credentials.{profile}", value, secret=True)

    async def aclear_provider_credentials(self, profile: str) -> None:
        await self._adelete(f"provider.credentials.{profile}")

    async def aset_personality(self, profile: str, source_hash: str, source: str = "inferred") -> None:
        await self._aset(
            "personality",
            {
                "profile": profile,
                "source_hash": source_hash,
                "source": source,
                "updated_at": time.time(),
            },
        )

    async def aactive_interrupts(self) -> list[dict[str, Any]]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT i.*, c.peer_name FROM escalation_interrupts i
                    LEFT JOIN conversations c ON c.channel_id=i.channel_id
                    WHERE i.state IN ('pending', 'claimed')
                    ORDER BY i.created_at DESC
                    """
                )
            ).fetchall()
        return [self._interrupt_dict(row) for row in rows]

    async def aescalation_interrupt(self, escalation_id: str) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT * FROM escalation_interrupts WHERE id=?", (escalation_id,))
            ).fetchone()
        return self._interrupt_dict(row) if row else None

    async def aactive_interrupt_for_channel(self, channel_id: str) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT * FROM escalation_interrupts
                    WHERE channel_id=? AND state IN ('pending', 'claimed')
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (channel_id,),
                )
            ).fetchone()
        return self._interrupt_dict(row) if row else None

    @staticmethod
    def _interrupt_dict(row: Mapping[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = json.loads(item["payload"])
        return item

    async def aconversation(self, channel_id: str) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT * FROM conversations WHERE channel_id=?", (channel_id,))
            ).fetchone()
        return self._conversation_dict(row) if row else None

    async def aconversations(self) -> list[dict[str, Any]]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute("SELECT * FROM conversations ORDER BY updated_at DESC LIMIT 200")
            ).fetchall()
        return [self._conversation_dict(row) for row in rows]

    async def acan_automate(self, channel_id: str, now: float | None = None) -> bool:
        conversation = await self.aconversation(channel_id)
        if not conversation or conversation["availability"] == "paused":
            return False
        current_time = time.time() if now is None else now
        return not conversation["snoozed_until"] or conversation["snoozed_until"] <= current_time

    async def ahistory(self, channel_id: str, limit: int) -> list[dict[str, Any]]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    "SELECT * FROM messages WHERE channel_id=? ORDER BY timestamp DESC LIMIT ?",
                    (channel_id, limit),
                )
            ).fetchall()
        return [self._message_dict(row) for row in reversed(rows)]

    async def alatest_incoming_message(self, channel_id: str) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    """SELECT * FROM messages
                       WHERE channel_id=? AND direction='in'
                       ORDER BY timestamp DESC LIMIT 1""",
                    (channel_id,),
                )
            ).fetchone()
        return self._message_dict(row) if row else None

    async def ais_assistant_message(self, message_id: str) -> bool:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT 1 FROM messages WHERE id=? AND source='assistant'", (message_id,)
                )
            ).fetchone()
        return row is not None

    async def areaction_allowed(
        self,
        channel_id: str,
        *,
        now: float | None = None,
        channel_cooldown_seconds: float = 6 * 60 * 60,
        recent_action_limit: int = 12,
    ) -> bool:
        current_time = time.time() if now is None else now
        async with self.database.transaction() as connection:
            recent_in_channel = await (
                await connection.execute(
                    """SELECT 1 FROM assistant_reactions
                       WHERE channel_id=? AND created_at>=? LIMIT 1""",
                    (channel_id, current_time - channel_cooldown_seconds),
                )
            ).fetchone()
            if recent_in_channel:
                return False
            recent_actions = await (
                await connection.execute(
                    """SELECT kind FROM (
                         SELECT timestamp AS action_time, 'message' AS kind
                           FROM messages WHERE source='assistant'
                         UNION ALL
                         SELECT created_at AS action_time, 'reaction' AS kind
                           FROM assistant_reactions
                       ) ORDER BY action_time DESC LIMIT ?""",
                    (max(1, recent_action_limit),),
                )
            ).fetchall()
        return all(row["kind"] != "reaction" for row in recent_actions)

    async def asave_agent_configuration(self, configuration) -> int:
        value = configuration.to_dict()
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        async with self._cache_write_lock:
            async with self.database.transaction() as connection:
                await connection.execute("UPDATE agent_configuration_versions SET active=0 WHERE active=1")
                cursor = await connection.execute(
                    "INSERT INTO agent_configuration_versions(created_at, configuration, active) "
                    "VALUES(?, ?, 1)",
                    (time.time(), payload),
                )
                configuration_id = int(cursor.lastrowid)
            with self._cache_lock:
                self._active_configuration = deepcopy(value)
        return configuration_id

    async def aactive_configuration_id(self) -> int | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT id FROM agent_configuration_versions WHERE active=1")
            ).fetchone()
        return int(row["id"]) if row else None

    async def aagent_configuration(self, configuration_id: int):
        from .providers import ModelConfiguration

        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT configuration FROM agent_configuration_versions WHERE id=?",
                    (configuration_id,),
                )
            ).fetchone()
        return ModelConfiguration.from_dict(json.loads(row["configuration"])) if row else None

    async def acreate_provider_setup_draft(
        self,
        draft_id: str,
        payload: dict[str, Any],
        fingerprint: str,
        *,
        expires_at: float,
    ) -> None:
        encoded = self._box.seal(json.dumps(payload, ensure_ascii=False))
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                INSERT INTO provider_setup_drafts(
                  id, payload, secret, fingerprint, created_at, expires_at
                ) VALUES(?, ?, 1, ?, ?, ?)
                """,
                (draft_id, encoded, fingerprint, time.time(), expires_at),
            )

    async def aprovider_setup_draft(self, draft_id: str) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT payload, secret, fingerprint, expires_at FROM provider_setup_drafts "
                    "WHERE id=? AND expires_at>?",
                    (draft_id, time.time()),
                )
            ).fetchone()
        if row is None:
            return None
        payload = self._decode_config(str(row["payload"]), bool(row["secret"]))
        return {
            "payload": payload,
            "fingerprint": str(row["fingerprint"]),
            "expires_at": float(row["expires_at"]),
        }

    async def aupdate_provider_setup_draft(self, draft_id: str, payload: dict[str, Any]) -> bool:
        encoded = self._box.seal(json.dumps(payload, ensure_ascii=False))
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                "UPDATE provider_setup_drafts SET payload=?, secret=1 WHERE id=? AND expires_at>?",
                (encoded, draft_id, time.time()),
            )
            return cursor.rowcount == 1

    async def adelete_provider_setup_draft(self, draft_id: str) -> None:
        async with self.database.transaction() as connection:
            await connection.execute("DELETE FROM provider_setup_drafts WHERE id=?", (draft_id,))

    async def acreate_admin_job_input(
        self,
        input_id: str,
        payload: dict[str, Any],
        *,
        expires_at: float,
    ) -> None:
        encoded = self._box.seal(json.dumps(payload, ensure_ascii=False))
        async with self.database.transaction() as connection:
            await connection.execute(
                "INSERT INTO admin_job_inputs(id, payload, secret, created_at, expires_at) "
                "VALUES(?, ?, 1, ?, ?)",
                (input_id, encoded, time.time(), expires_at),
            )

    async def aadmin_job_input(self, input_id: str) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT payload, secret FROM admin_job_inputs WHERE id=? AND expires_at>?",
                    (input_id, time.time()),
                )
            ).fetchone()
        return self._decode_config(str(row["payload"]), bool(row["secret"])) if row else None

    async def adelete_admin_job_input(self, input_id: str) -> None:
        async with self.database.transaction() as connection:
            await connection.execute("DELETE FROM admin_job_inputs WHERE id=?", (input_id,))

    async def astart_agent_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        channel_id: str,
        trace_id: str,
        trigger_kind: str | None = None,
        trigger_message_id: str | None = None,
    ) -> None:
        async with self.database.transaction() as connection:
            await connection.execute(
                """INSERT INTO agent_runs(
                     id, thread_id, channel_id, trace_id, status, started_at,
                     configuration_version_id, trigger_kind, trigger_message_id
                   ) VALUES(?, ?, ?, ?, 'running', ?,
                     (SELECT id FROM agent_configuration_versions WHERE active=1), ?, ?)""",
                (
                    run_id,
                    thread_id,
                    channel_id,
                    trace_id,
                    time.time(),
                    trigger_kind,
                    trigger_message_id,
                ),
            )

    async def afinish_agent_run(self, run_id: str, status: str, error: str | None = None) -> None:
        if status not in {"completed", "interrupted", "failed", "cancelled"}:
            raise ValueError("Invalid agent run status")
        async with self.database.transaction() as connection:
            await connection.execute(
                "UPDATE agent_runs SET status=?, completed_at=?, error=? WHERE id=?",
                (status, time.time(), error, run_id),
            )

    async def arecord_agent_trace(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM agent_trace_events WHERE run_id=?",
                    (run_id,),
                )
            ).fetchone()
            await connection.execute(
                "INSERT INTO agent_trace_events(run_id, sequence, kind, payload, recorded_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (
                    run_id,
                    int(row[0]),
                    kind,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    time.time(),
                ),
            )

    async def aupsert_conversation(self, channel_id: str, peer_id: str, peer_name: str) -> None:
        now = time.time()
        default_paused = not self.automation_settings().default_conversation_enabled
        async with self.database.transaction() as connection:
            await connection.execute(
                """INSERT INTO conversations
                   (channel_id, peer_id, peer_name, availability, paused_at, updated_at, snoozed_until)
                   VALUES(?,?,?,?,?,?,NULL)
                   ON CONFLICT(channel_id) DO UPDATE SET
                     peer_id=excluded.peer_id, peer_name=excluded.peer_name,
                     updated_at=excluded.updated_at""",
                (
                    channel_id,
                    peer_id,
                    peer_name,
                    "paused" if default_paused else "active",
                    now if default_paused else None,
                    now,
                ),
            )

    async def aset_permanent_pause(self, channel_id: str, paused: bool) -> None:
        now = time.time()
        async with self.database.transaction() as connection:
            await connection.execute(
                "UPDATE conversations SET availability=?, paused_at=?, updated_at=? WHERE channel_id=?",
                ("paused" if paused else "active", now if paused else None, now, channel_id),
            )

    async def ainteraction_policy(
        self,
        channel_id: str,
    ) -> tuple[InteractionPolicy, int, bool]:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT * FROM chat_interaction_policies WHERE channel_id=?",
                    (channel_id,),
                )
            ).fetchone()
        if row is None:
            policy = self.default_interaction_policy()
            return policy, self._effective_policy_version(policy, 0), True
        value = {
            "preset": row["preset"],
            "trigger_rules": json.loads(row["trigger_rules"]),
            "trigger_participants": json.loads(row["trigger_participants"]),
            "owner_handoff": json.loads(row["owner_handoff"]),
            "conversation_role": row["conversation_role"],
            "identity_marker": row["identity_marker"],
            "delivery": row["delivery"],
            "active_turn_input": json.loads(row["active_turn_input"]),
            "availability_schedule": json.loads(row["availability_schedule"]),
            "engagement_window": json.loads(row["engagement_window"]),
            "invocation_snooze_behavior": row["invocation_snooze_behavior"],
            "invocation_turn_lifetime": row["invocation_turn_lifetime"],
        }
        policy = InteractionPolicy.from_dict(value)
        return policy, self._effective_policy_version(policy, int(row["policy_version"])), False

    def _effective_policy_version(self, policy: InteractionPolicy, stored_version: int) -> int:
        profile = self.assistant_profile()
        material = "\0".join(
            (
                str(stored_version),
                policy.encoded(),
                profile.prompt_locale,
                profile.assistant_name,
            )
        )
        return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") & ((1 << 63) - 1)

    async def aset_interaction_policy(self, channel_id: str, policy: InteractionPolicy) -> bool:
        profile = self.assistant_profile()
        validate_policy(
            policy,
            assistant_name=assistant_name_for(profile.prompt_locale, profile.assistant_name),
            supported_attention_locales=frozenset(invocation_attention_words()),
        )
        now = time.time()
        async with self.database.transaction() as connection:
            if not await (
                await connection.execute("SELECT 1 FROM conversations WHERE channel_id=?", (channel_id,))
            ).fetchone():
                return False
            await connection.execute(
                """
                INSERT INTO chat_interaction_policies(
                  channel_id, preset, trigger_rules, trigger_participants, owner_handoff,
                  conversation_role, identity_marker, delivery, active_turn_input,
                  invocation_snooze_behavior, invocation_turn_lifetime, availability_schedule,
                  engagement_window, policy_version, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                  preset=excluded.preset,
                  trigger_rules=excluded.trigger_rules,
                  trigger_participants=excluded.trigger_participants,
                  owner_handoff=excluded.owner_handoff,
                  conversation_role=excluded.conversation_role,
                  identity_marker=excluded.identity_marker,
                  delivery=excluded.delivery,
                  active_turn_input=excluded.active_turn_input,
                  invocation_snooze_behavior=excluded.invocation_snooze_behavior,
                  invocation_turn_lifetime=excluded.invocation_turn_lifetime,
                  availability_schedule=excluded.availability_schedule,
                  engagement_window=excluded.engagement_window,
                  policy_version=chat_interaction_policies.policy_version + 1,
                  updated_at=excluded.updated_at
                """,
                (
                    channel_id,
                    policy.preset,
                    json.dumps([asdict(rule) for rule in policy.trigger_rules], ensure_ascii=False),
                    json.dumps(sorted(policy.trigger_participants)),
                    json.dumps(asdict(policy.owner_handoff)),
                    policy.conversation_role,
                    policy.identity_marker,
                    policy.delivery,
                    json.dumps(
                        {
                            "timing": policy.active_turn_input.timing,
                            "participants": sorted(policy.active_turn_input.participants),
                        }
                    ),
                    policy.invocation_snooze_behavior,
                    policy.invocation_turn_lifetime,
                    json.dumps(
                        {
                            "enabled": policy.availability_schedule.enabled,
                            "weekdays": sorted(policy.availability_schedule.weekdays),
                            "start_minute": policy.availability_schedule.start_minute,
                            "end_minute": policy.availability_schedule.end_minute,
                            "timezone": policy.availability_schedule.timezone,
                        }
                    ),
                    json.dumps(
                        {
                            "duration_seconds": policy.engagement_window.duration_seconds,
                            "max_followup_turns": policy.engagement_window.max_followup_turns,
                        }
                    ),
                    now,
                ),
            )
            await connection.execute("DELETE FROM conversation_engagements WHERE channel_id=?", (channel_id,))
        return True

    async def areset_interaction_policy(self, channel_id: str) -> bool:
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                "DELETE FROM chat_interaction_policies WHERE channel_id=?", (channel_id,)
            )
            await connection.execute("DELETE FROM conversation_engagements WHERE channel_id=?", (channel_id,))
        return cursor.rowcount > 0

    async def asnooze(self, channel_id: str, seconds: float) -> float:
        until = time.time() + max(0.0, seconds)
        async with self.database.transaction() as connection:
            await connection.execute(
                "UPDATE conversations SET snoozed_until=?, updated_at=? WHERE channel_id=?",
                (until, time.time(), channel_id),
            )
        return until

    async def aclear_snooze(self, channel_id: str) -> None:
        async with self.database.transaction() as connection:
            await connection.execute(
                "UPDATE conversations SET snoozed_until=NULL, updated_at=? WHERE channel_id=?",
                (time.time(), channel_id),
            )

    async def aengagement(self, channel_id: str, *, now: float | None = None) -> dict[str, Any] | None:
        timestamp = time.time() if now is None else now
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT * FROM conversation_engagements WHERE channel_id=?",
                    (channel_id,),
                )
            ).fetchone()
            if row is not None and (
                float(row["expires_at"]) <= timestamp or int(row["remaining_turns"]) <= 0
            ):
                await connection.execute(
                    "DELETE FROM conversation_engagements WHERE channel_id=?", (channel_id,)
                )
                row = None
        return dict(row) if row is not None else None

    async def aactivate_engagement(
        self,
        channel_id: str,
        *,
        duration_seconds: int,
        max_followup_turns: int,
        policy_version: int,
    ) -> None:
        now = time.time()
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                INSERT INTO conversation_engagements(
                  channel_id, expires_at, remaining_turns, policy_version, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                  expires_at=excluded.expires_at,
                  remaining_turns=excluded.remaining_turns,
                  policy_version=excluded.policy_version,
                  created_at=excluded.created_at,
                  updated_at=excluded.updated_at
                """,
                (
                    channel_id,
                    now + duration_seconds,
                    max_followup_turns,
                    policy_version,
                    now,
                    now,
                ),
            )

    async def atouch_engagement(self, channel_id: str, *, duration_seconds: int) -> None:
        now = time.time()
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                UPDATE conversation_engagements SET expires_at=?, updated_at=?
                WHERE channel_id=? AND remaining_turns>0
                """,
                (now + duration_seconds, now, channel_id),
            )

    async def aclose_engagement(self, channel_id: str) -> bool:
        async with self.database.transaction() as connection:
            changed = (
                await connection.execute(
                    "DELETE FROM conversation_engagements WHERE channel_id=?", (channel_id,)
                )
            ).rowcount
        return changed > 0

    async def aset_interrupt_state(self, escalation_id: str, state: str) -> bool:
        if state not in {"claimed", "resolved", "dismissed"}:
            raise ValueError("Unknown interrupt state")
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                "UPDATE escalation_interrupts SET state=?, updated_at=? "
                "WHERE id=? AND state IN ('pending', 'claimed')",
                (state, time.time(), escalation_id),
            )
            return cursor.rowcount > 0

    async def asave_message(
        self,
        *,
        id: str,
        channel_id: str,
        author_id: str,
        author_name: str,
        direction: str,
        source: str,
        content: str,
        timestamp: float,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        async with self.database.transaction() as connection:
            await connection.execute(
                """INSERT OR IGNORE INTO messages
                   (id, channel_id, author_id, author_name, direction, source, content,
                    timestamp, attachments) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    id,
                    channel_id,
                    author_id,
                    author_name,
                    direction,
                    source,
                    content,
                    timestamp,
                    json.dumps(attachments or [], ensure_ascii=False),
                ),
            )
            await connection.execute(
                "UPDATE conversations SET updated_at=? WHERE channel_id=?", (timestamp, channel_id)
            )

    async def aupdate_message_content(
        self, message_id: str, content: str, *, source: str | None = None
    ) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT * FROM messages WHERE id=?", (message_id,))
            ).fetchone()
            if row is None:
                return None
            new_source = source or row["source"]
            changed = row["content"] != content or row["source"] != new_source
            if changed:
                await connection.execute(
                    "UPDATE messages SET content=?, source=?, edited_at=? WHERE id=?",
                    (content, new_source, time.time(), message_id),
                )
        result = dict(row)
        result.update(content=content, source=new_source, changed=changed)
        return result

    async def amark_message_deleted(self, message_id: str, *, deleted_at: float | None = None) -> bool:
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                "UPDATE messages SET deleted_at=? WHERE id=? AND deleted_at IS NULL",
                (deleted_at or time.time(), message_id),
            )
            return cursor.rowcount > 0

    async def aremember_nonce(self, nonce: str) -> None:
        async with self.database.transaction() as connection:
            await connection.execute("INSERT OR REPLACE INTO bot_nonces VALUES(?,?)", (nonce, time.time()))

    async def aconsume_nonce(self, nonce: str) -> bool:
        async with self.database.transaction() as connection:
            found = await (
                await connection.execute("SELECT 1 FROM bot_nonces WHERE nonce=?", (nonce,))
            ).fetchone()
            if found:
                await connection.execute("DELETE FROM bot_nonces WHERE nonce=?", (nonce,))
        return found is not None

    async def ais_bot_message(self, message_id: str) -> bool:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT 1 FROM bot_message_ids WHERE id=?", (message_id,))
            ).fetchone()
            return row is not None

    async def aremember_bot_message(self, message_id: str) -> None:
        async with self.database.transaction() as connection:
            await connection.execute(
                "INSERT OR REPLACE INTO bot_message_ids VALUES(?,?)", (message_id, time.time())
            )

    async def arecord_assistant_reaction(
        self,
        *,
        trigger_message_id: str,
        channel_id: str,
        emoji: str,
        created_at: float | None = None,
    ) -> None:
        timestamp = time.time() if created_at is None else created_at
        async with self.database.transaction() as connection:
            await connection.execute(
                "INSERT OR IGNORE INTO assistant_reactions VALUES(?,?,?,?)",
                (trigger_message_id, channel_id, emoji, timestamp),
            )
            await connection.execute(
                "UPDATE conversations SET updated_at=? WHERE channel_id=?",
                (timestamp, channel_id),
            )

    async def ais_assistant_reaction(self, trigger_message_id: str, emoji: str) -> bool:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT 1 FROM assistant_reactions WHERE trigger_message_id=? AND emoji=?",
                    (trigger_message_id, emoji),
                )
            ).fetchone()
        return row is not None

    async def aremove_assistant_reaction(self, trigger_message_id: str, emoji: str) -> None:
        async with self.database.transaction() as connection:
            await connection.execute(
                "DELETE FROM assistant_reactions WHERE trigger_message_id=? AND emoji=?",
                (trigger_message_id, emoji),
            )

    async def aprune(self) -> None:
        cutoff = time.time() - 86400
        async with self.database.transaction() as connection:
            await connection.execute("DELETE FROM bot_nonces WHERE created_at<?", (cutoff,))
            await connection.execute("DELETE FROM bot_message_ids WHERE created_at<?", (cutoff,))
            await connection.execute("DELETE FROM provider_setup_drafts WHERE expires_at<?", (time.time(),))
            await connection.execute("DELETE FROM admin_job_inputs WHERE expires_at<?", (time.time(),))

    async def achat_threads(self) -> list[dict[str, Any]]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute("SELECT * FROM chat_threads ORDER BY updated_at DESC")
            ).fetchall()
        return [dict(row) for row in rows]

    async def achat_thread_generations(self, channel_id: str) -> list[dict[str, Any]]:
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    "SELECT * FROM chat_thread_generations WHERE channel_id=? ORDER BY generation DESC",
                    (channel_id,),
                )
            ).fetchall()
        return [dict(row) for row in rows]

    async def aindex_checkpoints(self, checkpoints: list[dict[str, Any]]) -> None:
        if not checkpoints:
            return
        async with self.database.transaction() as connection:
            for item in checkpoints:
                thread_id = str(item["thread_id"])
                generation = await (
                    await connection.execute(
                        "SELECT 1 FROM chat_thread_generations WHERE thread_id=?", (thread_id,)
                    )
                ).fetchone()
                if generation is None:
                    identity = self._checkpoint_thread_identity(thread_id)
                    if identity is None:
                        continue
                    account_id, channel_id, generation_number = identity
                    await connection.execute(
                        """
                        INSERT OR IGNORE INTO chat_thread_generations(
                          thread_id, channel_id, account_id, generation,
                          configuration_version_id, created_at, closed_at, close_reason
                        ) VALUES(?, ?, ?, ?,
                          (SELECT id FROM agent_configuration_versions WHERE active=1), ?, ?, ?)
                        """,
                        (
                            thread_id,
                            channel_id,
                            account_id,
                            generation_number,
                            float(item["created_at"]),
                            float(item["created_at"]),
                            "historical_backfill",
                        ),
                    )
                await connection.execute(
                    """
                    INSERT INTO checkpoint_index(
                      thread_id, checkpoint_id, parent_checkpoint_id, run_id,
                      created_at, step, source, message_count
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id, checkpoint_id) DO UPDATE SET
                      parent_checkpoint_id=excluded.parent_checkpoint_id,
                      run_id=COALESCE(excluded.run_id, checkpoint_index.run_id),
                      created_at=excluded.created_at,
                      step=excluded.step,
                      source=excluded.source,
                      message_count=excluded.message_count
                    """,
                    (
                        thread_id,
                        item["checkpoint_id"],
                        item.get("parent_checkpoint_id"),
                        item.get("run_id"),
                        item["created_at"],
                        item.get("step"),
                        item.get("source"),
                        item.get("message_count", 0),
                    ),
                )

    async def aupdate_run_checkpoints(
        self, run_id: str, first_checkpoint_id: str | None, final_checkpoint_id: str | None
    ) -> None:
        async with self.database.transaction() as connection:
            await connection.execute(
                "UPDATE agent_runs SET first_checkpoint_id=?, final_checkpoint_id=? WHERE id=?",
                (first_checkpoint_id, final_checkpoint_id, run_id),
            )

    @staticmethod
    def _checkpoint_thread_identity(thread_id: str) -> tuple[str, str, int] | None:
        if not thread_id.startswith("discord:") or ":g" not in thread_id:
            return None
        prefix, generation = thread_id.rsplit(":g", 1)
        parts = prefix.split(":")
        if len(parts) != 3 or not generation.isdigit():
            return None
        return parts[1], parts[2], int(generation)

    async def achat_thread_by_id(self, thread_id: str) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT * FROM chat_thread_generations WHERE thread_id=?", (thread_id,)
                )
            ).fetchone()
        return dict(row) if row else None

    async def achat_thread_for_channel(self, channel_id: str) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT * FROM chat_threads WHERE channel_id=?", (channel_id,))
            ).fetchone()
        return dict(row) if row else None

    async def aagent_run_for_trace(self, trace_id: str) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute("SELECT * FROM agent_runs WHERE trace_id=?", (trace_id,))
            ).fetchone()
        return dict(row) if row else None

    async def ahas_active_interrupt(self, channel_id: str) -> bool:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT 1 FROM escalation_interrupts "
                    "WHERE channel_id=? AND state IN ('pending','claimed') LIMIT 1",
                    (channel_id,),
                )
            ).fetchone()
        return row is not None

    async def alatest_capability_probe(self, capability: str) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT * FROM provider_capability_probes "
                    "WHERE capability=? ORDER BY completed_at DESC LIMIT 1",
                    (capability,),
                )
            ).fetchone()
        return dict(row) if row else None

    async def adatabase_tables(self) -> list[dict[str, Any]]:
        result = []
        async with self.database.transaction() as connection:
            for name, spec in DATABASE_TABLES.items():
                row = await (await connection.execute(f'SELECT COUNT(*) FROM "{name}"')).fetchone()
                result.append({"name": name, "count": int(row[0]), **spec})
        return result

    async def adatabase_rows(
        self,
        table: str,
        *,
        limit: int = 50,
        offset: int = 0,
        query: str = "",
    ) -> dict[str, Any]:
        spec = DATABASE_TABLES.get(table)
        if spec is None:
            raise ValueError("Unknown database table")
        page_limit = max(1, min(limit, 100))
        page_offset = max(0, offset)
        async with self.database.transaction() as connection:
            column_rows = await (await connection.execute(f'PRAGMA table_info("{table}")')).fetchall()
            columns = [str(row["name"]) for row in column_rows]
            searchable = ["key"] if table == "config" else columns
            parameters: list[Any] = []
            where = ""
            if query:
                where = " WHERE " + " OR ".join(f'CAST("{column}" AS TEXT) LIKE ?' for column in searchable)
                parameters.extend([f"%{query}%"] * len(searchable))
            total_row = await (
                await connection.execute(f'SELECT COUNT(*) FROM "{table}"{where}', parameters)
            ).fetchone()
            rows = await (
                await connection.execute(
                    f'SELECT * FROM "{table}"{where} ORDER BY "{spec["order_by"]}" DESC LIMIT ? OFFSET ?',
                    (*parameters, page_limit, page_offset),
                )
            ).fetchall()
        values = [dict(row) for row in rows]
        if table == "config":
            for row in values:
                if row["secret"]:
                    row["value"] = "[encrypted value redacted]"
        return {
            "name": table,
            "label": spec["label"],
            "primary_key": spec["primary_key"],
            "read_only": spec["read_only"],
            "columns": columns,
            "rows": values,
            "total": int(total_row[0]),
            "limit": page_limit,
            "offset": page_offset,
            "query": query,
        }

    async def adelete_database_row(self, table: str, row_key: str) -> bool:
        spec = DATABASE_TABLES.get(table)
        if spec is None:
            raise ValueError("Unknown database table")
        if spec["read_only"]:
            raise ValueError("This database table is read-only")
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                f'DELETE FROM "{table}" WHERE "{spec["primary_key"]}"=?',
                (row_key,),
            )
            return cursor.rowcount > 0
