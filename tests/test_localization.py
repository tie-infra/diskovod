import copy
import json
import re
from collections import Counter
from pathlib import Path
from string import Formatter
from typing import Any, cast

import pytest

from diskovod.agent import AgentPrompt
from diskovod.agent_tools import localized_agent_tools
from diskovod.localization import (
    ASSISTANT_IDENTITIES,
    ASSISTANT_NAMES,
    INLINE_TOOL_TEXT,
    PROMPTS,
    RUNTIME_CONTEXT_TEXT,
    SUPPORTED_LOCALES,
    TOOL_TEXT,
    TOOL_POLICIES,
    UI_TEXT,
    PromptBundle,
    assistant_identity,
    assistant_name_for,
    prompts_for,
    tool_policy,
    ui_text,
)
from diskovod.store import DATABASE_TABLES
from diskovod.web import localized_base_instructions, personality_source_hash


CATALOG_PATH = Path(__file__).parents[1] / "diskovod" / "localization.json"
TOP_LEVEL_FIELDS = {"schema_version", "default_locale", "locales"}
LOCALE_FIELDS = {
    "display_name",
    "assistant_name",
    "assistant_identity",
    "escalation_fallback",
    "tool_policy",
    "tool_text",
    "inline_tool_text",
    "summarization_prompt",
    "runtime_context",
    "ui",
    "prompts",
}
SIMPLE_STRING_FIELDS = {
    "display_name",
    "assistant_name",
    "assistant_identity",
    "escalation_fallback",
    "tool_policy",
    "summarization_prompt",
}
STRING_MAP_FIELDS = {"tool_text", "inline_tool_text", "runtime_context", "ui", "prompts"}
PROMPT_FIELDS = set(PromptBundle.__dataclass_fields__)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    assert len(keys) == len(set(keys)), f"duplicate JSON properties: {keys}"
    return dict(pairs)


def _read_catalog() -> dict[str, Any]:
    value = json.loads(CATALOG_PATH.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _mapping(value: object, path: str) -> dict[str, Any]:
    assert isinstance(value, dict), f"{path} must be an object"
    assert all(isinstance(key, str) for key in value), f"{path} has a non-string key"
    return cast(dict[str, Any], value)


def _nonempty_string(value: object, path: str) -> str:
    assert isinstance(value, str) and value, f"{path} must be a non-empty string"
    return value


def _format_fields(value: str) -> Counter[str]:
    return Counter(name for _, name, _, _ in Formatter().parse(value) if name is not None)


def _validate_localized_value(reference: object, value: object, path: str) -> None:
    if isinstance(reference, str):
        localized = _nonempty_string(value, path)
        assert _format_fields(localized) == _format_fields(reference), f"{path} format fields differ"
        return

    assert isinstance(reference, list), f"{path} default value must be a string or array"
    assert isinstance(value, list), f"{path} must be an array"
    assert len(value) == len(reference), f"{path} array length differs"
    for index, (reference_item, localized_item) in enumerate(zip(reference, value, strict=True)):
        assert isinstance(reference_item, str), f"{path}[{index}] default value must be a string"
        _validate_localized_value(reference_item, localized_item, f"{path}[{index}]")


def _validate_record(
    reference: dict[str, Any],
    value: object,
    path: str,
    *,
    required_fields: set[str] | None = None,
) -> None:
    localized = _mapping(value, path)
    expected_keys = required_fields if required_fields is not None else set(reference)
    assert set(reference) == expected_keys, f"default {path} keys differ from the schema"
    assert set(localized) == expected_keys, f"{path} keys differ from the default locale"
    for key, reference_value in reference.items():
        _validate_localized_value(reference_value, localized[key], f"{path}.{key}")


def _validate_catalog(catalog: dict[str, Any]) -> None:
    assert set(catalog) == TOP_LEVEL_FIELDS
    assert catalog["schema_version"] == 1
    default_locale = _nonempty_string(catalog["default_locale"], "default_locale")
    locales = _mapping(catalog["locales"], "locales")
    assert locales
    assert default_locale in locales

    locale_records = {locale: _mapping(value, f"locales.{locale}") for locale, value in locales.items()}
    for locale, record in locale_records.items():
        assert set(record) == LOCALE_FIELDS, f"locales.{locale} keys differ from the schema"

    reference = locale_records[default_locale]
    for locale, record in locale_records.items():
        for field in SIMPLE_STRING_FIELDS:
            _validate_localized_value(reference[field], record[field], f"locales.{locale}.{field}")
        for field in STRING_MAP_FIELDS:
            required_fields = PROMPT_FIELDS if field == "prompts" else None
            _validate_record(
                _mapping(reference[field], f"locales.{default_locale}.{field}"),
                record[field],
                f"locales.{locale}.{field}",
                required_fields=required_fields,
            )


def test_every_supported_locale_has_a_complete_prompt_bundle():
    assert (
        set(PROMPTS)
        == set(SUPPORTED_LOCALES)
        == {
            "en",
            "ru",
            "uk",
            "ja",
            "zh",
            "de",
            "fr",
        }
    )
    for locale, prompts in PROMPTS.items():
        assert all(getattr(prompts, field) for field in prompts.__dataclass_fields__), locale
        assert "details" in prompts.owner_details.format(details="details")
        assert "profile" in prompts.cached_personality.format(profile="profile")
        assert "examples" in prompts.owner_examples.format(examples="examples")
        assert "256" in prompts.length_budget.format(tokens=256)
    assert set(TOOL_POLICIES) == set(SUPPORTED_LOCALES)
    assert all(TOOL_POLICIES.values())


def test_assistant_identity_has_a_localized_default_name_in_every_locale():
    assert set(ASSISTANT_NAMES) == set(ASSISTANT_IDENTITIES) == set(SUPPORTED_LOCALES)
    assert ASSISTANT_NAMES == {
        "en": "Diskovod",
        "ru": "Дисковод",
        "uk": "Дисковод",
        "ja": "ディスコヴォド",
        "zh": "迪斯科沃德",
        "de": "Diskowod",
        "fr": "Disquovode",
    }
    for locale, name in ASSISTANT_NAMES.items():
        assert assistant_name_for(locale) == name
        assert name in assistant_identity(locale)
        assert assistant_name_for(locale, " Custom name ") == "Custom name"
        assert "Custom name" in assistant_identity(locale, "Custom name")


def test_reply_instructions_use_the_selected_prompt_locale():
    for locale in SUPPORTED_LOCALES:
        prompts = prompts_for(locale)
        instructions = AgentPrompt(
            locale,
            assistant_name_for(locale),
            prompts.base,
            "profile",
            "details",
        ).stable_prefix()

        assert prompts.base in instructions
        assert prompts.dm_style in instructions
        assert prompts.terminal_roleplay in instructions
        assert assistant_identity(locale) in instructions
        assert "react_to_message" in instructions
        assert tool_policy(locale) in instructions
        assert "<react>" not in instructions
        assert prompts.owner_details.format(details="details") in instructions
        assert prompts.cached_personality.format(profile="profile") in instructions


def test_prompt_locale_is_part_of_the_personality_cache_identity():
    assert personality_source_hash("samples", "en") != personality_source_hash("samples", "fr")


def test_unknown_locale_falls_back_to_english():
    assert prompts_for("unknown") is prompts_for("en")


def test_every_admin_string_supports_every_locale():
    for key, translations in UI_TEXT.items():
        assert set(translations) == set(SUPPORTED_LOCALES), key
        assert tuple(translations) == tuple(SUPPORTED_LOCALES), key
        assert all(translations.values()), key
        for locale in SUPPORTED_LOCALES:
            assert ui_text(locale, key) == translations[locale]


def test_database_explorer_can_label_every_managed_table_in_every_locale():
    for locale in SUPPORTED_LOCALES:
        for table in DATABASE_TABLES:
            specific_key = f"table_{table}"
            specific = ui_text(locale, specific_key)
            label = (
                ui_text(locale, "database_table_label", name=table) if specific == specific_key else specific
            )
            assert label and label != specific_key


def test_background_job_presentation_keys_are_localized():
    job_types = {
        "provider.capability_probe",
        "provider.setup_draft_probe",
        "assistant.personality_inference",
        "runtime.checkpoint_replay",
    }
    stages = {
        "building_probe_request",
        "testing_native_tools",
        "testing_hosted_web_search",
        "loading_personality_samples",
        "inferring_personality",
        "loading_checkpoint",
        "replaying_checkpoint",
        "recovered_after_restart",
    }
    for locale in SUPPORTED_LOCALES:
        for job_type in job_types:
            key = "job_type_" + job_type.replace(".", "_")
            assert ui_text(locale, key) != key
        for stage in stages:
            key = "job_stage_" + stage
            assert ui_text(locale, key) != key


def test_every_tool_string_supports_every_locale():
    assert set(TOOL_TEXT) == set(SUPPORTED_LOCALES)
    expected_keys = set(TOOL_TEXT["en"])
    for locale, translations in TOOL_TEXT.items():
        assert set(translations) == expected_keys, locale
        assert all(translations.values()), locale

    assert set(INLINE_TOOL_TEXT) == set(SUPPORTED_LOCALES)
    inline_keys = set(INLINE_TOOL_TEXT["en"])
    for locale, translations in INLINE_TOOL_TEXT.items():
        assert set(translations) == inline_keys, locale
        assert all(translations.values()), locale


def test_tool_schemas_and_validation_errors_use_the_selected_locale():
    parameter_keys = {
        "get_current_datetime": {"timezone": "timezone"},
        "calculate": {"expression": "expression"},
        "web_search": {"query": "web_query"},
        "fetch_url": {"url": "url"},
        "search_chat_attachments": {"query": "attachment_query"},
        "search_chat_memory": {"query": "memory_query"},
        "remember_chat_memory": {"key": "memory_key", "value": "memory_value"},
        "forget_chat_memory": {"key": "memory_key"},
        "react_to_message": {"emoji": "emoji"},
        "escalate_to_owner": {},
    }
    description_keys = {
        "get_current_datetime": "current_datetime",
        "calculate": "calculate",
        "web_search": "web_search",
        "fetch_url": "fetch_url",
        "search_chat_attachments": "attachment_search",
        "search_chat_memory": "memory_search",
        "remember_chat_memory": "remember_memory",
        "forget_chat_memory": "forget_memory",
        "react_to_message": "react",
        "escalate_to_owner": "escalate",
    }

    for locale in SUPPORTED_LOCALES:
        translations = TOOL_TEXT[locale]
        tools = localized_agent_tools(locale, cast(Any, None), cast(Any, None))
        for tool in tools:
            assert tool.description == translations[description_keys[tool.name]]
            assert callable(tool.handle_validation_error)
            assert tool.handle_validation_error(None) == translations["invalid_arguments"]
            for field, key in parameter_keys[tool.name].items():
                assert tool.args_schema.model_fields[field].description == translations[key]


def test_runtime_prompt_fragments_are_catalogued_for_every_locale():
    expected = set(RUNTIME_CONTEXT_TEXT["en"])
    for locale, translations in RUNTIME_CONTEXT_TEXT.items():
        assert set(translations) == expected, locale
        assert all(translations.values()), locale

    sources = "\n".join(
        (Path(__file__).parents[1] / "diskovod" / name).read_text(encoding="utf-8")
        for name in ("agent.py", "agent_tools.py", "discord.py", "runtime.py", "waits.py")
    )
    for obsolete_literal in (
        "standalone owner message",
        "continuation in an owner message burst",
        "A written reply is required because a reaction is unavailable.",
    ):
        assert obsolete_literal not in sources


def test_dynamic_admin_presentation_values_are_localized():
    keys = {
        "trace_category_model",
        "trace_category_tool",
        "trace_category_state",
        "trace_category_run",
        "trace_kind_model_error",
        "trace_kind_tool_request",
        "trace_kind_tool_response",
        "trace_kind_tool_error",
        "trace_kind_run_input",
        "trace_kind_run_output",
        "trace_kind_run_error",
        "trace_kind_interrupt_resume",
        "trace_kind_interrupt_resume_error",
        "trace_kind_emulated_actions",
        "trace_kind_public_text_extracted",
        "trace_kind_followup_wait_armed",
        "trace_kind_followup_wait_scheduled",
        "trace_kind_followup_wait_woken",
        "trace_kind_followup_wait_resume",
        "trace_kind_followup_wait_recovery",
        "trace_kind_followup_wait_result",
        "trace_kind_followup_wait_cancelled",
        "trace_kind_followup_wait_reconciled",
        "trace_kind_mailbox_injection",
        "trace_kind_outbound_action_reconciled",
        "trace_kind_outbound_action_operator_resolution",
        "trace_kind_public_output_cutover_claim_reconciled",
        "trace_kind_public_output_cutover_escalation_reconciled",
        "trace_kind_abandoned_run_reconciled",
        "trace_kind_historical_replay",
        "checkpoint_source_input",
        "checkpoint_source_loop",
        "checkpoint_source_update",
        "checkpoint_source_fork",
        "trigger_message",
        "trigger_edit",
        "trigger_delete",
        "delivery_action_send_messages",
        "delivery_action_discord_message",
        "delivery_action_discord_reaction",
        "delivery_action_react_to_message",
        "delivery_action_escalate_to_owner",
        "delivery_state_accepted",
        "delivery_state_ambiguous",
        "delivery_state_completed",
        "result_kind_agent_run",
        "result_kind_assistant_personality",
        "result_kind_provider_capability_probe",
        "provider_custom_openai",
        "role_peer",
        "role_assistant",
        "role_owner",
        "role_tool",
        "role_system",
    }
    for locale in SUPPORTED_LOCALES:
        for key in keys:
            assert ui_text(locale, key) != key


def test_templates_and_javascript_do_not_embed_ui_copy_fallbacks():
    root = Path(__file__).parents[1] / "diskovod"
    templates = "\n".join(
        template.read_text(encoding="utf-8") for template in (root / "templates").glob("*.html")
    )
    visible_literals = {
        match.group(1).strip() for match in re.finditer(r">\s*([A-Za-z][A-Za-z0-9 ._-]*)\s*<", templates)
    }
    assert visible_literals <= {
        "Diskovod",
        "Responses API",
        "Chat Completions",
        "SQLite",
        "run_id",
        "trace_id",
        "thread_id",
        "checkpoint_id",
    }

    javascript = (root / "static" / "app.js").read_text(encoding="utf-8")
    for fallback in (
        '|| "Copy"',
        '|| "Copied"',
        '|| "Could not load details"',
        '|| "Live"',
        '|| "New messages"',
        '|| "attachment"',
    ):
        assert fallback not in javascript


def test_every_literal_admin_template_key_exists_in_the_catalog():
    templates = Path(__file__).parents[1] / "diskovod" / "templates"
    referenced = {
        key
        for template in templates.glob("*.html")
        for key in re.findall(r"""\bt\(["']([^"']+)["']""", template.read_text())
    }

    dynamic_prefixes = {key for key in referenced if key.endswith("_")}
    assert referenced - dynamic_prefixes <= set(UI_TEXT)
    for prefix in dynamic_prefixes:
        assert any(key.startswith(prefix) for key in UI_TEXT), prefix


def test_strings_live_in_the_machine_editable_json_catalog():
    catalog = _read_catalog()
    source = (Path(__file__).parents[1] / "diskovod" / "localization.py").read_text()

    assert catalog["schema_version"] == 1
    assert catalog["default_locale"] == "en"
    assert tuple(catalog["locales"]) == tuple(SUPPORTED_LOCALES)
    assert catalog["locales"]["zh"]["ui"]["skip"] == "跳到内容"
    assert catalog["locales"]["en"]["prompts"]["base"] == PROMPTS["en"].base
    assert "_ZH_UI_TEXT" not in source
    assert "def _text(" not in source
    assert "Skip to content" not in source
    assert "Write as an AI assistant" not in source
    assert not (Path(__file__).parents[1] / "diskovod" / "ui_localization.py").exists()


def test_catalog_is_structurally_complete():
    _validate_catalog(_read_catalog())


def test_catalog_validation_rejects_placeholder_drift():
    broken = copy.deepcopy(_read_catalog())
    broken["locales"]["fr"]["assistant_identity"] = "Je suis un assistant sans nom."

    with pytest.raises(AssertionError, match="format fields"):
        _validate_catalog(broken)


def test_locale_switch_translates_only_the_stock_base_prompt():
    assert localized_base_instructions("en", "ja", prompts_for("en").base) == prompts_for("ja").base
    assert localized_base_instructions("en", "ja", "  custom instructions  ") == ("custom instructions")
