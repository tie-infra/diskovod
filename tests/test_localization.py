import copy
import json
import re
from collections import Counter
from pathlib import Path
from string import Formatter
from typing import Any, cast

import pytest

from diskovod.agent import AgentPrompt
from diskovod.localization import (
    ASSISTANT_IDENTITIES,
    ASSISTANT_NAMES,
    INLINE_TOOL_TEXT,
    PROMPTS,
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
        assert "send_messages" in instructions
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


def test_every_literal_admin_template_key_exists_in_the_catalog():
    template = (Path(__file__).parents[1] / "diskovod" / "templates" / "index.html").read_text()
    referenced = set(re.findall(r"""\bt\(["']([^"']+)["']""", template))

    assert referenced <= set(UI_TEXT)


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
