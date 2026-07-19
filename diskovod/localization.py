from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, fields
from importlib.resources import files
from string import Formatter
from typing import Any, cast


_CATALOG_RESOURCE = "localization.json"
_SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = {"schema_version", "default_locale", "locales"}
_LOCALE_FIELDS = {
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
_SIMPLE_STRING_FIELDS = {
    "display_name",
    "assistant_name",
    "assistant_identity",
    "escalation_fallback",
    "tool_policy",
    "summarization_prompt",
}
_STRING_MAP_FIELDS = {
    "tool_text",
    "inline_tool_text",
    "runtime_context",
    "ui",
    "prompts",
}


@dataclass(frozen=True, slots=True)
class PromptBundle:
    base: str
    dm_style: str
    terminal_roleplay: str
    forced_reply: str
    owner_details: str
    cached_personality: str
    owner_examples: str
    length_budget: str
    no_message_text: str
    attachments_heading: str
    personality: str


_PROMPT_FIELDS = {field.name for field in fields(PromptBundle)}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate property {key!r}")
        result[key] = value
    return result


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be a JSON object")
    return cast(dict[str, Any], value)


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _format_fields(value: str, path: str) -> Counter[str]:
    try:
        return Counter(name for _, name, _, _ in Formatter().parse(value) if name is not None)
    except ValueError as error:
        raise ValueError(f"{path} contains invalid format syntax: {error}") from error


def _validate_localized_value(reference: object, value: object, path: str) -> None:
    if isinstance(reference, str):
        localized = _nonempty_string(value, path)
        expected_fields = _format_fields(reference, f"{path} reference")
        actual_fields = _format_fields(localized, path)
        if actual_fields != expected_fields:
            raise ValueError(
                f"{path} format fields {sorted(actual_fields.elements())!r} do not match "
                f"the default locale {sorted(expected_fields.elements())!r}"
            )
        return

    if isinstance(reference, list):
        if not isinstance(value, list) or len(value) != len(reference):
            raise ValueError(f"{path} must be an array of {len(reference)} localized strings")
        for index, (reference_item, localized_item) in enumerate(zip(reference, value, strict=True)):
            if not isinstance(reference_item, str):
                raise ValueError(f"{path} reference contains a non-string array value")
            _validate_localized_value(reference_item, localized_item, f"{path}[{index}]")
        return

    raise ValueError(f"{path} reference must be a string or an array of strings")


def _validate_record(
    reference: dict[str, Any],
    value: object,
    path: str,
    *,
    required_fields: set[str] | None = None,
) -> None:
    localized = _mapping(value, path)
    expected_keys = required_fields if required_fields is not None else set(reference)
    if set(localized) != expected_keys:
        missing = sorted(expected_keys - set(localized))
        extra = sorted(set(localized) - expected_keys)
        raise ValueError(f"{path} has mismatched keys; missing={missing!r}, extra={extra!r}")
    if set(reference) != expected_keys:
        missing = sorted(expected_keys - set(reference))
        extra = sorted(set(reference) - expected_keys)
        raise ValueError(
            f"default locale {path.rsplit('.', 1)[-1]} has mismatched keys; "
            f"missing={missing!r}, extra={extra!r}"
        )
    for key, reference_value in reference.items():
        _validate_localized_value(reference_value, localized[key], f"{path}.{key}")


def _validate_catalog(catalog: dict[str, Any]) -> None:
    if set(catalog) != _TOP_LEVEL_FIELDS:
        missing = sorted(_TOP_LEVEL_FIELDS - set(catalog))
        extra = sorted(set(catalog) - _TOP_LEVEL_FIELDS)
        raise ValueError(f"catalog has mismatched keys; missing={missing!r}, extra={extra!r}")
    if catalog["schema_version"] != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {catalog['schema_version']!r}; expected {_SCHEMA_VERSION}"
        )

    default_locale = _nonempty_string(catalog["default_locale"], "default_locale")
    locales = _mapping(catalog["locales"], "locales")
    if not locales:
        raise ValueError("locales must contain at least one locale")
    if default_locale not in locales:
        raise ValueError("default_locale must identify an entry in locales")

    normalized_locales: dict[str, dict[str, Any]] = {}
    for locale, raw_locale in locales.items():
        locale_data = _mapping(raw_locale, f"locales.{locale}")
        if set(locale_data) != _LOCALE_FIELDS:
            missing = sorted(_LOCALE_FIELDS - set(locale_data))
            extra = sorted(set(locale_data) - _LOCALE_FIELDS)
            raise ValueError(f"locales.{locale} has mismatched keys; missing={missing!r}, extra={extra!r}")
        normalized_locales[locale] = locale_data

    reference = normalized_locales[default_locale]
    for field in _SIMPLE_STRING_FIELDS:
        _nonempty_string(reference[field], f"locales.{default_locale}.{field}")
    for field in _STRING_MAP_FIELDS:
        required = _PROMPT_FIELDS if field == "prompts" else None
        _validate_record(
            _mapping(reference[field], f"locales.{default_locale}.{field}"),
            reference[field],
            f"locales.{default_locale}.{field}",
            required_fields=required,
        )

    for locale, locale_data in normalized_locales.items():
        for field in _SIMPLE_STRING_FIELDS:
            _validate_localized_value(
                reference[field],
                locale_data[field],
                f"locales.{locale}.{field}",
            )
        for field in _STRING_MAP_FIELDS:
            required = _PROMPT_FIELDS if field == "prompts" else None
            _validate_record(
                _mapping(reference[field], f"locales.{default_locale}.{field}"),
                locale_data[field],
                f"locales.{locale}.{field}",
                required_fields=required,
            )


def _load_catalog() -> dict[str, Any]:
    try:
        raw = files("diskovod").joinpath(_CATALOG_RESOURCE).read_text(encoding="utf-8")
        catalog = json.loads(raw, object_pairs_hook=_unique_object)
        catalog = _mapping(catalog, "catalog")
        _validate_catalog(catalog)
        return catalog
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid {_CATALOG_RESOURCE}: {error}") from error


_CATALOG = _load_catalog()
_LOCALES = cast(dict[str, dict[str, Any]], _CATALOG["locales"])

DEFAULT_LOCALE = cast(str, _CATALOG["default_locale"])
SUPPORTED_LOCALES = {
    locale: cast(str, locale_data["display_name"]) for locale, locale_data in _LOCALES.items()
}
ASSISTANT_NAMES = {
    locale: cast(str, locale_data["assistant_name"]) for locale, locale_data in _LOCALES.items()
}
ASSISTANT_IDENTITIES = {
    locale: cast(str, locale_data["assistant_identity"]) for locale, locale_data in _LOCALES.items()
}
ESCALATION_FALLBACKS = {
    locale: cast(str, locale_data["escalation_fallback"]) for locale, locale_data in _LOCALES.items()
}
TOOL_POLICIES = {locale: cast(str, locale_data["tool_policy"]) for locale, locale_data in _LOCALES.items()}
TOOL_TEXT = {
    locale: cast(dict[str, Any], locale_data["tool_text"]) for locale, locale_data in _LOCALES.items()
}
INLINE_TOOL_TEXT = {
    locale: cast(dict[str, str], locale_data["inline_tool_text"]) for locale, locale_data in _LOCALES.items()
}
SUMMARIZATION_PROMPTS = {
    locale: cast(str, locale_data["summarization_prompt"]) for locale, locale_data in _LOCALES.items()
}
RUNTIME_CONTEXT_TEXT = {
    locale: cast(dict[str, str], locale_data["runtime_context"]) for locale, locale_data in _LOCALES.items()
}

_UI_KEYS = cast(dict[str, str], _LOCALES[DEFAULT_LOCALE]["ui"])
UI_TEXT: dict[str, dict[str, str]] = {
    key: {locale: cast(dict[str, str], locale_data["ui"])[key] for locale, locale_data in _LOCALES.items()}
    for key in _UI_KEYS
}

PROMPTS = {
    locale: PromptBundle(**cast(dict[str, str], locale_data["prompts"]))
    for locale, locale_data in _LOCALES.items()
}


def normalize_locale(locale: str) -> str:
    return locale if locale in SUPPORTED_LOCALES else DEFAULT_LOCALE


def escalation_fallback(locale: str) -> str:
    return ESCALATION_FALLBACKS[normalize_locale(locale)]


def assistant_name_for(locale: str, configured: str = "") -> str:
    return configured.strip() or ASSISTANT_NAMES[normalize_locale(locale)]


def assistant_identity(locale: str, configured: str = "") -> str:
    normalized = normalize_locale(locale)
    return ASSISTANT_IDENTITIES[normalized].format(name=assistant_name_for(normalized, configured))


def tool_policy(locale: str) -> str:
    return TOOL_POLICIES[normalize_locale(locale)]


def tool_text(locale: str) -> dict[str, Any]:
    return TOOL_TEXT[normalize_locale(locale)]


def inline_tool_text(locale: str) -> dict[str, str]:
    return INLINE_TOOL_TEXT[normalize_locale(locale)]


def summarization_prompt(locale: str) -> str:
    return SUMMARIZATION_PROMPTS[normalize_locale(locale)]


def runtime_context_text(locale: str) -> dict[str, str]:
    return RUNTIME_CONTEXT_TEXT[normalize_locale(locale)]


def ui_text(locale: str, key: str, **values: object) -> str:
    translations = UI_TEXT.get(key)
    if translations is None:
        return key
    text = translations.get(normalize_locale(locale), translations[DEFAULT_LOCALE])
    return text.format(**values) if values else text


def prompts_for(locale: str) -> PromptBundle:
    return PROMPTS[normalize_locale(locale)]
