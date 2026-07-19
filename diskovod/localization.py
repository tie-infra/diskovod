from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, cast


_CATALOG_RESOURCE = "localization.json"


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


def _load_catalog() -> dict[str, Any]:
    raw = files("diskovod").joinpath(_CATALOG_RESOURCE).read_text(encoding="utf-8")
    return cast(dict[str, Any], json.loads(raw))


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
