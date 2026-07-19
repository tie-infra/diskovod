import re
from pathlib import Path

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
    assistant_identity,
    assistant_name_for,
    prompts_for,
    tool_policy,
    ui_text,
)
from diskovod.web import localized_base_instructions, personality_source_hash


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


def test_admin_strings_use_explicit_locale_maps_in_the_shared_catalog():
    source = (Path(__file__).parents[1] / "diskovod" / "localization.py").read_text()

    assert "_ZH_UI_TEXT" not in source
    assert "def _text(" not in source
    assert '"zh": "跳到内容"' in source
    assert not (Path(__file__).parents[1] / "diskovod" / "ui_localization.py").exists()


def test_locale_switch_translates_only_the_stock_base_prompt():
    assert localized_base_instructions("en", "ja", prompts_for("en").base) == prompts_for("ja").base
    assert localized_base_instructions("en", "ja", "  custom instructions  ") == ("custom instructions")
