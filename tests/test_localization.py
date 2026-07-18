import re
from pathlib import Path

from diskovod.automation import build_reply_instructions
from diskovod.localization import (
    PROMPTS,
    SUPPORTED_LOCALES,
    TOOL_TEXT,
    TOOL_POLICIES,
    prompts_for,
    tool_policy,
)
from diskovod.models import AppSettings, attachment_context
from diskovod.ui_localization import UI_TEXT, ui_text
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


def test_reply_instructions_use_the_selected_prompt_locale():
    for locale in SUPPORTED_LOCALES:
        prompts = prompts_for(locale)
        settings = AppSettings(
            prompt_locale=locale,
            base_instructions=prompts.base,
            owner_details="details",
        )
        instructions = build_reply_instructions(
            settings,
            {"profile": "profile"},
            [
                {
                    "direction": "out",
                    "source": "human",
                    "content": "example",
                }
            ],
        )

        assert prompts.base in instructions
        assert prompts.dm_style in instructions
        assert "send_messages" in instructions
        assert "react_to_message" in instructions
        assert tool_policy(locale) in instructions
        assert "<react>" not in instructions
        assert "<message>" not in instructions
        assert prompts.owner_details.format(details="details") in instructions
        assert prompts.cached_personality.format(profile="profile") in instructions


def test_attachment_context_uses_the_selected_prompt_locale():
    result = attachment_context(
        "",
        [{"filename": "notes.txt", "content_type": "text/plain", "size": 4}],
        provider="custom",
        model="local",
        locale="uk",
    )

    assert prompts_for("uk").no_message_text in result
    assert prompts_for("uk").attachments_heading in result


def test_prompt_locale_is_part_of_the_personality_cache_identity():
    assert personality_source_hash("samples", "en") != personality_source_hash("samples", "fr")


def test_unknown_locale_falls_back_to_english():
    assert prompts_for("unknown") is prompts_for("en")


def test_every_admin_string_supports_every_locale():
    for key, translations in UI_TEXT.items():
        assert set(translations) == set(SUPPORTED_LOCALES), key
        assert all(translations.values()), key
        for locale in SUPPORTED_LOCALES:
            assert ui_text(locale, key) == translations[locale]


def test_every_tool_string_supports_every_locale():
    assert set(TOOL_TEXT) == set(SUPPORTED_LOCALES)
    expected_keys = set(TOOL_TEXT["en"])
    for locale, translations in TOOL_TEXT.items():
        assert set(translations) == expected_keys, locale
        assert all(translations.values()), locale


def test_every_literal_admin_template_key_exists_in_the_catalog():
    template = (Path(__file__).parents[1] / "diskovod" / "templates" / "index.html").read_text()
    referenced = set(re.findall(r"""\bt\(["']([^"']+)["']""", template))

    assert referenced <= set(UI_TEXT)


def test_chinese_admin_strings_are_declared_without_a_runtime_overlay():
    source = (Path(__file__).parents[1] / "diskovod" / "ui_localization.py").read_text()

    assert "_ZH_UI_TEXT" not in source
    assert '"zh": "跳到内容"' in source


def test_locale_switch_translates_only_the_stock_base_prompt():
    assert localized_base_instructions("en", "ja", prompts_for("en").base) == prompts_for("ja").base
    assert localized_base_instructions("en", "ja", "  custom instructions  ") == ("custom instructions")
