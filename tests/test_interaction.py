from dataclasses import replace

import pytest

from diskovod.interaction import (
    InvocationAlias,
    TriggerRule,
    evaluate_trigger,
    preset_policy,
)


ATTENTION = {
    "en": ("hey", "hello", "hi", "okay"),
    "ja": ("ねえ",),
    "zh": ("嘿", "你好"),
}


def decide(text: str, *, name: str = "Diskovod", participant: str = "peer", policy=None):
    return evaluate_trigger(
        policy or preset_policy("on_invocation"),
        participant=participant,
        content=text,
        assistant_name=name,
        attention_words=ATTENTION,
    )


@pytest.mark.parametrize(
    "text",
    (
        "Diskovod help me",
        "Diskovod, help me",
        "Hey Diskovod can you check this?",
        "Hey, Diskovod — can you check this?",
        "  HEY,   DISKOVOD: thoughts?",
        "Diskovod",
    ),
)
def test_natural_direct_address_accepts_conversational_punctuation(text: str):
    decision = decide(text)
    assert decision.matched
    assert decision.reason == "direct_address"


@pytest.mark.parametrize(
    "text",
    (
        "HeyDiskovod, help",
        "I mentioned Diskovod yesterday",
        "Diskovodder, help",
        "> Diskovod, answer this quote",
    ),
)
def test_direct_address_does_not_fuzzy_search_arbitrary_text(text: str):
    assert not decide(text).matched


@pytest.mark.parametrize(
    "text",
    (
        "Diskvod help",  # deletion
        "Diskovood help",  # insertion
        "Diskobod help",  # substitution
        "Diksovod help",  # adjacent transposition
    ),
)
def test_conservative_typo_tolerance_handles_one_edit_in_the_name_slot(text: str):
    decision = decide(text)
    assert decision.matched
    assert decision.reason == "fuzzy_direct_address"
    assert decision.distance == 1


def test_typo_tolerance_refuses_short_and_ambiguous_aliases():
    short = replace(
        preset_policy("on_invocation"),
        trigger_rules=(
            TriggerRule(
                "direct_address",
                aliases=(InvocationAlias("literal", "Bot"),),
                attention_locales=("en",),
            ),
        ),
    )
    assert not decide("Bpt help", policy=short).matched

    ambiguous = replace(
        preset_policy("on_invocation"),
        trigger_rules=(
            TriggerRule(
                "direct_address",
                aliases=(
                    InvocationAlias("literal", "Diskovod"),
                    InvocationAlias("literal", "Diskavod"),
                ),
                attention_locales=("en",),
            ),
        ),
    )
    decision = decide("Diskxvod help", policy=ambiguous)
    assert not decision.matched
    assert decision.reason == "ambiguous_fuzzy_address"


def test_dynamic_alias_tracks_the_configured_name_while_literals_remain_stable():
    policy = replace(
        preset_policy("on_invocation"),
        trigger_rules=(
            TriggerRule(
                "direct_address",
                aliases=(InvocationAlias(), InvocationAlias("literal", "Diskovod")),
                attention_locales=("en",),
            ),
        ),
    )
    assert decide("Nova help", name="Nova", policy=policy).matched
    assert decide("Diskovod help", name="Nova", policy=policy).matched
    assert not decide("Oldname help", name="Nova", policy=policy).matched


def test_typo_tolerance_uses_the_complete_multiword_alias_slot():
    policy = replace(
        preset_policy("on_invocation"),
        trigger_rules=(
            TriggerRule(
                "direct_address",
                aliases=(InvocationAlias("literal", "Helpful Robot"),),
                attention_locales=("en",),
            ),
        ),
    )
    decision = decide("Helpful Rboot, please check this", policy=policy)
    assert decision.matched
    assert decision.reason == "fuzzy_direct_address"


def test_typo_tolerance_is_unicode_aware_for_a_configured_localized_name():
    decision = decide("Дисковд, помоги", name="Дисковод")
    assert decision.matched
    assert decision.reason == "fuzzy_direct_address"


def test_custom_and_multilingual_attention_words_use_the_same_natural_grammar():
    policy = replace(
        preset_policy("on_invocation"),
        trigger_rules=(
            TriggerRule(
                "direct_address",
                aliases=(InvocationAlias(),),
                attention_locales=("zh",),
                additional_attention_words=("yo",),
            ),
        ),
    )
    assert decide("你好，Diskovod 帮忙", policy=policy).matched
    assert decide("Yo Diskovod help", policy=policy).matched


def test_participant_scope_is_independent_from_the_trigger_rule():
    owner_only = replace(
        preset_policy("on_invocation"),
        trigger_participants=frozenset({"owner"}),
    )
    assert decide("Diskovod help", participant="owner", policy=owner_only).matched
    peer = decide("Diskovod help", participant="peer", policy=owner_only)
    assert not peer.matched
    assert peer.reason == "participant_not_eligible"


def test_literal_prefix_is_strict_and_never_uses_typo_tolerance():
    policy = replace(
        preset_policy("on_invocation"),
        trigger_rules=(TriggerRule("literal_prefix", literal="Hey, Diskovod"),),
    )
    assert decide("hey, diskovod help", policy=policy).matched
    assert not decide("hey diskovod help", policy=policy).matched
    assert not decide("hey, diskvod help", policy=policy).matched


def test_cjk_names_match_without_requiring_spaces_or_western_punctuation():
    assert decide(
        "ディスコヴォド手伝って",
        name="ディスコヴォド",
        policy=preset_policy("on_invocation", prompt_locale="ja"),
    ).matched
    assert decide(
        "嘿迪斯科沃德帮我看看",
        name="迪斯科沃德",
        policy=preset_policy("on_invocation", prompt_locale="zh"),
    ).matched


def test_policy_round_trip_preserves_frozen_participant_sets():
    policy = preset_policy("on_invocation", prompt_locale="ja", inject_active_input=False)
    restored = type(policy).from_dict(policy.to_dict())
    assert restored == policy
    assert restored.active_turn_input.timing == "queue_for_next_turn"
