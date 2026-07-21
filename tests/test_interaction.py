from dataclasses import replace
from datetime import datetime, timezone

import pytest

from diskovod.interaction import (
    AvailabilitySchedule,
    InvocationAlias,
    TriggerRule,
    evaluate_trigger,
    preset_policy,
    schedule_allows,
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
    assert peer.rule_matched
    assert not peer.participant_eligible

    unrelated = decide("Talking about lunch", participant="peer", policy=owner_only)
    assert not unrelated.matched
    assert not unrelated.rule_matched
    assert not unrelated.participant_eligible
    assert unrelated.reason == "not_addressed"


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
    policy = replace(
        preset_policy("on_invocation", prompt_locale="ja", inject_active_input=False),
        trigger_rules=(
            TriggerRule("reply_to_assistant", id="reply"),
            TriggerRule("reaction_invocation", id="reaction", reactions=("👀", "🤖")),
        ),
    )
    restored = type(policy).from_dict(policy.to_dict())
    assert restored == policy
    assert restored.active_turn_input.timing == "queue_for_next_turn"


def test_reply_to_assistant_is_an_explicit_message_trigger():
    policy = replace(
        preset_policy("on_invocation"),
        trigger_rules=(TriggerRule("reply_to_assistant", id="reply"),),
    )
    ordinary = evaluate_trigger(
        policy,
        participant="peer",
        content="Following up",
        assistant_name="Diskovod",
        attention_words=ATTENTION,
    )
    reply = evaluate_trigger(
        policy,
        participant="peer",
        content="Following up",
        assistant_name="Diskovod",
        attention_words=ATTENTION,
        reply_to_assistant=True,
    )
    assert not ordinary.matched
    assert reply.matched
    assert reply.reason == "reply_to_assistant"
    assert reply.rule_id == "reply"


def test_reaction_invocation_matches_only_configured_reactions():
    policy = replace(
        preset_policy("on_invocation"),
        trigger_rules=(TriggerRule("reaction_invocation", id="reaction", reactions=("👀",)),),
    )
    matched = evaluate_trigger(
        policy,
        participant="owner",
        content="",
        assistant_name="Diskovod",
        attention_words=ATTENTION,
        event_kind="reaction",
        reaction="👀",
    )
    ignored = evaluate_trigger(
        policy,
        participant="owner",
        content="",
        assistant_name="Diskovod",
        attention_words=ATTENTION,
        event_kind="reaction",
        reaction="👍",
    )
    assert matched.matched
    assert matched.reason == "reaction_invocation"
    assert matched.alias == "👀"
    assert not ignored.matched


def test_availability_schedule_supports_daytime_and_overnight_ranges():
    monday = datetime(2026, 7, 20, tzinfo=timezone.utc)
    daytime = AvailabilitySchedule(
        enabled=True,
        weekdays=frozenset({0}),
        start_minute=9 * 60,
        end_minute=17 * 60,
        timezone="UTC",
    )
    assert schedule_allows(daytime, timestamp=monday.replace(hour=10).timestamp(), default_timezone="UTC")
    assert not schedule_allows(daytime, timestamp=monday.replace(hour=18).timestamp(), default_timezone="UTC")

    overnight = replace(daytime, start_minute=22 * 60, end_minute=2 * 60)
    assert schedule_allows(overnight, timestamp=monday.replace(hour=23).timestamp(), default_timezone="UTC")
    assert schedule_allows(
        overnight,
        timestamp=monday.replace(day=21, hour=1).timestamp(),
        default_timezone="UTC",
    )
    assert not schedule_allows(
        overnight,
        timestamp=monday.replace(day=21, hour=3).timestamp(),
        default_timezone="UTC",
    )


def test_private_and_approval_presets_have_distinct_delivery_contracts():
    manual = preset_policy("manual")
    draft = preset_policy("draft")
    assert manual.trigger_rules == ()
    assert manual.trigger_participants == frozenset()
    assert manual.conversation_role == "owner_copilot"
    assert manual.delivery == "dashboard_only"
    assert draft.trigger_participants == frozenset({"peer"})
    assert draft.conversation_role == "owner_copilot"
    assert draft.delivery == "owner_approval"
