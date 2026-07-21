from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Iterable, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import regex


Participant = Literal["owner", "peer"]
Preset = Literal["autonomous", "shared", "on_invocation", "manual", "draft"]


@dataclass(frozen=True, slots=True)
class InvocationAlias:
    kind: Literal["assistant_name", "literal"] = "assistant_name"
    value: str = ""


@dataclass(frozen=True, slots=True)
class TypoTolerance:
    enabled: bool = True
    maximum_distance: int = 1
    minimum_alias_graphemes: int = 6


@dataclass(frozen=True, slots=True)
class TriggerRule:
    kind: Literal[
        "every_message",
        "direct_address",
        "literal_prefix",
        "reply_to_assistant",
        "reaction_invocation",
    ]
    id: str = ""
    aliases: tuple[InvocationAlias, ...] = ()
    attention_locales: tuple[str, ...] = ()
    additional_attention_words: tuple[str, ...] = ()
    allow_bare_alias: bool = True
    literal: str = ""
    reactions: tuple[str, ...] = ()
    typo_tolerance: TypoTolerance = field(default_factory=TypoTolerance)


@dataclass(frozen=True, slots=True)
class OwnerHandoff:
    availability_transition: Literal["none", "snooze", "pause"] = "none"
    active_run_action: Literal["keep_or_inject", "cancel"] = "keep_or_inject"


@dataclass(frozen=True, slots=True)
class ActiveTurnInput:
    timing: Literal["inject_at_safe_points", "queue_for_next_turn"] = "inject_at_safe_points"
    participants: frozenset[Participant] = frozenset({"owner", "peer"})


@dataclass(frozen=True, slots=True)
class AvailabilitySchedule:
    enabled: bool = False
    weekdays: frozenset[int] = frozenset(range(7))
    start_minute: int = 9 * 60
    end_minute: int = 17 * 60
    timezone: str = ""


@dataclass(frozen=True, slots=True)
class InteractionPolicy:
    preset: Preset
    trigger_rules: tuple[TriggerRule, ...]
    trigger_participants: frozenset[Participant]
    owner_handoff: OwnerHandoff
    conversation_role: Literal["owner_delegate", "shared_assistant", "owner_copilot"]
    identity_marker: Literal["configurable", "forced"]
    delivery: Literal["immediate", "owner_approval", "dashboard_only"]
    active_turn_input: ActiveTurnInput
    availability_schedule: AvailabilitySchedule = field(default_factory=AvailabilitySchedule)
    invocation_snooze_behavior: Literal["bypass", "respect"] = "bypass"
    invocation_turn_lifetime: Literal["strict"] = "strict"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["trigger_participants"] = sorted(self.trigger_participants)
        value["active_turn_input"]["participants"] = sorted(self.active_turn_input.participants)
        value["availability_schedule"]["weekdays"] = sorted(self.availability_schedule.weekdays)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InteractionPolicy:
        rules = []
        for index, raw in enumerate(value.get("trigger_rules", [])):
            aliases = tuple(InvocationAlias(**item) for item in raw.get("aliases", []))
            typo = TypoTolerance(**raw.get("typo_tolerance", {}))
            rules.append(
                TriggerRule(
                    kind=raw["kind"],
                    id=str(raw.get("id") or f"{raw['kind']}-{index + 1}"),
                    aliases=aliases,
                    attention_locales=tuple(raw.get("attention_locales", [])),
                    additional_attention_words=tuple(raw.get("additional_attention_words", [])),
                    allow_bare_alias=bool(raw.get("allow_bare_alias", True)),
                    literal=str(raw.get("literal", "")),
                    reactions=tuple(str(item) for item in raw.get("reactions", [])),
                    typo_tolerance=typo,
                )
            )
        active = value.get("active_turn_input", {})
        schedule = value.get("availability_schedule", {})
        return cls(
            preset=value["preset"],
            trigger_rules=tuple(rules),
            trigger_participants=frozenset(value.get("trigger_participants", [])),
            owner_handoff=OwnerHandoff(**value.get("owner_handoff", {})),
            conversation_role=value["conversation_role"],
            identity_marker=value["identity_marker"],
            delivery=value["delivery"],
            active_turn_input=ActiveTurnInput(
                timing=active.get("timing", "inject_at_safe_points"),
                participants=frozenset(active.get("participants", ["owner", "peer"])),
            ),
            availability_schedule=AvailabilitySchedule(
                enabled=bool(schedule.get("enabled", False)),
                weekdays=frozenset(int(item) for item in schedule.get("weekdays", range(7))),
                start_minute=int(schedule.get("start_minute", 9 * 60)),
                end_minute=int(schedule.get("end_minute", 17 * 60)),
                timezone=str(schedule.get("timezone", "")),
            ),
            invocation_snooze_behavior=value.get("invocation_snooze_behavior", "bypass"),
            invocation_turn_lifetime=value.get("invocation_turn_lifetime", "strict"),
        )

    def encoded(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def preset_policy(
    preset: Preset,
    *,
    prompt_locale: str = "en",
    inject_active_input: bool = True,
) -> InteractionPolicy:
    timing: Literal["inject_at_safe_points", "queue_for_next_turn"] = (
        "inject_at_safe_points" if inject_active_input else "queue_for_next_turn"
    )
    if preset == "autonomous":
        return InteractionPolicy(
            preset=preset,
            trigger_rules=(TriggerRule("every_message", id="every-message"),),
            trigger_participants=frozenset({"peer"}),
            owner_handoff=OwnerHandoff("snooze", "cancel"),
            conversation_role="owner_delegate",
            identity_marker="configurable",
            delivery="immediate",
            active_turn_input=ActiveTurnInput(timing=timing, participants=frozenset({"peer"})),
        )
    if preset == "shared":
        return InteractionPolicy(
            preset=preset,
            trigger_rules=(TriggerRule("every_message", id="every-message"),),
            trigger_participants=frozenset({"owner", "peer"}),
            owner_handoff=OwnerHandoff(),
            conversation_role="shared_assistant",
            identity_marker="forced",
            delivery="immediate",
            active_turn_input=ActiveTurnInput(timing=timing),
        )
    if preset == "on_invocation":
        return InteractionPolicy(
            preset=preset,
            trigger_rules=(
                TriggerRule(
                    "direct_address",
                    id="assistant-name",
                    aliases=(InvocationAlias(),),
                    attention_locales=(prompt_locale,),
                ),
            ),
            trigger_participants=frozenset({"owner", "peer"}),
            owner_handoff=OwnerHandoff(),
            conversation_role="shared_assistant",
            identity_marker="forced",
            delivery="immediate",
            active_turn_input=ActiveTurnInput(timing=timing),
        )
    if preset == "manual":
        return InteractionPolicy(
            preset=preset,
            trigger_rules=(),
            trigger_participants=frozenset(),
            owner_handoff=OwnerHandoff(),
            conversation_role="owner_copilot",
            identity_marker="configurable",
            delivery="dashboard_only",
            active_turn_input=ActiveTurnInput(timing=timing),
        )
    if preset == "draft":
        return InteractionPolicy(
            preset=preset,
            trigger_rules=(TriggerRule("every_message", id="every-message"),),
            trigger_participants=frozenset({"peer"}),
            owner_handoff=OwnerHandoff("snooze", "cancel"),
            conversation_role="owner_copilot",
            identity_marker="forced",
            delivery="owner_approval",
            active_turn_input=ActiveTurnInput(timing=timing, participants=frozenset({"peer"})),
        )
    raise ValueError(f"Unknown interaction preset: {preset}")


def validate_policy(
    policy: InteractionPolicy,
    *,
    assistant_name: str,
    supported_attention_locales: frozenset[str] | None = None,
) -> None:
    participants = {"owner", "peer"}
    if policy.preset not in {"autonomous", "shared", "on_invocation", "manual", "draft"}:
        raise ValueError("Unknown interaction preset")
    if not policy.trigger_participants <= participants:
        raise ValueError("Unknown trigger participant")
    if not policy.active_turn_input.participants <= participants:
        raise ValueError("Unknown active-turn participant")
    if policy.active_turn_input.timing not in {
        "inject_at_safe_points",
        "queue_for_next_turn",
    }:
        raise ValueError("Unknown active-turn timing")
    if policy.owner_handoff.availability_transition not in {"none", "snooze", "pause"}:
        raise ValueError("Unknown owner-handoff transition")
    if policy.owner_handoff.active_run_action not in {"keep_or_inject", "cancel"}:
        raise ValueError("Unknown owner-handoff action")
    if policy.conversation_role not in {"owner_delegate", "shared_assistant", "owner_copilot"}:
        raise ValueError("Unknown conversation role")
    if policy.identity_marker not in {"configurable", "forced"}:
        raise ValueError("Unknown identity-marker policy")
    if policy.conversation_role == "shared_assistant" and policy.identity_marker != "forced":
        raise ValueError("A shared assistant must use a forced identity marker")
    if policy.delivery not in {"immediate", "owner_approval", "dashboard_only"}:
        raise ValueError("Unknown delivery policy")
    if policy.invocation_snooze_behavior not in {"bypass", "respect"}:
        raise ValueError("Unknown invocation snooze policy")
    if policy.invocation_turn_lifetime != "strict":
        raise ValueError("Only strict invocation turn lifetime is implemented")
    schedule = policy.availability_schedule
    if not schedule.weekdays <= frozenset(range(7)):
        raise ValueError("Unknown schedule weekday")
    if schedule.enabled and not schedule.weekdays:
        raise ValueError("An enabled schedule requires at least one weekday")
    if not 0 <= schedule.start_minute < 24 * 60 or not 0 <= schedule.end_minute < 24 * 60:
        raise ValueError("Schedule times must be within one day")
    if schedule.enabled and schedule.start_minute == schedule.end_minute:
        raise ValueError("An enabled schedule requires a non-empty time range")
    if schedule.timezone:
        try:
            ZoneInfo(schedule.timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError("Unknown schedule timezone") from error
    if len(policy.trigger_rules) > 16:
        raise ValueError("An interaction policy may contain at most sixteen trigger rules")
    for rule in policy.trigger_rules:
        if rule.kind not in {
            "every_message",
            "direct_address",
            "literal_prefix",
            "reply_to_assistant",
            "reaction_invocation",
        }:
            raise ValueError("Unknown trigger rule")
        if rule.kind in {"every_message", "reply_to_assistant"}:
            continue
        if rule.kind == "reaction_invocation":
            if not rule.reactions or len(rule.reactions) > 16:
                raise ValueError("Reaction invocation requires between one and sixteen emoji")
            if any(not item.strip() or len(item) > 64 for item in rule.reactions):
                raise ValueError("Invalid reaction invocation emoji")
            continue
        if rule.kind == "literal_prefix":
            literal = _normalize(rule.literal)
            if not literal or not any(character.isalnum() for character in literal):
                raise ValueError("A literal trigger must contain text")
            if len(literal) > 80:
                raise ValueError("A literal trigger may contain at most 80 characters")
            continue
        if not rule.aliases or len(rule.aliases) > 16:
            raise ValueError("Direct address requires between one and sixteen aliases")
        if len(rule.attention_locales) > 16 or len(rule.additional_attention_words) > 32:
            raise ValueError("Too many invocation attention entries")
        if supported_attention_locales is not None and not set(rule.attention_locales) <= set(
            supported_attention_locales
        ):
            raise ValueError("Unknown invocation attention locale")
        for word in rule.additional_attention_words:
            normalized_word = _normalize(word)
            if (
                not normalized_word
                or len(normalized_word) > 80
                or not any(character.isalnum() for character in normalized_word)
            ):
                raise ValueError("Invocation attention words must contain text")
        normalized: set[str] = set()
        for entry in rule.aliases:
            if entry.kind not in {"assistant_name", "literal"}:
                raise ValueError("Unknown invocation-alias kind")
            value = assistant_name if entry.kind == "assistant_name" else entry.value
            alias = _normalize(value)
            if not alias or not any(character.isalnum() for character in alias):
                raise ValueError("Invocation aliases must contain letters or numbers")
            if len(alias) > 80:
                raise ValueError("Invocation aliases may contain at most 80 characters")
            if alias in normalized:
                raise ValueError("Invocation aliases must be distinct after normalization")
            normalized.add(alias)
        if rule.typo_tolerance.maximum_distance not in {0, 1}:
            raise ValueError("Only a maximum typo distance of one is supported")
        if rule.typo_tolerance.minimum_alias_graphemes < 1:
            raise ValueError("The typo alias length threshold must be positive")


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    matched: bool
    reason: str
    rule_kind: str = ""
    rule_id: str = ""
    alias: str = ""
    distance: int = 0
    rule_matched: bool = False
    participant_eligible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_trigger(
    policy: InteractionPolicy,
    *,
    participant: str,
    content: str,
    assistant_name: str,
    attention_words: dict[str, Iterable[str]],
    event_kind: str = "message",
    reply_to_assistant: bool = False,
    reaction: str = "",
) -> TriggerDecision:
    participant_eligible = participant in policy.trigger_participants
    abstention: TriggerDecision | None = None
    for index, rule in enumerate(policy.trigger_rules):
        rule_id = rule.id or f"{rule.kind}-{index + 1}"
        if rule.kind == "every_message" and event_kind in {"message", "edit"}:
            decision = TriggerDecision(
                True,
                "every_message",
                rule_kind=rule.kind,
                rule_id=rule_id,
            )
            break
        if rule.kind == "reply_to_assistant":
            if event_kind in {"message", "edit"} and reply_to_assistant:
                decision = TriggerDecision(
                    True,
                    "reply_to_assistant",
                    rule_kind=rule.kind,
                    rule_id=rule_id,
                )
                break
            continue
        if rule.kind == "reaction_invocation":
            if event_kind == "reaction" and reaction in rule.reactions:
                decision = TriggerDecision(
                    True,
                    "reaction_invocation",
                    rule_kind=rule.kind,
                    rule_id=rule_id,
                    alias=reaction,
                )
                break
            continue
        if event_kind not in {"message", "edit"}:
            continue
        if rule.kind == "literal_prefix":
            literal = _normalize(rule.literal)
            text = _normalize(content)
            if literal and text.startswith(literal) and _has_boundary(text, len(literal), literal):
                decision = TriggerDecision(
                    True,
                    "literal_match",
                    rule_kind=rule.kind,
                    rule_id=rule_id,
                    alias=rule.literal,
                )
                break
            continue
        aliases = _resolve_aliases(rule.aliases, assistant_name)
        words = (
            tuple(word for locale in rule.attention_locales for word in attention_words.get(locale, ()))
            + rule.additional_attention_words
        )
        result = _match_direct_address(content, aliases, words, rule, rule_id=rule_id)
        if result.matched:
            decision = result
            break
        if result.reason != "not_addressed":
            abstention = result
    else:
        decision = abstention or TriggerDecision(False, "not_addressed")
    rule_matched = decision.matched
    if rule_matched and not participant_eligible:
        return replace(
            decision,
            matched=False,
            reason="participant_not_eligible",
            rule_matched=True,
            participant_eligible=False,
        )
    return replace(
        decision,
        rule_matched=rule_matched,
        participant_eligible=participant_eligible,
    )


def schedule_allows(
    schedule: AvailabilitySchedule,
    *,
    timestamp: float,
    default_timezone: str,
) -> bool:
    if not schedule.enabled:
        return True
    try:
        timezone = ZoneInfo(schedule.timezone or default_timezone)
    except ZoneInfoNotFoundError:
        timezone = ZoneInfo("UTC")
    local = datetime.fromtimestamp(timestamp, timezone)
    minute = local.hour * 60 + local.minute
    if schedule.start_minute < schedule.end_minute:
        return local.weekday() in schedule.weekdays and schedule.start_minute <= minute < schedule.end_minute
    if minute >= schedule.start_minute:
        return local.weekday() in schedule.weekdays
    previous_weekday = (local.weekday() - 1) % 7
    return previous_weekday in schedule.weekdays and minute < schedule.end_minute


def _resolve_aliases(entries: tuple[InvocationAlias, ...], assistant_name: str) -> tuple[str, ...]:
    aliases = [assistant_name if entry.kind == "assistant_name" else entry.value for entry in entries]
    normalized: dict[str, str] = {}
    for alias in aliases:
        clean = alias.strip()
        key = _normalize(clean)
        if clean and key and any(character.isalnum() for character in key):
            normalized.setdefault(key, clean)
    return tuple(sorted(normalized.values(), key=lambda item: len(_normalize(item)), reverse=True))


def _match_direct_address(
    content: str,
    aliases: tuple[str, ...],
    attention_words: tuple[str, ...],
    rule: TriggerRule,
    *,
    rule_id: str,
) -> TriggerDecision:
    text = _normalize(content)
    starts = [0] if rule.allow_bare_alias else []
    for word in sorted(
        {_normalize(item) for item in attention_words if _normalize(item)}, key=len, reverse=True
    ):
        if text.startswith(word) and _has_boundary(text, len(word), word):
            starts.append(_skip_separators(text, len(word)))
    for start in dict.fromkeys(starts):
        for alias in aliases:
            normalized = _normalize(alias)
            if text.startswith(normalized, start) and _has_boundary(
                text, start + len(normalized), normalized
            ):
                return TriggerDecision(
                    True,
                    "direct_address",
                    rule_kind=rule.kind,
                    rule_id=rule_id,
                    alias=alias,
                )
    tolerance = rule.typo_tolerance
    if not tolerance.enabled or tolerance.maximum_distance < 1:
        return TriggerDecision(False, "not_addressed")
    candidates: list[tuple[int, int, str]] = []
    for start in dict.fromkeys(starts):
        for alias in aliases:
            normalized = _normalize(alias)
            alias_clusters = _graphemes(normalized)
            if len(alias_clusters) < tolerance.minimum_alias_graphemes:
                continue
            for candidate_clusters in _fuzzy_candidates(
                text,
                start,
                alias=normalized,
                maximum_distance=tolerance.maximum_distance,
            ):
                distance = _bounded_damerau_levenshtein(
                    candidate_clusters,
                    alias_clusters,
                    tolerance.maximum_distance,
                )
                if distance is not None:
                    candidates.append((distance, -len(alias_clusters), alias))
    if not candidates:
        return TriggerDecision(False, "not_addressed")
    candidates.sort()
    best_distance, best_length, best_alias = candidates[0]
    equally_good = {
        alias for distance, length, alias in candidates if (distance, length) == (best_distance, best_length)
    }
    if len(equally_good) != 1:
        return TriggerDecision(False, "ambiguous_fuzzy_address")
    return TriggerDecision(
        True,
        "fuzzy_direct_address",
        rule_kind=rule.kind,
        rule_id=rule_id,
        alias=best_alias,
        distance=best_distance,
    )


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().strip().split())


def normalize_invocation_text(value: str) -> str:
    """Return the exact normalized form used by deterministic trigger rules."""
    return _normalize(value)


def _is_cjk(value: str) -> bool:
    return any(
        "CJK" in unicodedata.name(character, "")
        or "HIRAGANA" in unicodedata.name(character, "")
        or "KATAKANA" in unicodedata.name(character, "")
        for character in value
    )


def _has_boundary(text: str, end: int, matched: str) -> bool:
    if end >= len(text):
        return True
    following = text[end]
    if _is_cjk(matched):
        return True
    return not (following.isalnum() or following == "_")


def _skip_separators(text: str, start: int) -> int:
    while start < len(text) and (text[start].isspace() or unicodedata.category(text[start]).startswith("P")):
        start += 1
    return start


def _fuzzy_candidates(
    text: str,
    start: int,
    *,
    alias: str,
    maximum_distance: int,
) -> tuple[tuple[str, ...], ...]:
    start = _skip_separators(text, start)
    remainder = _graphemes(text[start:])
    alias_length = len(_graphemes(alias))
    result: list[tuple[str, ...]] = []
    for length in range(
        max(1, alias_length - maximum_distance),
        min(len(remainder), alias_length + maximum_distance) + 1,
    ):
        candidate = remainder[:length]
        end = start + sum(len(cluster) for cluster in candidate)
        if not _is_cjk(alias) and not _has_boundary(text, end, alias):
            continue
        result.append(candidate)
    return tuple(result)


def _graphemes(value: str) -> tuple[str, ...]:
    return tuple(regex.findall(r"\X", value))


def _bounded_damerau_levenshtein(
    left: tuple[str, ...],
    right: tuple[str, ...],
    maximum: int,
) -> int | None:
    if abs(len(left) - len(right)) > maximum:
        return None
    previous_previous: list[int] | None = None
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        row_minimum = i
        for j, right_item in enumerate(right, 1):
            value = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_item != right_item),
            )
            if (
                previous_previous is not None
                and i > 1
                and j > 1
                and left_item == right[j - 2]
                and left[i - 2] == right_item
            ):
                value = min(value, previous_previous[j - 2] + 1)
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > maximum:
            return None
        previous_previous, previous = previous, current
    result = previous[-1]
    return result if result <= maximum else None
