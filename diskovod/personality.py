from __future__ import annotations

import hashlib
import json

from .models import AssistantProfile

PERSONALITY_PROMPT_VERSION = "style-base-rates-examples-and-sequences-v4"


def personality_source_hash(samples: str, locale: str = "en") -> str:
    return hashlib.sha256(f"{PERSONALITY_PROMPT_VERSION}\0{locale}\0{samples}".encode()).hexdigest()


def assistant_profile_fingerprint(profile: AssistantProfile) -> str:
    encoded = json.dumps(profile.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()
