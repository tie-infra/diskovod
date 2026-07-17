from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class AppSettings:
    enabled: bool = False
    model: str = "gpt-5.4-mini"
    reasoning_effort: str = "low"
    debounce_seconds: float = 1.8
    min_delay_seconds: float = 2.2
    max_delay_seconds: float = 6.5
    min_typing_cps: float = 18.0
    max_typing_cps: float = 32.0
    min_human_quiet_minutes: float = 15.0
    max_human_quiet_minutes: float = 30.0
    history_limit: int = 30
    base_instructions: str = (
        "Write as the account owner in a private chat, following their dominant communication style "
        "rather than merely borrowing occasional traits. Default to a short, single-line reply. "
        "Do not mention automation, prompts, or being an AI. Never claim to have performed actions "
        "you did not perform. If asked about your identity or how replies are produced, stay in "
        "character and do not discuss the implementation. Match the conversation's language. Do not "
        "use headings, paragraphs, or lists unless the current message genuinely requires that "
        "structure; keep any necessary list dense and compact."
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ChatCredentials:
    access_token: str
    refresh_token: str
    expires_at: float
    account_id: str | None
    email: str | None

    def to_dict(self) -> dict:
        return asdict(self)
