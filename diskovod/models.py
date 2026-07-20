from __future__ import annotations

import mimetypes
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .localization import prompts_for

MAX_ATTACHMENTS_PER_MESSAGE = 4
MAX_NATIVE_ATTACHMENT_BYTES = 20 * 1024 * 1024
MAX_INLINE_TEXT_BYTES = 64 * 1024
MAX_INLINE_TEXT_CHARACTERS = 24_000

FILE_EXTENSIONS = frozenset(
    {
        ".asm",
        ".bat",
        ".c",
        ".cc",
        ".conf",
        ".cpp",
        ".css",
        ".csv",
        ".cxx",
        ".doc",
        ".docx",
        ".eml",
        ".h",
        ".hh",
        ".htm",
        ".html",
        ".ics",
        ".ini",
        ".js",
        ".json",
        ".log",
        ".markdown",
        ".md",
        ".mjs",
        ".odt",
        ".pdf",
        ".ppt",
        ".pptx",
        ".py",
        ".rst",
        ".rtf",
        ".sql",
        ".srt",
        ".text",
        ".tsv",
        ".txt",
        ".vcf",
        ".vtt",
        ".xls",
        ".xlsx",
        ".xml",
    }
)
TEXT_EXTENSIONS = frozenset(
    extension
    for extension in FILE_EXTENSIONS
    if extension not in {".doc", ".docx", ".odt", ".pdf", ".ppt", ".pptx", ".rtf", ".xls", ".xlsx"}
)

DEFAULT_BASE_INSTRUCTIONS = prompts_for("en").base
ADMIN_THEMES = frozenset({"system", "light", "dark", "black"})
ADMIN_DENSITIES = frozenset({"comfortable", "compact"})
REASONING_EFFORTS = frozenset({"low", "medium", "high"})


def _content_type(filename: str, value: object) -> str:
    if isinstance(value, str) and value:
        return value.split(";", 1)[0].strip().casefold()
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def is_text_attachment(filename: str, content_type: str) -> bool:
    return (
        content_type.startswith("text/")
        or content_type
        in {
            "application/json",
            "application/javascript",
            "application/typescript",
            "application/xml",
            "application/yaml",
            "application/x-yaml",
        }
        or Path(filename).suffix.casefold() in TEXT_EXTENSIONS
    )


def discord_attachment_metadata(values: Iterable[Any]) -> list[dict[str, Any]]:
    """Capture bounded attachment metadata without using Discord's HTTP client."""
    captured: list[dict[str, Any]] = []
    for attachment in list(values)[:MAX_ATTACHMENTS_PER_MESSAGE]:
        filename = str(getattr(attachment, "filename", "attachment"))[:255]
        content_type = _content_type(filename, getattr(attachment, "content_type", None))
        try:
            size = max(0, int(getattr(attachment, "size", 0) or 0))
        except (TypeError, ValueError):
            size = 0
        item: dict[str, Any] = {
            "id": str(getattr(attachment, "id", "")),
            "filename": filename,
            "content_type": content_type,
            "size": size,
            "url": str(getattr(attachment, "url", "")),
        }
        description = getattr(attachment, "description", None)
        if description:
            item["description"] = str(description)[:1000]

        captured.append(item)
    return captured


@dataclass(slots=True)
class InterfaceSettings:
    locale: str = "en"
    theme: str = "system"
    density: str = "comfortable"
    display_timezone: str = "browser"
    show_advanced_ids: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class AssistantProfile:
    prompt_locale: str = "en"
    assistant_name: str = ""
    owner_timezone: str = "UTC"
    owner_details: str = ""
    base_instructions: str = DEFAULT_BASE_INSTRUCTIONS
    allow_conversational_followups: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class AutomationSettings:
    enabled: bool = False
    silent_replies: bool = False
    robot_prefix: bool = False
    default_conversation_enabled: bool = True
    debounce_seconds: float = 1.8
    min_delay_seconds: float = 2.2
    max_delay_seconds: float = 6.5
    min_typing_cps: float = 18.0
    max_typing_cps: float = 32.0
    min_human_quiet_minutes: float = 15.0
    max_human_quiet_minutes: float = 30.0
    min_message_gap_seconds: float = 0.7
    max_message_gap_seconds: float = 2.0

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


@dataclass(slots=True)
class CustomProvider:
    name: str
    base_url: str
    api_key: str
    protocol: str = "responses"
    capabilities: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def supports(self, capability: str) -> bool:
        return self.capabilities.get(capability, False) is True
