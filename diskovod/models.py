from __future__ import annotations

import logging
import mimetypes
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .localization import prompts_for

log = logging.getLogger(__name__)

MAX_ATTACHMENTS_PER_MESSAGE = 4
MAX_NATIVE_ATTACHMENT_BYTES = 20 * 1024 * 1024
MAX_INLINE_TEXT_BYTES = 64 * 1024
MAX_INLINE_TEXT_CHARACTERS = 24_000

IMAGE_CONTENT_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
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
ADMIN_THEMES = frozenset({"light", "dark", "black"})


def _content_type(filename: str, value: object) -> str:
    if isinstance(value, str) and value:
        return value.split(";", 1)[0].strip().casefold()
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _is_text_attachment(filename: str, content_type: str) -> bool:
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


async def capture_discord_attachments(values: Iterable[Any]) -> list[dict[str, Any]]:
    """Capture bounded metadata and small textual attachment bodies from a Discord message."""
    captured: list[dict[str, Any]] = []
    inline_bytes = 0
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

        remaining = MAX_INLINE_TEXT_BYTES - inline_bytes
        if _is_text_attachment(filename, content_type) and (not size or size <= remaining):
            try:
                raw = await attachment.read(use_cached=True)
                if len(raw) <= remaining and b"\0" not in raw:
                    text = raw.decode("utf-8", errors="replace").strip()
                    if text:
                        item["text"] = text[:MAX_INLINE_TEXT_CHARACTERS]
                        inline_bytes += len(raw)
            except Exception as exc:
                log.warning("Could not read Discord text attachment %s: %s", filename, exc)
        captured.append(item)
    return captured


def model_supports_vision(model: str) -> bool:
    """Conservative capability check for model families with documented image input."""
    name = model.strip().casefold()
    if "mini" in name and name.startswith("o1"):
        return False
    return name.startswith(("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4"))


def can_send_image(attachment: dict[str, Any], model: str) -> bool:
    return (
        model_supports_vision(model)
        and attachment.get("content_type") in IMAGE_CONTENT_TYPES
        and 0 < int(attachment.get("size") or 0) <= MAX_NATIVE_ATTACHMENT_BYTES
        and bool(attachment.get("url"))
    )


def can_send_file(attachment: dict[str, Any], provider: str, model: str) -> bool:
    """The subscription transport uses Responses input_file; custom Chat Completions does not."""
    return (
        provider == "chatgpt"
        and model_supports_vision(model)
        and Path(str(attachment.get("filename", ""))).suffix.casefold() in FILE_EXTENSIONS
        and 0 < int(attachment.get("size") or 0) <= MAX_NATIVE_ATTACHMENT_BYTES
        and bool(attachment.get("url"))
    )


def attachment_context(
    content: str,
    attachments: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    locale: str = "en",
) -> str:
    """Add stable metadata and retrieval text for attachments not sent as native files."""
    notes: list[str] = []
    for attachment in attachments:
        filename = str(attachment.get("filename") or "attachment")
        content_type = str(attachment.get("content_type") or "unknown type")
        size = int(attachment.get("size") or 0)
        description = attachment.get("description")
        note = f"- {filename} ({content_type}, {size} bytes)"
        if description:
            note += f": {description}"
        if text := attachment.get("text"):
            if not can_send_file(attachment, provider, model):
                note += f"\n<attachment filename={filename!r}>\n{text}\n</attachment>"
        notes.append(note)
    if not notes:
        return content
    prompts = prompts_for(locale)
    prefix = content.strip() or prompts.no_message_text
    return prefix + f"\n\n{prompts.attachments_heading}\n" + "\n".join(notes)


@dataclass(slots=True)
class AppSettings:
    enabled: bool = False
    silent_replies: bool = False
    robot_prefix: bool = False
    admin_locale: str = "en"
    admin_theme: str = "light"
    prompt_locale: str = "en"
    owner_timezone: str = "UTC"
    default_conversation_enabled: bool = True
    provider: str = "chatgpt"
    model: str = "gpt-5.4-mini"
    reasoning_effort: str = "low"
    max_reply_tokens: int = 256
    debounce_seconds: float = 1.8
    min_delay_seconds: float = 2.2
    max_delay_seconds: float = 6.5
    min_typing_cps: float = 18.0
    max_typing_cps: float = 32.0
    min_human_quiet_minutes: float = 15.0
    max_human_quiet_minutes: float = 30.0
    history_limit: int = 30
    multi_message_replies: bool = False
    max_reply_messages: int = 3
    min_message_gap_seconds: float = 0.7
    max_message_gap_seconds: float = 2.0
    owner_details: str = ""
    base_instructions: str = DEFAULT_BASE_INSTRUCTIONS

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


@dataclass(slots=True)
class TextOutput:
    text: str
    annotations: list[dict[str, Any]]


@dataclass(slots=True)
class FunctionCall:
    call_id: str
    name: str
    arguments: str
    parsed_arguments: dict[str, Any] | None


@dataclass(slots=True)
class HostedToolCall:
    kind: str
    status: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ModelResult:
    text_outputs: list[TextOutput]
    function_calls: list[FunctionCall]
    hosted_tool_calls: list[HostedToolCall]
    usage: dict[str, Any] | None = None
    provider_response_id: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(output.text for output in self.text_outputs if output.text).strip()
