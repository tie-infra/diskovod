from __future__ import annotations

from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Any, NotRequired

from langchain.agents import AgentState


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    """Capabilities verified for one saved provider/model/transport selection."""

    native_tools: bool = True
    hosted_web_search: bool = False
    image_input: bool = False
    file_input: bool = False
    prompt_cache: bool = False
    standard_content_blocks: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentRuntimeContext:
    """Trusted invocation data that must not be persisted as conversation content."""

    account_id: str
    channel_id: str
    participant_ids: tuple[str, ...]
    owner_id: str
    ui_locale: str
    prompt_locale: str
    assistant_name: str
    automation_mode: str
    force_reply: bool
    provider_id: str
    model_id: str
    transport_profile: str
    capabilities: CapabilityProfile
    trace_id: str
    owner_timezone: str = "UTC"
    permissions: frozenset[str] = frozenset()


class DiskovodAgentState(AgentState):
    """Provider-neutral durable state for one Discord chat generation."""

    logical_request_id: NotRequired[str]
    claimed_event_ids: NotRequired[list[str]]
    successful_written_sends: NotRequired[Annotated[int, add]]
    terminate_after_send: NotRequired[bool]
    live_injection_batches: NotRequired[int]
    summary_metadata: NotRequired[dict[str, Any]]
