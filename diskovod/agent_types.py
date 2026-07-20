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
    thread_id: str = ""
    owner_timezone: str = "UTC"
    trigger_message_id: str = ""
    permissions: frozenset[str] = frozenset()


class DiskovodAgentState(AgentState):
    """Provider-neutral durable state for one Discord chat generation."""

    logical_request_id: NotRequired[str]
    claimed_event_ids: NotRequired[Annotated[list[str], add]]
    outbound_delivery_count: NotRequired[Annotated[int, add]]
    model_call_count: NotRequired[int]
    tool_call_count: NotRequired[int]
    counter_run_id: NotRequired[str]
    reaction_target_message_id: NotRequired[str]
    model_step_route: NotRequired[str]
    live_injection_batches: NotRequired[Annotated[int, add]]
    summary_metadata: NotRequired[dict[str, Any]]
