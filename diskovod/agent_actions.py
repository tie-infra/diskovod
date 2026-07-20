from __future__ import annotations

from typing import Literal, Protocol

from .agent_types import AgentRuntimeContext
from .outbound import DeliveryRecord


DeliveryStatus = Literal["accepted", "failed", "ambiguous"]


class AgentActionGateway(Protocol):
    async def send_messages(
        self,
        context: AgentRuntimeContext,
        messages: tuple[str, ...],
        *,
        tool_call_id: str,
    ) -> list[DeliveryRecord]:
        """Deliver messages and return one stable record for every requested message."""

    async def react_to_message(
        self,
        context: AgentRuntimeContext,
        emoji: str,
        *,
        tool_call_id: str,
    ) -> DeliveryRecord: ...

    async def record_escalation(
        self,
        context: AgentRuntimeContext,
        payload: dict[str, object],
        *,
        tool_call_id: str,
    ) -> None: ...
