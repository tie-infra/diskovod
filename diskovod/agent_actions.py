from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from .agent_types import AgentRuntimeContext


DeliveryStatus = Literal["accepted", "failed", "ambiguous"]


@dataclass(frozen=True, slots=True)
class DeliveryRecord:
    status: DeliveryStatus
    message_index: int
    discord_message_id: str | None = None
    error_code: str | None = None
    error_detail: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and self.discord_message_id is not None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AgentActionGateway(Protocol):
    async def send_messages(
        self,
        context: AgentRuntimeContext,
        messages: tuple[str, ...],
        *,
        tool_call_id: str,
    ) -> list[DeliveryRecord]:
        """Deliver messages and return one stable record for every requested message."""
