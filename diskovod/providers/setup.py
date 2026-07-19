from __future__ import annotations

import json
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core._api import LangChainBetaWarning

from diskovod.store import Store
from diskovod.localization import tool_text

from .base import ModelConfiguration, ProviderCapabilities, ProviderCredentials
from .service import ModelService


def normalize_custom_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    try:
        parsed = urlparse(base_url)
        hostname = parsed.hostname
        parsed.port
    except ValueError as error:
        raise ValueError("Invalid API base URL") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in base_url)
    ):
        raise ValueError("Invalid API base URL")
    return base_url


@dataclass(frozen=True, slots=True)
class CapabilityProbe:
    id: str
    capability: str
    supported: bool
    conclusion: str
    response: dict[str, Any] | None


class ProviderSetup:
    """Explicit setup probes. Results never change a live configuration implicitly."""

    def __init__(self, store: Store, models: ModelService):
        self.store = store
        self.models = models

    async def probe_client_tools(
        self,
        configuration: ModelConfiguration,
        credentials: ProviderCredentials,
    ) -> CapabilityProbe:
        text = tool_text(self.store.app_settings().prompt_locale)

        @tool("diskovod_setup_probe", description=text["connection_test_tool"])
        def diskovod_setup_probe(value: str) -> str:
            return value

        model = self.models.build_configuration(configuration, credentials)
        prompt = f"{text['connection_test_system']} {text['connection_test_tool']} value=ready."
        request = {
            "messages": [{"role": "user", "content": prompt}],
            "tool_choice": "required",
            "tools": ["diskovod_setup_probe"],
        }
        started = time.time()
        probe_id = str(uuid.uuid4())
        response_payload: dict[str, Any] | None = None
        try:
            response, events = await self._invoke_with_events(
                model.bind_tools(
                    [diskovod_setup_probe],
                    tool_choice="required",
                ),
                [HumanMessage(prompt)],
            )
            response_payload = {
                "message": self._message_payload(response),
                "langchain_events_v3": events,
                "raw_provider_transport": "unavailable",
            }
            supported = bool(
                isinstance(response, AIMessage)
                and len(response.tool_calls) == 1
                and response.tool_calls[0].get("name") == "diskovod_setup_probe"
                and response.tool_calls[0].get("args", {}).get("value") == "ready"
            )
            conclusion = "client_tool_call_verified" if supported else "expected_setup_tool_call_missing"
            status = "supported" if supported else "unsupported"
        except Exception as error:
            supported = False
            conclusion = f"{type(error).__name__}: {error}"[:4000]
            status = "error"
        self._record(
            probe_id,
            configuration,
            "native_tools",
            status,
            request,
            response_payload,
            conclusion,
            started,
        )
        return CapabilityProbe(probe_id, "native_tools", supported, conclusion, response_payload)

    async def probe_hosted_web_search(
        self,
        configuration: ModelConfiguration,
        credentials: ProviderCredentials,
    ) -> CapabilityProbe:
        model = self.models.build_configuration(configuration, credentials)
        text = tool_text(self.store.app_settings().prompt_locale)
        prompt = f"{text['web_test_system']} {text['web_test_input']}"
        request = {
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"type": "web_search"}],
        }
        started = time.time()
        probe_id = str(uuid.uuid4())
        response_payload: dict[str, Any] | None = None
        try:
            response, events = await self._invoke_with_events(
                model.bind_tools([{"type": "web_search"}]),
                [HumanMessage(prompt)],
            )
            response_payload = {
                "message": self._message_payload(response),
                "langchain_events_v3": events,
                "raw_provider_transport": "unavailable",
            }
            blocks = getattr(response, "content_blocks", [])
            block_types = {str(block.get("type")) for block in blocks if isinstance(block, dict)}
            supported = bool(
                block_types
                & {
                    "server_tool_call",
                    "server_tool_result",
                    "web_search_call",
                    "web_search_result",
                }
            )
            conclusion = (
                "hosted_web_search_observed"
                if supported
                else "no_hosted_search_blocks_in_normalized_response"
            )
            status = "supported" if supported else "unsupported"
        except Exception as error:
            supported = False
            conclusion = f"{type(error).__name__}: {error}"[:4000]
            status = "error"
        self._record(
            probe_id,
            configuration,
            "hosted_web_search",
            status,
            request,
            response_payload,
            conclusion,
            started,
        )
        return CapabilityProbe(
            probe_id,
            "hosted_web_search",
            supported,
            conclusion,
            response_payload,
        )

    def configuration_with_capability(
        self,
        configuration: ModelConfiguration,
        capability: str,
        supported: bool,
        probe_id: str,
    ) -> ModelConfiguration:
        values = configuration.capabilities.to_dict()
        if capability not in values or not isinstance(values[capability], bool):
            raise ValueError("Unknown boolean capability")
        values[capability] = supported
        values["probed_at"] = time.time()
        values["probe_trace_id"] = probe_id
        return ModelConfiguration(
            provider_id=configuration.provider_id,
            model_id=configuration.model_id,
            transport_profile=configuration.transport_profile,
            credential_profile=configuration.credential_profile,
            endpoint=configuration.endpoint,
            options=configuration.options,
            capabilities=ProviderCapabilities.from_dict(values),
            integration_version=configuration.integration_version,
        )

    def _record(
        self,
        probe_id: str,
        configuration: ModelConfiguration,
        capability: str,
        status: str,
        request: dict[str, Any],
        response: dict[str, Any] | None,
        conclusion: str,
        started: float,
    ) -> None:
        with self.store._lock, self.store._db:
            self.store._db.execute(
                """
                INSERT INTO provider_capability_probes(
                  id, configuration, capability, status, request_payload,
                  response_payload, conclusion, started_at, completed_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    probe_id,
                    json.dumps(configuration.to_dict(), ensure_ascii=False),
                    capability,
                    status,
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(response, ensure_ascii=False) if response is not None else None,
                    conclusion,
                    started,
                    time.time(),
                ),
            )

    @staticmethod
    def _message_payload(message: Any) -> dict[str, Any]:
        if hasattr(message, "model_dump"):
            payload = message.model_dump(mode="json")
            return payload if isinstance(payload, dict) else {"value": payload}
        return {"type": type(message).__name__, "content": str(message)[:20_000]}

    async def _invoke_with_events(self, runnable, messages) -> tuple[Any, list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LangChainBetaWarning)
            stream = await runnable.astream_events(messages, version="v3")
        async for event in stream:
            if len(events) < 500:
                value = self._event_value(event)
                events.append(value if isinstance(value, dict) else {"value": value})
        response = await stream
        return response, events

    @classmethod
    def _event_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value if not isinstance(value, str) else value[:20_000]
        if hasattr(value, "model_dump"):
            return cls._event_value(value.model_dump(mode="json"))
        if isinstance(value, dict):
            return {str(key): cls._event_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._event_value(item) for item in value[:200]]
        return str(value)[:20_000]
