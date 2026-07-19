from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    native_tools: bool = True
    hosted_web_search: bool = False
    image_input: bool = False
    file_input: bool = False
    prompt_cache: bool = False
    standard_content_blocks: bool = True
    probed_at: float | None = None
    probe_trace_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderCapabilities:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    """Saved, immutable selection used for every call in one configuration version."""

    provider_id: str
    model_id: str
    transport_profile: str
    credential_profile: str
    endpoint: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)
    integration_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["capabilities"] = self.capabilities.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelConfiguration:
        fields = cls.__dataclass_fields__
        normalized = {key: item for key, item in value.items() if key in fields}
        capabilities = normalized.get("capabilities", {})
        if isinstance(capabilities, dict):
            normalized["capabilities"] = ProviderCapabilities.from_dict(capabilities)
        return cls(**normalized)


@dataclass(frozen=True, slots=True)
class ProviderCredentials:
    """Resolved secrets passed directly to one provider adapter, never graph state."""

    api_key: str | None = None
    oauth_token_provider: Any = None


class ProviderBuildError(RuntimeError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(detail)


class ProviderAdapter(ABC):
    id: str
    integration_package: str
    transport_profiles: frozenset[str]

    @abstractmethod
    def build_model(
        self,
        configuration: ModelConfiguration,
        credentials: ProviderCredentials,
    ) -> BaseChatModel:
        """Construct the exact saved model; implementations must not fall back."""

    def validate(self, configuration: ModelConfiguration) -> None:
        if configuration.provider_id != self.id:
            raise ProviderBuildError(
                "provider_mismatch",
                f"Configuration selects {configuration.provider_id!r}, not {self.id!r}",
            )
        if not configuration.model_id.strip():
            raise ProviderBuildError("missing_model", "A provider model ID is required")
        if configuration.transport_profile not in self.transport_profiles:
            raise ProviderBuildError(
                "unsupported_transport",
                f"Provider {self.id!r} does not support transport {configuration.transport_profile!r}",
            )


class ProviderRegistry:
    """Provider-neutral construction boundary used by the agent runtime."""

    def __init__(self, adapters: list[ProviderAdapter] | tuple[ProviderAdapter, ...] = ()):
        self._adapters: dict[str, ProviderAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ProviderAdapter) -> None:
        if not adapter.id or adapter.id in self._adapters:
            raise ValueError(f"Provider adapter {adapter.id!r} is already registered or invalid")
        self._adapters[adapter.id] = adapter

    def adapter(self, provider_id: str) -> ProviderAdapter:
        try:
            return self._adapters[provider_id]
        except KeyError as exc:
            raise ProviderBuildError(
                "unknown_provider", f"Provider adapter {provider_id!r} is not installed"
            ) from exc

    def build_model(
        self,
        configuration: ModelConfiguration,
        credentials: ProviderCredentials,
    ) -> BaseChatModel:
        adapter = self.adapter(configuration.provider_id)
        adapter.validate(configuration)
        return adapter.build_model(configuration, credentials)

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))
