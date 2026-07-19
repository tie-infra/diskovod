from __future__ import annotations

from importlib.metadata import version

from langchain_core.language_models.chat_models import BaseChatModel

from diskovod.models import CustomProvider
from diskovod.oauth import ChatGPTAccount
from diskovod.store import Store

from .base import (
    ModelConfiguration,
    ProviderBuildError,
    ProviderCapabilities,
    ProviderCredentials,
    ProviderRegistry,
)
from .openai import ChatGPTSubscriptionAdapter, OpenAIAdapter


class ModelService:
    """Resolve one immutable saved model configuration; never choose a fallback."""

    def __init__(self, store: Store, account: ChatGPTAccount):
        self.store = store
        self.account = account
        self.registry = ProviderRegistry(
            [
                ChatGPTSubscriptionAdapter(),
                OpenAIAdapter(),
                OpenAIAdapter("custom_openai", custom_endpoint=True),
            ]
        )

    @property
    def configuration(self) -> ModelConfiguration | None:
        return self.store.active_agent_configuration()

    @property
    def ready(self) -> bool:
        configuration = self.configuration
        if configuration is None:
            return False
        try:
            self._credentials(configuration)
            self.registry.adapter(configuration.provider_id).validate(configuration)
        except (ProviderBuildError, RuntimeError):
            return False
        return True

    @property
    def provider_label(self) -> str:
        configuration = self.configuration
        if configuration is None:
            return "Not configured"
        if configuration.provider_id == "chatgpt_subscription":
            return "ChatGPT subscription"
        if configuration.provider_id == "custom_openai":
            custom = self.store.custom_provider()
            return custom.name if custom else "Custom OpenAI-compatible API"
        return "OpenAI API"

    def build_model(self) -> BaseChatModel:
        configuration = self.configuration
        if configuration is None:
            raise ProviderBuildError("not_configured", "No model configuration has been saved")
        return self.registry.build_model(configuration, self._credentials(configuration))

    def build_configuration(
        self,
        configuration: ModelConfiguration,
        credentials: ProviderCredentials,
    ) -> BaseChatModel:
        return self.registry.build_model(configuration, credentials)

    def credentials_for(self, configuration: ModelConfiguration) -> ProviderCredentials:
        return self._credentials(configuration)

    def save_subscription(
        self,
        *,
        model_id: str,
        reasoning_effort: str,
        max_output_tokens: int,
        capabilities: ProviderCapabilities | None = None,
    ) -> int:
        if not self.account.connected:
            raise ProviderBuildError("missing_credentials", "ChatGPT Subscription is not connected")
        return self.store.save_agent_configuration(
            ModelConfiguration(
                provider_id="chatgpt_subscription",
                model_id=model_id,
                transport_profile="responses",
                credential_profile="chatgpt_subscription",
                options={
                    "reasoning_effort": reasoning_effort,
                    "max_completion_tokens": max_output_tokens,
                },
                capabilities=capabilities or ProviderCapabilities(),
                integration_version=version("langchain-openai"),
            )
        )

    def save_custom_openai(
        self,
        provider: CustomProvider,
        *,
        model_id: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> int:
        profile = "custom_openai_default"
        self.store.set_provider_credentials(profile, {"api_key": provider.api_key})
        capabilities = ProviderCapabilities(
            native_tools=provider.supports("native_function_calls"),
            hosted_web_search=provider.supports("hosted_web_search"),
            image_input=provider.supports("image_input"),
            file_input=provider.supports("file_input"),
            prompt_cache=provider.supports("prompt_cache_key"),
            details={"setup_probe": dict(provider.capabilities)},
        )
        return self.store.save_agent_configuration(
            ModelConfiguration(
                provider_id="custom_openai",
                model_id=model_id,
                transport_profile=provider.protocol,
                credential_profile=profile,
                endpoint=provider.base_url,
                options={
                    "reasoning_effort": reasoning_effort,
                    "max_completion_tokens": max_output_tokens,
                },
                capabilities=capabilities,
                integration_version=version("langchain-openai"),
            )
        )

    def migrate_legacy_selection(self) -> int | None:
        """One-time transformer for installations created before configuration versions."""
        if self.configuration is not None:
            return None
        settings = self.store.app_settings()
        if settings.provider == "custom":
            provider = self.store.custom_provider()
            if provider is None:
                return None
            return self.save_custom_openai(
                provider,
                model_id=settings.model,
                reasoning_effort=settings.reasoning_effort,
                max_output_tokens=settings.max_reply_tokens,
            )
        if self.account.connected:
            capabilities = ProviderCapabilities(
                native_tools=True,
                hosted_web_search=(self.store.subscription_web_search_capability(settings.model) is True),
            )
            return self.save_subscription(
                model_id=settings.model,
                reasoning_effort=settings.reasoning_effort,
                max_output_tokens=settings.max_reply_tokens,
                capabilities=capabilities,
            )
        return None

    def _credentials(self, configuration: ModelConfiguration) -> ProviderCredentials:
        if configuration.provider_id == "chatgpt_subscription":
            from .openai import StoredChatGPTTokenProvider

            return ProviderCredentials(
                oauth_token_provider=StoredChatGPTTokenProvider(
                    self.store.chat_credentials,
                    self.account.credentials,
                )
            )
        stored = self.store.provider_credentials(configuration.credential_profile) or {}
        api_key = stored.get("api_key")
        if not isinstance(api_key, str) or not api_key:
            raise ProviderBuildError("missing_credentials", "The selected API key is unavailable")
        return ProviderCredentials(api_key=api_key)
