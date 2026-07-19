from __future__ import annotations

import hashlib
import json
from dataclasses import replace
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
        return self.store.save_agent_configuration(
            self._subscription_configuration(model_id, reasoning_effort, max_output_tokens, capabilities)
        )

    async def asave_subscription(
        self,
        *,
        model_id: str,
        reasoning_effort: str,
        max_output_tokens: int,
        capabilities: ProviderCapabilities | None = None,
    ) -> int:
        return await self.store.asave_agent_configuration(
            self._subscription_configuration(model_id, reasoning_effort, max_output_tokens, capabilities)
        )

    def _subscription_configuration(
        self,
        model_id: str,
        reasoning_effort: str,
        max_output_tokens: int,
        capabilities: ProviderCapabilities | None,
    ) -> ModelConfiguration:
        if not self.account.connected:
            raise ProviderBuildError("missing_credentials", "ChatGPT Subscription is not connected")
        resolved_capabilities = replace(
            capabilities or ProviderCapabilities(),
            output_token_limit=False,
        )
        return ModelConfiguration(
            provider_id="chatgpt_subscription",
            model_id=model_id,
            transport_profile="responses",
            credential_profile="chatgpt_subscription",
            options=self._completion_options(
                reasoning_effort,
                max_output_tokens,
                resolved_capabilities,
            ),
            capabilities=resolved_capabilities,
            integration_version=version("langchain-openai"),
        )

    def save_custom_openai(
        self,
        provider: CustomProvider,
        *,
        model_id: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> int:
        profile, configuration = self._custom_openai_configuration(
            provider, model_id, reasoning_effort, max_output_tokens
        )
        self.store.set_provider_credentials(profile, {"api_key": provider.api_key})
        return self.store.save_agent_configuration(configuration)

    async def asave_custom_openai(
        self,
        provider: CustomProvider,
        *,
        model_id: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> int:
        profile, configuration = self._custom_openai_configuration(
            provider, model_id, reasoning_effort, max_output_tokens
        )
        await self.store.aset_provider_credentials(profile, {"api_key": provider.api_key})
        return await self.store.asave_agent_configuration(configuration)

    def _custom_openai_configuration(
        self,
        provider: CustomProvider,
        model_id: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> tuple[str, ModelConfiguration]:
        profile = "custom_openai_default"
        capabilities = ProviderCapabilities(
            native_tools=provider.supports("native_function_calls"),
            hosted_web_search=provider.supports("hosted_web_search"),
            image_input=provider.supports("image_input"),
            file_input=provider.supports("file_input"),
            prompt_cache=provider.supports("prompt_cache_key"),
            output_token_limit=provider.supports("output_token_limit"),
            details={"setup_probe": dict(provider.capabilities)},
        )
        options = self._completion_options(reasoning_effort, max_output_tokens, capabilities)
        if capabilities.prompt_cache:
            options["prompt_cache_key"] = self._prompt_cache_key(
                provider_id="custom_openai",
                model_id=model_id,
                transport_profile=provider.protocol,
            )
        return profile, ModelConfiguration(
            provider_id="custom_openai",
            model_id=model_id,
            transport_profile=provider.protocol,
            credential_profile=profile,
            endpoint=provider.base_url,
            options=options,
            capabilities=capabilities,
            integration_version=version("langchain-openai"),
        )

    @staticmethod
    def _completion_options(
        reasoning_effort: str,
        max_output_tokens: int,
        capabilities: ProviderCapabilities,
    ) -> dict[str, object]:
        options: dict[str, object] = {"reasoning_effort": reasoning_effort}
        if capabilities.output_token_limit:
            options["max_completion_tokens"] = max_output_tokens
        return options

    def migrate_legacy_selection(self) -> int | None:
        """One-time transformer for installations created before configuration versions."""
        if self.configuration is not None:
            return None
        raw = self.store._get("app.settings", {})
        provider_id = str(raw.get("provider") or "chatgpt")
        model_id = str(raw.get("model") or "gpt-5.4-mini")
        effort = str(raw.get("reasoning_effort") or "low")
        if effort not in {"low", "medium", "high"}:
            effort = "low"
        try:
            max_output_tokens = max(32, min(int(raw.get("max_reply_tokens") or 256), 2048))
        except (TypeError, ValueError):
            max_output_tokens = 256
        if provider_id == "custom":
            provider = self.store.custom_provider()
            if provider is None:
                return None
            return self.save_custom_openai(
                provider,
                model_id=model_id,
                reasoning_effort=effort,
                max_output_tokens=max_output_tokens,
            )
        if self.account.connected:
            capabilities = ProviderCapabilities(
                native_tools=True,
                hosted_web_search=False,
            )
            return self.save_subscription(
                model_id=model_id,
                reasoning_effort=effort,
                max_output_tokens=max_output_tokens,
                capabilities=capabilities,
            )
        return None

    def refresh_prompt_cache_identity(self) -> int | None:
        configuration = self._refreshed_prompt_cache_configuration()
        if configuration is None:
            return None
        return self.store.save_agent_configuration(configuration)

    async def arefresh_prompt_cache_identity(self) -> int | None:
        configuration = self._refreshed_prompt_cache_configuration()
        if configuration is None:
            return None
        return await self.store.asave_agent_configuration(configuration)

    def _refreshed_prompt_cache_configuration(self) -> ModelConfiguration | None:
        configuration = self.configuration
        if configuration is None or not configuration.capabilities.prompt_cache:
            return None
        options = dict(configuration.options)
        options["prompt_cache_key"] = self._prompt_cache_key(
            provider_id=configuration.provider_id,
            model_id=configuration.model_id,
            transport_profile=configuration.transport_profile,
        )
        if options == configuration.options:
            return None
        return ModelConfiguration(
            provider_id=configuration.provider_id,
            model_id=configuration.model_id,
            transport_profile=configuration.transport_profile,
            credential_profile=configuration.credential_profile,
            endpoint=configuration.endpoint,
            options=options,
            capabilities=configuration.capabilities,
            integration_version=configuration.integration_version,
        )

    def _prompt_cache_key(self, **model_identity: str) -> str:
        settings = self.store.app_settings()
        personality = self.store.personality() or {}
        identity = {
            **model_identity,
            "locale": settings.prompt_locale,
            "assistant_name": settings.assistant_name,
            "base_instructions": settings.base_instructions,
            "owner_details": settings.owner_details,
            "personality_hash": personality.get("source_hash"),
            "tool_schema": "langgraph-v1",
        }
        digest = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"diskovod:{digest}"

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
        if not isinstance(api_key, str) or (not api_key and configuration.provider_id != "custom_openai"):
            raise ProviderBuildError("missing_credentials", "The selected API key is unavailable")
        return ProviderCredentials(api_key=api_key)
