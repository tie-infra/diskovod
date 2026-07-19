from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.codex import _ChatOpenAICodex
from langchain_openai.chatgpt_oauth import _ChatGPTToken

from diskovod.models import ChatCredentials

from .base import (
    ModelConfiguration,
    ProviderAdapter,
    ProviderBuildError,
    ProviderCredentials,
)


RESPONSES = "responses"
CHAT_COMPLETIONS = "chat_completions"


class OpenAIAdapter(ProviderAdapter):
    integration_package = "langchain-openai"
    transport_profiles = frozenset({RESPONSES, CHAT_COMPLETIONS})

    def __init__(self, provider_id: str = "openai", *, custom_endpoint: bool = False):
        self.id = provider_id
        self.custom_endpoint = custom_endpoint

    def validate(self, configuration: ModelConfiguration) -> None:
        super().validate(configuration)
        if self.custom_endpoint and not configuration.endpoint:
            raise ProviderBuildError("missing_endpoint", "A custom OpenAI endpoint is required")
        if not self.custom_endpoint and configuration.endpoint:
            raise ProviderBuildError(
                "unexpected_endpoint", "The official OpenAI adapter does not accept a custom endpoint"
            )

    def build_model(
        self,
        configuration: ModelConfiguration,
        credentials: ProviderCredentials,
    ) -> BaseChatModel:
        if credentials.api_key is None and not self.custom_endpoint:
            raise ProviderBuildError("missing_credentials", "An OpenAI API key is required")
        options = _model_options(configuration.options)
        prompt_cache_key = options.pop("prompt_cache_key", None)
        return ChatOpenAI(
            model=configuration.model_id,
            api_key=credentials.api_key or "diskovod-keyless-endpoint",
            base_url=configuration.endpoint,
            use_responses_api=configuration.transport_profile == RESPONSES,
            output_version=("responses/v1" if configuration.transport_profile == RESPONSES else None),
            max_retries=0,
            model_kwargs=({"prompt_cache_key": prompt_cache_key} if prompt_cache_key is not None else {}),
            **options,
        )


class ChatGPTSubscriptionAdapter(ProviderAdapter):
    id = "chatgpt_subscription"
    integration_package = "langchain-openai"
    transport_profiles = frozenset({RESPONSES})

    def validate(self, configuration: ModelConfiguration) -> None:
        super().validate(configuration)
        if configuration.endpoint:
            raise ProviderBuildError(
                "unexpected_endpoint", "ChatGPT Subscription uses its fixed Codex endpoint"
            )

    def build_model(
        self,
        configuration: ModelConfiguration,
        credentials: ProviderCredentials,
    ) -> BaseChatModel:
        if credentials.oauth_token_provider is None:
            raise ProviderBuildError(
                "missing_credentials", "Encrypted ChatGPT Subscription credentials are required"
            )
        options = _model_options(configuration.options)
        prompt_cache_key = options.pop("prompt_cache_key", None)
        return _ChatOpenAICodex(
            model=configuration.model_id,
            token_provider=credentials.oauth_token_provider,
            output_version="responses/v1",
            max_retries=0,
            originator="diskovod",
            model_kwargs=({"prompt_cache_key": prompt_cache_key} if prompt_cache_key is not None else {}),
            **options,
        )


class StoredChatGPTTokenProvider:
    """Refresh-aware bridge from Diskovod's encrypted store to LangChain OAuth."""

    def __init__(
        self,
        load: Callable[[], ChatCredentials | None],
        refresh: Callable[[], Awaitable[ChatCredentials]],
        *,
        refresh_skew: timedelta = timedelta(minutes=5),
    ):
        self._load = load
        self._refresh = refresh
        self._refresh_skew = refresh_skew
        self._lock = asyncio.Lock()

    def get_token(self) -> _ChatGPTToken:
        credentials = self._credentials()
        if self._needs_refresh(credentials):
            raise RuntimeError(
                "ChatGPT credentials require asynchronous refresh; use the async agent runtime"
            )
        return self._token(credentials)

    async def aget_token(self) -> _ChatGPTToken:
        credentials = self._credentials()
        if self._needs_refresh(credentials):
            async with self._lock:
                credentials = self._credentials()
                if self._needs_refresh(credentials):
                    credentials = await self._refresh()
        return self._token(credentials)

    def get_access_token(self) -> str:
        return self.get_token().access_token

    async def aget_access_token(self) -> str:
        return (await self.aget_token()).access_token

    def _credentials(self) -> ChatCredentials:
        credentials = self._load()
        if credentials is None:
            raise RuntimeError("ChatGPT Subscription is not connected")
        return credentials

    def _needs_refresh(self, credentials: ChatCredentials) -> bool:
        return datetime.now(UTC) >= datetime.fromtimestamp(credentials.expires_at, UTC) - self._refresh_skew

    @staticmethod
    def _token(credentials: ChatCredentials) -> _ChatGPTToken:
        return _ChatGPTToken(
            access_token=credentials.access_token,
            refresh_token=credentials.refresh_token,
            expires_at=datetime.fromtimestamp(credentials.expires_at, UTC),
            account_id=credentials.account_id,
        )


def _model_options(options: dict[str, object]) -> dict[str, object]:
    allowed = {
        "reasoning_effort",
        "max_completion_tokens",
        "timeout",
        "temperature",
        "prompt_cache_key",
    }
    unknown = set(options) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ProviderBuildError("unsupported_options", f"Unsupported provider options: {names}")
    return dict(options)
