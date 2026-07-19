from __future__ import annotations

import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

from diskovod.models import ChatCredentials, CustomProvider
from diskovod.providers import (
    ChatGPTSubscriptionAdapter,
    ModelConfiguration,
    OpenAIAdapter,
    ProviderAdapter,
    ProviderBuildError,
    ProviderCapabilities,
    ProviderCredentials,
    ProviderRegistry,
    ProviderSetup,
    ModelService,
    StoredChatGPTTokenProvider,
)
from diskovod.store import Store
from test_agent import ScriptedChatModel


class StreamingChatModel(ScriptedChatModel):
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        del messages, stop, run_manager, kwargs
        response = self.responses[self.index]
        self.index += 1
        yield ChatGenerationChunk(message=AIMessageChunk(content=response.content))


class FakeAdapter(ProviderAdapter):
    id = "fake"
    integration_package = "test"
    transport_profiles = frozenset({"native"})

    def __init__(self, model: BaseChatModel):
        self.model = model

    def build_model(self, configuration, credentials):
        return self.model


def configuration(
    provider_id: str,
    transport: str,
    *,
    endpoint: str | None = None,
    options: dict | None = None,
    capabilities: ProviderCapabilities | None = None,
) -> ModelConfiguration:
    return ModelConfiguration(
        provider_id=provider_id,
        model_id="test-model",
        transport_profile=transport,
        credential_profile="default",
        endpoint=endpoint,
        options=options or {},
        capabilities=capabilities or ProviderCapabilities(),
        integration_version="test",
    )


def test_registry_is_provider_neutral_and_never_falls_back():
    model = ChatOpenAI(model="test-model", api_key="test", max_retries=0)
    registry = ProviderRegistry([FakeAdapter(model)])

    assert registry.available() == ("fake",)
    assert registry.build_model(configuration("fake", "native"), ProviderCredentials()) is model
    with pytest.raises(ProviderBuildError) as unknown:
        registry.build_model(configuration("missing", "native"), ProviderCredentials())
    assert unknown.value.code == "unknown_provider"
    with pytest.raises(ProviderBuildError) as transport:
        registry.build_model(configuration("fake", "other"), ProviderCredentials())
    assert transport.value.code == "unsupported_transport"


def test_openai_adapter_pins_saved_transport_and_disables_retries():
    adapter = OpenAIAdapter()
    responses = adapter.build_model(
        configuration("openai", "responses"), ProviderCredentials(api_key="secret")
    )
    chat = adapter.build_model(
        configuration("openai", "chat_completions"), ProviderCredentials(api_key="secret")
    )

    assert isinstance(responses, BaseChatModel)
    assert responses.use_responses_api is True
    assert responses.max_retries == 0
    assert chat.use_responses_api is False
    assert chat.max_retries == 0


def test_public_responses_adapter_preserves_output_token_limit():
    model = OpenAIAdapter().build_model(
        configuration(
            "openai",
            "responses",
            options={"max_completion_tokens": 512},
        ),
        ProviderCredentials(api_key="secret"),
    )

    payload = model._get_request_payload([HumanMessage("hello")])

    assert payload["max_output_tokens"] == 512


def test_custom_openai_adapter_requires_and_preserves_endpoint():
    adapter = OpenAIAdapter("custom_openai", custom_endpoint=True)
    selected = configuration("custom_openai", "responses", endpoint="https://models.example/v1")
    adapter.validate(selected)
    model = adapter.build_model(selected, ProviderCredentials(api_key="secret"))

    assert str(model.openai_api_base).rstrip("/") == "https://models.example/v1"
    with pytest.raises(ProviderBuildError, match="endpoint"):
        adapter.validate(configuration("custom_openai", "responses"))


def test_custom_openai_adapter_supports_keyless_local_endpoints():
    adapter = OpenAIAdapter("custom_openai", custom_endpoint=True)
    model = adapter.build_model(
        configuration("custom_openai", "responses", endpoint="http://localhost:8000/v1"),
        ProviderCredentials(api_key=""),
    )

    assert isinstance(model, BaseChatModel)
    assert model.openai_api_key.get_secret_value() == "diskovod-keyless-endpoint"


def test_provider_options_are_allowlisted_instead_of_forwarded_blindly():
    adapter = OpenAIAdapter()
    with pytest.raises(ProviderBuildError) as error:
        adapter.build_model(
            configuration("openai", "responses", options={"dangerous": True}),
            ProviderCredentials(api_key="secret"),
        )
    assert error.value.code == "unsupported_options"


@pytest.mark.asyncio
async def test_subscription_token_provider_refreshes_from_encrypted_store_callback():
    current = ChatCredentials("old", "refresh", time.time() - 10, "account", None)
    refreshes = 0

    def load():
        return current

    async def refresh():
        nonlocal current, refreshes
        refreshes += 1
        current = ChatCredentials("new", "refresh", time.time() + 3600, "account", None)
        return current

    provider = StoredChatGPTTokenProvider(load, refresh, refresh_skew=timedelta(seconds=0))
    token = await provider.aget_token()

    assert token.access_token == "new"
    assert token.account_id == "account"
    assert refreshes == 1
    assert await provider.aget_access_token() == "new"
    assert refreshes == 1


def test_subscription_adapter_is_responses_only_and_uses_private_surface_narrowly():
    credentials = ChatCredentials("access", "refresh", time.time() + 3600, "account", None)

    async def refresh():
        return credentials

    token_provider = StoredChatGPTTokenProvider(lambda: credentials, refresh)
    adapter = ChatGPTSubscriptionAdapter()
    selected = configuration(
        "chatgpt_subscription",
        "responses",
        options={"reasoning_effort": "low"},
        capabilities=ProviderCapabilities(output_token_limit=False),
    )
    model = adapter.build_model(selected, ProviderCredentials(oauth_token_provider=token_provider))
    payload = model._get_request_payload([HumanMessage("hello")])

    assert isinstance(model, BaseChatModel)
    assert model.max_retries == 0
    assert model.use_responses_api is True
    assert "max_output_tokens" not in payload
    with pytest.raises(ProviderBuildError, match="output token limit"):
        adapter.build_model(
            configuration(
                "chatgpt_subscription",
                "responses",
                options={"max_completion_tokens": 256},
                capabilities=ProviderCapabilities(output_token_limit=False),
            ),
            ProviderCredentials(oauth_token_provider=token_provider),
        )
    with pytest.raises(ProviderBuildError) as transport:
        adapter.validate(configuration("chatgpt_subscription", "chat_completions"))
    assert transport.value.code == "unsupported_transport"


async def test_subscription_configuration_omits_unsupported_token_limit(tmp_path):
    store = Store(tmp_path / "diskovod.sqlite3", "x" * 32)
    models = ModelService(store, SimpleNamespace(connected=True))

    await models.asave_subscription(
        model_id="test-model",
        reasoning_effort="low",
        max_output_tokens=256,
    )

    configuration = models.configuration
    assert configuration.options == {"reasoning_effort": "low"}
    assert configuration.capabilities.output_token_limit is False
    await store.aclose()


async def test_model_configuration_round_trips_as_an_immutable_active_version(tmp_path):
    store = Store(tmp_path / "diskovod.sqlite3", "x" * 32)
    first = configuration("openai", "responses")
    second = configuration("custom_openai", "chat_completions", endpoint="https://example.test/v1")

    first_id = await store.asave_agent_configuration(first)
    second_id = await store.asave_agent_configuration(second)

    assert second_id > first_id
    assert store.active_agent_configuration() == second
    async with store.database.transaction() as connection:
        versions = await (
            await connection.execute("SELECT active FROM agent_configuration_versions")
        ).fetchall()
    assert sum(int(row["active"]) for row in versions) == 1
    await store.aclose()


async def test_provider_credentials_are_encrypted_and_profile_scoped(tmp_path):
    store = Store(tmp_path / "diskovod.sqlite3", "x" * 32)
    await store.aset_provider_credentials("custom_openai_default", {"api_key": "secret-value"})

    assert store.provider_credentials("custom_openai_default") == {"api_key": "secret-value"}
    async with store.database.transaction() as connection:
        raw = await (
            await connection.execute(
                "SELECT value, secret FROM config WHERE key='provider.credentials.custom_openai_default'"
            )
        ).fetchone()
    assert raw["secret"] == 1
    assert "secret-value" not in raw["value"]
    await store.aclose()


async def test_prompt_cache_identity_is_shared_by_configuration_and_rotates_with_personality(tmp_path):
    store = Store(tmp_path / "diskovod.sqlite3", "x" * 32)
    models = ModelService(store, SimpleNamespace(connected=False))
    provider = CustomProvider(
        "Local",
        "http://localhost:8000/v1",
        "",
        "responses",
        {
            "native_function_calls": True,
            "prompt_cache_key": True,
            "output_token_limit": True,
        },
    )

    await models.asave_custom_openai(
        provider,
        model_id="local-model",
        reasoning_effort="low",
        max_output_tokens=256,
    )
    first = models.configuration.options["prompt_cache_key"]
    assert models.configuration.options["max_completion_tokens"] == 256
    assert models.configuration.capabilities.output_token_limit is True
    await store.aset_personality("A durable style profile", "personality-v2")
    assert await models.arefresh_prompt_cache_identity() is not None
    second = models.configuration.options["prompt_cache_key"]

    assert first.startswith("diskovod:")
    assert second.startswith("diskovod:")
    assert first != second
    model = models.build_model()
    assert model.model_kwargs["prompt_cache_key"] == second
    await store.aclose()


async def test_custom_provider_without_token_limit_capability_omits_option(tmp_path):
    store = Store(tmp_path / "diskovod.sqlite3", "x" * 32)
    models = ModelService(store, SimpleNamespace(connected=False))
    provider = CustomProvider(
        "Limited",
        "http://localhost:8000/v1",
        "",
        "responses",
        {"native_function_calls": True, "output_token_limit": False},
    )

    await models.asave_custom_openai(
        provider,
        model_id="local-model",
        reasoning_effort="low",
        max_output_tokens=256,
    )

    assert "max_completion_tokens" not in models.configuration.options
    assert models.configuration.capabilities.output_token_limit is False
    await store.aclose()


@pytest.mark.asyncio
async def test_provider_probe_capture_includes_v3_stream_events(tmp_path):
    store = Store(tmp_path / "diskovod.sqlite3", "x" * 32)
    setup = ProviderSetup(store, SimpleNamespace())
    model = StreamingChatModel(responses=[AIMessage(content="probe complete")])

    response, events = await setup._invoke_with_events(model, [HumanMessage("probe")])

    assert "probe complete" in response.text
    assert events
    assert any(event.get("event") == "content-block-delta" for event in events)
    await store.aclose()
