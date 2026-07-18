import base64
import json
import time
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest

from diskovod.chatgpt import (
    CALLBACK_URL,
    ORIGINATOR,
    ChatGPTClient,
    make_prompt_cache_key,
    normalize_custom_base_url,
)
from diskovod.models import AppSettings, ChatCredentials, CustomProvider
from diskovod.store import Store


class FakeContent:
    def __init__(self, data: bytes):
        self.data = data

    async def iter_any(self):
        yield self.data


class FakeResponse:
    status = 200

    def __init__(self, data: bytes):
        self.content = FakeContent(data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.last_kwargs = None

    def post(self, *_args, **kwargs):
        self.last_args = _args
        self.last_kwargs = kwargs
        return self.response


class FakeJSONResponse(FakeResponse):
    def __init__(self, payload: object, status: int = 200):
        super().__init__(b"")
        self.payload = payload
        self.status = status

    async def json(self, *, content_type=None):
        return self.payload


def test_extracts_unverified_display_claims_only():
    payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                {
                    "email": "me@example.test",
                    "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
                }
            ).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    claims = ChatGPTClient._jwt_claims(f"header.{payload}.signature")
    assert claims["email"] == "me@example.test"
    assert claims["https://api.openai.com/auth"]["chatgpt_account_id"] == "acct_123"


def test_malformed_jwt_has_no_claims():
    assert ChatGPTClient._jwt_claims("not-a-jwt") == {}


def test_prompt_cache_keys_are_stable_and_do_not_expose_conversation_ids():
    first = make_prompt_cache_key("dm", "gpt-5.4-mini\0private-channel-123")
    second = make_prompt_cache_key("dm", "gpt-5.4-mini\0private-channel-123")

    assert first == second
    assert first.startswith("diskovod:dm:")
    assert "private-channel-123" not in first
    assert len(first) <= 64


def test_subscription_messages_use_native_image_and_file_inputs():
    message = ChatGPTClient._responses_message(
        {
            "role": "user",
            "content": "what do you think?",
            "attachments": [
                {
                    "filename": "photo.png",
                    "content_type": "image/png",
                    "size": 1024,
                    "url": "https://cdn.example/photo.png",
                },
                {
                    "filename": "brief.pdf",
                    "content_type": "application/pdf",
                    "size": 2048,
                    "url": "https://cdn.example/brief.pdf",
                },
            ],
        },
        "gpt-5.4-mini",
    )

    assert message["content"][0]["type"] == "input_text"
    assert message["content"][1] == {
        "type": "input_image",
        "image_url": "https://cdn.example/photo.png",
        "detail": "auto",
    }
    assert message["content"][2] == {
        "type": "input_file",
        "file_url": "https://cdn.example/brief.pdf",
    }


def test_custom_text_only_model_gets_bounded_retrieval_context():
    message = ChatGPTClient._custom_message(
        {
            "role": "user",
            "content": "review this",
            "attachments": [
                {
                    "filename": "sample.py",
                    "content_type": "text/x-python",
                    "size": 20,
                    "url": "https://cdn.example/sample.py",
                    "text": "print('hello')",
                }
            ],
        },
        "local-text-model",
    )

    assert message["role"] == "user"
    assert "sample.py (text/x-python, 20 bytes)" in message["content"]
    assert "print('hello')" in message["content"]


def test_custom_vision_model_uses_chat_completions_image_url_shape():
    message = ChatGPTClient._custom_message(
        {
            "role": "user",
            "content": "describe it",
            "attachments": [
                {
                    "filename": "photo.webp",
                    "content_type": "image/webp",
                    "size": 1024,
                    "url": "https://cdn.example/photo.webp",
                }
            ],
        },
        "gpt-4o-mini",
    )

    assert message["content"][0]["type"] == "text"
    assert message["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://cdn.example/photo.webp", "detail": "auto"},
    }


def test_extracts_usage_from_completed_response():
    usage = ChatGPTClient._usage_from_response(
        {
            "id": "resp_123",
            "model": "gpt-5",
            "usage": {
                "input_tokens": 120,
                "input_tokens_details": {"cached_tokens": 80},
                "output_tokens": 45,
                "output_tokens_details": {"reasoning_tokens": 15},
                "total_tokens": 165,
            },
        }
    )

    assert usage == {
        "response_id": "resp_123",
        "model": "gpt-5",
        "input_tokens": 120,
        "cached_input_tokens": 80,
        "output_tokens": 45,
        "reasoning_tokens": 15,
        "total_tokens": 165,
    }


def test_missing_usage_is_not_recorded():
    assert ChatGPTClient._usage_from_response({"id": "resp_123", "usage": None}) is None


@pytest.mark.asyncio
async def test_completed_stream_records_usage(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_chat_credentials(ChatCredentials("access", "refresh", time.time() + 3600, "account", None))
    events = [
        {"type": "response.output_text.delta", "delta": "Hello"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_stream",
                "model": "gpt-5-resolved",
                "usage": {
                    "input_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 12},
                    "output_tokens": 8,
                    "output_tokens_details": {"reasoning_tokens": 3},
                    "total_tokens": 28,
                },
            },
        },
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
    client = ChatGPTClient(store)
    session = FakeSession(FakeResponse(stream))
    client.session = cast(aiohttp.ClientSession, session)

    result = await client.complete(
        [],
        "instructions",
        "gpt-5",
        "low",
        purpose="dm_reply",
        max_output_tokens=256,
        cache_key="diskovod:dm:cache-key",
    )

    assert result == "Hello"
    stats = store.chatgpt_usage_stats()
    assert stats["all_time"]["total_tokens"] == 28
    assert stats["by_model"][0]["name"] == "gpt-5-resolved"
    assert stats["by_purpose"][0]["name"] == "dm_reply"
    assert session.last_kwargs["headers"]["Originator"] == ORIGINATOR
    assert "max_output_tokens" not in session.last_kwargs["json"]
    assert session.last_kwargs["json"]["prompt_cache_key"] == "diskovod:dm:cache-key"
    assert "within approximately 256 tokens" in session.last_kwargs["json"]["instructions"]
    store.close()


@pytest.mark.asyncio
async def test_oauth_uses_registered_codex_callback_url():
    client = ChatGPTClient(None)

    authorize_url = await client.begin_oauth()

    query = parse_qs(urlparse(authorize_url).query)
    assert query["redirect_uri"] == [CALLBACK_URL]
    assert query["originator"] == [ORIGINATOR]
    assert client.oauth.redirect_uri == CALLBACK_URL


@pytest.mark.parametrize(
    ("value", "normalized"),
    (
        ("https://models.example/v1/", "https://models.example/v1"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1"),
    ),
)
def test_custom_api_base_url_normalization(value: str, normalized: str):
    assert normalize_custom_base_url(value) == normalized


@pytest.mark.parametrize(
    "value",
    (
        "localhost:8000/v1",
        "ftp://models.example/v1",
        "https://user:secret@models.example/v1",
        "https://models.example/v1?key=secret",
        "https://models example/v1",
        "http://[::1/v1",
    ),
)
def test_custom_api_base_url_rejects_ambiguous_or_secret_urls(value: str):
    with pytest.raises(ValueError, match="API base URL"):
        normalize_custom_base_url(value)


@pytest.mark.asyncio
async def test_custom_provider_uses_chat_completions_and_records_usage(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_app_settings(AppSettings(provider="custom", model="local-model"))
    store.set_custom_provider(CustomProvider("Local", "http://localhost:8000/v1", "provider-key"))
    payload = {
        "id": "chatcmpl-local",
        "model": "resolved-model",
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {
            "prompt_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 5},
            "completion_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 27,
        },
    }
    client = ChatGPTClient(store)
    session = FakeSession(FakeJSONResponse(payload))
    client.session = cast(aiohttp.ClientSession, session)

    result = await client.complete(
        [{"role": "user", "content": "hi"}],
        "system instructions",
        "local-model",
        "high",
        purpose="dm_reply",
        max_output_tokens=192,
        cache_key="must-not-be-sent-to-custom-providers",
    )

    assert result == "hello"
    assert session.last_args == ("http://localhost:8000/v1/chat/completions",)
    assert session.last_kwargs["headers"]["Authorization"] == "Bearer provider-key"
    assert session.last_kwargs["json"] == {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "system instructions"},
            {"role": "user", "content": "hi"},
        ],
        "stream": False,
        "max_completion_tokens": 192,
    }
    stats = store.chatgpt_usage_stats()
    assert stats["all_time"]["total_tokens"] == 27
    assert stats["all_time"]["cached_input_tokens"] == 5
    assert stats["all_time"]["reasoning_tokens"] == 2
    store.close()


@pytest.mark.asyncio
async def test_keyless_custom_provider_omits_authorization_header(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_app_settings(AppSettings(provider="custom"))
    store.set_custom_provider(CustomProvider("Keyless", "http://localhost:8000/v1", ""))
    payload = {"choices": [{"message": {"content": "ok"}}]}
    client = ChatGPTClient(store)
    session = FakeSession(FakeJSONResponse(payload))
    client.session = cast(aiohttp.ClientSession, session)

    assert await client.complete([], "instructions", "model", "low") == "ok"
    assert "Authorization" not in session.last_kwargs["headers"]
    store.close()


@pytest.mark.asyncio
async def test_custom_provider_surfaces_api_errors(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_app_settings(AppSettings(provider="custom"))
    store.set_custom_provider(CustomProvider("Gateway", "https://models.example/v1", "key"))
    client = ChatGPTClient(store)
    client.session = cast(
        aiohttp.ClientSession,
        FakeSession(FakeJSONResponse({"error": {"message": "unknown model"}}, status=400)),
    )

    with pytest.raises(RuntimeError, match="Gateway returned HTTP 400: unknown model"):
        await client.complete([], "instructions", "missing-model", "low")

    assert client.last_error == "Gateway returned HTTP 400: unknown model"
    store.close()


def test_custom_usage_falls_back_to_prompt_plus_completion_total():
    usage = ChatGPTClient._usage_from_chat_completion(
        {"usage": {"prompt_tokens": 11, "completion_tokens": 4}}
    )

    assert usage["total_tokens"] == 15
