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
    ProviderHTTPError,
    make_prompt_cache_key,
    normalize_custom_base_url,
)
from diskovod.models import AppSettings, ChatCredentials, CustomProvider
from diskovod.store import Store
from diskovod.tooling import WEB_SEARCH_TOOL


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


class SequenceSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = iter(responses)
        self.calls: list[tuple[tuple, dict]] = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return next(self.responses)


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


def test_request_log_error_redaction_removes_common_credentials():
    detail = ChatGPTClient._safe_request_error(
        RuntimeError("Bearer access.credential sk-secret access_token=private refresh_token:refresh")
    )

    assert "access.credential" not in detail
    assert "sk-secret" not in detail
    assert "private" not in detail
    assert ":refresh" not in detail
    assert detail.count("[redacted]") == 4


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
    store.set_custom_provider(
        CustomProvider("Local", "http://localhost:8000/v1", "provider-key", "chat_completions")
    )
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
    request_log = store.model_request_logs()[0]
    assert request_log["provider"] == "Local"
    assert request_log["protocol"] == "chat_completions"
    assert request_log["request_summary"]["messages"] == [
        {
            "role": "user",
            "content_characters": 2,
            "attachments": 0,
            "attachment_types": [],
        }
    ]
    assert request_log["request_summary"]["instructions_characters"] == len("system instructions")
    assert request_log["request_summary"]["cache_key_present"] is True
    assert request_log["response_summary"]["text_outputs"] == [{"characters": 5, "annotations": 0}]
    assert "system instructions" not in str(request_log)
    assert '"hi"' not in str(request_log)
    store.close()


@pytest.mark.asyncio
async def test_keyless_custom_provider_omits_authorization_header(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_app_settings(AppSettings(provider="custom"))
    store.set_custom_provider(CustomProvider("Keyless", "http://localhost:8000/v1", "", "chat_completions"))
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
    store.set_custom_provider(
        CustomProvider("Gateway", "https://models.example/v1", "key", "chat_completions")
    )
    client = ChatGPTClient(store)
    client.session = cast(
        aiohttp.ClientSession,
        FakeSession(FakeJSONResponse({"error": {"message": "unknown model"}}, status=400)),
    )

    with pytest.raises(RuntimeError, match="Gateway returned HTTP 400: unknown model"):
        await client.complete([], "instructions", "missing-model", "low")

    assert client.last_error == "Gateway returned HTTP 400: unknown model"
    request_log = store.model_request_logs()[0]
    assert request_log["status"] == "error"
    assert request_log["error_type"] == "ProviderHTTPError"
    assert request_log["error_detail"] == "Gateway returned HTTP 400: unknown model"
    store.close()


def test_custom_usage_falls_back_to_prompt_plus_completion_total():
    usage = ChatGPTClient._usage_from_chat_completion(
        {"usage": {"prompt_tokens": 11, "completion_tokens": 4}}
    )

    assert usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_custom_responses_provider_is_pinned_and_normalized(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_app_settings(AppSettings(provider="custom", model="responses-model"))
    store.set_custom_provider(
        CustomProvider("Responses", "https://models.example/v1", "provider-key", "responses")
    )
    payload = {
        "id": "resp-local",
        "model": "resolved-model",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
    }
    client = ChatGPTClient(store)
    session = FakeSession(FakeJSONResponse(payload))
    client.session = cast(aiohttp.ClientSession, session)

    result = await client.complete(
        [{"role": "user", "content": "hi"}],
        "system instructions",
        "responses-model",
        "low",
        max_output_tokens=128,
        cache_key="not-enabled-for-custom-yet",
    )

    assert result == "hello"
    assert session.last_args == ("https://models.example/v1/responses",)
    assert session.last_kwargs["json"] == {
        "model": "responses-model",
        "instructions": "system instructions",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ],
        "stream": False,
        "store": False,
        "max_output_tokens": 128,
    }
    store.close()


@pytest.mark.asyncio
async def test_custom_responses_preserves_hosted_search_and_function_tools(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_app_settings(AppSettings(provider="custom", model="responses-model"))
    store.set_custom_provider(
        CustomProvider(
            "Responses",
            "https://models.example/v1",
            "provider-key",
            "responses",
            {
                "native_function_calls": True,
                "strict_function_schemas": True,
                "hosted_web_search": True,
            },
        )
    )
    payload = {
        "id": "resp-web",
        "output": [
            {"type": "web_search_call", "status": "completed", "action": {"query": "latest"}},
            {
                "type": "function_call",
                "call_id": "send-call",
                "name": "send_messages",
                "arguments": '{"messages":["Latest: https://example.test"]}',
            },
        ],
    }
    client = ChatGPTClient(store)
    session = FakeSession(FakeJSONResponse(payload))
    client.session = cast(aiohttp.ClientSession, session)
    send_tool = {
        "type": "function",
        "name": "send_messages",
        "description": "Send",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    }

    result = await client.complete_result(
        [{"role": "user", "content": "what is latest?"}],
        "instructions",
        "responses-model",
        "low",
        tools=[send_tool, WEB_SEARCH_TOOL],
        tool_choice="required",
    )

    assert result.hosted_tool_calls[0].kind == "web_search_call"
    assert result.function_calls[0].name == "send_messages"
    assert session.last_kwargs["json"]["tools"] == [send_tool, WEB_SEARCH_TOOL]
    assert client.hosted_web_search_available is True
    store.close()


def test_chat_completions_never_enables_responses_hosted_search(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_app_settings(AppSettings(provider="custom"))
    store.set_custom_provider(
        CustomProvider(
            "Chat only",
            "https://models.example/v1",
            "key",
            "chat_completions",
            {"native_function_calls": True, "hosted_web_search": True},
        )
    )

    assert ChatGPTClient(store).hosted_web_search_available is False
    store.close()


@pytest.mark.asyncio
async def test_subscription_web_search_probe_is_model_scoped_and_requires_terminal_call(
    tmp_path: Path,
):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_chat_credentials(ChatCredentials("access", "refresh", time.time() + 3600, "account", None))
    completed = {
        "type": "response.completed",
        "response": {
            "id": "resp-probe",
            "output": [
                {
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"query": "OpenAI homepage"},
                },
                {
                    "type": "function_call",
                    "call_id": "probe-call",
                    "name": "connection_test",
                    "arguments": '{"ok":true}',
                },
            ],
        },
    }
    stream = f"data: {json.dumps(completed)}\n\n".encode()
    client = ChatGPTClient(store)
    session = FakeSession(FakeResponse(stream))
    client.session = cast(aiohttp.ClientSession, session)

    assert await client.detect_subscription_web_search("gpt-model", "low") is True
    assert store.subscription_web_search_capability("gpt-model") is True
    diagnostics = store.subscription_web_search_probe("gpt-model")["diagnostics"]
    request_log_id = diagnostics.pop("request_log_id")
    assert diagnostics == {
        "outcome": "verified",
        "response_id": "resp-probe",
        "response_text_present": False,
        "function_call_count": 1,
        "function_call_names": ["connection_test"],
        "connection_test_ok": True,
        "hosted_call_count": 1,
        "hosted_calls": [{"kind": "web_search_call", "status": "completed"}],
        "effort": "low",
    }
    request_log = store.model_request_logs()[0]
    assert request_log["id"] == request_log_id
    assert request_log["validation_status"] == "probe_verified"
    assert request_log["response_summary"]["hosted_tool_calls"] == [
        {"kind": "web_search_call", "status": "completed"}
    ]
    assert client.hosted_web_search_available is False
    store.set_app_settings(AppSettings(model="gpt-model"))
    assert client.hosted_web_search_available is True
    assert session.last_kwargs["json"]["tools"][0] == WEB_SEARCH_TOOL
    store.close()


@pytest.mark.asyncio
async def test_subscription_web_search_probe_records_inconclusive_response_diagnostics(
    tmp_path: Path,
):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_chat_credentials(ChatCredentials("access", "refresh", time.time() + 3600, "account", None))
    completed = {
        "type": "response.completed",
        "response": {
            "id": "resp-mismatch",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}],
        },
    }
    client = ChatGPTClient(store)
    client.session = cast(
        aiohttp.ClientSession,
        FakeSession(FakeResponse(f"data: {json.dumps(completed)}\n\n".encode())),
    )

    assert await client.detect_subscription_web_search("gpt-model", "medium") is False
    assert store.subscription_web_search_capability("gpt-model") is False
    diagnostics = store.subscription_web_search_probe("gpt-model")["diagnostics"]
    assert diagnostics["outcome"] == "response_mismatch"
    assert diagnostics["response_id"] == "resp-mismatch"
    assert diagnostics["response_text_present"] is True
    assert diagnostics["function_call_count"] == 0
    assert diagnostics["hosted_call_count"] == 0
    assert diagnostics["effort"] == "medium"
    request_log = store.model_request_logs()[0]
    assert request_log["validation_status"] == "probe_inconclusive"
    assert request_log["validation_detail"] == "response_mismatch"
    store.close()


@pytest.mark.asyncio
async def test_subscription_web_search_probe_records_request_errors_as_inconclusive(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_chat_credentials(ChatCredentials("access", "refresh", time.time() + 3600, "account", None))
    event = {"type": "error", "error": {"message": "backend tool unavailable"}}
    client = ChatGPTClient(store)
    client.session = cast(
        aiohttp.ClientSession,
        FakeSession(FakeResponse(f"data: {json.dumps(event)}\n\n".encode())),
    )

    with pytest.raises(RuntimeError, match="backend tool unavailable"):
        await client.detect_subscription_web_search("gpt-model", "high")

    assert store.subscription_web_search_capability("gpt-model") is None
    diagnostics = store.subscription_web_search_probe("gpt-model")["diagnostics"]
    assert diagnostics["outcome"] == "request_error"
    assert diagnostics["effort"] == "high"
    assert "backend tool unavailable" in diagnostics["error"]
    request_log = store.model_request_logs()[0]
    assert request_log["status"] == "error"
    assert request_log["validation_status"] == "probe_inconclusive"
    store.close()


@pytest.mark.asyncio
async def test_setup_detection_preselects_verified_responses_web_search(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    client = ChatGPTClient(store)
    session = SequenceSession(
        [
            FakeJSONResponse(
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "OK"}],
                        }
                    ]
                }
            ),
            FakeJSONResponse(
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "native-call",
                            "name": "connection_test",
                            "arguments": '{"ok":true}',
                        }
                    ]
                }
            ),
            FakeJSONResponse(
                {
                    "output": [
                        {"type": "web_search_call", "status": "completed"},
                        {
                            "type": "function_call",
                            "call_id": "web-call",
                            "name": "connection_test",
                            "arguments": '{"ok":true}',
                        },
                    ]
                }
            ),
        ]
    )
    client.session = cast(aiohttp.ClientSession, session)

    detection = await client.detect_custom_protocol(
        name="Gateway",
        base_url="https://models.example/v1",
        api_key="key",
        model="model",
    )

    assert detection.protocol == "responses"
    assert detection.native_function_calls is True
    assert detection.hosted_web_search is True
    assert len(session.calls) == 3
    assert store.custom_provider() is None
    store.close()


@pytest.mark.asyncio
async def test_setup_detection_may_select_chat_completions_on_conclusive_404(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    client = ChatGPTClient(store)
    session = SequenceSession(
        [
            FakeJSONResponse({"error": {"message": "missing"}}, status=404),
            FakeJSONResponse({"choices": [{"message": {"content": "OK"}}]}),
            FakeJSONResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "probe-call",
                                        "type": "function",
                                        "function": {
                                            "name": "connection_test",
                                            "arguments": '{"ok":true}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            ),
        ]
    )
    client.session = cast(aiohttp.ClientSession, session)

    detection = await client.detect_custom_protocol(
        name="Gateway",
        base_url="https://models.example/v1",
        api_key="key",
        model="model",
    )

    assert detection.protocol == "chat_completions"
    assert detection.native_function_calls is True
    assert detection.hosted_web_search is False
    assert [call[0][0] for call in session.calls] == [
        "https://models.example/v1/responses",
        "https://models.example/v1/chat/completions",
        "https://models.example/v1/chat/completions",
    ]
    assert store.custom_provider() is None
    store.close()


@pytest.mark.asyncio
async def test_setup_detection_does_not_cross_fallback_on_ambiguous_failure(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    client = ChatGPTClient(store)
    session = SequenceSession([FakeJSONResponse({"error": {"message": "temporary outage"}}, status=503)])
    client.session = cast(aiohttp.ClientSession, session)

    with pytest.raises(ProviderHTTPError, match="HTTP 503"):
        await client.detect_custom_protocol(
            name="Gateway",
            base_url="https://models.example/v1",
            api_key="key",
            model="model",
        )

    assert len(session.calls) == 1
    assert store.custom_provider() is None
    store.close()


@pytest.mark.asyncio
async def test_custom_chat_tools_and_continuation_use_protocol_native_shapes(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.set_app_settings(AppSettings(provider="custom"))
    store.set_custom_provider(
        CustomProvider(
            "Gateway",
            "https://models.example/v1",
            "key",
            "chat_completions",
            {"native_function_calls": True, "prompt_cache_key": True},
        )
    )
    payload = {
        "id": "chatcmpl-tools",
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "send-call",
                            "type": "function",
                            "function": {
                                "name": "send_messages",
                                "arguments": '{"messages":["done"]}',
                            },
                        }
                    ],
                }
            }
        ],
    }
    client = ChatGPTClient(store)
    session = FakeSession(FakeJSONResponse(payload))
    client.session = cast(aiohttp.ClientSession, session)
    tool = {
        "type": "function",
        "name": "send_messages",
        "description": "Send",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    }

    result = await client.complete_result(
        [{"role": "user", "content": "hi"}],
        "instructions",
        "model",
        "low",
        cache_key="shared-key",
        tools=[tool],
        tool_choice={"type": "function", "name": "send_messages"},
        continuation_items=[
            {
                "type": "function_call",
                "call_id": "time-call",
                "name": "get_current_datetime",
                "arguments": '{"timezone":null}',
            },
            {
                "type": "function_call_output",
                "call_id": "time-call",
                "output": '{"date":"2026-07-19"}',
            },
        ],
    )

    assert result.function_calls[0].parsed_arguments == {"messages": ["done"]}
    request = session.last_kwargs["json"]
    assert request["prompt_cache_key"] == "shared-key"
    assert "strict" not in request["tools"][0]["function"]
    assert "parallel_tool_calls" not in request
    assert request["tool_choice"] == {
        "type": "function",
        "function": {"name": "send_messages"},
    }
    assert [message["role"] for message in request["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    store.close()


def test_responses_parser_retains_structured_outputs_and_bounded_hosted_calls():
    payload = {
        "id": "resp-tools",
        "output": [
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {"query": "latest news", "ignored": ["raw result"]},
            },
            {
                "type": "function_call",
                "call_id": "send-call",
                "name": "send_messages",
                "arguments": '{"messages":["natural reply"]}',
            },
        ],
    }

    result = ChatGPTClient._model_result_from_response(payload)

    assert result.provider_response_id == "resp-tools"
    assert result.function_calls[0].parsed_arguments == {"messages": ["natural reply"]}
    assert result.hosted_tool_calls[0].kind == "web_search_call"
    assert result.hosted_tool_calls[0].metadata == {"action": {"query": "latest news"}}
