import base64
import json
import time
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest

from diskovod.chatgpt import CALLBACK_URL, ORIGINATOR, ChatGPTClient
from diskovod.models import ChatCredentials
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
        self.last_kwargs = kwargs
        return self.response


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

    result = await client.complete([], "instructions", "gpt-5", "low", purpose="dm_reply")

    assert result == "Hello"
    stats = store.chatgpt_usage_stats()
    assert stats["all_time"]["total_tokens"] == 28
    assert stats["by_model"][0]["name"] == "gpt-5-resolved"
    assert stats["by_purpose"][0]["name"] == "dm_reply"
    assert session.last_kwargs["headers"]["Originator"] == ORIGINATOR
    store.close()


@pytest.mark.asyncio
async def test_oauth_uses_registered_codex_callback_url():
    client = ChatGPTClient(None)

    authorize_url = await client.begin_oauth()

    query = parse_qs(urlparse(authorize_url).query)
    assert query["redirect_uri"] == [CALLBACK_URL]
    assert query["originator"] == [ORIGINATOR]
    assert client.oauth.redirect_uri == CALLBACK_URL
